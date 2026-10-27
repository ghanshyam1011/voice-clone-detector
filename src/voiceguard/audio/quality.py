"""Is this audio fit to be judged?  Sanitise it, then measure how much speech it holds.

Every transport hands raw audio to ``SessionManager.assess``. Real streams carry
things none of the models were trained on -- NaN/Inf from a bad decoder, a DC
offset, samples in 16-bit integer range instead of [-1, 1], a muted line, hold
music, line hum, a recording 60 dB too quiet. Judged blindly, several of these
produced the loudest alarm in the product (ESCALATE on a muted line) and two
crashed outright; others returned a clean ALLOW on one sample of audio.

``analyze`` is the single serving-side gate:

  1. sanitise   non-finite samples, int16-range scaling, DC offset (only when
                needed -- normal audio is returned untouched, bit for bit)
  2. speech     Silero-VAD (a small neural detector; white noise, a tone or a
                muted line contain 0 s of speech). Falls back to an energy /
                spectral-flatness heuristic if ``silero-vad`` is not installed.
  3. level      speech far below the loudness meter's absolute gate would reach
                the (frozen) front-end un-normalised, so it is brought up first.
  4. verdict    ``usable`` + a plain-English reason, and how much speech there is.

It never changes the evaluated front-end (``voiceguard.audio.preprocess``): for
audio in the normal range this module is the identity.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import numpy as np

SR = 16000

MIN_SPEECH_S = 1.0  # less speech than this -> no read at all (WAITING)
FULL_SPEECH_S = 2.0  # less than this -> a provisional read (never ESCALATE on its own)
MIN_ENROL_SPEECH_S = 3.0  # an enrolment needs real speech; ~6 s of clean speech is best
LOW_LEVEL_DBFS = -50.0  # speech RMS below this is raised before the front-end sees it
TARGET_LEVEL_DBFS = -30.0
_DC_LIMIT = 0.005
_INT16_PEAK = 32768.0
_BAD_FRACTION = 0.2  # more non-finite samples than this -> the stream is corrupt

_vad_model = None
_vad_lock = threading.Lock()


@dataclass
class InputQuality:
    usable: bool  # enough real speech to make an acoustic judgement
    status: str  # ok | no_audio | no_speech | too_short
    reason: str  # plain English, empty when ok
    duration_s: float
    speech_s: float
    level_dbfs: float  # RMS of the speech (whole clip if none was found)
    clipped_frac: float
    method: str  # "silero" | "energy" -- how speech was found
    notes: list[str] = field(default_factory=list)  # fixes applied / things worth knowing

    @property
    def provisional(self) -> bool:
        """Usable, but with so little speech the read must not stand alone."""
        return self.usable and self.speech_s < FULL_SPEECH_S

    def as_dict(self) -> dict:
        return {
            "usable": self.usable,
            "status": self.status,
            "reason": self.reason,
            "duration_s": round(self.duration_s, 2),
            "speech_s": round(self.speech_s, 2),
            "level_dbfs": round(self.level_dbfs, 1),
            "clipped_frac": round(self.clipped_frac, 4),
            "method": self.method,
            "notes": list(self.notes),
        }


def _rms_dbfs(y: np.ndarray) -> float:
    if len(y) == 0:
        return -120.0
    return float(10 * np.log10(np.mean(np.square(y, dtype=np.float64)) + 1e-12))


def sanitise(y: np.ndarray) -> tuple[np.ndarray, list[str], bool]:
    """-> (audio, notes, corrupt). Touches the signal only if it needs fixing."""
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    notes: list[str] = []
    if len(y) == 0:
        return y, notes, False
    bad = ~np.isfinite(y)
    if bad.any():
        if bad.mean() > _BAD_FRACTION:
            return np.zeros_like(y), ["audio is mostly NaN/Inf (corrupt stream)"], True
        y = np.where(bad, 0.0, y).astype(np.float32)
        notes.append(f"{int(bad.sum())} non-finite samples replaced")
    peak = float(np.max(np.abs(y)))
    if peak > 1.5:
        if peak <= _INT16_PEAK * 1.001:
            y = y / _INT16_PEAK
            notes.append("samples were in 16-bit integer range; rescaled to [-1, 1]")
        else:
            y = y / peak
            notes.append("amplitude far outside [-1, 1]; normalised")
    dc = float(np.mean(y))
    if abs(dc) > _DC_LIMIT:
        y = (y - dc).astype(np.float32)
        notes.append(f"DC offset {dc:+.3f} removed")
    return np.ascontiguousarray(y, dtype=np.float32), notes, False


class PcmStream:
    """Incremental decoder for streamed little-endian float32 PCM (WebSocket / gRPC).

    A chunk boundary may split a 4-byte sample; the odd bytes wait for the next
    chunk instead of misaligning everything after them. NaN/Inf become silence so one
    bad sample cannot crash the models downstream."""

    def __init__(self) -> None:
        self._carry = b""

    def feed(self, data: bytes) -> np.ndarray:
        data = self._carry + bytes(data)
        n = len(data) // 4 * 4
        self._carry = data[n:]
        y = np.frombuffer(data[:n], dtype="<f4").astype(np.float32)
        if len(y) and not np.isfinite(y).all():
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        return y


def stream_rate(value, default: int = SR) -> int:
    """A client-declared sample rate, kept only if it is one we can resample from."""
    try:
        r = int(value)
    except (TypeError, ValueError):
        return default
    return r if 8000 <= r <= 192000 else default


def _load_vad():
    global _vad_model
    if _vad_model is None:
        try:
            from silero_vad import load_silero_vad

            _vad_model = load_silero_vad()
        except Exception:  # noqa: BLE001 -- optional dependency: fall back to the heuristic
            _vad_model = False
    return _vad_model


def _energy_segments(y: np.ndarray, sr: int) -> list[tuple[float, float]]:
    """Fallback detector: loud, non-flat 25 ms frames, merged across short gaps. A
    stretch only counts as speech if its loudness fluctuates (syllables) -- a steady
    tone (DTMF, ringback, hum) is loud and non-flat but is not speech."""
    n = 400
    if len(y) < 2 * n:
        return []
    fr = y[: len(y) // n * n].reshape(-1, n)
    db = 10 * np.log10(np.mean(fr**2, axis=1) + 1e-10)
    spec = np.abs(np.fft.rfft(fr * np.hanning(n), axis=1)) + 1e-9
    flat = np.exp(np.mean(np.log(spec), axis=1)) / np.mean(spec, axis=1)
    floor = np.percentile(db, 10)
    voiced = (db > floor + 12.0) & (flat < 0.35)  # relative: a quiet call must still be found
    runs, start = [], None
    for i, v in enumerate(voiced):
        if v and start is None:
            start = i
        if (not v or i == len(voiced) - 1) and start is not None:
            runs.append((start, i + (1 if v else 0)))
            start = None
    merged: list[list[int]] = []
    for s, e in runs:
        if merged and (s - merged[-1][1]) * n / sr < 0.3:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    keep = [
        (s * n / sr, e * n / sr)
        for s, e in merged
        if (e - s) * n / sr >= 0.2 and float(np.std(db[s:e])) >= 2.0
    ]
    return keep


_VAD_FRAME = 512  # Silero's frame at 16 kHz: 32 ms
_VAD_MIN_SEGMENT_FRAMES = 20  # never split finer than ~0.64 s per independent segment
_VAD_MAX_SEGMENTS = 16


def _vad_probs(y: np.ndarray, model) -> np.ndarray:
    """Speech probability per 32 ms frame.

    Silero is recurrent and is normally driven one frame at a time from Python, which
    costs ~1 ms of overhead per frame (0.3 s for a 10 s window). Here the window is cut
    into up to 16 independent segments that run in lockstep as one batch: ~16x fewer
    sequential model calls, speech seconds agreeing with Silero's own
    ``get_speech_timestamps`` to within ~0.1 s (tests/test_quality.py). Cost: the
    recurrent state restarts at each cut -- a few frames of context, irrelevant to
    "how many seconds of speech are in this window"."""
    import torch

    n = len(y) // _VAD_FRAME
    if n == 0:
        return np.zeros(0, np.float32)
    # Silero loses speech that is far below normal level; give the detector (only) a
    # usable copy. Normal-level audio is not touched. Noise stays noise when amplified.
    peak = float(np.percentile(np.abs(y), 99.5))
    if 1e-7 < peak < 0.05:
        y = (y * min(0.3 / peak, 1000.0)).astype(np.float32)
    k = int(max(1, min(_VAD_MAX_SEGMENTS, n // _VAD_MIN_SEGMENT_FRAMES)))
    per = -(-n // k)
    x = np.concatenate([y[: n * _VAD_FRAME], np.zeros((per * k - n) * _VAD_FRAME, np.float32)])
    xs = torch.from_numpy(x).reshape(k, per * _VAD_FRAME)
    with _vad_lock, torch.no_grad():
        model.reset_states()
        out = torch.stack(
            [
                model(xs[:, i * _VAD_FRAME : (i + 1) * _VAD_FRAME], SR).reshape(k)
                for i in range(per)
            ],
            dim=1,
        )
    return out.reshape(-1).numpy()[:n]


def _segments_from_probs(
    p: np.ndarray,
    thr: float = 0.5,
    neg: float = 0.35,
    min_speech: float = 0.25,
    min_gap: float = 0.1,
    pad: float = 0.03,
) -> list[tuple[float, float]]:
    """Silero's own hysteresis: a frame turns speech on above ``thr`` and off below ``neg``."""
    fs = _VAD_FRAME / SR
    on, runs, start = False, [], None
    for i, v in enumerate(p):
        if v >= thr:
            on = True
        elif v < neg:
            on = False
        if on and start is None:
            start = i
        if start is not None and (not on or i == len(p) - 1):
            runs.append([start * fs, (i + (1 if on else 0)) * fs])
            start = None
    merged: list[list[float]] = []
    for a, b in runs:
        if merged and a - merged[-1][1] < min_gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(max(0.0, a - pad), b + pad) for a, b in merged if b - a >= min_speech]


def speech_segments(y: np.ndarray, sr: int = SR) -> tuple[list[tuple[float, float]], str]:
    """Speech intervals in seconds, and which detector found them."""
    y = np.ascontiguousarray(y, dtype=np.float32)
    model = _load_vad()
    if model and sr == SR:
        return _segments_from_probs(_vad_probs(y, model)), "silero"
    return _energy_segments(y, sr), "energy"


def _speech_samples(y: np.ndarray, segs: list[tuple[float, float]], sr: int) -> np.ndarray:
    if not segs:
        return y
    return np.concatenate([y[int(s * sr) : int(e * sr)] for s, e in segs])


def analyze(
    y: np.ndarray, sr: int = SR, *, window_s: float | None = None
) -> tuple[np.ndarray, InputQuality, list[tuple[float, float]]]:
    """-> (conditioned audio, quality report, speech segments of the analysed window).

    ``window_s`` limits the speech measurement to the most recent seconds (what the
    models actually look at) so a long stream stays cheap; the returned audio is the
    full sanitised input. Segment times are relative to the start of that window."""
    y, notes, corrupt = sanitise(y)
    dur = len(y) / sr
    if len(y) == 0 or corrupt:
        reason = "no audio received" if len(y) == 0 else notes[0]
        return y, InputQuality(False, "no_audio", reason, dur, 0.0, -120.0, 0.0, "-", notes), []

    clipped = float(np.mean(np.abs(y) >= 0.99))
    win = y if window_s is None else y[-int(window_s * sr) :]
    segs, method = speech_segments(win, sr)
    speech_s = float(sum(e - s for s, e in segs))
    level = _rms_dbfs(_speech_samples(win, segs, sr))

    if segs and level < LOW_LEVEL_DBFS:
        gain = 10 ** ((TARGET_LEVEL_DBFS - level) / 20)
        y = np.clip(y * gain, -1.0, 1.0).astype(np.float32)
        notes.append(f"very quiet audio ({level:.0f} dBFS); raised to {TARGET_LEVEL_DBFS:.0f} dBFS")
        level = TARGET_LEVEL_DBFS
    if clipped > 0.02:
        notes.append(f"{clipped:.1%} of samples are clipped")

    if speech_s < 0.25:
        status, usable = "no_speech", False
        reason = "no speech detected (silence, noise or music only)"
    elif speech_s < MIN_SPEECH_S:
        status, usable = "too_short", False
        reason = f"only {speech_s:.1f} s of speech so far (need {MIN_SPEECH_S:.1f} s)"
    else:
        status, usable, reason = "ok", True, ""
    quality = InputQuality(usable, status, reason, dur, speech_s, level, clipped, method, notes)
    return y, quality, segs
