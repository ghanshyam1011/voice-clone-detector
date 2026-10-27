"""Acoustic situations a real call is recorded in, as reproducible degradations.

One definition shared by the *evaluation* (``scripts/build_system_trials.py``)
and the *training* augmentation (``scripts/finetune_robust.py``), so "what we
test on" and "what we train on" cannot drift apart.

Every function takes and returns float32 mono audio at 16 kHz. Degradations are
applied identically to genuine and attack audio by the callers -- never to one
class only (docs/evaluation_protocol.md) -- so "is it degraded?" cannot become a
shortcut. All backgrounds here are *synthetic* (noise, hum, tone sequences,
bursts): they exercise the failure modes, but they are not recordings of real
rooms, streets or phone networks.

Named situations (see ``apply_condition``):

    reverb        small-to-large room, direct + exponentially decaying tail
    speakerphone  reverb, then a narrow loudspeaker/mic band, plus room noise
    narrow        300-3400 Hz band-limiting (telephone band, no codec)
    clip          overdriven microphone / loud talker (hard clipping)
    quiet         very low recording level (far from the mic)
    loss          VoIP packet loss (20-60 ms dropouts)
    hum           50/60 Hz mains hum with harmonics
    music         music-like harmonic background
    burst         transient noise (door slams, keyboard clacks)
    amr, opus8k, g722, ulaw     telephony codecs (ffmpeg)
    opus16, opus24              messaging-app voice notes: Opus, 16 kHz, 16 / 24 kbps (ffmpeg)
    tel_babble    background voices, then an AMR-NB codec (a noisy room on a phone)
    tel_reverb    room reverb, then an AMR-NB codec
    babble<snr>, pink<snr>      background voices / broadband noise at <snr> dB
    first<N>      not a degradation: only the first N seconds of the call (duration study)
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.signal import butter, fftconvolve, sosfiltfilt

SR = 16000
Babble = Callable[[np.random.Generator, int], np.ndarray]

# Voice notes from messaging apps (WhatsApp and the like), not in the stress set above: added
# after a real voice note was called fake (docs/real_world_audio.md). The shipped detector saw
# Opus only at 8 kbps in training, so these are held-out for it.
MESSAGING_CONDITIONS = ("opus16", "opus24")

# Situations never seen in training by the shipped detector/fusion: the stress set.
STRESS_CONDITIONS = (
    "reverb",
    "speakerphone",
    "narrow",
    "clip",
    "quiet",
    "loss",
    "hum",
    "music",
    "burst",
    "amr",
    "opus8k",
    "g722",
    "tel_babble",
    "tel_reverb",
)


def rms(y: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y, dtype=np.float64)) + 1e-12))


def _f32(y: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(y, dtype=np.float32)


def peak_limit(y: np.ndarray, ceiling: float = 0.95) -> np.ndarray:
    """Scale down only if the signal would clip; never amplifies."""
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    return _f32(y * (ceiling / peak)) if peak > 0.99 else _f32(y)


def mix_at_snr(y: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """``y`` plus ``noise`` scaled so that 10*log10(P_y / P_noise) == snr_db."""
    noise = np.resize(np.asarray(noise, dtype=np.float32), len(y))
    g = rms(y) / (rms(noise) * 10 ** (snr_db / 20))
    return peak_limit(y + g * noise)


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    f = np.fft.rfftfreq(n)
    f[0] = f[1] if len(f) > 1 else 1.0
    spec = (rng.standard_normal(len(f)) + 1j * rng.standard_normal(len(f))) / np.sqrt(f)
    x = np.fft.irfft(spec, n)
    return _f32(x / (np.std(x) + 1e-9))


def synth_rir(
    rng: np.random.Generator,
    t60: float | None = None,
    drr_db: float | None = None,
    sr: int = SR,
) -> np.ndarray:
    """Synthetic room impulse response: a direct impulse plus Gaussian noise whose
    envelope falls 60 dB in ``t60`` seconds; ``drr_db`` is the direct-to-reverberant
    energy ratio (0 dB ~ talker a couple of metres away)."""
    t60 = float(rng.uniform(0.25, 0.9)) if t60 is None else t60
    drr_db = float(rng.uniform(-2.0, 8.0)) if drr_db is None else drr_db
    n = int(1.2 * t60 * sr)
    t = np.arange(n) / sr
    tail = rng.standard_normal(n) * np.exp(-6.9078 * t / t60)  # ln(1000) = 60 dB
    onset = int(0.003 * sr)  # first reflections arrive after the direct path
    tail[:onset] = 0.0
    e_tail = float(np.sum(tail**2)) + 1e-12
    rir = tail
    rir[0] = np.sqrt(e_tail * 10 ** (drr_db / 10))
    return _f32(rir / np.max(np.abs(rir)))


def reverberate(
    y: np.ndarray, rng: np.random.Generator, t60: float | None = None, drr_db: float | None = None
) -> np.ndarray:
    rir = synth_rir(rng, t60, drr_db)
    out = fftconvolve(y, rir)[: len(y)]
    out *= rms(y) / rms(out)  # reverb colours the signal; keep the loudness
    return peak_limit(out)


def band_limit(y: np.ndarray, lo: float = 300.0, hi: float = 3400.0, sr: int = SR) -> np.ndarray:
    sos = butter(4, [lo, hi], btype="band", fs=sr, output="sos")
    return _f32(sosfiltfilt(sos, y))


def hard_clip(y: np.ndarray, drive_db: float = 18.0, ceiling: float = 0.4) -> np.ndarray:
    return _f32(np.clip(y * 10 ** (drive_db / 20), -ceiling, ceiling))


def set_level(y: np.ndarray, level_dbfs: float) -> np.ndarray:
    """Scale so the RMS sits at ``level_dbfs`` (typical speech is about -25 dBFS)."""
    return _f32(y * (10 ** (level_dbfs / 20) / rms(y)))


def packet_loss(
    y: np.ndarray, rng: np.random.Generator, loss: float = 0.10, sr: int = SR
) -> np.ndarray:
    """Zero out random 20-60 ms stretches until about ``loss`` of the audio is gone."""
    out = y.copy()
    target = int(loss * len(y))
    gone = 0
    while gone < target:
        n = int(rng.uniform(0.02, 0.06) * sr)
        start = int(rng.integers(0, max(1, len(y) - n)))
        out[start : start + n] = 0.0
        gone += n
    return _f32(out)


def mains_hum(n: int, rng: np.random.Generator, sr: int = SR) -> np.ndarray:
    f0 = float(rng.choice([50.0, 60.0]))
    t = np.arange(n) / sr
    phase = rng.uniform(0, 2 * np.pi, 6)
    x = sum(np.sin(2 * np.pi * f0 * k * t + phase[k]) / k for k in range(1, 6))
    return _f32(x / (np.std(x) + 1e-9))


def music_like(n: int, rng: np.random.Generator, sr: int = SR) -> np.ndarray:
    """Chord sequence with harmonics and soft attack/decay: tonal, non-speech, wideband."""
    out = np.zeros(n, dtype=np.float64)
    i = 0
    scale = np.array([0, 2, 4, 7, 9, 12, 14, 16, 19])  # major pentatonic-ish, in semitones
    while i < n:
        dur = int(rng.uniform(0.4, 1.4) * sr)
        notes = rng.choice(scale, size=3, replace=False) + 12 * int(rng.integers(-1, 2))
        t = np.arange(min(dur, n - i)) / sr
        env = np.minimum(1.0, t / 0.03) * np.exp(-1.8 * t)
        for s in notes:
            f = 261.63 * 2 ** (s / 12)
            tone = sum(np.sin(2 * np.pi * f * h * t) / h for h in (1, 2, 3, 4))
            out[i : i + len(t)] += env * tone
        i += dur
    return _f32(out / (np.std(out) + 1e-9))


def bursts(n: int, rng: np.random.Generator, sr: int = SR, rate_hz: float = 2.0) -> np.ndarray:
    """Short noise bursts (door slams, keyboard clacks) at about ``rate_hz`` per second."""
    out = np.zeros(n, dtype=np.float64)
    for _ in range(max(1, int(rate_hz * n / sr))):
        d = int(rng.uniform(0.03, 0.2) * sr)
        s = int(rng.integers(0, max(1, n - d)))
        out[s : s + d] += rng.standard_normal(min(d, n - s)) * np.hanning(min(d, n - s))
    return _f32(out / (np.std(out) + 1e-9))


def apply_condition(
    cond: str,
    y: np.ndarray,
    rng: np.random.Generator,
    babble: Babble | None = None,
    sr: int = SR,
) -> np.ndarray:
    """Degrade ``y`` as situation ``cond``. ``babble(rng, n)`` supplies background
    voices (3 other speakers summed); only the ``babble*`` / ``tel_babble``
    conditions need it."""
    y = _f32(y)
    n = len(y)

    def voices() -> np.ndarray:
        if babble is None:
            raise ValueError(f"condition {cond!r} needs a babble source")
        return babble(rng, n)

    def codec(name: str, x: np.ndarray) -> np.ndarray:
        from voiceguard.audio.codec import codec_roundtrip

        return _f32(codec_roundtrip(x, sr, name))

    if cond == "clean":
        return y
    if cond.startswith("first"):  # only the first N seconds of the call have been heard
        return _f32(y[: int(float(cond[len("first") :]) * sr)])
    if cond.startswith("babble"):
        return mix_at_snr(y, voices(), float(cond[len("babble") :]))
    if cond.startswith("pink"):
        return mix_at_snr(y, pink_noise(n, rng), float(cond[len("pink") :]))
    if cond == "reverb":
        return reverberate(y, rng)
    if cond == "speakerphone":
        x = reverberate(y, rng, t60=float(rng.uniform(0.3, 0.6)))
        return mix_at_snr(band_limit(x, 200.0, 4000.0), pink_noise(n, rng), 15.0)
    if cond == "narrow":
        return band_limit(y)
    if cond == "clip":
        return hard_clip(y)
    if cond == "quiet":
        return set_level(y, -45.0)
    if cond == "loss":
        return packet_loss(y, rng)
    if cond == "hum":
        return mix_at_snr(y, mains_hum(n, rng), 12.0)
    if cond == "music":
        return mix_at_snr(y, music_like(n, rng), 8.0)
    if cond == "burst":
        return mix_at_snr(y, bursts(n, rng), 8.0)
    if cond in ("amr", "opus8k", "g722", "ulaw", *MESSAGING_CONDITIONS):
        names = {
            "amr": "amr_nb",
            "opus8k": "opus_8k",
            "ulaw": "g711_ulaw",
            "opus16": "opus_16k",
            "opus24": "opus_24k",
        }
        return codec(names.get(cond, cond), y)
    if cond == "tel_babble":
        return codec("amr_nb", mix_at_snr(y, voices(), 10.0))
    if cond == "tel_reverb":
        return codec("amr_nb", reverberate(y, rng))
    raise ValueError(f"unknown condition {cond!r}")
