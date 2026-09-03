"""The anti-shortcut preprocessing front-end.

Why this module exists
----------------------
`src/silence_ablation.py` showed the old ensemble scoring 13-15% EER using
*only the non-speech parts* of dev clips (chance = 50%). The model had
learned the ASVspoof-2019 silence-duration / loudness artifact, not
synthesis. Every feature vector and every model trained on them is
contaminated.

The fix: one deterministic preprocessing function that every path calls,
applying the *same* silence trim and loudness normalisation to bonafide
and spoof. After this, `silence_ablation` on a model trained through here
must land within ~5 points of 50%.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import librosa
import numpy as np

try:
    import pyloudnorm as _pyln

    _HAVE_PYLN = True
except ImportError:  # pragma: no cover - fallback path is tested instead
    _HAVE_PYLN = False


@dataclass(frozen=True)
class PreprocessConfig:
    sample_rate: int = 16000
    trim_enabled: bool = True
    trim_top_db: float = 30.0
    trim_frame_length: int = 2048
    trim_hop_length: int = 512
    trim_pad_ms: float = 40.0
    loudness_enabled: bool = True
    loudness_method: str = "ebu_r128"  # "ebu_r128" | "rms"
    target_lufs: float = -23.0
    peak_ceiling_dbfs: float = -1.0

    def fingerprint(self) -> str:
        """10-char deterministic hash of these settings.

        Stamped onto every results row and into the processed-feature
        directory name, so a number can never be silently compared across
        two different front-ends.
        """
        payload = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha1(payload).hexdigest()[:10]


def trim_silence(y: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """Remove leading/trailing near-silence, then re-pad a fixed margin.

    The re-pad is identical for both classes, so "speech starts at sample
    0" cannot itself become a cue.
    """
    if not cfg.trim_enabled or y.size == 0:
        return y
    trimmed, _ = librosa.effects.trim(
        y,
        top_db=cfg.trim_top_db,
        frame_length=cfg.trim_frame_length,
        hop_length=cfg.trim_hop_length,
    )
    if trimmed.size == 0:  # entire clip below threshold - leave it alone
        return y
    pad = int(round(cfg.sample_rate * cfg.trim_pad_ms / 1000.0))
    if pad > 0:
        trimmed = np.pad(trimmed, pad)
    return np.ascontiguousarray(trimmed, dtype=np.float32)


def _apply_peak_ceiling(y: np.ndarray, ceiling_dbfs: float) -> np.ndarray:
    ceiling = 10.0 ** (ceiling_dbfs / 20.0)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > ceiling > 0:
        y = y * (ceiling / peak)
    return y


def normalize_loudness(y: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """Bring every clip to the same integrated loudness, then peak-limit.

    EBU R128 (pyloudnorm) when available and the clip is long enough for
    its block gating (>= 400 ms); otherwise an RMS match to an equivalent
    target. Digital silence is passed through untouched.
    """
    if not cfg.loudness_enabled or y.size == 0:
        return y
    if float(np.max(np.abs(y))) < 1e-6:
        return y

    use_r128 = (
        cfg.loudness_method == "ebu_r128" and _HAVE_PYLN and y.size >= int(cfg.sample_rate * 0.4)
    )
    if use_r128:
        meter = _pyln.Meter(cfg.sample_rate)
        loudness = meter.integrated_loudness(y)
        if np.isfinite(loudness):
            y = _pyln.normalize.loudness(y, loudness, cfg.target_lufs)
    else:
        rms = float(np.sqrt(np.mean(np.square(y, dtype=np.float64))))
        if rms > 0:
            # speech LUFS ~ dBFS RMS; keep a little headroom
            target_rms = 10.0 ** ((cfg.target_lufs + 3.0) / 20.0)
            y = y * (target_rms / rms)

    y = _apply_peak_ceiling(y, cfg.peak_ceiling_dbfs)
    return np.ascontiguousarray(np.nan_to_num(y), dtype=np.float32)


def preprocess_wave(y: np.ndarray, cfg: PreprocessConfig | None = None) -> np.ndarray:
    """trim -> loudness. The order matters: trim first so silence doesn't
    drag the loudness measurement."""
    cfg = cfg or PreprocessConfig()
    y = np.asarray(y, dtype=np.float32)
    y = trim_silence(y, cfg)
    y = normalize_loudness(y, cfg)
    return y


def preprocess_file(path, cfg: PreprocessConfig | None = None) -> np.ndarray:
    from voiceguard.audio.io import load_wave

    cfg = cfg or PreprocessConfig()
    return preprocess_wave(load_wave(path, cfg.sample_rate), cfg)
