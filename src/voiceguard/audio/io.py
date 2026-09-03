"""Loading and resampling. Format-agnostic (flac / wav / mp3 / ogg) via librosa."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

DEFAULT_SR = 16000


def load_wave(
    path: str | Path, sample_rate: int = DEFAULT_SR, *, res_type: str = "soxr_hq"
) -> np.ndarray:
    """Load ``path``, downmix to mono, resample to ``sample_rate``.

    Returns a 1-D float32 array in roughly [-1, 1].
    """
    y, _ = librosa.load(str(path), sr=sample_rate, mono=True, res_type=res_type)
    return np.ascontiguousarray(y, dtype=np.float32)


def rms_dbfs(y: np.ndarray) -> float:
    """RMS level in dBFS. Returns -inf for digital silence."""
    if y.size == 0:
        return float("-inf")
    rms = float(np.sqrt(np.mean(np.square(y, dtype=np.float64))))
    return 20.0 * np.log10(rms) if rms > 0 else float("-inf")
