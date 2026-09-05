"""Loading and resampling.

Deliberately NOT librosa.load(..., res_type=...): measured librosa 0.10.2's
soxr_hq resample path at ~1s/call even for a 16kHz -> 16kHz no-op (every
clip in this project's corpora is already 16 kHz). The identical no-op
through the `soxr` package directly is ~1ms -- three orders of magnitude
faster -- and that one call was the dominant cost of the entire training
and feature-extraction pipeline. soundfile handles the read; soxr handles
resampling only when a clip's native rate actually differs; librosa.load
is kept only as a fallback for formats libsndfile can't read.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

DEFAULT_SR = 16000

_QUALITY = {
    "soxr_qq": "QQ",
    "soxr_lq": "LQ",
    "soxr_mq": "MQ",
    "soxr_hq": "HQ",
    "soxr_vhq": "VHQ",
}


def load_wave(
    path: str | Path, sample_rate: int = DEFAULT_SR, *, res_type: str = "soxr_hq"
) -> np.ndarray:
    """Load ``path``, downmix to mono, resample to ``sample_rate`` (only if
    the file's native rate actually differs). Returns 1-D float32, roughly
    in [-1, 1].
    """
    try:
        y, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)  # (frames, channels) -> (frames,)
    except Exception:
        import librosa  # fallback for formats libsndfile can't read

        y, sr = librosa.load(str(path), sr=None, mono=True)  # sr=None: no resample here

    y = np.ascontiguousarray(y, dtype=np.float32)
    if sr != sample_rate:
        y = soxr.resample(y, sr, sample_rate, quality=_QUALITY.get(res_type, "HQ"))
    return np.ascontiguousarray(y, dtype=np.float32)


def rms_dbfs(y: np.ndarray) -> float:
    """RMS level in dBFS. Returns -inf for digital silence."""
    if y.size == 0:
        return float("-inf")
    rms = float(np.sqrt(np.mean(np.square(y, dtype=np.float64))))
    return 20.0 * np.log10(rms) if rms > 0 else float("-inf")
