"""The 61-dim handcrafted feature vector: 40 MFCC + spectral rolloff + ZCR
+ 12 chroma + 7 spectral-contrast, each time-averaged.

This is the frozen "Generation 1" baseline representation ONLY. It is not
on the serving path -- time-averaged MFCC means discard the temporal and
phase structure where vocoder artifacts live. It stays in the repo as the
honest handcrafted-baseline row in the model comparison, sharing one
implementation with the silence ablation.
"""

from __future__ import annotations

import librosa
import numpy as np

N_MFCC = 40
DIM = N_MFCC + 1 + 1 + 12 + 7  # == 61


def extract(y: np.ndarray, sr: int = 16000) -> np.ndarray:
    """``y`` must already be through voiceguard.audio.preprocess_*."""
    y = np.asarray(y, dtype=np.float32)
    if y.size < sr // 20:  # < 50 ms - pad so librosa framing doesn't fail
        y = np.pad(y, (0, sr // 20 - y.size))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC).mean(axis=1)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
    zcr = librosa.feature.zero_crossing_rate(y).mean()
    chroma = librosa.feature.chroma_stft(y=y, sr=sr).mean(axis=1)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr).mean(axis=1)

    vec = np.concatenate([mfcc, [rolloff], [zcr], chroma, contrast])
    return vec.astype(np.float32)
