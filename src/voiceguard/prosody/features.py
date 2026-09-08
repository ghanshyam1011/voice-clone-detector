"""Prosodic / voice-quality features via Praat (parselmouth).

One fixed-length vector per clip. Designed to be cheap (a few ms/clip)
and to capture the things synthetic speech tends to get wrong: F0 that is
too flat or too smooth, jitter/shimmer that are unnaturally low, HNR that
is unnaturally high, rhythm/pausing that is too regular.
"""

from __future__ import annotations

import warnings

import numpy as np

FEATURE_NAMES = [
    "f0_mean",
    "f0_std",
    "f0_range",
    "f0_slope_abs",
    "voiced_fraction",
    "jitter_local",
    "jitter_rap",
    "shimmer_local",
    "shimmer_apq5",
    "hnr_mean",
    "intensity_std",
    "pause_ratio",
    "energy_flux",
]
N_FEATURES = len(FEATURE_NAMES)

_F0_MIN, _F0_MAX = 60.0, 500.0


def prosody_features(wave: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Mono waveform -> (N_FEATURES,) float32. NaNs where a feature is
    undefined (too little voiced speech); the scorer imputes them."""
    import parselmouth
    from parselmouth.praat import call

    y = np.ascontiguousarray(np.asarray(wave, dtype=np.float64).reshape(-1))
    out = np.full(N_FEATURES, np.nan, dtype=np.float64)
    if y.size < sr * 0.4 or not np.any(np.abs(y) > 1e-4):
        return out.astype(np.float32)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        snd = parselmouth.Sound(y, sampling_frequency=sr)
        try:
            pitch = snd.to_pitch(pitch_floor=_F0_MIN, pitch_ceiling=_F0_MAX)
            f0 = pitch.selected_array["frequency"]
        except Exception:  # noqa: BLE001
            f0 = np.zeros(1)
        voiced = f0[f0 > 0]
        vf = float(len(voiced)) / max(len(f0), 1)
        out[4] = vf
        if len(voiced) >= 5:
            out[0] = float(np.mean(voiced))
            out[1] = float(np.std(voiced))
            out[2] = float(np.percentile(voiced, 95) - np.percentile(voiced, 5))
            d = np.diff(voiced)
            out[3] = float(np.mean(np.abs(d))) if len(d) else 0.0

        try:
            pp = call(snd, "To PointProcess (periodic, cc)", _F0_MIN, _F0_MAX)
            out[5] = call(pp, "Get jitter (local)", 0, 0, 1e-4, 0.02, 1.3)
            out[6] = call(pp, "Get jitter (rap)", 0, 0, 1e-4, 0.02, 1.3)
            out[7] = call([snd, pp], "Get shimmer (local)", 0, 0, 1e-4, 0.02, 1.3, 1.6)
            out[8] = call([snd, pp], "Get shimmer (apq5)", 0, 0, 1e-4, 0.02, 1.3, 1.6)
        except Exception:  # noqa: BLE001
            pass

        try:
            hnr = snd.to_harmonicity_cc(minimum_pitch=_F0_MIN)
            out[9] = float(call(hnr, "Get mean", 0, 0))
        except Exception:  # noqa: BLE001
            pass

    # intensity dynamics + pausing, from a simple frame energy envelope
    frame = int(0.025 * sr)
    hop = int(0.010 * sr)
    if len(y) >= frame:
        frames = np.lib.stride_tricks.sliding_window_view(y, frame)[::hop]
        rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
        db = 20 * np.log10(rms / (np.max(rms) + 1e-12) + 1e-6)
        out[10] = float(np.std(db))
        out[11] = float(np.mean(db < -35))  # fraction of near-silent frames
        out[12] = float(np.mean(np.abs(np.diff(db))))  # how jerky the energy is

    out = np.where(np.isfinite(out), out, np.nan)
    return out.astype(np.float32)
