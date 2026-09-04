"""Channel / device augmentation for anti-spoofing training.

RawBoost (Tak et al., ICASSP 2022) via the vendored reference
implementation, plus an ffmpeg telephony-codec pass. Everything here must
be applied IDENTICALLY to bonafide and spoof at the call site -- augmenting
one class teaches the model to detect augmentation instead of synthesis.

RawBoost `algo`:
  1 = LnL convolutive     2 = ISD additive        3 = SSI additive
  4 = series(1,2,3)       5 = series(1,2)         6 = parallel(1,2)
  7 = series(1, parallel(2,3))                    8 = no-op (A/B baseline)
The SSL-AASIST recipe uses algo 5 for LA.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from voiceguard.models._vendor import rawboost_ref as _rb

# reference default hyper-parameters (from the RawBoost paper / repo args)
_LNL = dict(
    N_f=5,
    nBands=5,
    minF=20,
    maxF=8000,
    minBW=100,
    maxBW=1000,
    minCoeff=10,
    maxCoeff=100,
    minG=0,
    maxG=0,
    minBiasLinNonLin=5,
    maxBiasLinNonLin=20,
)
_ISD = dict(P=10, g_sd=2)
_SSI = dict(
    SNRmin=10,
    SNRmax=40,
    nBands=5,
    minF=20,
    maxF=8000,
    minBW=100,
    maxBW=1000,
    minCoeff=10,
    maxCoeff=100,
    minG=0,
    maxG=0,
)


def rawboost(x: np.ndarray, sr: int = 16000, algo: int = 5) -> np.ndarray:
    """Apply the RawBoost combination `algo` to a 1-D float waveform.

    Randomness comes from numpy's global RNG (seed it upstream for
    reproducibility). Returns float32.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0 or algo == 8:
        return x.astype(np.float32)

    def lnl(sig):
        return _rb.LnL_convolutive_noise(sig, fs=sr, **_LNL)

    def isd(sig):
        return _rb.ISD_additive_noise(sig, _ISD["P"], _ISD["g_sd"])

    def ssi(sig):
        return _rb.SSI_additive_noise(sig, fs=sr, **_SSI)

    if algo == 1:
        y = lnl(x)
    elif algo == 2:
        y = isd(x)
    elif algo == 3:
        y = ssi(x)
    elif algo == 4:
        y = ssi(isd(lnl(x)))
    elif algo == 5:
        y = isd(lnl(x))
    elif algo == 6:
        y = lnl(x) + isd(x)
        y = _rb.normWav(y, 0)
    elif algo == 7:
        t = lnl(x)
        y = _rb.normWav(isd(t) + ssi(t), 0)
    else:
        raise ValueError(f"unknown RawBoost algo {algo}")

    return np.nan_to_num(y).astype(np.float32)


@dataclass(frozen=True)
class AugmentConfig:
    enabled: bool = True
    rawboost_prob: float = 0.8
    rawboost_algo: int = 5
    codec_prob: float = 0.5
    codecs: tuple[str, ...] = field(
        default_factory=lambda: ("g711_ulaw", "amr_nb", "opus_8k", "g722")
    )


def augment_waveform(
    x: np.ndarray, sr: int, cfg: AugmentConfig, rng: np.random.Generator | None = None
) -> np.ndarray:
    """RawBoost (prob) then a random telephony codec (prob). Same cfg + rng
    policy must be used for both classes at the call site."""
    if not cfg.enabled:
        return np.asarray(x, dtype=np.float32)
    rng = rng or np.random.default_rng()
    y = np.asarray(x, dtype=np.float32)

    if rng.random() < cfg.rawboost_prob:
        y = rawboost(y, sr, cfg.rawboost_algo)

    if cfg.codecs and rng.random() < cfg.codec_prob:
        from voiceguard.audio.codec import codec_roundtrip

        codec = cfg.codecs[int(rng.integers(len(cfg.codecs)))]
        try:
            y = codec_roundtrip(y, sr, codec)
        except Exception:
            pass  # ffmpeg / codec failure -> skip, never crash training

    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 1.0:
        y = y / peak
    return np.nan_to_num(y).astype(np.float32)
