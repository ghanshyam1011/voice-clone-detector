"""Prosody / behaviour branch.

`prosody_features` pulls a small set of voice-quality and rhythm features
(F0 statistics, jitter, shimmer, HNR, intensity dynamics, pause ratio)
via Praat/Parselmouth. `ProsodyScorer` wraps a tiny logistic-regression
fit on ASVspoof19 train and turns the features into a synthetic-likeness
risk in [0, 1] — a *contributing* signal in the fusion, not a detector on
its own.

Needs the ``[prosody]`` extra (``praat-parselmouth``).
"""

from voiceguard.prosody.features import FEATURE_NAMES, prosody_features
from voiceguard.prosody.scorer import ProsodyResult, ProsodyScorer

__all__ = ["prosody_features", "FEATURE_NAMES", "ProsodyScorer", "ProsodyResult"]
