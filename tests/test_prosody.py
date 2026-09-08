"""Prosody features + scorer.

The scorer needs `models/prosody/prosody_lr.joblib` (tracked, produced by
`scripts/train_prosody.py`); tests that need it skip cleanly when absent."""

import numpy as np
import pytest

from voiceguard.prosody import FEATURE_NAMES, ProsodyScorer, prosody_features
from voiceguard.prosody.scorer import DEFAULT_MODEL_PATH

needs_model = pytest.mark.skipif(
    not DEFAULT_MODEL_PATH.exists(), reason="prosody model not trained yet"
)


def _speechy(seed: int, secs: float = 3.0, sr: int = 16000) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(secs * sr)) / sr
    f0 = 120 + 40 * rng.random() + 15 * np.sin(2 * np.pi * 3 * t)  # moving pitch
    sig = sum((1.0 / k) * np.sin(2 * np.pi * f0 * k * t) for k in (1, 2, 3, 4))
    env = (np.sin(2 * np.pi * 2.5 * t) > -0.3).astype(float)  # syllable-ish gating
    sig = sig * env + 0.03 * rng.standard_normal(t.shape)
    return (0.25 * sig / np.max(np.abs(sig))).astype(np.float32)


def test_feature_vector_shape():
    f = prosody_features(_speechy(1))
    assert f.shape == (len(FEATURE_NAMES),)
    assert f.dtype == np.float32
    assert np.isfinite(f).sum() >= len(FEATURE_NAMES) - 3  # most features defined


def test_silence_returns_all_nan():
    f = prosody_features(np.zeros(16000, dtype=np.float32))
    assert np.all(np.isnan(f))


@needs_model
def test_scorer_risk_in_range():
    sc = ProsodyScorer()
    sc.warm()
    r = sc.score(_speechy(3))
    assert r.available is True
    assert 0.0 <= r.risk <= 1.0
    assert r.detail


@needs_model
def test_scorer_unavailable_on_silence():
    r = ProsodyScorer().score(np.zeros(16000, dtype=np.float32))
    assert r.available is False and r.risk is None
