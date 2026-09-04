import numpy as np
import pytest

from voiceguard.audio.augment import AugmentConfig, augment_waveform, rawboost

SR = 16000


def _speech(seconds=3.0):
    rng = np.random.default_rng(0)
    t = np.arange(int(SR * seconds)) / SR
    tone = 0.2 * np.sin(2 * np.pi * 160 * t) + 0.1 * np.sin(2 * np.pi * 320 * t)
    return (tone + 0.01 * rng.standard_normal(t.shape)).astype(np.float32)


@pytest.mark.parametrize("algo", [1, 2, 3, 4, 5, 6, 7])
def test_rawboost_algos_finite_and_same_length(algo):
    x = _speech()
    np.random.seed(0)
    y = rawboost(x, SR, algo)
    assert y.shape == x.shape
    assert np.all(np.isfinite(y))
    assert not np.array_equal(y, x)  # something changed


def test_rawboost_passthrough():
    x = _speech()
    assert np.array_equal(rawboost(x, SR, 8), x.astype(np.float32))


def test_rawboost_deterministic_under_seed():
    x = _speech()
    np.random.seed(123)
    a = rawboost(x, SR, 5)
    np.random.seed(123)
    b = rawboost(x, SR, 5)
    assert np.allclose(a, b)


def test_augment_disabled_is_identity():
    x = _speech()
    cfg = AugmentConfig(enabled=False)
    assert np.array_equal(augment_waveform(x, SR, cfg), x.astype(np.float32))


def test_augment_rawboost_only_bounded():
    x = _speech()
    cfg = AugmentConfig(rawboost_prob=1.0, codec_prob=0.0)
    y = augment_waveform(x, SR, cfg, np.random.default_rng(0))
    assert np.all(np.isfinite(y))
    assert np.max(np.abs(y)) <= 1.0 + 1e-4
    assert len(y) == len(x)


def test_augment_same_rng_same_output():
    x = _speech()
    cfg = AugmentConfig(rawboost_prob=1.0, codec_prob=0.0)
    y1 = augment_waveform(x, SR, cfg, np.random.default_rng(7))
    y2 = augment_waveform(x, SR, cfg, np.random.default_rng(7))
    # note: rawboost also touches numpy global RNG; seed it for full determinism
    np.random.seed(0)
    a = augment_waveform(x, SR, cfg, np.random.default_rng(7))
    np.random.seed(0)
    b = augment_waveform(x, SR, cfg, np.random.default_rng(7))
    assert np.allclose(a, b)
    assert y1.shape == y2.shape
