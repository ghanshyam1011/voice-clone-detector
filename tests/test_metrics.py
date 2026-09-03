import numpy as np
import pytest

from voiceguard.eval.metrics import compute_eer, far_frr_at_threshold


def test_eer_perfectly_separable():
    labels = [1] * 50 + [0] * 50
    scores = [1.0] * 50 + [0.0] * 50
    eer, threshold = compute_eer(labels, scores)
    assert eer == pytest.approx(0.0, abs=1e-6)
    assert 0.0 <= threshold <= 1.0


def test_eer_random_guessing():
    rng = np.random.default_rng(0)
    n = 20000
    labels = rng.integers(0, 2, size=n)
    scores = rng.random(n)
    eer, _ = compute_eer(labels, scores)
    assert eer == pytest.approx(0.5, abs=0.03)


def test_eer_known_overlap_case():
    rng = np.random.default_rng(1)
    n = 5000
    bonafide_scores = rng.normal(0.6, 0.15, n)
    spoof_scores = rng.normal(0.4, 0.15, n)
    labels = [1] * n + [0] * n
    scores = np.concatenate([bonafide_scores, spoof_scores])
    eer, _ = compute_eer(labels, scores)
    assert 0.0 < eer < 0.5


def test_eer_raises_on_single_class():
    with pytest.raises(ValueError):
        compute_eer([1, 1, 1], [0.9, 0.8, 0.7])


def test_far_frr_at_threshold_matches_manual_count():
    labels = [1, 1, 1, 1, 0, 0, 0, 0]
    scores = [0.9, 0.8, 0.3, 0.2, 0.95, 0.6, 0.1, 0.05]
    far, frr = far_frr_at_threshold(labels, scores, 0.5)
    assert far == pytest.approx(0.5)  # spoof >= 0.5: 0.95, 0.6 -> 2/4
    assert frr == pytest.approx(0.5)  # bonafide < 0.5: 0.3, 0.2 -> 2/4
