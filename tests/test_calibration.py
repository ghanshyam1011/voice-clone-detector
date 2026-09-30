"""Calibration fitting + quality metrics. Pure numpy/sklearn, no model
weights needed."""

import numpy as np

from voiceguard.eval.calibration import (
    Calibrator,
    brier_score,
    expected_calibration_error,
    fit_calibrator,
    reliability_bins,
)


def _synthetic(n=4000, seed=0):
    """Scores from a detector whose raw output is a *monotonic but
    miscalibrated* function of the true probability -- like an
    overconfident softmax: it ranks well but the numbers themselves lie."""
    rng = np.random.default_rng(seed)
    true_p = rng.uniform(0, 1, n)
    labels = (rng.random(n) < true_p).astype(int)
    # squash true_p toward the extremes -- classic overconfidence
    raw = 1.0 / (1.0 + np.exp(-8 * (true_p - 0.5)))
    return raw, labels, true_p


def test_platt_improves_calibration_error():
    raw, labels, _ = _synthetic()
    tr = slice(0, 2000)
    te = slice(2000, 4000)

    ece_before = expected_calibration_error(raw[te], labels[te])
    cal = fit_calibrator(raw[tr], labels[tr], method="platt")
    calibrated = cal.apply(raw[te])
    ece_after = expected_calibration_error(calibrated, labels[te])

    assert ece_after < ece_before


def test_isotonic_improves_calibration_error():
    raw, labels, _ = _synthetic(seed=1)
    tr, te = slice(0, 2000), slice(2000, 4000)

    ece_before = expected_calibration_error(raw[te], labels[te])
    cal = fit_calibrator(raw[tr], labels[tr], method="isotonic")
    ece_after = expected_calibration_error(cal.apply(raw[te]), labels[te])

    assert ece_after < ece_before


def test_calibrator_output_is_monotonic_in_raw_score():
    raw, labels, _ = _synthetic(seed=2)
    for method in ("platt", "isotonic"):
        cal = fit_calibrator(raw, labels, method=method)
        xs = np.linspace(raw.min(), raw.max(), 50)
        ys = cal.apply(xs)
        assert np.all(np.diff(ys) >= -1e-9), method  # non-decreasing


def test_platt_round_trips_through_dict():
    raw, labels, _ = _synthetic(seed=3)
    cal = fit_calibrator(raw, labels, method="platt")
    reloaded = Calibrator.from_dict(cal.to_dict())
    probe = np.linspace(raw.min(), raw.max(), 20)
    assert np.allclose(cal.apply(probe), reloaded.apply(probe), atol=1e-6)


def test_isotonic_round_trips_through_dict():
    raw, labels, _ = _synthetic(seed=4)
    cal = fit_calibrator(raw, labels, method="isotonic")
    reloaded = Calibrator.from_dict(cal.to_dict())
    probe = np.linspace(raw.min(), raw.max(), 20)
    assert np.allclose(cal.apply(probe), reloaded.apply(probe), atol=1e-6)


def test_fit_calibrator_rejects_a_single_class():
    import pytest

    with pytest.raises(ValueError):
        fit_calibrator([0.1, 0.2, 0.3], [1, 1, 1])


def test_brier_score_perfect_and_worst_case():
    labels = np.array([1, 0, 1, 0])
    assert brier_score(labels, labels) == 0.0  # predicting the truth exactly
    assert brier_score(1 - labels, labels) == 1.0  # predicting the opposite


def test_ece_zero_for_a_perfectly_calibrated_predictor():
    rng = np.random.default_rng(5)
    probs = np.repeat([0.1, 0.3, 0.5, 0.7, 0.9], 2000)
    labels = (rng.random(len(probs)) < probs).astype(int)
    # with enough samples per bin, ECE should be small (not exactly 0 --
    # finite-sample noise)
    assert expected_calibration_error(probs, labels, n_bins=5) < 0.03


def test_reliability_bins_shapes_and_empty_bins_are_nan():
    probs = np.array([0.05, 0.05, 0.95, 0.95])
    labels = np.array([0, 1, 1, 1])
    centers, rates, counts = reliability_bins(probs, labels, n_bins=10)
    assert len(centers) == len(rates) == len(counts) == 10
    assert counts.sum() == 4
    assert np.isnan(rates[5])  # a middle bin with no samples
    assert counts[0] == 2 and rates[0] == 0.5  # both low-prob points, one positive
