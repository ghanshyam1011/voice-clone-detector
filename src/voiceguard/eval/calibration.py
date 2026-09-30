"""Probability calibration: fit raw_score -> genuine P(positive), and
measure whether the result is actually trustworthy.

A ranking metric (EER, AUC) says a detector separates two classes; it says
nothing about whether a "0.9" it emits means "90% likely" in any real
sense. Every risk number this system fuses is read by a human (and by the
policy engine's fixed thresholds) as a probability, so calibration quality
is measured here the same way ranking quality is measured in `metrics.py`:
on labeled data, with a documented method, unit-tested.

Two fitting methods:
  - "platt"    -- 1-D logistic regression (sigmoid). Smooth, extrapolates
                  sanely, needs less data. Good default.
  - "isotonic" -- monotonic step function. Fits the empirical curve more
                  exactly given enough data; can overfit on small sets and
                  is flat (unhelpful) outside the training score range.

Both must be fit on a split the raw scores' own model never trained on,
and are only trustworthy where the calibration set's score distribution
actually covers -- fitting on ASVspoof dev and reporting ECE on ASVspoof
eval is a fair test; reporting it on In-the-Wild is a harder, honest one
(the calibration may not transfer -- report it anyway, per
docs/evaluation_protocol.md's rule that the cross-dataset number is the
one that matters).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


@dataclass
class Calibrator:
    """A fitted 1-D mapping: raw score -> calibrated P(positive class)."""

    method: str  # "platt" | "isotonic"
    coef: float | None = None  # platt only, for a human-readable summary
    intercept: float | None = None  # platt only
    _model: object = None

    def apply(self, raw) -> np.ndarray:
        x = np.atleast_1d(np.asarray(raw, dtype=np.float64))
        if self.method == "platt":
            p = self._model.predict_proba(x.reshape(-1, 1))[:, 1]
        else:
            p = self._model.predict(x)
        return np.clip(p, 1e-6, 1 - 1e-6)

    def to_dict(self) -> dict:
        """JSON-serialisable summary (isotonic keeps its full fitted curve
        so it can be reloaded exactly; platt is two numbers)."""
        d = {"method": self.method}
        if self.method == "platt":
            d["coef"] = float(self._model.coef_[0, 0])
            d["intercept"] = float(self._model.intercept_[0])
        else:
            d["x_thresholds"] = self._model.X_thresholds_.tolist()
            d["y_thresholds"] = self._model.y_thresholds_.tolist()
        return d

    @staticmethod
    def from_dict(d: dict) -> Calibrator:
        if d["method"] == "platt":
            m = LogisticRegression()
            m.coef_ = np.array([[d["coef"]]])
            m.intercept_ = np.array([d["intercept"]])
            m.classes_ = np.array([0, 1])
            return Calibrator("platt", d["coef"], d["intercept"], m)
        m = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        m.X_thresholds_ = np.array(d["x_thresholds"])
        m.y_thresholds_ = np.array(d["y_thresholds"])
        m.X_min_, m.X_max_ = m.X_thresholds_[0], m.X_thresholds_[-1]
        m.f_ = None
        from scipy.interpolate import interp1d

        m.f_ = interp1d(
            m.X_thresholds_, m.y_thresholds_, kind="linear", bounds_error=False,
            fill_value=(m.y_thresholds_[0], m.y_thresholds_[-1]),
        )
        m.predict = lambda x: m.f_(x)  # bind the reloaded curve as .predict
        return Calibrator("isotonic", _model=m)


def fit_calibrator(raw, labels, method: str = "platt") -> Calibrator:
    """labels: 1 = positive class (e.g. spoof, or "different speaker").
    raw: uncalibrated score, higher = more positive."""
    raw = np.asarray(raw, dtype=np.float64).reshape(-1, 1)
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        raise ValueError("fit_calibrator needs both classes present")
    if method == "platt":
        model = LogisticRegression(max_iter=2000)
        model.fit(raw, labels)
        return Calibrator("platt", float(model.coef_[0, 0]), float(model.intercept_[0]), model)
    if method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        model.fit(raw.ravel(), labels)
        return Calibrator("isotonic", _model=model)
    raise ValueError(f"unknown calibration method: {method!r}")


def expected_calibration_error(probs, labels, n_bins: int = 10) -> float:
    """ECE: bin predictions by confidence, compare each bin's mean
    predicted probability to its empirical positive rate, weight by bin
    size. 0 = perfectly calibrated."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(probs)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        mask = (probs >= lo) & (probs < hi) if hi < 1.0 else (probs >= lo) & (probs <= hi)
        if not mask.any():
            continue
        ece += (mask.sum() / n) * abs(probs[mask].mean() - labels[mask].mean())
    return float(ece)


def brier_score(probs, labels) -> float:
    """Mean squared error between predicted probability and the 0/1
    outcome. 0 = perfect, 0.25 = the score of a constant p=0.5 predictor
    on a balanced set -- a ceiling any real calibration should beat."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    return float(np.mean((probs - labels) ** 2))


def reliability_bins(probs, labels, n_bins: int = 10):
    """(bin_centers, empirical_rate, bin_counts) for a reliability
    diagram; empirical_rate is NaN for empty bins."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    rates = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        mask = (probs >= lo) & (probs < hi) if hi < 1.0 else (probs >= lo) & (probs <= hi)
        counts[i] = int(mask.sum())
        if mask.any():
            rates[i] = float(labels[mask].mean())
    return centers, rates, counts
