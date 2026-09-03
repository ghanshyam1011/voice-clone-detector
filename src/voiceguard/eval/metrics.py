"""Anti-spoofing evaluation metrics.

Everything downstream (results tables, calibration, thresholds) depends on
compute_eer being correct, so it is unit-tested in tests/test_metrics.py.

Moved verbatim from the old src/metrics.py -- it was already correct.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve


def compute_eer(labels, scores) -> tuple[float, float]:
    """Equal Error Rate.

    labels: 1 = bonafide, 0 = spoof.
    scores: higher = more bonafide-like.

    Returns (eer, threshold) where threshold is the score cutoff at which
    the false-acceptance rate (spoof accepted as bonafide) equals the
    false-rejection rate (bonafide rejected as spoof).
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if len(np.unique(labels)) < 2:
        raise ValueError("compute_eer needs both classes present in labels")

    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr

    # fpr is monotonically non-decreasing, fnr non-increasing -> unique crossing
    eer = brentq(lambda x: interp1d(fpr, fnr)(x) - x, 0.0, 1.0)
    threshold = float(interp1d(fpr, thresholds)(eer))
    return float(eer), threshold


def far_frr_at_threshold(labels, scores, threshold) -> tuple[float, float]:
    """False accept rate (spoof called bonafide) and false reject rate
    (bonafide called spoof) at a fixed score threshold."""
    labels = np.asarray(labels)
    scores = np.asarray(scores)

    spoof_mask = labels == 0
    bonafide_mask = labels == 1

    far = float((scores[spoof_mask] >= threshold).mean()) if spoof_mask.any() else float("nan")
    frr = float((scores[bonafide_mask] < threshold).mean()) if bonafide_mask.any() else float("nan")
    return far, frr


def det_curve(labels, scores):
    """Returns (fpr, fnr, thresholds) for plotting a DET curve."""
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    return fpr, fnr, thresholds
