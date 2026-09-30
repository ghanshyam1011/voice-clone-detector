"""Evaluation: one harness, one metric implementation, provenance on every row."""

from voiceguard.eval.calibration import (
    Calibrator,
    brier_score,
    expected_calibration_error,
    fit_calibrator,
    reliability_bins,
)
from voiceguard.eval.metrics import compute_eer, det_curve, far_frr_at_threshold
from voiceguard.eval.provenance import git_commit, stamp
from voiceguard.eval.tdcf import (
    DEFAULT_COST_MODEL,
    MinTDCFResult,
    compute_min_tdcf,
    load_asv_scores,
)

__all__ = [
    "compute_eer",
    "far_frr_at_threshold",
    "det_curve",
    "git_commit",
    "stamp",
    "Calibrator",
    "fit_calibrator",
    "expected_calibration_error",
    "brier_score",
    "reliability_bins",
    "compute_min_tdcf",
    "load_asv_scores",
    "MinTDCFResult",
    "DEFAULT_COST_MODEL",
]
