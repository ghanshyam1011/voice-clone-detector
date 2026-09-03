"""Evaluation: one harness, one metric implementation, provenance on every row."""

from voiceguard.eval.metrics import compute_eer, det_curve, far_frr_at_threshold
from voiceguard.eval.provenance import git_commit, stamp

__all__ = ["compute_eer", "far_frr_at_threshold", "det_curve", "git_commit", "stamp"]
