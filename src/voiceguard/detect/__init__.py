"""Streaming / windowed scoring on top of a countermeasure model.

Minimal for now (demo + the start of P2): sliding-window inference with an
exponential-moving-average risk timeline. VAD, latency budgets and the
two-tier design come later.
"""

from voiceguard.detect.scorer import CMScorer, Verdict, build_scorer

__all__ = ["CMScorer", "Verdict", "build_scorer"]
