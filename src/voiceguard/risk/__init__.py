"""Risk fusion + policy — combine the detection signals into one decision.

    signals -> fuse() -> FusedRisk -> decide(scenario, context) -> Decision

Everything here is deterministic, dependency-free and unit-tested. The
numbers (weights, thresholds, context rules) are documented operating
points chosen from a fraud-vs-friction trade-off, not learned — real
calibration is later work.
"""

from voiceguard.risk.context import CallContext
from voiceguard.risk.fusion import DEFAULT_WEIGHTS, FusedRisk, fuse
from voiceguard.risk.policy import SCENARIOS, Decision, decide
from voiceguard.risk.signals import Signal

__all__ = [
    "CallContext",
    "Signal",
    "FusedRisk",
    "fuse",
    "DEFAULT_WEIGHTS",
    "Decision",
    "decide",
    "SCENARIOS",
]
