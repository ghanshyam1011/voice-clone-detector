"""Combine the available signals into one fused risk score.

Weighted mean over the signals that are actually present (weights
renormalised), with one override: if any single signal is very high on
its own, the fused score is floored so a strong lone detector still
escalates instead of being averaged away.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from voiceguard.risk.signals import Signal

# relative weights; only the present signals' weights count (renormalised)
DEFAULT_WEIGHTS: dict[str, float] = {
    "spoof": 0.45,
    "speaker": 0.35,
    "context": 0.20,
    "prosody": 0.25,
}

_LONE_SIGNAL_ALARM = 0.80  # a single signal at/above this...
_ALARM_FLOOR = 0.68  # ...floors the fused score here


@dataclass
class FusedRisk:
    score: float
    signals: list[Signal] = field(default_factory=list)
    note: str = ""
    has_provisional: bool = False  # true if any present signal is still building
    # confidence (see Signal.provisional) -- policy.decide() reads this to cap
    # the action at VERIFY instead of ESCALATE.

    @property
    def present(self) -> list[Signal]:
        return [s for s in self.signals if s.available]


def fuse(signals: list[Signal]) -> FusedRisk:
    present = [s for s in signals if s.available]
    if not present:
        return FusedRisk(0.0, signals, "no signals available yet")

    total_w = sum(s.weight for s in present) or 1.0
    score = sum(s.value * s.weight for s in present) / total_w

    note = ""
    loudest = max(present, key=lambda s: s.value)
    if loudest.value >= _LONE_SIGNAL_ALARM and score < _ALARM_FLOOR:
        score = _ALARM_FLOOR
        note = f"{loudest.label} alone is high enough to act on"

    has_provisional = any(s.provisional for s in present)
    return FusedRisk(round(min(1.0, score), 4), signals, note, has_provisional)
