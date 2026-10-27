"""Turn a fused risk score into an operator decision with reason codes.

Three scenarios, each with its own (warn, escalate) thresholds on the
fused score. Higher-stakes scenarios act on less evidence. The output is
a defensible recommendation for a human, never an automated block.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from voiceguard.risk.fusion import FusedRisk

# scenario -> (warn threshold, escalate threshold) on fused risk in [0, 1]
SCENARIOS: dict[str, tuple[float, float]] = {
    "routine": (0.55, 0.75),  # ordinary support call
    "transaction": (0.40, 0.65),  # a payment / transfer is on the table
    "privileged": (0.30, 0.55),  # password reset, beneficiary change, large limit
}

_RECOMMENDATION = {
    "ALLOW": "Proceed. Keep monitoring the call.",
    "VERIFY": "Ask a knowledge-based verification question before continuing.",
    "ESCALATE": "Pause approval and verify the caller through the registered number.",
    "WAITING": "No verdict yet — not enough clear speech to judge. Keep the caller talking.",
}


@dataclass
class Decision:
    action: str  # ALLOW | VERIFY | ESCALATE | WAITING (no verdict: nothing usable to judge)
    score: float
    scenario: str
    reasons: list[str] = field(default_factory=list)
    recommendation: str = ""

    def as_dict(self) -> dict:
        return {
            "action": self.action,
            "score": round(self.score, 4),
            "scenario": self.scenario,
            "reasons": self.reasons,
            "recommendation": self.recommendation,
        }


def waiting(scenario: str, reason: str) -> Decision:
    """No verdict because there is nothing usable to judge (silence, noise, a few
    hundred milliseconds of audio). Deliberately NOT ALLOW: ALLOW tells the agent
    the call has been checked and cleared, which is false for a call nobody has
    heard -- and not ESCALATE either, which would cry fraud at a muted line."""
    return Decision("WAITING", 0.0, scenario, [reason], _RECOMMENDATION["WAITING"])


def decide(fused: FusedRisk, scenario: str, context_reasons: list[str] | None = None) -> Decision:
    warn, escalate = SCENARIOS.get(scenario, SCENARIOS["routine"])
    score = fused.score
    action = "ESCALATE" if score >= escalate else "VERIFY" if score >= warn else "ALLOW"

    reasons: list[str] = []
    for s in fused.signals:
        if s.name == "context":
            continue  # the context signal's specifics come from context_reasons
        r = s.reason()
        if r:
            reasons.append(r)
    reasons.extend(context_reasons or [])
    if fused.note:
        reasons.append(fused.note)

    if action == "ESCALATE" and fused.has_provisional:
        # a read still built partly from a not-yet-full audio window can raise
        # concern (VERIFY) but must never pause a real transaction on its own --
        # it upgrades to ESCALATE automatically once the read is confirmed.
        action = "VERIFY"
        reasons.append("Early read — escalation available once the read is confirmed")

    return Decision(action, score, scenario, reasons, _RECOMMENDATION[action])
