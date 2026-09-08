"""Call / transaction context -> a risk contribution + reason codes.

A small, explicit rules table. Context alone never decides anything; it
is one signal among the acoustic ones, but it is what turns a "the voice
sounds a bit off" into "pause this ₹75,000 transfer".
"""

from __future__ import annotations

from dataclasses import dataclass

# transfer-amount bands (INR) -> (risk add, phrase)
_AMOUNT_BANDS = (
    (100_000, 0.45, "high-value transfer requested"),
    (25_000, 0.30, "large transfer requested"),
    (5_000, 0.15, "transfer requested"),
)


@dataclass
class CallContext:
    caller_known: bool = True  # number in the customer's known contacts / prior calls
    amount: float = 0.0  # transfer amount discussed on the call, INR
    channel: str = "pstn"  # pstn | voip | mobile
    prior_flags: int = 0  # past suspicious events attributed to this caller

    def assess(self) -> tuple[float, list[str]]:
        """Return (risk in [0, 1], reason codes)."""
        risk = 0.0
        reasons: list[str] = []

        if not self.caller_known:
            risk += 0.35
            reasons.append("Caller number not recognised")

        for floor, add, phrase in _AMOUNT_BANDS:
            if self.amount >= floor:
                risk += add
                reasons.append(f"{phrase} (₹{self.amount:,.0f})")
                break

        if self.prior_flags:
            risk += min(0.40, 0.20 * self.prior_flags)
            reasons.append(f"{self.prior_flags} prior flag(s) on this caller")

        if self.channel == "voip" and not self.caller_known:
            risk += 0.10
            reasons.append("Unrecognised VoIP call")

        return min(1.0, round(risk, 4)), reasons
