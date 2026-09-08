"""A single risk signal feeding the fusion engine."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Signal:
    """One contributing signal.

    ``value`` is in [0, 1], higher = more suspicious. ``None`` (or NaN)
    means the signal is not available this frame (e.g. speaker check
    before enrolment) and it is dropped from fusion rather than treated
    as zero risk.
    """

    name: str  # machine key: "spoof" | "speaker" | "context" | "prosody"
    label: str  # human: "Synthetic voice", "Speaker mismatch", ...
    value: float | None
    weight: float
    detail: str = ""  # short phrase for the reason list, e.g. "similarity 0.41 vs enrolled"

    @property
    def available(self) -> bool:
        return self.value is not None and not math.isnan(self.value)

    def reason(self, threshold: float = 0.5) -> str | None:
        """Reason-code string when this signal is elevated, else None."""
        if not self.available or self.value < threshold:
            return None
        tail = self.detail or f"score {self.value:.2f}"
        return f"{self.label}: {tail}"
