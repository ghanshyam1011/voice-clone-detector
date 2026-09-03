"""Provenance stamping.

docs/evaluation_protocol.md, rule: every result row records train set,
test set, model, seed, front-end fingerprint, git commit and score
direction. A number without those does not go in a table.
"""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime


def git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = subprocess.run(["git", "diff", "--quiet"], stderr=subprocess.DEVNULL).returncode
        return f"{out}{'+dirty' if dirty else ''}"
    except Exception:
        return "nogit"


def stamp(
    row: dict,
    *,
    train_set: str,
    test_set: str,
    model: str,
    seed: int,
    front_end: str,
    score_direction: str = "higher=bonafide",
    **extra,
) -> dict:
    """Return ``row`` with the mandatory provenance fields attached."""
    return {
        **row,
        "train_set": train_set,
        "test_set": test_set,
        "model": model,
        "seed": seed,
        "front_end": front_end,
        "score_direction": score_direction,
        "git_commit": git_commit(),
        "utc": datetime.now(UTC).isoformat(timespec="seconds"),
        **extra,
    }
