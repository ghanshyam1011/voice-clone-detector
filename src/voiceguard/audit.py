"""Feature-only audit log.

Every decision writes one JSONL row: the scores, the action, the reason
codes, and provenance. **Never** the audio, a transcript, or an
embedding — that is the privacy posture, enforced here by rejecting any
such key. Rows carry a retention horizon; ``prune`` drops expired ones.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from voiceguard.eval.provenance import git_commit

# keys that must never appear in an audit row, at any nesting level
_FORBIDDEN = {
    "audio",
    "waveform",
    "wave",
    "pcm",
    "samples",
    "embedding",
    "embeddings",
    "transcript",
    "text",
    "utterance",
}

_DEFAULT_RETENTION_S = 24 * 3600


def _now() -> datetime:
    return datetime.now(UTC)


def _has_forbidden(obj) -> str | None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() in _FORBIDDEN:
                return str(k)
            hit = _has_forbidden(v)
            if hit:
                return hit
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            hit = _has_forbidden(v)
            if hit:
                return hit
    return None


@dataclass
class AuditEvent:
    event_id: str
    utc: str
    session_id: str
    scenario: str
    model: str
    front_end: str
    git_commit: str
    signals: dict  # {"spoof": 0.78, "speaker": 0.59, "context": 0.45}
    fused: float
    action: str
    reasons: list[str] = field(default_factory=list)
    retention_seconds: int = _DEFAULT_RETENTION_S


class AuditLog:
    def __init__(self, path: str | Path, *, model: str = "", front_end: str = ""):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.front_end = front_end
        self._commit = git_commit()

    def write(
        self,
        *,
        session_id: str,
        scenario: str,
        signals: dict,
        fused: float,
        action: str,
        reasons: list[str],
        retention_seconds: int = _DEFAULT_RETENTION_S,
    ) -> AuditEvent:
        ev = AuditEvent(
            event_id=uuid.uuid4().hex[:12],
            utc=_now().isoformat(timespec="seconds"),
            session_id=session_id,
            scenario=scenario,
            model=self.model,
            front_end=self.front_end,
            git_commit=self._commit,
            signals={k: (round(v, 4) if isinstance(v, float) else v) for k, v in signals.items()},
            fused=round(float(fused), 4),
            action=action,
            reasons=list(reasons),
            retention_seconds=retention_seconds,
        )
        row = asdict(ev)
        hit = _has_forbidden(row)
        if hit:
            raise ValueError(f"audit rows are feature-only; refusing key {hit!r}")
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return ev

    def read(self, session_id: str | None = None) -> list[dict]:
        if not self.path.exists():
            return []
        rows = [
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if session_id is not None:
            rows = [r for r in rows if r.get("session_id") == session_id]
        return rows

    def prune(self, *, now: datetime | None = None) -> int:
        """Drop expired rows in place. Returns the number removed."""
        if not self.path.exists():
            return 0
        now = now or _now()
        kept, removed = [], 0
        for r in self.read():
            try:
                age = (now - datetime.fromisoformat(r["utc"])).total_seconds()
                alive = age < r.get("retention_seconds", _DEFAULT_RETENTION_S)
            except (KeyError, ValueError):
                alive = True
            if alive:
                kept.append(r)
            else:
                removed += 1
        if removed:
            self.path.write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept), encoding="utf-8"
            )
        return removed
