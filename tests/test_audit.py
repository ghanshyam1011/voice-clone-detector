"""Feature-only audit log."""

from datetime import UTC, datetime, timedelta

import pytest

from voiceguard.audit import AuditLog


def _log(tmp_path):
    return AuditLog(tmp_path / "audit.jsonl", model="aasist", front_end="640b0644ae")


def test_write_and_read_roundtrip(tmp_path):
    log = _log(tmp_path)
    ev = log.write(
        session_id="s1",
        scenario="transaction",
        signals={"spoof": 0.781, "speaker": 0.59, "context": 0.45},
        fused=0.66,
        action="ESCALATE",
        reasons=["Synthetic voice: score 0.78", "Caller number not recognised"],
    )
    rows = log.read("s1")
    assert len(rows) == 1
    assert rows[0]["event_id"] == ev.event_id
    assert rows[0]["action"] == "ESCALATE"
    assert rows[0]["signals"]["spoof"] == 0.781
    assert rows[0]["git_commit"]  # provenance stamped


def test_rejects_raw_audio_keys(tmp_path):
    log = _log(tmp_path)
    with pytest.raises(ValueError):
        log.write(
            session_id="s1",
            scenario="routine",
            signals={"spoof": 0.2, "embedding": [0.1, 0.2]},  # forbidden nested key
            fused=0.2,
            action="ALLOW",
            reasons=[],
        )


def test_no_audio_field_anywhere(tmp_path):
    log = _log(tmp_path)
    log.write(
        session_id="s1",
        scenario="routine",
        signals={"spoof": 0.1},
        fused=0.1,
        action="ALLOW",
        reasons=[],
    )
    raw = (tmp_path / "audit.jsonl").read_text()
    for k in ("audio", "waveform", "pcm", "transcript", "embedding"):
        assert k not in raw


def test_prune_drops_expired_rows(tmp_path):
    log = _log(tmp_path)
    log.write(
        session_id="s1",
        scenario="routine",
        signals={"spoof": 0.1},
        fused=0.1,
        action="ALLOW",
        reasons=[],
        retention_seconds=60,
    )
    assert log.prune(now=datetime.now(UTC) + timedelta(seconds=30)) == 0
    assert log.prune(now=datetime.now(UTC) + timedelta(seconds=120)) == 1
    assert log.read() == []
