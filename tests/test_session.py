"""Operator-console session layer: create → enrol → analyse → audit.

Model-backed; skips without the pretrained AASIST weights (the speaker
model downloads on first use)."""

import io
import wave
from pathlib import Path

import numpy as np
import pytest

_W = Path(__file__).resolve().parents[1] / "models" / "pretrained" / "AASIST.pth"
needs_weights = pytest.mark.skipif(not _W.exists(), reason="pretrained AASIST weights absent")


def _wav(seconds: float, seed: int = 0, sr: int = 16000) -> bytes:
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    f0 = 110 + 30 * rng.random()
    y = sum((1.0 / k) * np.sin(2 * np.pi * f0 * k * t) for k in (1, 2, 3))
    y = (0.2 * y / np.max(np.abs(y)) + 0.02 * rng.standard_normal(t.shape)).astype(np.float32)
    pcm = (np.clip(y, -1, 1) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from voiceguard.serve import create_app

    try:
        app = create_app("aasist", pretrained=True)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"app unavailable: {exc}")
    return TestClient(app)


@needs_weights
def test_session_lifecycle_and_audit(client, tmp_path):
    sid = client.post("/api/session", json={"scenario": "transaction"}).json()["session_id"]
    assert sid

    up = {"file": ("enrol.wav", _wav(6.0, seed=1), "audio/wav")}
    r = client.post(f"/api/session/{sid}/enroll", files=up)
    assert r.status_code == 200 and r.json()["enrolled"] is True

    # benign-ish clip, no context risk
    a = client.post(
        f"/api/session/{sid}/analyze", files={"file": ("call.wav", _wav(6.0, seed=1), "audio/wav")}
    ).json()
    need = {"spoof", "speaker", "prosody", "context", "fused", "action", "reasons", "signals"}
    assert need <= set(a)
    assert {s["name"] for s in a["signals"]} == {"spoof", "speaker", "prosody", "context"}
    assert a["enrolled"] is True
    assert a["action"] in {"ALLOW", "VERIFY", "ESCALATE"}
    assert a["context"] == 0.0

    # raise the stakes: unknown caller + large transfer
    client.post(
        f"/api/session/{sid}/context", json={"caller_known": False, "amount": 75000}
    )
    b = client.post(
        f"/api/session/{sid}/analyze",
        files={"file": ("call2.wav", _wav(6.0, seed=42), "audio/wav")},
    ).json()
    assert b["context"] > 0.0
    assert any("75,000" in reason for reason in b["reasons"])
    assert b["fused"] >= a["fused"]  # context pushed it up

    ev = client.get(f"/api/session/{sid}/audit").json()["events"]
    assert len(ev) >= 1
    for e in ev:
        assert "audio" not in e and "embedding" not in e and "transcript" not in e
        assert {"spoof", "speaker", "context"} <= set(e["signals"])  # + prosody if active
        assert e["git_commit"]


@needs_weights
def test_short_clip_is_provisional_long_clip_is_not(client):
    sid = client.post("/api/session", json={"scenario": "transaction"}).json()["session_id"]

    short = client.post(
        f"/api/session/{sid}/analyze",
        files={"file": ("short.wav", _wav(1.0, seed=7), "audio/wav")},
    ).json()
    assert short["provisional"] is True
    # a still-building read can flag VERIFY but must never unilaterally ESCALATE
    assert short["action"] != "ESCALATE"

    long = client.post(
        f"/api/session/{sid}/analyze",
        files={"file": ("long.wav", _wav(6.0, seed=7), "audio/wav")},
    ).json()
    assert long["provisional"] is False


@needs_weights
def test_speech_onset_and_first_read_latency_reported(client):
    sid = client.post("/api/session", json={"scenario": "routine"}).json()["session_id"]
    a = client.post(
        f"/api/session/{sid}/analyze",
        files={"file": ("call.wav", _wav(6.0, seed=3), "audio/wav")},
    ).json()
    # the synthetic clip has no leading silence, so onset should land near t=0
    assert a["speech_onset_s"] is not None
    assert 0.0 <= a["speech_onset_s"] < 0.5
    # a full (non-provisional) clip confirms immediately
    assert a["first_confirmed_read_s"] is not None
    assert a["first_confirmed_read_s"] >= 0.0


@needs_weights
def test_analyze_unknown_session_404(client):
    r = client.post(
        "/api/session/nope/analyze", files={"file": ("x.wav", _wav(3.0), "audio/wav")}
    )
    assert r.status_code == 404


@needs_weights
def test_raw_detector_still_works(client):
    """The console layer must not break the original endpoints."""
    assert client.get("/api/info").json()["sample_rate"] == 16000
    r = client.post("/api/score", files={"file": ("x.wav", _wav(5.0), "audio/wav")})
    assert r.status_code == 200 and r.json()["label"] in {"GENUINE", "UNCERTAIN", "SYNTHETIC"}
