"""Smoke tests for the demo web app. The model-backed checks need the
pretrained AASIST weights (committed to the repo, so normally present)."""

import io
import wave
from pathlib import Path

import numpy as np
import pytest

_W = Path(__file__).resolve().parents[1] / "models" / "pretrained" / "AASIST.pth"
needs_weights = pytest.mark.skipif(not _W.exists(), reason="pretrained AASIST weights absent")


def _wav_bytes(seconds: float, sr: int = 16000) -> bytes:
    n = int(seconds * sr)
    t = np.arange(n) / sr
    y = (0.2 * np.sin(2 * np.pi * 160 * t)).astype(np.float32)
    pcm = (np.clip(y, -1, 1) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def test_static_pages_present():
    static = Path(__file__).resolve().parents[1] / "src/voiceguard/serve/static"
    idx = (static / "index.html").read_text(encoding="utf-8")
    assert "/api/score" in idx and "/api/stream" in idx
    assert idx.count("<script") == idx.count("</script>")
    console = (static / "console.html").read_text(encoding="utf-8")
    assert "/api/session" in console and "/api/session/${SID}/analyze" in console
    assert console.count("<script") == console.count("</script>")


def test_create_app_importable():
    from voiceguard.serve import create_app  # noqa: F401


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from voiceguard.serve import create_app

    return TestClient(create_app("aasist", pretrained=True))


@needs_weights
def test_info(client):
    d = client.get("/api/info").json()
    assert d["sample_rate"] == 16000
    assert 0.0 < d["genuine_below"] < d["synthetic_above"] < 1.0
    assert d["window_s"] > 3.0


@needs_weights
def test_score_wav(client):
    r = client.post("/api/score", files={"file": ("tone.wav", _wav_bytes(6.0), "audio/wav")})
    assert r.status_code == 200
    d = r.json()
    assert d["label"] in {"GENUINE", "UNCERTAIN", "SYNTHETIC"}
    assert 0.0 <= d["risk"] <= 1.0
    assert len(d["window_risks"]) == d["n_windows"] >= 3
    assert len(d["smoothed"]) == len(d["window_times"]) == d["n_windows"]


@needs_weights
def test_score_rejects_junk(client):
    r = client.post("/api/score", files={"file": ("x.wav", b"not audio at all", "audio/wav")})
    assert r.status_code == 400


@needs_weights
def test_stream_ws(client):
    sr = 16000
    tone = (0.2 * np.sin(2 * np.pi * 150 * np.arange(sr * 6) / sr)).astype(np.float32)
    with client.websocket_connect("/api/stream") as ws:
        ws.send_text('{"sampleRate": 16000}')
        for i in range(0, len(tone), 4000):
            ws.send_bytes(tone[i : i + 4000].tobytes())
        msg = ws.receive_json()
        assert 0.0 <= msg["risk"] <= 1.0
        assert msg["label"] in {"GENUINE", "UNCERTAIN", "SYNTHETIC"}
        assert "filled_s" in msg
        assert msg["level"] > 0.05  # the tone is well above the silence floor


@needs_weights
def test_stream_resamples_non_16k(client):
    """Browser sends native-rate PCM (e.g. 48 kHz); the server resamples."""
    sr = 48000
    tone = (0.2 * np.sin(2 * np.pi * 150 * np.arange(sr * 6) / sr)).astype(np.float32)
    with client.websocket_connect("/api/stream") as ws:
        ws.send_text('{"sampleRate": 48000}')
        for i in range(0, len(tone), 8192):
            ws.send_bytes(tone[i : i + 8192].tobytes())
        msg = ws.receive_json()
        assert 0.0 <= msg["risk"] <= 1.0
        assert msg["level"] > 0.05
