"""Operator-console session layer: create → enrol → analyse → audit.

Model-backed; skips without the pretrained AASIST weights (the speaker
model downloads on first use)."""

import io
import wave
from pathlib import Path

import numpy as np
import pytest

import voiceguard.serve.session as session_mod
from voiceguard.audio.quality import InputQuality
from voiceguard.audio.quality import analyze as real_analyze

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


def _treat_as_speech(y, sr=16000, *, window_s=None):
    """The lifecycle tests drive the API with synthetic tones. The real input gate
    (rightly) refuses anything that is not speech, so for these tests it is stubbed;
    the gate itself is tested in test_quality.py and by the *_real_gate tests below."""
    y = np.asarray(y, np.float32).reshape(-1)
    d = len(y) / sr
    return y, InputQuality(True, "ok", "", d, d, -25.0, 0.0, "stub", []), []


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from voiceguard.serve import create_app

    mp = pytest.MonkeyPatch()
    mp.setattr(session_mod, "analyze", _treat_as_speech)
    try:
        app = create_app("aasist", pretrained=True)
    except Exception as exc:  # noqa: BLE001
        mp.undo()
        pytest.skip(f"app unavailable: {exc}")
    yield TestClient(app)
    mp.undo()


@pytest.fixture
def real_gate(monkeypatch):
    monkeypatch.setattr(session_mod, "analyze", real_analyze)


def _float_wav(y, sr=16000) -> bytes:
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV", subtype="FLOAT")
    return buf.getvalue()


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
    client.post(f"/api/session/{sid}/context", json={"caller_known": False, "amount": 75000})
    b = client.post(
        f"/api/session/{sid}/analyze",
        files={"file": ("call2.wav", _wav(6.0, seed=1), "audio/wav")},
    ).json()
    assert b["context"] > 0.0
    assert any("75,000" in reason for reason in b["reasons"])
    assert b["fused"] >= a["fused"]  # same clip: only the context changed, so it can only add risk

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
    r = client.post("/api/session/nope/analyze", files={"file": ("x.wav", _wav(3.0), "audio/wav")})
    assert r.status_code == 404


@needs_weights
def test_raw_detector_still_works(client):
    """The console layer must not break the original endpoints."""
    assert client.get("/api/info").json()["sample_rate"] == 16000
    r = client.post("/api/score", files={"file": ("x.wav", _wav(5.0), "audio/wav")})
    assert r.status_code == 200 and r.json()["label"] in {"GENUINE", "UNCERTAIN", "SYNTHETIC"}


# ---------- the real input gate, through the API ----------


@needs_weights
def test_silence_is_waiting_never_allow_or_escalate_real_gate(client, real_gate):
    sid = client.post("/api/session", json={"scenario": "transaction"}).json()["session_id"]
    silence = _float_wav(np.zeros(6 * 16000, np.float32))
    out = client.post(
        f"/api/session/{sid}/analyze", files={"file": ("quiet.wav", silence, "audio/wav")}
    ).json()
    assert out["action"] == "WAITING"
    assert out["quality"]["usable"] is False and out["quality"]["status"] == "no_speech"
    assert out["spoof"] is None and out["speaker"] is None
    assert out["clip"] is None and out["audit_event"] is None  # nothing was judged or logged
    assert client.get(f"/api/session/{sid}/audit").json()["events"] == []


@needs_weights
def test_context_still_counts_while_the_line_is_silent_real_gate(client, real_gate):
    sid = client.post(
        "/api/session",
        json={"scenario": "transaction", "caller_known": False, "amount": 75000},
    ).json()["session_id"]
    silence = _float_wav(np.zeros(6 * 16000, np.float32))
    out = client.post(
        f"/api/session/{sid}/analyze", files={"file": ("quiet.wav", silence, "audio/wav")}
    ).json()
    # an unknown caller asking for a large transfer is a fact whether or not anyone spoke
    assert out["action"] in {"VERIFY", "ESCALATE"}
    assert any("context alone" in r for r in out["reasons"])
    assert out["spoof"] is None


@needs_weights
def test_enrolling_silence_is_refused_real_gate(client, real_gate):
    sid = client.post("/api/session", json={}).json()["session_id"]
    silence = _float_wav(np.zeros(6 * 16000, np.float32))
    r = client.post(f"/api/session/{sid}/enroll", files={"file": ("e.wav", silence, "audio/wav")})
    assert r.status_code == 422 and "enrolment needs" in r.json()["detail"]


@needs_weights
def test_nan_audio_never_crashes_the_api_real_gate(client, real_gate):
    sid = client.post("/api/session", json={}).json()["session_id"]
    some_bad = np.zeros(6 * 16000, np.float32)
    some_bad[::4000] = np.nan
    r = client.post(
        f"/api/session/{sid}/analyze", files={"file": ("n.wav", _float_wav(some_bad), "audio/wav")}
    )
    assert r.status_code == 200 and r.json()["action"] == "WAITING"
    all_bad = np.full(6 * 16000, np.nan, np.float32)
    r = client.post(
        f"/api/session/{sid}/analyze", files={"file": ("n.wav", _float_wav(all_bad), "audio/wav")}
    )
    assert r.status_code == 400 and "corrupt" in r.json()["detail"]


@needs_weights
def test_the_response_carries_what_the_console_explains_with(client):
    sid = client.post("/api/session", json={"scenario": "routine"}).json()["session_id"]
    client.post(
        f"/api/session/{sid}/enroll", files={"file": ("e.wav", _wav(6.0, seed=1), "audio/wav")}
    )
    a = client.post(
        f"/api/session/{sid}/analyze", files={"file": ("c.wav", _wav(6.0, seed=1), "audio/wav")}
    ).json()
    assert set(a["evidence"]) == {"synthetic", "voice_mismatch"}  # the two learned heads
    assert set(a["evidence_level"]) == {"synthetic", "voice_mismatch"}
    assert set(a["evidence_level"].values()) <= {"none", "some", "likely", "strong"}
    # how the recording looks, for support: numbers only, and never the absolute level
    assert {"floor_rel_db", "env_acf", "hf_ratio", "flatness"} <= set(a["channel"])
    assert "level_db" not in a["channel"]
    assert all(0.0 <= v <= 1.0 for v in a["evidence"].values())
    assert isinstance(a["audio_notes"], list)
    assert all(isinstance(n, str) for n in a["audio_notes"])


@needs_weights
def test_info_describes_the_risk_zones_the_console_draws_from(client):
    from voiceguard.risk.policy import SCENARIOS

    d = client.get("/api/info").json()
    assert set(d["scenarios"]) == set(SCENARIOS)
    for name, (warn, stop) in SCENARIOS.items():
        assert d["scenarios"][name] == {"check_from": warn, "stop_from": stop}
        assert 0 < warn < stop < 1


@needs_weights
def test_demo_clip_routes_serve_only_the_catalogue(client):
    listed = client.get("/api/demo/clips").json()["clips"]
    for c in listed:  # whatever is on disk is described and downloadable
        assert {"id", "role", "title", "text", "truth", "seconds"} <= set(c)
        r = client.get(f"/api/demo/clips/{c['id']}")
        assert r.status_code == 200 and r.headers["content-type"] == "audio/wav"
    assert client.get("/api/demo/clips/not_a_sample").status_code == 404
    assert client.get("/api/demo/clips/..%2F..%2Fpyproject.toml").status_code == 404


@needs_weights
def test_reset_route_starts_a_new_call_and_unknown_sessions_404(client):
    sid = client.post("/api/session", json={}).json()["session_id"]
    assert client.post(f"/api/session/{sid}/reset").json() == {"reset": True}
    assert client.post("/api/session/nosuchsession/reset").status_code == 404


def test_reset_call_forgets_the_call_but_keeps_the_voice_and_the_call_details(tmp_path):
    from voiceguard.audit import AuditLog

    class _Verifier:
        def is_enrolled(self, sid):
            return True

        def drop(self, sid):
            pass

    mgr = session_mod.SessionManager(
        None, _Verifier(), AuditLog(tmp_path / "a.jsonl"), sr=16000, nb_samp=64600
    )
    s = mgr.create("transaction", caller_known=False, amount=75000)
    s.fused_ema, s.sig_ema, s.last_action = 0.9, {"spoof": 0.8}, "ESCALATE"
    s.last_signals, s.last_fused, s.last_reasons = {"spoof": 0.8}, 0.9, ["x"]
    s.speech_onset_s, s.first_confirmed_read_s = 1.2, 4.0
    mgr.reset_call(s.session_id)
    assert s.fused_ema is None and s.sig_ema == {} and s.last_action is None
    assert s.last_signals == {} and s.last_fused == 0.0 and s.last_reasons == []
    assert s.speech_onset_s is None and s.first_confirmed_read_s is None
    assert (s.scenario, s.context.caller_known, s.context.amount) == ("transaction", False, 75000)


@needs_weights
def test_the_stream_reports_how_much_of_the_call_each_verdict_covers(client):
    import json

    sid = client.post("/api/session", json={"scenario": "routine"}).json()["session_id"]
    client.post(
        f"/api/session/{sid}/enroll", files={"file": ("e.wav", _wav(6.0, seed=1), "audio/wav")}
    )
    pcm = np.frombuffer(_wav(5.0, seed=1)[44:], dtype="<i2").astype(np.float32) / 32768.0
    heard = []
    with client.websocket_connect(f"/api/session/{sid}/stream") as ws:
        ws.send_text(json.dumps({"sampleRate": 16000}))
        for i in range(0, len(pcm), 4096):  # all of it at once: far faster than real time
            ws.send_bytes(pcm[i : i + 4096].astype("<f4").tobytes())
        for _ in range(40):
            m = ws.receive_json()
            heard.append(m["heard_s"])
            if m["heard_s"] >= 4.9:
                break
    assert heard == sorted(heard)
    assert heard[-1] == pytest.approx(5.0, abs=0.1)  # the latest verdict covers the whole call
    assert {"action", "fused", "evidence", "audio_notes", "quality"} <= set(m)


@needs_weights
def test_the_end_message_brings_a_last_verdict_that_covers_the_whole_call(client):
    import json

    sid = client.post("/api/session", json={"scenario": "routine"}).json()["session_id"]
    client.post(
        f"/api/session/{sid}/enroll", files={"file": ("e.wav", _wav(6.0, seed=1), "audio/wav")}
    )
    pcm = np.frombuffer(_wav(4.3, seed=1)[44:], dtype="<i2").astype(np.float32) / 32768.0
    msgs = []
    with client.websocket_connect(f"/api/session/{sid}/stream") as ws:
        ws.send_text(json.dumps({"sampleRate": 16000}))
        for i in range(0, len(pcm), 4096):
            ws.send_bytes(pcm[i : i + 4096].astype("<f4").tobytes())
        ws.send_text(json.dumps({"end": True}))  # a 0.3 s tail would never get a read of its own
        for _ in range(40):
            msgs.append(ws.receive_json())
            if msgs[-1].get("final"):
                break
    assert msgs[-1].get("final") is True
    assert msgs[-1]["heard_s"] == pytest.approx(4.3, abs=0.05)
    assert not any("final" in m for m in msgs[:-1])


def test_end_message_is_recognised_only_when_it_says_end():
    from voiceguard.serve.app import _is_end_message

    assert _is_end_message('{"end": true}')
    for text in ('{"end": false}', '{"sampleRate": 16000}', "end", "[1]", "", '{"end": 1}'):
        assert not _is_end_message(text)
