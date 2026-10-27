"""VoiceGuard over gRPC: the same SessionManager test_session.py drives over
REST, reached through voiceguard.serve.grpc_service instead. Runs a real
grpc.aio server in a background thread and drives it with a sync client --
this is proof the two transports produce the same decisions, not just that
the generated stubs import."""

from __future__ import annotations

import asyncio
import io
import threading
import wave
from pathlib import Path

import grpc
import numpy as np
import pytest

import voiceguard.serve.session as session_mod
from voiceguard.audio.quality import InputQuality
from voiceguard.audio.quality import analyze as real_analyze
from voiceguard.serve.proto import voiceguard_pb2 as pb
from voiceguard.serve.proto import voiceguard_pb2_grpc as pb_grpc

_W = Path(__file__).resolve().parents[1] / "models" / "pretrained" / "AASIST.pth"
needs_weights = pytest.mark.skipif(not _W.exists(), reason="pretrained AASIST weights absent")
_PORT = 50061  # distinct from serve_grpc.py's default 50051


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
    """These tests drive the API with synthetic tones; the real input gate refuses
    non-speech, so it is stubbed here (the gate has its own tests, and the
    *_real_gate tests below put it back)."""
    y = np.asarray(y, np.float32).reshape(-1)
    d = len(y) / sr
    return y, InputQuality(True, "ok", "", d, d, -25.0, 0.0, "stub", []), []


@pytest.fixture
def real_gate(monkeypatch):
    monkeypatch.setattr(session_mod, "analyze", real_analyze)


@pytest.fixture(scope="module")
def stub():
    from voiceguard.serve.grpc_service import _MAX_BYTES, VoiceGuardServicer
    from voiceguard.serve.session import build_session_stack

    mp = pytest.MonkeyPatch()
    mp.setattr(session_mod, "analyze", _treat_as_speech)
    try:
        sessions = build_session_stack("aasist", pretrained=True)
    except Exception as exc:  # noqa: BLE001
        mp.undo()
        pytest.skip(f"session stack unavailable: {exc}")

    loop = asyncio.new_event_loop()
    ready = threading.Event()
    state: dict = {}

    async def _start():
        server = grpc.aio.server(
            options=[
                ("grpc.max_receive_message_length", _MAX_BYTES),
                ("grpc.max_send_message_length", _MAX_BYTES),
            ]
        )
        pb_grpc.add_VoiceGuardServicer_to_server(VoiceGuardServicer(sessions), server)
        server.add_insecure_port(f"127.0.0.1:{_PORT}")
        await server.start()
        state["server"] = server
        ready.set()
        await server.wait_for_termination()

    def _run_loop():
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_start())

    thread = threading.Thread(target=_run_loop, daemon=True)
    thread.start()
    if not ready.wait(timeout=30):
        pytest.skip("gRPC server did not start in time")

    channel = grpc.insecure_channel(f"127.0.0.1:{_PORT}")
    grpc.channel_ready_future(channel).result(timeout=10)
    yield pb_grpc.VoiceGuardStub(channel)

    channel.close()
    # stopping waits for assess() calls still running in executor threads (a streamed-garbage
    # test leaves some), which takes longer when the whole suite is loading the GPU
    asyncio.run_coroutine_threadsafe(state["server"].stop(None), loop).result(timeout=90)
    thread.join(timeout=30)
    mp.undo()


@needs_weights
def test_session_lifecycle_and_audit(stub):
    sid = stub.CreateSession(pb.CreateSessionRequest(scenario="transaction")).session_id
    assert sid

    enr = stub.Enroll(
        pb.EnrollRequest(session_id=sid, audio=_wav(6.0, seed=1), filename="enrol.wav")
    )
    assert enr.enrolled is True

    a = stub.Analyze(
        pb.AnalyzeRequest(session_id=sid, audio=_wav(6.0, seed=1), filename="call.wav")
    )
    assert a.enrolled is True
    assert a.action in {"ALLOW", "VERIFY", "ESCALATE"}
    assert a.context == 0.0

    stub.UpdateContext(pb.UpdateContextRequest(session_id=sid, caller_known=False, amount=75000))
    b = stub.Analyze(
        pb.AnalyzeRequest(session_id=sid, audio=_wav(6.0, seed=1), filename="call2.wav")
    )
    assert b.context > 0.0
    assert any("75,000" in reason for reason in b.reasons)
    assert b.fused >= a.fused  # same clip: context can only add risk, as in the REST test

    trail = stub.GetAudit(pb.GetAuditRequest(session_id=sid))
    assert len(trail.events) >= 1
    for e in trail.events:
        assert {"spoof", "speaker", "context"} <= set(e.signals)  # + prosody if active
        assert e.git_commit


@needs_weights
def test_short_clip_is_provisional_long_clip_is_not(stub):
    sid = stub.CreateSession(pb.CreateSessionRequest(scenario="transaction")).session_id

    short = stub.Analyze(
        pb.AnalyzeRequest(session_id=sid, audio=_wav(1.0, seed=7), filename="short.wav")
    )
    assert short.provisional is True
    assert short.action != "ESCALATE"

    long = stub.Analyze(
        pb.AnalyzeRequest(session_id=sid, audio=_wav(6.0, seed=7), filename="long.wav")
    )
    assert long.provisional is False


@needs_weights
def test_analyze_unknown_session_not_found(stub):
    with pytest.raises(grpc.RpcError) as exc_info:
        stub.Analyze(pb.AnalyzeRequest(session_id="nope", audio=_wav(3.0), filename="x.wav"))
    assert exc_info.value.code() == grpc.StatusCode.NOT_FOUND


@needs_weights
def test_rest_and_grpc_agree_on_the_same_clip(stub):
    """Both transports call the same SessionManager -- the same clip through
    each should land on the same action, not just a similar-looking one."""
    from fastapi.testclient import TestClient

    from voiceguard.serve import create_app

    app = create_app("aasist", pretrained=True)
    client = TestClient(app)
    clip = _wav(6.0, seed=99)

    rest_sid = client.post("/api/session", json={"scenario": "routine"}).json()["session_id"]
    rest_out = client.post(
        f"/api/session/{rest_sid}/analyze", files={"file": ("c.wav", clip, "audio/wav")}
    ).json()

    grpc_sid = stub.CreateSession(pb.CreateSessionRequest(scenario="routine")).session_id
    grpc_out = stub.Analyze(pb.AnalyzeRequest(session_id=grpc_sid, audio=clip, filename="c.wav"))

    assert grpc_out.action == rest_out["action"]
    assert grpc_out.fused == pytest.approx(rest_out["fused"], abs=1e-6)


def _silence_wav(sec=6.0, sr=16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(np.zeros(int(sec * sr), "<i2").tobytes())
    return buf.getvalue()


@needs_weights
def test_silence_is_waiting_and_carries_the_quality_report_real_gate(stub, real_gate):
    sid = stub.CreateSession(pb.CreateSessionRequest(scenario="transaction")).session_id
    a = stub.Analyze(pb.AnalyzeRequest(session_id=sid, audio=_silence_wav(), filename="s.wav"))
    assert a.action == "WAITING"
    assert a.quality.usable is False and a.quality.status == "no_speech"
    assert "no speech" in a.quality.reason
    assert not a.HasField("spoof") and not a.HasField("speaker")  # nothing was scored
    assert len(stub.GetAudit(pb.GetAuditRequest(session_id=sid)).events) == 0


@needs_weights
def test_enrolling_silence_is_invalid_argument_real_gate(stub, real_gate):
    sid = stub.CreateSession(pb.CreateSessionRequest()).session_id
    with pytest.raises(grpc.RpcError) as exc_info:
        stub.Enroll(pb.EnrollRequest(session_id=sid, audio=_silence_wav(), filename="e.wav"))
    assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    assert "enrolment needs" in exc_info.value.details()


@needs_weights
def test_streamed_garbage_does_not_kill_the_stream_real_gate(stub, real_gate):
    sid = stub.CreateSession(pb.CreateSessionRequest()).session_id
    nan = np.full(16000, np.nan, "<f4").tobytes()
    odd = np.zeros(16000, "<f4").tobytes() + b"\x00\x00"  # a chunk ending mid-sample

    def chunks():
        for _ in range(3):
            yield pb.AudioChunk(session_id=sid, pcm_f32le=nan, sample_rate=16000)
            yield pb.AudioChunk(session_id=sid, pcm_f32le=odd, sample_rate=999999999)  # absurd rate

    got = list(stub.StreamAudio(chunks()))
    assert got and all(a.action == "WAITING" for a in got)
