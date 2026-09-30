"""gRPC transport for the same session decision engine the REST/WS API in
:mod:`voiceguard.serve.app` uses. Both call :func:`build_session_stack`
and drive the returned :class:`~voiceguard.serve.session.SessionManager` --
this module only adds a second way to reach it. See docs/grpc.md.

    python scripts/serve_grpc.py

Inference only, not hardened for the public internet -- bind to localhost,
same posture as the REST demo server.
"""

from __future__ import annotations

import asyncio
from functools import partial

import grpc
import numpy as np
import soxr

from voiceguard.serve.app import _decode
from voiceguard.serve.proto import voiceguard_pb2 as pb
from voiceguard.serve.proto import voiceguard_pb2_grpc as pb_grpc
from voiceguard.serve.session import SessionManager, build_session_stack

_MAX_BYTES = 30 * 1024 * 1024
_MAX_SECONDS = 150.0
_MIN_SECONDS = 0.2
_SESSION_BUF_S = 12.0  # rolling audio kept per streaming session, matches app.py


def _context_to_proto(ctx: dict) -> pb.ContextState:
    return pb.ContextState(
        caller_known=ctx["caller_known"],
        amount=ctx["amount"],
        channel=ctx["channel"],
        prior_flags=ctx.get("prior_flags", 0),
    )


def _audit_event_to_proto(ev: dict | None) -> pb.AuditEvent | None:
    if ev is None:
        return None
    return pb.AuditEvent(
        event_id=ev["event_id"],
        utc=ev["utc"],
        session_id=ev["session_id"],
        action=ev["action"],
        fused=ev["fused"],
        signals={k: v for k, v in ev["signals"].items() if v is not None},
        reasons=ev["reasons"],
        scenario=ev.get("scenario", ""),
        model=ev.get("model", ""),
        front_end=ev.get("front_end", ""),
        git_commit=ev.get("git_commit", ""),
    )


def _assessment_to_proto(d: dict) -> pb.Assessment:
    kw = {
        "context": d["context"],
        "similarity": d["similarity"] or 0.0,
        "enrolled": d["enrolled"],
        "fused": d["fused"],
        "action": d["action"],
        "scenario": d["scenario"],
        "reasons": d["reasons"],
        "recommendation": d["recommendation"],
        "context_state": _context_to_proto(d["context_state"]),
        "level": d["level"] or 0.0,
        "warming": d["warming"],
        "provisional": d["provisional"],
    }
    # proto3 `optional` scalars: only set the ones that have a real reading,
    # so the client can tell "not scored this hop" (unset) from "scored 0.0".
    for name in ("spoof", "speaker", "prosody", "speech_onset_s", "first_confirmed_read_s"):
        if d.get(name) is not None:
            kw[name] = d[name]
    audit_event = _audit_event_to_proto(d.get("audit_event"))
    if audit_event is not None:
        kw["audit_event"] = audit_event
    return pb.Assessment(**kw)


async def _session_or_abort(sessions: SessionManager, sid: str, context):
    # grpc.aio's context.abort() is itself a coroutine that always raises
    # (NoReturn) once awaited -- it must be awaited to actually take effect,
    # and once it does, nothing after it in the caller runs.
    s = sessions.get(sid)
    if s is None:
        await context.abort(grpc.StatusCode.NOT_FOUND, "unknown or expired session")
    return s


async def _decode_or_abort(raw: bytes, filename: str, sr: int, context, loop):
    if len(raw) > _MAX_BYTES:
        await context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, "audio too large (30 MB max)")
    try:
        return await loop.run_in_executor(None, _decode, raw, filename, sr)
    except Exception as exc:  # noqa: BLE001 -- surfaced to the caller as INVALID_ARGUMENT
        await context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"could not decode audio: {exc}")


class VoiceGuardServicer(pb_grpc.VoiceGuardServicer):
    def __init__(self, sessions: SessionManager):
        self.sessions = sessions

    async def CreateSession(self, request, context):
        s = self.sessions.create(
            request.scenario or "routine",
            caller_known=request.caller_known if request.HasField("caller_known") else True,
            amount=request.amount,
            channel=request.channel or "pstn",
        )
        return pb.SessionState(
            session_id=s.session_id,
            scenario=s.scenario,
            context=_context_to_proto(
                {
                    "caller_known": s.context.caller_known,
                    "amount": s.context.amount,
                    "channel": s.context.channel,
                    "prior_flags": s.context.prior_flags,
                }
            ),
        )

    async def UpdateContext(self, request, context):
        await _session_or_abort(self.sessions, request.session_id, context)
        kw = {}
        if request.HasField("scenario"):
            kw["scenario"] = request.scenario
        if request.HasField("caller_known"):
            kw["caller_known"] = request.caller_known
        if request.HasField("amount"):
            kw["amount"] = request.amount
        if request.HasField("channel"):
            kw["channel"] = request.channel
        if request.HasField("prior_flags"):
            kw["prior_flags"] = request.prior_flags
        ctx = self.sessions.set_context(request.session_id, **kw)
        return _context_to_proto(
            {
                "caller_known": ctx.caller_known,
                "amount": ctx.amount,
                "channel": ctx.channel,
                "prior_flags": ctx.prior_flags,
            }
        )

    async def Enroll(self, request, context):
        await _session_or_abort(self.sessions, request.session_id, context)
        loop = asyncio.get_event_loop()
        y = await _decode_or_abort(
            request.audio, request.filename, self.sessions.sr, context, loop
        )
        secs = await loop.run_in_executor(None, self.sessions.enroll, request.session_id, y)
        return pb.EnrollResponse(enrolled=True, seconds=round(secs, 2), short=secs < 4.0)

    async def Analyze(self, request, context):
        await _session_or_abort(self.sessions, request.session_id, context)
        loop = asyncio.get_event_loop()
        y = await _decode_or_abort(
            request.audio, request.filename, self.sessions.sr, context, loop
        )
        dur = len(y) / self.sessions.sr
        if dur < _MIN_SECONDS:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "clip too short")
        if dur > _MAX_SECONDS:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"clip too long ({dur:.0f}s)")
        short_clip = len(y) < self.sessions.nb_samp
        out = await loop.run_in_executor(
            None,
            partial(
                self.sessions.assess,
                request.session_id,
                y,
                ema=False,
                warming=short_clip,
            ),
        )
        return _assessment_to_proto(out)

    async def Acknowledge(self, request, context):
        await _session_or_abort(self.sessions, request.session_id, context)
        ev = self.sessions.acknowledge(request.session_id, by=request.by or "agent")
        return pb.AcknowledgeResponse(acknowledged=True, audit_event=_audit_event_to_proto(ev))

    async def GetAudit(self, request, context):
        await _session_or_abort(self.sessions, request.session_id, context)
        events = self.sessions.audit.read(request.session_id)
        return pb.AuditTrail(events=[_audit_event_to_proto(e) for e in events])

    async def StreamAudio(self, request_iterator, context):
        # Mirrors app.py's WS /api/session/{sid}/stream hop-by-hop: same
        # rolling buffer, same hop size, same SessionManager.assess() call --
        # only the transport differs.
        loop = asyncio.get_event_loop()
        sid: str | None = None
        in_sr = self.sessions.sr
        hop_samples = self.sessions.sr // 2
        cap = int(_SESSION_BUF_S * self.sessions.sr)
        buf = np.zeros(0, dtype=np.float32)
        since = 0
        async for chunk in request_iterator:
            if sid is None:
                sid = chunk.session_id
                if await _session_or_abort(self.sessions, sid, context) is None:
                    return
            if chunk.sample_rate:
                in_sr = chunk.sample_rate
            pcm = np.frombuffer(chunk.pcm_f32le, dtype=np.float32)
            if in_sr != self.sessions.sr and len(pcm):
                pcm = soxr.resample(pcm, in_sr, self.sessions.sr).astype(np.float32)
            buf = np.concatenate([buf, pcm])[-cap:]
            since += len(pcm)
            if since < hop_samples or len(buf) < self.sessions.sr // 2:
                continue
            since = 0
            level = float(np.abs(buf[-self.sessions.sr :]).max())
            warming = len(buf) < self.sessions.nb_samp
            out = await loop.run_in_executor(
                None,
                partial(self.sessions.assess, sid, buf, level=round(level, 4), warming=warming),
            )
            yield _assessment_to_proto(out)


async def serve(
    model_name: str = "aasist",
    *,
    pretrained: bool = True,
    weights: str | None = None,
    host: str = "127.0.0.1",
    port: int = 50051,
) -> None:
    sessions = build_session_stack(model_name, pretrained=pretrained, weights=weights)
    server = grpc.aio.server(
        options=[
            ("grpc.max_receive_message_length", _MAX_BYTES),
            ("grpc.max_send_message_length", _MAX_BYTES),
        ]
    )
    pb_grpc.add_VoiceGuardServicer_to_server(VoiceGuardServicer(sessions), server)
    addr = f"{host}:{port}"
    server.add_insecure_port(addr)
    print(f"\n  VoiceGuard gRPC  ->  {addr}\n  Ctrl+C to stop\n")
    await server.start()
    await server.wait_for_termination()
