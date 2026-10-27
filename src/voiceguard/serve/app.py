"""FastAPI app for the local demo UI. See :mod:`voiceguard.serve` for the routes."""

from __future__ import annotations

import asyncio
import io
import json
import threading
from functools import partial
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
from fastapi import (
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from voiceguard.audio.quality import PcmStream, sanitise, stream_rate
from voiceguard.detect import NB_SAMP
from voiceguard.detect.scorer import _classify  # keep verdict banding in one place
from voiceguard.risk.policy import SCENARIOS
from voiceguard.serve import demo_clips
from voiceguard.serve.session import LIVE_BUFFER_S, LiveBuffer, build_session_stack

_STATIC = Path(__file__).parent / "static"
_MAX_BYTES = 30 * 1024 * 1024
_MAX_SECONDS = 150.0
_MIN_SECONDS = 0.2
_SESSION_BUF_S = LIVE_BUFFER_S  # rolling audio kept per streaming session


async def _json_body(request: Request) -> dict:
    try:
        body = await request.json()
        return body if isinstance(body, dict) else {}
    except Exception:  # noqa: BLE001 -- empty / non-JSON body
        return {}


def _decode(raw: bytes, filename: str, target_sr: int) -> np.ndarray:
    """bytes of an audio file -> mono float32 at ``target_sr``."""
    try:
        y, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    except Exception:  # noqa: BLE001 -- exotic container (mp3/m4a): retry via librosa
        import os
        import tempfile
        import warnings

        import librosa

        suffix = Path(filename or "clip").suffix or ".wav"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)  # noqa: SIM115
        try:
            tmp.write(raw)
            tmp.close()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                y, sr = librosa.load(tmp.name, sr=None, mono=True)
        finally:
            os.unlink(tmp.name)
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    # a float WAV can carry NaN/Inf, a bad decoder int16-range samples: fix them here,
    # before a resampler smears one bad sample across the clip or a model crashes on it
    y, notes, corrupt = sanitise(y)
    if corrupt:
        raise ValueError(notes[0])
    if sr != target_sr:
        y = soxr.resample(y, sr, target_sr).astype(np.float32)
    return np.ascontiguousarray(y, dtype=np.float32)


def _rate_from_message(text: str, default: int) -> int:
    """The ``{"sampleRate": N}`` control message of the streaming sockets; anything
    malformed (not JSON, not an object, absurd rate) leaves the current rate alone."""
    try:
        msg = json.loads(text)
    except ValueError:
        return default
    return stream_rate(msg.get("sampleRate"), default) if isinstance(msg, dict) else default


def _is_end_message(text: str) -> bool:
    """The ``{"end": true}`` control message: the caller's audio is over, give the last verdict."""
    try:
        msg = json.loads(text)
    except ValueError:
        return False
    return isinstance(msg, dict) and msg.get("end") is True


def _verdict_payload(v, duration_s: float, window_s: float) -> dict:
    return {
        "label": v.label,
        "risk": round(float(v.risk), 4),
        "first_high_s": None if v.first_high_s is None else round(float(v.first_high_s), 2),
        "duration_s": round(float(duration_s), 2),
        "window_s": round(float(window_s), 2),
        "n_windows": int(len(v.window_risks)),
        "window_times": [round(float(t), 3) for t in v.window_times],
        "window_risks": [round(float(r), 4) for r in v.window_risks],
        "smoothed": [round(float(s), 4) for s in v.smoothed],
    }


def create_app(
    model_name: str = "aasist",
    *,
    pretrained: bool = True,
    weights: str | None = None,
    prosody: bool | None = None,
    warm: bool = False,
) -> FastAPI:
    """``warm=True`` pushes one synthetic call through every model before returning, so the
    first real caller does not wait on CUDA start-up and lazy model loads (the servers pass
    it; tests, which build many apps, do not)."""
    sessions = build_session_stack(
        model_name, pretrained=pretrained, weights=weights, prosody=prosody
    )
    if warm:
        print(f"[voiceguard.serve] warmed up in {sessions.warm():.1f} s", flush=True)
    scorer = sessions.scorer
    window_s = NB_SAMP / scorer.sr
    lock = threading.Lock()  # serialise model access across the upload + socket handlers

    def _score_windows(w: np.ndarray) -> np.ndarray:
        with lock:
            return scorer.score_windows(w)

    def _score_waveform(y: np.ndarray, hop_s: float):
        with lock:
            return scorer.score_waveform(y, hop_s=hop_s)

    app = FastAPI(title="VoiceGuard demo", docs_url=None, redoc_url=None)

    @app.get("/", response_class=HTMLResponse)
    def console() -> str:
        return (_STATIC / "console.html").read_text(encoding="utf-8")

    @app.get("/detector", response_class=HTMLResponse)
    def detector() -> str:
        return (_STATIC / "index.html").read_text(encoding="utf-8")

    @app.get("/api/info")
    def info() -> dict:
        return {
            "model": scorer.name,
            "device": str(scorer.device),
            "dev_eer": scorer.dev_eer,
            "sample_rate": scorer.sr,
            "window_s": round(window_s, 2),
            "genuine_below": 0.35,
            "synthetic_above": 0.65,
            # where the risk meter turns from "looks genuine" to "double-check" to "stop",
            # per kind of call: the console draws its zones from these, never from a copy
            "scenarios": {
                name: {"check_from": warn, "stop_from": stop}
                for name, (warn, stop) in SCENARIOS.items()
            },
            "learned_fusion": sessions.fusion is not None,
        }

    @app.get("/api/demo/clips")
    def demo_clip_list() -> dict:
        return {"clips": demo_clips.available()}

    @app.get("/api/demo/clips/{clip_id}")
    def demo_clip(clip_id: str):
        path = demo_clips.clip_path(clip_id)  # a catalogue lookup, never a path join
        if path is None:
            raise HTTPException(404, "No such sample clip.")
        return FileResponse(path, media_type="audio/wav")

    @app.post("/api/score")
    async def score(file: UploadFile = File(...), hop: float = 1.0):  # noqa: B008
        raw = await file.read()
        if len(raw) > _MAX_BYTES:
            raise HTTPException(413, "File too large (30 MB max).")
        loop = asyncio.get_event_loop()
        try:
            y = await loop.run_in_executor(None, _decode, raw, file.filename or "", scorer.sr)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(400, f"Could not decode audio: {exc}") from exc
        dur = len(y) / scorer.sr
        if dur < _MIN_SECONDS:
            raise HTTPException(400, "Clip too short.")
        if dur > _MAX_SECONDS:
            raise HTTPException(400, f"Clip too long ({dur:.0f}s; {_MAX_SECONDS:.0f}s max).")
        hop = float(np.clip(hop, 0.25, 5.0))
        v = await loop.run_in_executor(None, _score_waveform, y, hop)
        return JSONResponse(_verdict_payload(v, dur, window_s))

    @app.websocket("/api/stream")
    async def stream(ws: WebSocket) -> None:
        await ws.accept()
        loop = asyncio.get_event_loop()
        in_sr = scorer.sr
        hop_samples = scorer.sr // 2  # re-score every ~0.5 s of new audio
        buf = np.zeros(0, dtype=np.float32)
        pcm = PcmStream()
        ema: float | None = None
        since = 0
        try:
            while True:
                msg = await ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                if msg.get("text") is not None:
                    in_sr = _rate_from_message(msg["text"], in_sr)
                    continue
                data = msg.get("bytes")
                if not data:
                    continue

                chunk = pcm.feed(data)
                if in_sr != scorer.sr and len(chunk):
                    chunk = soxr.resample(chunk, in_sr, scorer.sr).astype(np.float32)
                buf = np.concatenate([buf, chunk])[-NB_SAMP:]
                since += len(chunk)
                if since < hop_samples or len(buf) < scorer.sr // 2:
                    continue
                since = 0

                level = float(np.abs(buf[-scorer.sr :]).max())  # peak over the last ~1 s
                warming = len(buf) < NB_SAMP
                w = np.tile(buf, NB_SAMP // len(buf) + 1)[:NB_SAMP] if warming else buf
                risk = float((await loop.run_in_executor(None, _score_windows, w[None, :]))[0])
                ema = risk if ema is None else 0.6 * ema + 0.4 * risk
                await ws.send_json(
                    {
                        "risk": round(ema, 4),
                        "instant": round(risk, 4),
                        "label": _classify(ema),
                        "warming": warming,
                        "filled_s": round(len(buf) / scorer.sr, 1),
                        "window_s": round(window_s, 1),
                        "level": round(level, 4),
                    }
                )
        except WebSocketDisconnect:
            pass
        except Exception as exc:  # noqa: BLE001
            try:
                await ws.send_json({"error": str(exc)})
            except Exception:  # noqa: BLE001
                pass

    # ---------- operator-console session layer ----------

    def _session_or_404(sid: str):
        s = sessions.get(sid)
        if s is None:
            raise HTTPException(404, "unknown or expired session")
        return s

    def _ctx_out(ctx) -> dict:
        return {
            "caller_known": ctx.caller_known,
            "amount": ctx.amount,
            "channel": ctx.channel,
            "prior_flags": ctx.prior_flags,
        }

    @app.post("/api/session")
    async def new_session(request: Request):
        b = await _json_body(request)
        s = sessions.create(
            b.get("scenario", "routine"),
            caller_known=b.get("caller_known", True),
            amount=b.get("amount", 0.0),
            channel=b.get("channel", "pstn"),
        )
        return {"session_id": s.session_id, "scenario": s.scenario, "context": _ctx_out(s.context)}

    @app.post("/api/session/{sid}/context")
    async def session_context(sid: str, request: Request):
        _session_or_404(sid)
        ctx = sessions.set_context(sid, **(await _json_body(request)))
        return {"context": _ctx_out(ctx)}

    @app.post("/api/session/{sid}/enroll")
    async def session_enroll(sid: str, file: UploadFile = File(...)):  # noqa: B008
        _session_or_404(sid)
        raw = await file.read()
        if len(raw) > _MAX_BYTES:
            raise HTTPException(413, "File too large (30 MB max).")
        loop = asyncio.get_event_loop()
        try:
            y = await loop.run_in_executor(None, _decode, raw, file.filename or "", scorer.sr)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(400, f"Could not decode audio: {exc}") from exc
        try:
            res = await loop.run_in_executor(None, sessions.enroll, sid, y)
        except ValueError as exc:  # decodes fine, but it is not clear speech
            raise HTTPException(422, str(exc)) from exc
        return {
            "enrolled": True,
            "seconds": round(res.seconds, 2),
            "speech_seconds": round(res.speech_seconds, 2),
            "short": res.short,
        }

    @app.post("/api/session/{sid}/analyze")
    async def session_analyze(sid: str, file: UploadFile = File(...), hop: float = 1.0):  # noqa: B008
        _session_or_404(sid)
        raw = await file.read()
        if len(raw) > _MAX_BYTES:
            raise HTTPException(413, "File too large (30 MB max).")
        loop = asyncio.get_event_loop()
        try:
            y = await loop.run_in_executor(None, _decode, raw, file.filename or "", scorer.sr)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(400, f"Could not decode audio: {exc}") from exc
        dur = len(y) / scorer.sr
        if dur < _MIN_SECONDS:
            raise HTTPException(400, "Clip too short.")
        # a clip shorter than the model's fixed window was tiled/repeated to score it
        # (see CMScorer.score_waveform) -- flag it provisional so a short upload can't
        # unilaterally ESCALATE off a read built from repeated, not real, audio.
        short_clip = len(y) < NB_SAMP
        out = await loop.run_in_executor(
            None,
            partial(sessions.assess, sid, y, ema=False, warming=short_clip),
        )
        # the per-window risk timeline is only meaningful for audio that holds speech
        out["clip"] = None
        if out["quality"]["usable"]:
            v = await loop.run_in_executor(None, _score_waveform, y, float(np.clip(hop, 0.25, 5.0)))
            out["clip"] = _verdict_payload(v, dur, window_s)
        return JSONResponse(out)

    @app.post("/api/session/{sid}/reset")
    def session_reset(sid: str):
        """A new call on the same session: keeps the registered voice and call details."""
        _session_or_404(sid)
        sessions.reset_call(sid)
        return {"reset": True}

    @app.post("/api/session/{sid}/acknowledge")
    async def session_acknowledge(sid: str, request: Request):
        _session_or_404(sid)
        by = (await _json_body(request)).get("by", "agent")
        return {"acknowledged": True, "audit_event": sessions.acknowledge(sid, by=str(by))}

    @app.get("/api/session/{sid}/audit")
    def session_audit(sid: str):
        _session_or_404(sid)
        return {"events": sessions.audit.read(sid)}

    @app.websocket("/api/session/{sid}/stream")
    async def session_stream(ws: WebSocket, sid: str) -> None:
        await ws.accept()
        if sessions.get(sid) is None:
            await ws.send_json({"error": "unknown or expired session"})
            await ws.close()
            return
        loop = asyncio.get_event_loop()
        live = LiveBuffer(scorer.sr)
        wake, closed, ending = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def reader() -> None:
            """Takes audio off the socket the moment it arrives, whatever the assessor is
            doing, so no stale audio queues up behind a slow assessment."""
            try:
                while True:
                    msg = await ws.receive()
                    if msg.get("type") == "websocket.disconnect":
                        break
                    if msg.get("text") is not None:
                        if _is_end_message(msg["text"]):
                            ending.set()
                            wake.set()
                        else:
                            live.set_rate(_rate_from_message(msg["text"], live.in_sr))
                        continue
                    data = msg.get("bytes")
                    if data and live.feed(data):
                        wake.set()
            except WebSocketDisconnect:
                pass
            finally:
                closed.set()
                wake.set()

        reader_task = asyncio.create_task(reader())
        last: dict | None = None
        try:
            while True:
                await wake.wait()
                wake.clear()
                if closed.is_set():
                    break
                finishing = ending.is_set()
                # a tail shorter than one hop never triggers a read of its own: at the end of
                # the audio, assess whatever has not been assessed so the last verdict covers
                # the whole call, or say that the previous one already does
                snap = live.snapshot() if live.unassessed or not finishing else None
                if snap is None:
                    if finishing:
                        if last is not None:
                            await ws.send_json({**last, "final": True})
                        break
                    continue
                buf, heard_s = snap
                level = float(np.abs(buf[-scorer.sr :]).max())
                out = await loop.run_in_executor(
                    None,
                    partial(
                        sessions.assess,
                        sid,
                        buf,
                        level=round(level, 4),
                        warming=len(buf) < NB_SAMP,
                    ),
                )
                out["heard_s"] = round(heard_s, 2)  # how much of the call this verdict covers
                if finishing:
                    out["final"] = True
                last = out
                await ws.send_json(out)
                if finishing:
                    break
        except WebSocketDisconnect:
            pass
        except Exception as exc:  # noqa: BLE001
            try:
                await ws.send_json({"error": str(exc)})
            except Exception:  # noqa: BLE001
                pass
        finally:
            reader_task.cancel()

    return app
