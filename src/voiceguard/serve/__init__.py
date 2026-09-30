"""Local web UI for the VoiceGuard demo (P2 serving layer, minimal).

`create_app()` returns a FastAPI app that wraps :class:`voiceguard.detect.CMScorer`:

    GET  /            single-page UI
    GET  /api/info    model / device / verdict thresholds
    POST /api/score   multipart audio file  -> verdict + per-window risk timeline
    WS   /api/stream  raw float32 PCM       -> rolling EMA risk updates

The session-oriented API (POST /api/session, .../enroll, .../context,
.../analyze, .../acknowledge, WS .../stream, GET .../audit) is also built
here (also in app.py) and additionally reachable over gRPC -- see
:mod:`voiceguard.serve.grpc_service` (`scripts/serve_grpc.py`,
`examples/grpc_client.py`, `docs/grpc.md`). Both transports call
:func:`voiceguard.serve.session.build_session_stack` and drive the same
:class:`~voiceguard.serve.session.SessionManager` -- one decision engine,
two doors.

Inference only. Not hardened for the public internet -- bind to localhost.
"""

from voiceguard.serve.app import create_app

__all__ = ["create_app"]
