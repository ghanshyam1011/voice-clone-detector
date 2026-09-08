"""Local web UI for the VoiceGuard demo (P2 serving layer, minimal).

`create_app()` returns a FastAPI app that wraps :class:`voiceguard.detect.CMScorer`:

    GET  /            single-page UI
    GET  /api/info    model / device / verdict thresholds
    POST /api/score   multipart audio file  -> verdict + per-window risk timeline
    WS   /api/stream  raw float32 PCM       -> rolling EMA risk updates

Inference only. Not hardened for the public internet -- bind to localhost.
"""

from voiceguard.serve.app import create_app

__all__ = ["create_app"]
