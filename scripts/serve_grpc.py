"""Serve the VoiceGuard session API over gRPC on localhost.

    python scripts/serve_grpc.py
    python scripts/serve_grpc.py --port 50052 --model aasist-l

Same decision engine as scripts/serve_demo.py's REST/WS API (both call
voiceguard.serve.session.build_session_stack) -- this is a second transport,
not a second implementation. Bind to localhost only -- this is a demo
server, not a hardened service.
"""

# ruff: noqa: E402
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse
import asyncio


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=50051)
    ap.add_argument("--model", default="aasist", choices=["aasist", "aasist-l"])
    ap.add_argument("--ours", action="store_true", help="use models/cm/<model>/best.pt instead")
    ap.add_argument(
        "--weights",
        default=None,
        help="a .pth under models/pretrained/, e.g. AASIST_modern.pth (a fine-tune)",
    )
    args = ap.parse_args()

    from voiceguard.serve.grpc_service import serve

    asyncio.run(
        serve(
            args.model,
            pretrained=not args.ours,
            weights=args.weights,
            host=args.host,
            port=args.port,
        )
    )


if __name__ == "__main__":
    main()
