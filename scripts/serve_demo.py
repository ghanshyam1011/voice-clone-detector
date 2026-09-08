"""Serve the VoiceGuard demo UI on localhost.

    python scripts/serve_demo.py
    python scripts/serve_demo.py --port 8080 --model aasist-l --no-open

Loads the pretrained AASIST (inference only), then opens
http://127.0.0.1:8000 in a browser. Two panels: analyse an audio file, or
run a live-microphone risk meter. Bind to localhost only -- this is a demo
server, not a hardened service.
"""

# ruff: noqa: E402
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse
import threading
import webbrowser


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="aasist", choices=["aasist", "aasist-l"])
    ap.add_argument("--ours", action="store_true", help="use models/cm/<model>/best.pt instead")
    ap.add_argument(
        "--weights",
        default=None,
        help="a .pth under models/pretrained/, e.g. AASIST_modern.pth (a fine-tune)",
    )
    ap.add_argument("--no-open", action="store_true", help="don't open a browser")
    args = ap.parse_args()

    import uvicorn

    from voiceguard.serve import create_app

    app = create_app(args.model, pretrained=not args.ours, weights=args.weights)
    url = f"http://{args.host}:{args.port}"
    if not args.no_open:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    print(f"\n  VoiceGuard demo  ->  {url}\n  Ctrl+C to stop\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
