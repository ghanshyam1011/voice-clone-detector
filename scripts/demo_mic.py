"""Live microphone voice-integrity monitor.

    python scripts/demo_mic.py

Captures the default mic, scores a rolling ~4 s window every ~1 s with the
pretrained AASIST, and prints an EMA-smoothed risk bar that updates in
place. Ctrl-C to stop.

Demo script: talk (bar stays low / green), then play a cloned/synthetic
clip through your speakers (bar climbs, verdict flips to SYNTHETIC).
"""

# ruff: noqa: E402
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse
import queue
import sys

import numpy as np

from voiceguard.detect import NB_SAMP, build_scorer

_BAR_W = 40


def _line(risk: float, label: str) -> str:
    fill = int(round(risk * _BAR_W))
    ch = "#" if risk >= 0.60 else ("-" if risk >= 0.35 else "=")
    colour = "\033[91m" if risk >= 0.60 else ("\033[93m" if risk >= 0.35 else "\033[92m")
    return f"\r{colour}[{ch * fill}{' ' * (_BAR_W - fill)}] {risk * 100:5.1f}%  {label:9s}\033[0m"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="aasist", choices=["aasist", "aasist-l"])
    ap.add_argument("--ours", action="store_true")
    ap.add_argument("--hop", type=float, default=1.0)
    ap.add_argument("--ema", type=float, default=0.6)
    args = ap.parse_args()

    import sounddevice as sd

    scorer = build_scorer(args.model, pretrained=not args.ours)
    sr = scorer.sr
    win = NB_SAMP
    hop = int(args.hop * sr)
    print(
        f"model: {scorer.name} | device: {scorer.device} | window {win / sr:.1f}s hop {args.hop}s"
    )
    print("listening -- Ctrl+C to stop\n")

    q: queue.Queue[np.ndarray] = queue.Queue()

    def cb(indata, frames, t, status):  # noqa: ARG001
        q.put(indata[:, 0].copy())

    buf = np.zeros(0, dtype=np.float32)
    ema = None
    with sd.InputStream(samplerate=sr, channels=1, dtype="float32", callback=cb, blocksize=hop):
        try:
            while True:
                buf = np.concatenate([buf, q.get()])
                if len(buf) < win:
                    continue
                window = buf[-win:]
                buf = buf[-win:]  # keep one window of context
                risk = float(scorer.score_windows(window[None, :])[0])
                ema = risk if ema is None else args.ema * ema + (1 - args.ema) * risk
                label = "SYNTHETIC" if ema >= 0.60 else ("UNCERTAIN" if ema >= 0.35 else "GENUINE")
                sys.stdout.write(_line(ema, label))
                sys.stdout.flush()
        except KeyboardInterrupt:
            print("\nstopped.")


if __name__ == "__main__":
    main()
