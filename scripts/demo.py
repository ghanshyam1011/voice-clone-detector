"""Score audio files for synthetic / cloned speech.

    python scripts/demo.py path/to/audio.wav [more.flac ...]
    python scripts/demo.py --sample 6      # 6 real + 6 fake random ASVspoof clips
    python scripts/demo.py --ours clip.wav # use our trained checkpoint

Prints a verdict, an overall risk score, and a per-window risk timeline.
Runs the pretrained AASIST by default -- inference only, GPU if available.
Best on known synthesis methods (~87% on ASVspoof dev); generalisation to
novel in-the-wild deepfakes is weaker and is the open problem (see plan).
"""

# ruff: noqa: E402
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse
from pathlib import Path

from voiceguard.config import load_config, resolve
from voiceguard.detect import build_scorer

_BAR_W = 32


def _sample_clips(n: int, wild: bool):
    import pandas as pd

    man = resolve(load_config(), "manifests")
    f = man / ("in_the_wild_eval.csv" if wild else "asvspoof19_la_eval.csv")
    d = pd.read_csv(f)
    out = []
    for lab in ("bonafide", "spoof"):
        g = d[d.label == lab]
        for _, r in g.sample(min(n, len(g)), random_state=7).iterrows():
            out.append((r["path"], lab))
    return out


def _bar(risk: float) -> str:
    fill = int(round(risk * _BAR_W))
    mark = "!" if risk >= 0.60 else ("?" if risk >= 0.35 else "=")
    return f"[{mark * fill}{' ' * (_BAR_W - fill)}] {risk * 100:5.1f}%"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--sample", type=int, metavar="N", help="score N real + N fake random clips")
    ap.add_argument("--wild", action="store_true", help="--sample from In-the-Wild (harder)")
    ap.add_argument("--model", default="aasist", choices=["aasist", "aasist-l"])
    ap.add_argument("--ours", action="store_true", help="use models/cm/<model>/best.pt")
    ap.add_argument("--hop", type=float, default=1.0)
    args = ap.parse_args()

    if not args.files and not args.sample:
        ap.error("pass audio file(s) or --sample N")

    scorer = build_scorer(args.model, pretrained=not args.ours)
    eer = f" | dev EER {scorer.dev_eer * 100:.1f}%" if scorer.dev_eer else ""
    print(f"model: {scorer.name}  |  device: {scorer.device}{eer}\n")

    items = [(Path(f), None) for f in args.files]
    if args.sample:
        items += [(Path(p), t) for p, t in _sample_clips(args.sample, args.wild)]

    hits = total = 0
    for p, truth in items:
        if not p.exists():
            print(f"  {p}: NOT FOUND")
            continue
        v = scorer.score_file(p, hop_s=args.hop)
        mark = ""
        if truth is not None:
            pred = "spoof" if v.risk >= 0.5 else "bonafide"
            total += 1
            hits += pred == truth
            mark = f"   [actual: {truth} - {'correct' if pred == truth else 'WRONG'}]"

        print(f"=== {p.name} ===")
        print(f"  VERDICT: {v.label:9s} ({v.emoji})   risk {v.risk * 100:.1f}%{mark}")
        if v.first_high_s is not None:
            print(f"  crossed 'synthetic' at ~{v.first_high_s:.1f}s into the clip")
        if len(v.window_risks) > 1:
            print("  timeline:")
            for t, r in zip(v.window_times, v.window_risks, strict=False):
                print(f"    {t:5.1f}s  {_bar(float(r))}")
        print()

    if total:
        print(f"--- {hits}/{total} correct ---")


if __name__ == "__main__":
    main()
