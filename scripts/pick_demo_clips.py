"""Assemble a small set of demo clips for the console, from the ASVspoof
2019 data already on disk.

    python scripts/pick_demo_clips.py

Writes to demo_audio/:
    enrol.wav   ~8 s, one real speaker (for "Enrol from a file")
    benign.wav  ~5 s, the SAME speaker (drop as the benign call)
    cloned_1.wav / cloned_2.wav  synthetic-speech clips that the detector
                                 flags clearly (drop as the cloned call)

All 16 kHz mono WAV. Drag them onto the console at http://127.0.0.1:8000.
"""

# ruff: noqa: E402
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from voiceguard.audio import load_wave
from voiceguard.config import REPO_ROOT, load_config, resolve

OUT = REPO_ROOT / "demo_audio"
SR = 16000


def _dur(path: str) -> float:
    info = sf.info(path)
    return info.frames / info.samplerate


def _save(name: str, y: np.ndarray) -> None:
    y = np.asarray(y, dtype=np.float32)
    peak = float(np.max(np.abs(y))) or 1.0
    sf.write(OUT / name, (0.95 * y / peak).astype(np.float32), SR, subtype="PCM_16")
    print(f"  demo_audio/{name:14s} {len(y) / SR:4.1f}s")


def main() -> None:
    cfg = load_config()
    man = resolve(cfg, "manifests")
    OUT.mkdir(exist_ok=True)

    eval_df = pd.read_csv(man / "asvspoof19_la_eval.csv")
    bona = eval_df[eval_df.label == "bonafide"]

    # --- a real speaker with several clips ---
    speaker = bona.speaker_id.value_counts().idxmax()
    rows = bona[bona.speaker_id == speaker].head(20)
    long_enough = [r.path for r in rows.itertuples(index=False) if _dur(r.path) >= 2.5]
    if len(long_enough) < 4:
        long_enough = list(rows.path)
    print(f"real speaker {speaker}: {len(long_enough)} clips")

    enrol = np.concatenate([load_wave(p, SR) for p in long_enough[:3]])[: SR * 10]
    benign = np.concatenate([load_wave(p, SR) for p in long_enough[3:6]])[: SR * 8]
    _save("enrol.wav", enrol)
    _save("benign.wav", benign)

    # --- synthetic clips the detector flags clearly ---
    try:
        from voiceguard.detect import build_scorer

        scorer = build_scorer("aasist", pretrained=True)
    except Exception as exc:  # noqa: BLE001
        scorer = None
        print(f"(scorer unavailable: {exc} -- picking spoof clips by attack type only)")

    spoof = eval_df[eval_df.label == "spoof"]
    seed = cfg["seed"]
    # A08/A10/A16/A19 are among the strongest for AASIST; sample a short list first
    prefer = spoof[spoof.attack_id.isin(["A10", "A08", "A16", "A19"])].sample(
        30, random_state=seed
    )
    shortlist = [
        r.path for r in prefer.itertuples(index=False) if _dur(r.path) >= 3.0
    ][:20]
    print(f"scoring {len(shortlist)} synthetic candidates ...", flush=True)

    picked: list[tuple[str, float]] = []
    for path in shortlist:
        if len(picked) == 2:
            break
        if scorer is None:
            picked.append((path, float("nan")))
            continue
        risk = scorer.score_file(path).risk
        if risk >= 0.6:
            picked.append((path, risk))
    if not picked:  # scorer strict on this sample -- take the two loudest hits anyway
        scored = sorted(
            ((p, scorer.score_file(p).risk if scorer else float("nan")) for p in shortlist),
            key=lambda kv: (kv[1] if kv[1] == kv[1] else 0),
            reverse=True,
        )
        picked = scored[:2]

    for i, (path, risk) in enumerate(picked, 1):
        _save(f"cloned_{i}.wav", load_wave(path, SR))
        tag = f"risk {risk * 100:.0f}%" if risk == risk else "attack-type pick"
        print(f"     ({Path(path).stem}, {tag})")

    print("\nDone. Drag demo_audio/ clips onto the console:")
    print("  1. Enrol from a file  -> enrol.wav")
    print("  2. drop               -> benign.wav      (expect ALLOW)")
    print("  3. uncheck caller, amount 75000, drop -> cloned_1.wav   (expect ESCALATE)")


if __name__ == "__main__":
    main()
