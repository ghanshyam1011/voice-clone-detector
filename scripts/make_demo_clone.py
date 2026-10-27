"""Make the console's "a voice cloned with a free tool" sample, for the SAME customer as the
sample voice from ``scripts/pick_demo_clips.py``.

    python scripts/make_demo_clone.py

The clone is what an attacker with open-source tools does: text-to-speech reads a fraud script
(MMS-TTS), then kNN voice conversion (kNN-VC) turns that voice into the customer's, using about
25 seconds of the customer's real speech as the reference. The reference clips are different
from the ones in ``enrol.wav`` and ``benign.wav``, so the clone never hears the audio it is
tested against. Output: ``demo_audio/cloned_knnvc.wav`` (git-ignored, like all audio).

Needs the pieces ``docs/modern_attacks.md`` describes: the vetted kNN-VC copy and weights, and
the MMS-TTS clips from ``scripts/make_modern_clones.py --gen mms``.
"""

# ruff: noqa: E402
from __future__ import annotations

import importlib.util
import os
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np
import pandas as pd
import soundfile as sf

from voiceguard.audio import load_wave
from voiceguard.config import REPO_ROOT, load_config, resolve

SR = 16000
OUT = REPO_ROOT / "demo_audio"
# a believable fraud call, in two sentences
SCRIPT = (
    "Hello, I am calling about my bank account and a recent transaction.",
    "I need to transfer seventy five thousand rupees to a new account today.",
)


def _clone_tools():
    """The generator script's kNN-VC loader (vetted local copy, weights loaded as tensors)."""
    path = REPO_ROOT / "scripts" / "make_modern_clones.py"
    spec = importlib.util.spec_from_file_location("make_modern_clones", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["make_modern_clones"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    import torch

    mc = _clone_tools()
    cfg = load_config()
    man = pd.read_csv(resolve(cfg, "manifests") / "asvspoof19_la_eval.csv")
    bona = man[man.label == "bonafide"]
    speaker = bona.speaker_id.value_counts().idxmax()  # pick_demo_clips.py's rule
    theirs = bona[bona.speaker_id == speaker].path.tolist()
    refs = [REPO_ROOT / p for p in theirs[20:]][:10]  # clips 0-19 are enrol / benign
    if len(refs) < 6:
        raise SystemExit(f"speaker {speaker} has too few spare reference clips ({len(refs)})")

    tts = pd.read_csv(REPO_ROOT / "data" / "modern" / "manifest.csv")
    tts = tts[tts.generator == "mms"]
    sources = []
    for sentence in SCRIPT:
        hit = tts[tts.note == sentence]
        if hit.empty:
            raise SystemExit("MMS clips missing: run scripts/make_modern_clones.py --gen mms")
        sources.append(REPO_ROOT / hit.path.iloc[0])

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vc = mc._load_knnvc(dev)
    match = vc.get_matching_set([mc._tensor(p) for p in refs])
    parts = []
    for src in sources:
        q = vc.get_features(mc._tensor(src))
        with torch.no_grad():
            parts += [
                vc.match(q, match, topk=4).cpu().numpy(),
                np.zeros(int(0.35 * SR), np.float32),
            ]
    y = np.concatenate(parts[:-1]).astype(np.float32)
    OUT.mkdir(exist_ok=True)
    sf.write(OUT / "cloned_knnvc.wav", 0.95 * y / (np.abs(y).max() or 1.0), SR, subtype="PCM_16")
    # a clone's tail audio sanity check: it must be readable audio of a sensible length
    back = load_wave(OUT / "cloned_knnvc.wav", SR)
    print(f"speaker {speaker}: wrote demo_audio/cloned_knnvc.wav ({len(back) / SR:.1f} s)")


if __name__ == "__main__":
    main()
