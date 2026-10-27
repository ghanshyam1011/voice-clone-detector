"""How much of the customer's speech does an attacker need before the system stops catching a clone?

    python scripts/eval_clone_effort.py --split dev --sizes 4,8,16,32

The kNN-VC clones in ``docs/modern_attacks.md`` used about 25 s of the target's speech. A
better-prepared attacker has more. This builds clones of each speaker from nested reference sets
of growing size (so each set contains the previous one), puts every clone through the real
session stack against that speaker's enrolment, and reports per reference size how close the
clone's voice is to the enrolled one and how many clone calls the system flags.

Genuine calls of the same speakers are scored the same way as the baseline (``k = 0``).
Reference audio never includes the enrolment clips or the genuine test calls. Writes
``results/tables/clone_effort_<split>.csv``.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from voiceguard.audio import load_wave
from voiceguard.config import REPO_ROOT, load_config, resolve
from voiceguard.data import load_manifest
from voiceguard.eval.trials import choose_speakers, eligible_speakers, plan_speaker
from voiceguard.serve.session import build_session_stack

SR = 16000
TAB = REPO_ROOT / "results" / "tables"


def _tools():
    path = REPO_ROOT / "scripts" / "make_modern_clones.py"
    spec = importlib.util.spec_from_file_location("make_modern_clones", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["make_modern_clones"] = mod
    spec.loader.exec_module(mod)
    return mod


def _cat(paths):
    return np.concatenate([load_wave(REPO_ROOT / p, SR) for p in paths])


def _row(r, **kw):
    ev = r.get("evidence") or {}
    return dict(
        **kw,
        similarity=r["similarity"],
        speaker_risk=r["speaker"],
        spoof=r["spoof"],
        p_clone=ev.get("synthetic"),
        p_mismatch=ev.get("voice_mismatch"),
        fused=r["fused"],
        action=r["action"],
    )


def main() -> None:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "eval"])
    ap.add_argument("--speakers", type=int, default=0, help="0 = all eligible")
    ap.add_argument("--sizes", default="4,8,16,32", help="reference clips per clone, nested")
    ap.add_argument("--calls", type=int, default=2, help="clone calls per speaker per size")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    sizes = [int(x) for x in args.sizes.split(",")]

    cfg = load_config()
    df = load_manifest(resolve(cfg, "manifests") / f"asvspoof19_la_{args.split}.csv")
    bona, spoof = df[df.label == "bonafide"], df[df.label == "spoof"]
    ok = eligible_speakers(bona, spoof)
    spk = choose_speakers(ok, args.speakers, args.seed)

    tts = pd.read_csv(REPO_ROOT / "data" / "modern" / "manifest.csv")
    sources = tts[tts.generator == "mms"].path.tolist()
    if len(sources) < 20:
        raise SystemExit("MMS clips missing: run scripts/make_modern_clones.py --gen mms")

    mc = _tools()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vc = mc._load_knnvc(dev)
    mgr = build_session_stack(
        "aasist", pretrained=True, audit_path=Path(tempfile.gettempdir()) / "vg_effort_audit.jsonl"
    )
    rng = np.random.default_rng(args.seed)
    rows = []
    for s in tqdm(spk, desc="speakers"):
        plan = plan_speaker(bona, spoof, ok, s, args.seed)
        used = set(plan.enrol) | {p for k, _, ps in plan.trials if k == "genuine" for p in ps}
        spare = [p for p in bona[bona.speaker_id == s].path if p not in used]
        spare = [spare[i] for i in rng.permutation(len(spare))]
        sess = mgr.create("routine")
        mgr.enroll(sess.session_id, _cat(plan.enrol))
        for kind, _att, paths in plan.trials:  # baseline: the speaker's own genuine calls
            if kind == "genuine":
                mgr.reset_call(sess.session_id)
                r = mgr.assess(sess.session_id, _cat(paths), ema=False)
                rows.append(_row(r, speaker=s, k=0, ref_s=0.0, kind="genuine"))
        for k in sizes:
            refs = spare[:k]
            ref_s = sum(sf.info(str(REPO_ROOT / p)).duration for p in refs)
            match = vc.get_matching_set([mc._tensor(REPO_ROOT / p) for p in refs])
            for _ in range(args.calls):
                parts = []
                for src in rng.choice(sources, size=3, replace=False):
                    q = vc.get_features(mc._tensor(REPO_ROOT / src))
                    with torch.no_grad():
                        parts.append(vc.match(q, match, topk=4).cpu().numpy())
                wave = np.concatenate(parts).astype(np.float32)
                mgr.reset_call(sess.session_id)
                r = mgr.assess(sess.session_id, wave, ema=False)
                rows.append(_row(r, speaker=s, k=k, ref_s=ref_s, kind="clone"))
        mgr.end(sess.session_id)

    out = pd.DataFrame(rows)
    path = TAB / f"clone_effort_{args.split}.csv"
    out.to_csv(path, index=False)
    out["flagged"] = out.action.isin(["VERIFY", "ESCALATE"])
    out["escalated"] = out.action == "ESCALATE"
    g = out.groupby(["kind", "k"]).agg(
        calls=("fused", "size"),
        ref_seconds=("ref_s", "mean"),
        cosine=("similarity", "mean"),
        speaker_risk=("speaker_risk", "mean"),
        p_clone=("p_clone", "mean"),
        flagged=("flagged", "mean"),
        escalated=("escalated", "mean"),
    )
    print("\nclone quality and what the system does about it, by amount of reference speech")
    print(g.round(3).to_string())
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
