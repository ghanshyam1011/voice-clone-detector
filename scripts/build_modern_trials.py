"""System-level trials where the attacks are MODERN voice clones, not 2019 ASVspoof attacks.

    python scripts/build_modern_trials.py --split eval --gens knnvc,speecht5x,bark,mms
    python scripts/eval_stress.py --tag _modern          # per-condition table, shipped fusion

Same trial design and genuine / impostor calls as ``build_system_trials.py`` (so the
numbers are directly comparable), with the attack calls replaced by clips made by
``scripts/make_modern_clones.py``:

  clone   a clip of a TARGETED generator (knnvc, speecht5x): made to sound like the
          enrolled speaker. The honest test of "does the system catch a cloned voice?"
  synth   a clip of an UNTARGETED generator (mms, bark): synthetic, but in someone
          else's voice. A synthetic impostor.

The ``attack`` column carries the generator, so results are reported per generator --
and a generator the detector never saw in training is a genuinely held-out test.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from voiceguard.audio import load_wave
from voiceguard.audio.channel import channel_features
from voiceguard.audio.degrade import apply_condition
from voiceguard.config import REPO_ROOT, load_config, resolve
from voiceguard.data import load_manifest
from voiceguard.eval import git_commit
from voiceguard.eval.trials import (
    CALL_CLIPS,
    choose_speakers,
    eligible_speakers,
    plan_speaker,
    window_stats,
)
from voiceguard.serve.session import build_session_stack

SR = 16000
TARGETED = {"knnvc", "speecht5x"}
MANIFEST = REPO_ROOT / "data" / "modern" / "manifest.csv"


def _cat(paths):
    return np.concatenate([load_wave(REPO_ROOT / p, SR) for p in paths])


def _calls(clips: list[str], rng, n_calls: int) -> list[list[str]]:
    clips = list(clips)
    rng.shuffle(clips)
    return [
        clips[i * CALL_CLIPS : (i + 1) * CALL_CLIPS]
        for i in range(n_calls)
        if len(clips) >= (i + 1) * CALL_CLIPS
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="eval", choices=["dev", "eval"])
    ap.add_argument("--gens", default="knnvc,speecht5x,bark,mms")
    ap.add_argument("--conds", default="clean,babble10,ulaw")
    ap.add_argument("--speakers", type=int, default=0)
    ap.add_argument(
        "--calls-per-gen", type=int, default=2, help="attack calls per speaker per generator"
    )
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--tag", default="_modern")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    gens, conds = args.gens.split(","), tuple(args.conds.split(","))

    man = pd.read_csv(MANIFEST)
    man = man[(man.split == args.split) & man.generator.isin(gens)]
    cfg = load_config()
    df = load_manifest(resolve(cfg, "manifests") / f"asvspoof19_la_{args.split}.csv")
    bona, spoof = df[df.label == "bonafide"], df[df.label == "spoof"]
    ok = eligible_speakers(bona, spoof)
    have = (
        set(man[man.generator.isin(TARGETED)].target_speaker) if TARGETED & set(gens) else set(ok)
    )
    spk = [
        s
        for s in choose_speakers(ok, args.speakers, args.seed)
        if s in have or not (TARGETED & set(gens))
    ]
    print(f"{args.split}: {len(spk)} speakers, generators {gens}, conditions {conds}", flush=True)

    mgr = build_session_stack(
        "aasist",
        pretrained=True,
        weights=args.weights,
        audit_path=Path(tempfile.gettempdir()) / "voiceguard_trials_audit.jsonl",
    )
    out = REPO_ROOT / "results" / "tables" / f"system_trials{args.tag}_{args.split}.csv"
    rows, done = [], set()
    if args.resume and out.exists():
        prev = pd.read_csv(out)
        rows, done = prev.to_dict("records"), set(prev.speaker.unique())

    for s in tqdm(spk, desc="speakers"):
        if s in done:
            continue
        plan = plan_speaker(bona, spoof, ok, s, args.seed)
        # keep the standard genuine / impostor calls; swap the attack calls for modern ones
        trials = [t for t in plan.trials if t[0] != "clone"]
        for g in gens:
            if g in TARGETED:
                clips = man[(man.generator == g) & (man.target_speaker == s)].path.tolist()
                kind = "clone"
            else:
                clips = man[man.generator == g].path.tolist()
                kind = "synth"
            for paths in _calls(clips, plan.rng, args.calls_per_gen):
                trials.append((kind, g, paths))

        sess = mgr.create("routine")
        mgr.enroll(sess.session_id, _cat(plan.enrol))
        for ti, (kind, att, paths) in enumerate(trials):
            base = _cat(paths)
            for cond in conds:
                babble = lambda r, n, pool=plan.pool: _cat(  # noqa: E731
                    list(r.choice(pool, size=3, replace=False))
                )
                y = apply_condition(cond, base, plan.rng, babble=babble)
                r = mgr.assess(sess.session_id, y, ema=False)
                rows.append(
                    dict(
                        split=args.split,
                        speaker=s,
                        trial=ti,
                        kind=kind,
                        attack=att,
                        cond=cond,
                        spoof=r["spoof"],
                        speaker_risk=r["speaker"],
                        prosody=r["prosody"],
                        similarity=r.get("similarity"),
                        speech_s=r["quality"]["speech_s"],
                        detector=mgr.scorer.name,
                        op_point=round(mgr.scorer.operating_point, 6),
                        **window_stats(r.get("spoof_windows")),
                        **channel_features(y),
                    )
                )
        mgr.end(sess.session_id)
        pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}  ({len(rows)} reads, {git_commit()})", flush=True)


if __name__ == "__main__":
    main()
