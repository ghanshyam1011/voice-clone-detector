"""Replay whole calls through the live streaming path and log every update.

    python scripts/build_stream_trials.py --split eval --speakers 10 --conds clean,babble10
    python scripts/eval_stream.py --tag _stream

Every other table in this repo scores ONE read of a ~10 s call. The product runs for the
whole call, re-scoring every 0.5 s through ``SessionManager.assess`` (the WebSocket path:
per-signal EMAs, then a fused EMA, then thresholds). Two things only show up over time:

  * how soon a clone is caught (time to first alert), and
  * how often a GENUINE caller is flagged at some moment of a long call -- a per-read
    false-alarm rate compounds across the dozens of reads a call has.

This replays calls exactly as ``serve/app.py`` feeds the model (same 0.5 s cadence, same
rolling buffer, same ``warming`` / ``level`` arguments, a fresh enrolled session per call)
and writes one row per update. Policy variants (persistence, longer memory) are then
scored offline from the logged scores by ``scripts/eval_stream.py``.

A call is ``--clips`` clips of ONE source: the enrolled speaker (genuine), one ASVspoof
attack id targeting them (clone) or one other human (impostor). The degradation is applied
to the whole call, identically for every class.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from voiceguard.audio import load_wave
from voiceguard.audio.degrade import apply_condition
from voiceguard.config import REPO_ROOT, load_config, resolve
from voiceguard.data import load_manifest
from voiceguard.data.torch_dataset import NB_SAMP
from voiceguard.eval import git_commit
from voiceguard.eval.trials import ENROL_CLIPS, choose_speakers, eligible_speakers, plan_speaker
from voiceguard.serve.app import _SESSION_BUF_S
from voiceguard.serve.session import build_session_stack

SR = 16000


def _cat(paths):
    return np.concatenate([load_wave(REPO_ROOT / p, SR) for p in paths])


def plan_calls(bona, spoof, ok, speaker, enrol, seed, clips, n_gen, n_clone):
    """-> [(kind, attack_id, [manifest paths])], every call from a single source."""
    rng = np.random.default_rng([seed, 4242, int(speaker.split("_")[-1])])
    mine = bona[(bona.speaker_id == speaker) & ~bona.path.isin(set(enrol))].path
    mine = mine.sample(frac=1, random_state=int(rng.integers(1e9))).tolist()
    calls = [
        ("genuine", "bonafide", mine[i * clips : (i + 1) * clips])
        for i in range(n_gen)
        if len(mine) >= (i + 1) * clips
    ]
    sp = spoof[spoof.speaker_id == speaker]
    size = sp.groupby("attack_id").size()
    usable = sorted(size[size >= clips].index)
    for a in rng.choice(usable, size=min(n_clone, len(usable)), replace=False) if usable else []:
        pick = sp[sp.attack_id == a].path.sample(clips, random_state=int(rng.integers(1e9)))
        calls.append(("clone", str(a), pick.tolist()))
    other = rng.choice([o for o in ok if o != speaker])
    theirs = bona[bona.speaker_id == other].path
    if len(theirs) >= clips:
        pick = theirs.sample(clips, random_state=int(rng.integers(1e9)))
        calls.append(("impostor", "bonafide", pick.tolist()))
    return calls


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="eval", choices=["dev", "eval"])
    ap.add_argument("--speakers", type=int, default=10)
    ap.add_argument("--clips", type=int, default=8, help="clips per call (~3.5 s each)")
    ap.add_argument("--genuine-calls", type=int, default=2)
    ap.add_argument("--clone-calls", type=int, default=2)
    ap.add_argument("--conds", default="clean,babble10")
    ap.add_argument("--step", type=float, default=0.5, help="seconds between updates (live: 0.5)")
    ap.add_argument("--start", type=float, default=1.0, help="first update after this many s")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--tag", default="_stream")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    conds = tuple(args.conds.split(","))

    cfg = load_config()
    df = load_manifest(resolve(cfg, "manifests") / f"asvspoof19_la_{args.split}.csv")
    bona, spoof = df[df.label == "bonafide"], df[df.label == "spoof"]
    ok = eligible_speakers(bona, spoof)
    need = ENROL_CLIPS + args.genuine_calls * args.clips
    has = bona.groupby("speaker_id").size()
    ok_long = [s for s in ok if has[s] >= need]
    spk = choose_speakers(ok_long, args.speakers, args.seed)
    print(f"{args.split}: {len(spk)} speakers, {args.clips} clips/call, conditions {conds}")

    mgr = build_session_stack(
        "aasist",
        pretrained=True,
        weights=args.weights,
        audit_path=Path(tempfile.gettempdir()) / "voiceguard_trials_audit.jsonl",
    )
    out = REPO_ROOT / "results" / "tables" / f"stream_trials{args.tag}_{args.split}.csv"
    rows, done = [], set()
    if args.resume and out.exists():
        prev = pd.read_csv(out)
        rows, done = prev.to_dict("records"), set(prev.speaker.unique())

    cap, step, start = int(_SESSION_BUF_S * SR), int(args.step * SR), int(args.start * SR)
    for s in tqdm(spk, desc="speakers"):
        if s in done:
            continue
        plan = plan_speaker(bona, spoof, ok, s, args.seed)
        enrol = _cat(plan.enrol)
        calls = plan_calls(
            bona,
            spoof,
            ok,
            s,
            plan.enrol,
            args.seed,
            args.clips,
            args.genuine_calls,
            args.clone_calls,
        )
        for ci, (kind, att, paths) in enumerate(calls):
            base = _cat(paths)
            for cond in conds:
                babble = lambda r, n, pool=plan.pool: _cat(  # noqa: E731
                    list(r.choice(pool, size=3, replace=False))
                )
                y = apply_condition(cond, base, plan.rng, babble=babble)
                sess = mgr.create("routine")  # a fresh call: no smoothing carried over
                mgr.enroll(sess.session_id, enrol)
                for end in range(start, len(y) + 1, step):
                    buf = y[:end][-cap:]
                    level = round(float(np.abs(buf[-SR:]).max()), 4)
                    t0 = time.perf_counter()
                    r = mgr.assess(sess.session_id, buf, level=level, warming=len(buf) < NB_SAMP)
                    q = r["quality"]
                    rows.append(
                        dict(
                            split=args.split,
                            speaker=s,
                            call=ci,
                            kind=kind,
                            attack=att,
                            cond=cond,
                            t=round(end / SR, 2),
                            fused=r["fused"],
                            instant=r["instant_fused"],
                            action=r["action"],
                            spoof=r["spoof"],
                            speaker_risk=r["speaker"],
                            similarity=r["similarity"],
                            warming=r["warming"],
                            provisional=r["provisional"],
                            speech_s=q["speech_s"],
                            status=q["status"],
                            detector=mgr.scorer.name,
                            op_point=round(mgr.scorer.operating_point, 6),
                            ms=round(1000 * (time.perf_counter() - t0), 1),
                        )
                    )
                mgr.end(sess.session_id)
        pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}  ({len(rows)} updates, {git_commit()})", flush=True)


if __name__ == "__main__":
    main()
