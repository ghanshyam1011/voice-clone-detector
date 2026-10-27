"""System-level evaluation: the fused VoiceGuard decision, not one signal.

    python scripts/eval_system.py --split eval --speakers 25

For each target speaker in an ASVspoof 2019 LA split (official protocol files
only), enrol the session from real bona fide clips, then run held-out "calls"
through the real SessionManager (spoof + speaker + prosody + fusion + policy):

    genuine   bona fide clips of the enrolled speaker (not used to enrol)  -> should ALLOW
    clone     spoof clips targeting the enrolled speaker                   -> should be flagged
    impostor  bona fide clips of a *different* human                       -> should be flagged

Each call concatenates CALL_CLIPS clips (~8 s) so the read is a full,
non-provisional window. Call context is held neutral (known caller, no
transfer) so the number reflects the acoustic signals + fusion only; real
context would only add risk to attacks.

Reports, per trial type: ALLOW/VERIFY/ESCALATE rates; and for the fused score
and for each signal alone: EER (genuine vs attack) -- so the fusion is judged
against its own components on the *same trials*.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

from voiceguard.audio import load_wave
from voiceguard.config import REPO_ROOT, load_config, resolve
from voiceguard.data import load_manifest
from voiceguard.eval import compute_eer, git_commit
from voiceguard.serve.session import build_session_stack

CALL_CLIPS = 3
ENROL_CLIPS = 4
SR = 16000


def _cat(paths):
    return np.concatenate([load_wave(REPO_ROOT / p, SR) for p in paths])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="eval", choices=["dev", "eval"])
    ap.add_argument("--speakers", type=int, default=25)
    ap.add_argument(
        "--scenario", default="routine", choices=["routine", "transaction", "privileged"]
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    cfg = load_config()
    df = load_manifest(resolve(cfg, "manifests") / f"asvspoof19_la_{args.split}.csv")
    bona = df[df.label == "bonafide"]
    spoof = df[df.label == "spoof"]

    need = ENROL_CLIPS + 2 * CALL_CLIPS
    ok = [
        s
        for s, g in bona.groupby("speaker_id")
        if len(g) >= need and (spoof.speaker_id == s).sum() >= 2 * CALL_CLIPS
    ]
    spk = list(rng.choice(sorted(ok), size=min(args.speakers, len(ok)), replace=False))
    print(f"{args.split}: {len(spk)} speakers with enough audio (of {len(ok)})")

    mgr = build_session_stack("aasist", pretrained=True)
    rows = []
    for s in tqdm(spk, desc="speakers"):
        b = bona[bona.speaker_id == s].sample(frac=1, random_state=int(rng.integers(1e9)))
        enrol_p = b.path.iloc[:ENROL_CLIPS].tolist()
        gen = [
            b.path.iloc[ENROL_CLIPS + i * CALL_CLIPS : ENROL_CLIPS + (i + 1) * CALL_CLIPS].tolist()
            for i in range(2)
        ]
        sp = spoof[spoof.speaker_id == s].sample(frac=1, random_state=int(rng.integers(1e9)))
        clones = [
            sp.iloc[i * CALL_CLIPS : (i + 1) * CALL_CLIPS]
            for i in range(4)
            if (i + 1) * CALL_CLIPS <= len(sp)
        ]
        others = rng.choice([o for o in ok if o != s], size=2, replace=False)
        imp = [
            bona[bona.speaker_id == o]
            .sample(CALL_CLIPS, random_state=int(rng.integers(1e9)))
            .path.tolist()
            for o in others
        ]

        trials = (
            [("genuine", "bonafide", p) for p in gen]
            + [("clone", c.attack_id.iloc[0], c.path.tolist()) for c in clones]
            + [("impostor", "bonafide", p) for p in imp]
        )
        enrol_wave = _cat(enrol_p)
        for kind, att, paths in trials:
            sess = mgr.create(args.scenario)
            mgr.enroll(sess.session_id, enrol_wave)
            r = mgr.assess(sess.session_id, _cat(paths), ema=False)
            rows.append(
                dict(
                    speaker=s,
                    kind=kind,
                    attack=att,
                    action=r["action"],
                    fused=r["fused"],
                    spoof=r["spoof"],
                    speaker_risk=r["speaker"],
                    prosody=r["prosody"],
                )
            )
            mgr.end(sess.session_id)

    res = pd.DataFrame(rows)
    out = REPO_ROOT / "results" / "tables" / f"system_eval_{args.split}.csv"
    res.to_csv(out, index=False)

    print(f"\n=== {args.split}  n={len(res)}  scenario={args.scenario}  ({git_commit()}) ===")
    for k in ("genuine", "clone", "impostor"):
        g = res[res.kind == k]
        rates = {a: f"{(g.action == a).mean():.1%}" for a in ("ALLOW", "VERIFY", "ESCALATE")}
        print(f"{k:9s} n={len(g):3d}  {rates}")

    gen = res[res.kind == "genuine"]
    flagged = lambda g: (g.action != "ALLOW").mean()  # noqa: E731
    print(f"\nfalse-flag rate on genuine calls : {flagged(gen):.1%}")
    print(f"clone detection (VERIFY+ESCALATE): {flagged(res[res.kind == 'clone']):.1%}")
    print(f"impostor detection               : {flagged(res[res.kind == 'impostor']):.1%}")
    att = res[res.kind != "genuine"]
    sys_acc = ((gen.action == "ALLOW").sum() + (att.action != "ALLOW").sum()) / len(res)
    bal = ((gen.action == "ALLOW").mean() + (att.action != "ALLOW").mean()) / 2
    print(f"overall accuracy (ALLOW=genuine, else attack): {sys_acc:.1%}  balanced: {bal:.1%}")

    print("\nEER, genuine vs (clone+impostor), same trials:")
    for col in ("fused", "spoof", "speaker_risk", "prosody"):
        d = res.dropna(subset=[col])
        y = (d.kind == "genuine").astype(int).values  # 1 = bonafide
        eer = compute_eer(y, -d[col].values)[0]
        print(f"  {col:13s} {100 * eer:5.1f}%")
    print("\nEER, genuine vs clone only:")
    d0 = res[res.kind != "impostor"]
    for col in ("fused", "spoof", "speaker_risk", "prosody"):
        d = d0.dropna(subset=[col])
        y = (d.kind == "genuine").astype(int).values
        e = compute_eer(y, -d[col].values)[0]
        print(f"  {col:13s} {100 * e:5.1f}%")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
