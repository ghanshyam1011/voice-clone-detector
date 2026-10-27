"""How does the SHIPPED system behave in situations it was never fitted on?

    python scripts/build_system_trials.py --split eval --speakers 20 --tag _stress --conds stress
    python scripts/eval_stress.py --tag _stress

Scores the cached reads (results/tables/system_trials<tag>_eval.csv) with the
shipped learned fusion (models/fusion/fusion.json) at its shipped operating
points -- no refitting, no threshold tuning on these reads -- and prints, per
situation: decision accuracy, genuine accepted, clones / impostors flagged,
ESCALATE rates, and the threshold-free EER (genuine vs attack).

Accuracy at the shipped threshold says how the product behaves; EER says whether
the signals still *separate* the classes in that situation. A situation with a
good EER but poor accuracy means the operating point shifted (fixable by fusion);
a poor EER means the detectors themselves lose the evidence (needs better signals
or training data).

Reference rows come from the untouched ``dev`` split of the standard conditions.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from voiceguard.config import REPO_ROOT
from voiceguard.eval import compute_eer
from voiceguard.risk.learned import DEFAULT_PATH, LearnedFusion

TAB = REPO_ROOT / "results" / "tables"


def _none_if_nan(x):
    return None if pd.isna(x) else float(x)


def score(df: pd.DataFrame, lf: LearnedFusion) -> pd.DataFrame:
    """A read the system declined to judge (no usable speech: spoof is NaN) is WAITING,
    not a decision -- it is excluded from accuracy and counted in ``ans``."""
    if "floor_rel_db" not in df.columns:  # tables built before the gain-invariant descriptor
        df = df.assign(floor_rel_db=df["floor_db"] - df["level_db"])
    p = np.array(
        [
            np.nan
            if pd.isna(r.spoof)
            else lf.probs(
                r.spoof, _none_if_nan(r.speaker_risk), {k: getattr(r, k) for k in lf.channel}
            )[0]
            for r in df.itertuples()
        ]
    )
    out = df.copy()
    out["attack_id"] = out["attack"]  # A07.. / knnvc..; ``attack`` below is the boolean class
    out["p"] = p
    out["waiting"] = np.isnan(p)
    out["flag"] = np.nan_to_num(p, nan=0.0) >= lf.t_warn
    out["esc"] = np.nan_to_num(p, nan=0.0) >= lf.t_esc
    out["attack"] = out.kind != "genuine"
    return out


def per_condition(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cond, g_all in d.groupby("cond", sort=False):
        g = g_all[~g_all.waiting]
        if g.empty or g.attack.all() or (~g.attack).all():
            rows.append(dict(cond=cond, n=len(g_all), ans=len(g) / len(g_all)))
            continue
        gen, atk = g[~g.attack], g[g.attack]
        cl, im = g[g.kind == "clone"], g[g.kind == "impostor"]
        eer = compute_eer((~g.attack).astype(int).values, -g.p.values)[0]
        rows.append(
            dict(
                cond=cond,
                n=len(g_all),
                ans=len(g) / len(g_all),
                acc=(g.flag == g.attack).mean(),
                bal=((~gen.flag).mean() + atk.flag.mean()) / 2,
                genOK=(~gen.flag).mean(),
                clone=cl.flag.mean(),
                imp=im.flag.mean(),
                ESCgen=gen.esc.mean(),
                ESCatk=atk.esc.mean(),
                EER=eer,
            )
        )
    return pd.DataFrame(rows)


def per_attack(d: pd.DataFrame) -> pd.DataFrame:
    """Per attack id / generator, against the genuine calls of the same situation: how many of
    its calls are flagged, how many reach ESCALATE, and the threshold-free EER."""
    rows = []
    for cond, g_all in d.groupby("cond", sort=False):
        g = g_all[~g_all.waiting]
        gen = g[~g.attack]
        for name, a in g[g.attack].groupby("attack_id", sort=True):
            if gen.empty or a.empty:
                continue
            eer = compute_eer(np.r_[np.ones(len(gen)), np.zeros(len(a))], np.r_[-gen.p, -a.p])[0]
            rows.append(
                dict(
                    cond=cond,
                    attack=name,
                    n=len(a),
                    flagged=a.flag.mean(),
                    ESC=a.esc.mean(),
                    genOK=(~gen.flag).mean(),
                    EER=eer,
                )
            )
    return pd.DataFrame(rows)


def show(t: pd.DataFrame, title: str) -> None:
    print(f"\n{title}")
    f = t.copy()
    for c in ("ans", "acc", "bal", "genOK", "clone", "imp", "ESCgen", "ESCatk", "EER"):
        if c in f:
            f[c] = (f[c] * 100).map("{:5.1f}%".format)
    for c in ("flagged", "ESC"):
        if c in f:
            f[c] = (f[c] * 100).map("{:5.1f}%".format)
    print(f.to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="_stress")
    ap.add_argument("--split", default="eval")
    ap.add_argument("--fusion", default=None, help="fusion json (default: the shipped one)")
    ap.add_argument("--csv", default=None, help="also write the per-condition table here")
    ap.add_argument("--by-attack-csv", default=None, help="write the per-attack table here")
    ap.add_argument(
        "--by-attack",
        action="store_true",
        help="also break the attack calls down per attack id / generator (A07..A19, knnvc, ...). "
        "For ASVspoof trials the id is the call's FIRST clip: a clone call mixes 3 clips of "
        "possibly different attacks, so read that breakdown as indicative; modern-generator "
        "calls are single-source and exact",
    )
    args = ap.parse_args()

    lf = LearnedFusion.load(args.fusion)
    if lf is None:
        raise SystemExit(f"no fusion model at {args.fusion or DEFAULT_PATH}")
    spec = lf.spec
    print(
        f"fusion: t_warn={lf.t_warn:.3f} t_esc={lf.t_esc:.3f} "
        f"(fit on trials{spec.get('trials_tag')}, commit {spec.get('git_commit')})"
    )

    d = score(pd.read_csv(TAB / f"system_trials{args.tag}_{args.split}.csv"), lf)
    t = per_condition(d)
    show(t, f"SITUATIONS  trials{args.tag}_{args.split}  ({d.speaker.nunique()} speakers)")
    answered = d[~d.waiting]
    pooled = answered.flag.eq(answered.attack).mean()
    worst = t.loc[t.acc.idxmin()]
    print(
        f"\npooled accuracy over answered reads: {pooled:.1%}   "
        f"worst situation: {worst.cond} {worst.acc:.1%}"
    )

    if args.by_attack or args.by_attack_csv:
        pa = per_attack(d)
        show(pa, "PER ATTACK  (flagged / ESCALATE share of that attack's calls)")
        if args.by_attack_csv:
            pa.to_csv(args.by_attack_csv, index=False)
            print("wrote", args.by_attack_csv)

    ref = TAB / f"system_trials{spec.get('trials_tag', '')}_dev.csv"
    if ref.exists():
        show(
            per_condition(score(pd.read_csv(ref), lf)),
            "REFERENCE  standard conditions, untouched dev split",
        )
    if args.csv:
        t.to_csv(args.csv, index=False)
        print("wrote", args.csv)


if __name__ == "__main__":
    main()
