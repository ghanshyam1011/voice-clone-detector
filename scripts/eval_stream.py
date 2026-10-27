"""Call-level behaviour of the live system: time to alert, and false alarms over a whole call.

    python scripts/build_stream_trials.py --split eval --speakers 10 --conds clean,babble10
    python scripts/eval_stream.py --tag _stream

Reads the update-by-update traces written by ``build_stream_trials.py`` and answers what
a single-read table cannot:

  * a genuine caller who is wrongly flagged on 3% of reads is flagged at SOME moment of a
    28 s call far more often than 3% -- ``FA`` below is the share of genuine CALLS that were
    ever flagged, by call length;
  * a clone is only useful to catch if it is caught early -- ``det`` is the share of attack
    calls ever flagged and ``TTD`` the median seconds from the start of the call.

Policies are scored from the logged smoothed score ``fused`` (thresholds from
``risk.policy.SCENARIOS['routine']``), so the effect of requiring evidence to persist
(``k`` consecutive updates above the threshold) is measured on the very same traces:

    live     the action the system actually produced
    k=N      flag only after N consecutive updates (0.5 s apart in the live path) above warn;
             ESCALATE likewise above the escalate threshold, never while provisional
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from voiceguard.config import REPO_ROOT
from voiceguard.risk.policy import SCENARIOS

TAB = REPO_ROOT / "results" / "tables"
WARN, ESC = SCENARIOS["routine"]
HORIZONS = (5, 10, 20)
PERSIST = (1, 2, 4, 6, 10)


def _persist(x: np.ndarray, k: int) -> np.ndarray:
    """True where the last ``k`` updates were all True."""
    if k <= 1:
        return x
    return np.convolve(x.astype(int), np.ones(k, int), "full")[: len(x)] >= k


def _flags(g: pd.DataFrame, k: int | None) -> tuple[np.ndarray, np.ndarray]:
    """-> (flagged, escalated) per update of one call. ``k=None`` = the logged live action."""
    if k is None:
        return g.action.isin(["VERIFY", "ESCALATE"]).to_numpy(), (g.action == "ESCALATE").to_numpy()
    ok = (g.status == "ok").to_numpy()
    f = (g.fused.to_numpy() >= WARN) & ok
    e = (g.fused.to_numpy() >= ESC) & ok & ~g.provisional.to_numpy().astype(bool)
    return _persist(f, k), _persist(e, k)


def _first(t: np.ndarray, x: np.ndarray) -> float:
    return float(t[np.argmax(x)]) if x.any() else np.nan


def per_call(d: pd.DataFrame, k: int | None) -> pd.DataFrame:
    rows = []
    for (spk, call, cond), g in d.groupby(["speaker", "call", "cond"], sort=False):
        g = g.sort_values("t")
        f, e = _flags(g, k)
        t = g.t.to_numpy()
        rows.append(
            dict(
                speaker=spk,
                call=call,
                cond=cond,
                kind=g.kind.iloc[0],
                length=float(t[-1]),
                first_flag=_first(t, f),
                first_esc=_first(t, e),
                flag_rate=float(f.mean()),
            )
        )
    return pd.DataFrame(rows)


def summarise(calls: pd.DataFrame) -> dict:
    gen, clone, imp = (calls[calls.kind == k] for k in ("genuine", "clone", "impostor"))
    out = {}
    for h in (*HORIZONS, "end"):
        lim = np.inf if h == "end" else h
        out[f"FA@{h}"] = float((gen.first_flag <= lim).mean()) if len(gen) else np.nan
    out["FAesc"] = float(gen.first_esc.notna().mean()) if len(gen) else np.nan
    for name, a in (("clone", clone), ("imp", imp)):
        out[f"{name}_det"] = float(a.first_flag.notna().mean()) if len(a) else np.nan
        out[f"{name}_TTD"] = float(a.first_flag.median()) if a.first_flag.notna().any() else np.nan
        out[f"{name}_esc"] = float(a.first_esc.notna().mean()) if len(a) else np.nan
    out["n_gen"], out["n_clone"], out["n_imp"] = len(gen), len(clone), len(imp)
    return out


def _fmt(t: pd.DataFrame) -> pd.DataFrame:
    f = t.copy()
    for c in f.columns:
        if c.startswith(("FA", "clone_det", "imp_det", "clone_esc", "imp_esc")):
            f[c] = (f[c] * 100).map("{:5.1f}%".format)
        elif c.endswith("TTD"):
            f[c] = f[c].map("{:5.1f}s".format)
    return f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="_stream")
    ap.add_argument("--split", default="eval")
    ap.add_argument("--csv", default=None, help="also write the policy table here")
    args = ap.parse_args()
    d = pd.read_csv(TAB / f"stream_trials{args.tag}_{args.split}.csv")
    d["provisional"] = d.provisional.astype(bool)
    n_calls = d.groupby(["speaker", "call", "cond"]).ngroups
    print(
        f"{d.speaker.nunique()} speakers, {n_calls} calls, {len(d)} updates; "
        f"median update {d.ms.median():.0f} ms (p95 {d.ms.quantile(0.95):.0f} ms)"
    )

    rows = []
    for cond in d.cond.unique():
        sub = d[d.cond == cond]
        for label, k in [("live", None), *[(f"k={k}", k) for k in PERSIST]]:
            rows.append({"cond": cond, "policy": label, **summarise(per_call(sub, k))})
    t = pd.DataFrame(rows)
    cols = [
        "cond",
        "policy",
        "FA@5",
        "FA@10",
        "FA@20",
        "FA@end",
        "FAesc",
        "clone_det",
        "clone_TTD",
        "clone_esc",
        "imp_det",
        "imp_TTD",
        "n_gen",
        "n_clone",
        "n_imp",
    ]
    print(
        "\nCALL-LEVEL  FA@T = genuine calls ever flagged within T s; FAesc = ever ESCALATED;\n"
        "det = attack calls ever flagged; TTD = median seconds to the first flag"
    )
    print(_fmt(t[cols]).to_string(index=False))

    # how the live system's genuine false alarms accumulate with call length
    print("\nLIVE: share of GENUINE calls flagged by T seconds, and of ATTACK calls caught")
    for cond in d.cond.unique():
        c = per_call(d[d.cond == cond], None)
        grid = [3, 5, 8, 12, 16, 20, 24]
        line = {"T": grid}
        for kind in ("genuine", "clone", "impostor"):
            s = c[c.kind == kind]
            line[kind] = [f"{100 * (s.first_flag <= x).mean():5.1f}%" for x in grid]
        print(f"[{cond}]")
        print(pd.DataFrame(line).to_string(index=False))
    if args.csv:
        t.to_csv(args.csv, index=False)
        print("wrote", args.csv)


if __name__ == "__main__":
    main()
