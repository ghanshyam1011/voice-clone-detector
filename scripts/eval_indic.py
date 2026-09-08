"""IndicCall-Eval — score the pilot per language.

    python scripts/make_indic_spoof.py --per-lang 40
    # (optional) drop consented real clips in data/indic_eval/real/<lang>/*.wav
    python scripts/eval_indic.py

For each language reports the synthetic-voice detector's recall on the
generated attacks (fraction flagged at the demo threshold) and the mean
prosody score; where real clips exist it also reports a proper EER.
Writes results/tables/indic_eval.csv (provenance stamped).
"""

# ruff: noqa: E402
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse

import numpy as np
import pandas as pd

from voiceguard.config import REPO_ROOT, load_config, preprocess_config, resolve
from voiceguard.eval.metrics import compute_eer
from voiceguard.eval.provenance import stamp

INDIC = REPO_ROOT / "data" / "indic_eval"
_SYNTHETIC_ABOVE = 0.65  # the console's "synthetic" band


def _real_clips(lang: str) -> list[str]:
    d = INDIC / "real" / lang
    return [str(p) for p in sorted(d.glob("*.wav"))] if d.exists() else []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None, help="a .pth under models/pretrained/ (a fine-tune)")
    args = ap.parse_args()

    man = INDIC / "spoof_manifest.csv"
    if not man.exists():
        raise SystemExit("run scripts/make_indic_spoof.py first")

    from voiceguard.detect import build_scorer
    from voiceguard.prosody import ProsodyScorer

    cfg = load_config()
    scorer = build_scorer("aasist", pretrained=True, weights=args.weights)
    try:
        prosody = ProsodyScorer()
        prosody.warm()
    except Exception as exc:  # noqa: BLE001
        prosody = None
        print(f"(prosody off: {exc})")

    fe = preprocess_config(cfg).fingerprint()
    tag = args.weights or "AASIST-upstream"
    spoof = pd.read_csv(man)
    spoof["path"] = spoof["path"].map(lambda p: str(REPO_ROOT / p))

    rows, table = [], []
    for lang, g in spoof.groupby("language"):
        s_risk = np.array([float(scorer.score_file(p).risk) for p in g.path])
        recall = float((s_risk >= _SYNTHETIC_ABOVE).mean())
        pros = np.nan
        if prosody is not None:
            pv = [prosody.score_file(p).risk for p in g.path]
            pv = [x for x in pv if x is not None]
            pros = float(np.mean(pv)) if pv else np.nan

        eer = np.nan
        reals = _real_clips(lang)
        if len(reals) >= 5:
            r_risk = np.array([float(scorer.score_file(p).risk) for p in reals])
            labels = np.r_[np.zeros(len(s_risk)), np.ones(len(r_risk))]  # 1 = bonafide
            scores = 1.0 - np.r_[s_risk, r_risk]  # higher = bonafide
            eer = compute_eer(labels, scores)[0]

        table.append(
            {
                "language": lang,
                "n_spoof": len(g),
                "n_real": len(reals),
                "synth_recall_%": round(recall * 100, 1),
                "mean_spoof_risk_%": round(float(s_risk.mean()) * 100, 1),
                "mean_prosody_%": round(pros * 100, 1) if pros == pros else "-",
                "eer_%": round(eer * 100, 1) if eer == eer else "-",
            }
        )
        rows.append(
            stamp(
                {
                    "metric": "synthetic_recall",
                    "split": f"indic_{lang}",
                    "value_pct": recall * 100,
                    "n": len(g),
                    "eer_pct": None if eer != eer else eer * 100,
                },
                train_set="asvspoof19_la_train",
                test_set=f"indic_eval_{lang}",
                model=tag,
                seed=cfg["seed"],
                front_end=fe,
            )
        )

    out = resolve(cfg, "tables") / "indic_eval.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    df = pd.DataFrame(table)
    print("\nIndicCall-Eval (pilot) — synthetic-voice detector per language\n")
    print(df.to_string(index=False))
    print(f"\n-> {out}")
    if (df["n_real"] == 0).all():
        print("\nNo real clips yet — recall on synthetic only. Add consented recordings to")
        print("data/indic_eval/real/<lang>/ for a per-language EER (see docs/indic_eval.md).")


if __name__ == "__main__":
    main()
