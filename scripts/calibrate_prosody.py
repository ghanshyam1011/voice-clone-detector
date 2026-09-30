"""Fit a real probability calibration on top of the prosody LR's
predict_proba, and measure calibration quality on data it never
influenced at all (train_prosody.py's dev split *did* inform early
stopping/inspection during development; eval did not).

    python scripts/calibrate_prosody.py                  # report only
    python scripts/calibrate_prosody.py --ship platt     # + activate

Reports ECE/Brier before (raw predict_proba) vs after each fitted method
(Platt, isotonic) on dev (the calibrator's own fit set -- expected to look
best) and on eval (a genuinely untouched split). Nothing is written
without --ship -- see scripts/calibrate_cm.py's docstring for why a fit
that helps on its own fit set is not automatically worth shipping. Writes
the chosen calibrator into models/prosody/prosody_lr.joblib.

Honesty note, not fixed here: both dev and eval are drawn from the same
1:1 bonafide:spoof subsampling train_prosody.py uses, not the far more
skewed real-world base rate a deployed system would see. This calibrates
the model to be *honest about the evaluation protocol's class balance* --
recalibrating for an actual deployment's true spoof prevalence would need
a labeled sample drawn from that deployment, which does not exist yet.
"""

# ruff: noqa: E402
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from voiceguard.audio import load_wave, preprocess_wave
from voiceguard.config import load_config, preprocess_config, resolve
from voiceguard.data import load_manifest
from voiceguard.eval import brier_score, expected_calibration_error, fit_calibrator, git_commit
from voiceguard.prosody.features import FEATURE_NAMES, prosody_features
from voiceguard.prosody.scorer import DEFAULT_MODEL_PATH


def _subset(df: pd.DataFrame, per_class: int, seed: int) -> pd.DataFrame:
    return pd.concat(
        [g.sample(min(per_class, len(g)), random_state=seed) for _, g in df.groupby("label")]
    ).reset_index(drop=True)


def _score(df, pipe, medians, pcfg, sr):
    """-> (labels: 1=spoof, raw predict_proba(spoof))."""
    raw_p, labels = [], []
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="prosody", leave=False):
        w = preprocess_wave(load_wave(row.path, sr), pcfg)
        feats = prosody_features(w, sr)
        x = np.where(np.isfinite(feats), feats, medians).reshape(1, -1)
        raw_p.append(float(pipe.predict_proba(x)[0, 1]))
        labels.append(0 if row.label == "bonafide" else 1)
    return np.asarray(labels), np.clip(np.asarray(raw_p), 1e-6, 1 - 1e-6)


def _report(name, raw, labels, cals: dict):
    row = {
        "split": name,
        "n": len(labels),
        "ece_before": expected_calibration_error(raw, labels),
        "brier_before": brier_score(raw, labels),
    }
    line = (
        f"{name:14s} n={row['n']:5d}  "
        f"before ECE {row['ece_before']:.3f} Brier {row['brier_before']:.3f}"
    )
    for method, cal in cals.items():
        after = cal.apply(raw)
        row[f"ece_{method}"] = expected_calibration_error(after, labels)
        row[f"brier_{method}"] = brier_score(after, labels)
        line += f"  |  {method} ECE {row[f'ece_{method}']:.3f} Brier {row[f'brier_{method}']:.3f}"
    print(line)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-per-class", type=int, default=800)
    ap.add_argument("--eval-per-class", type=int, default=800)
    ap.add_argument(
        "--ship",
        default="none",
        choices=["none", "platt", "isotonic"],
        help="which fitted method (if any) to write into the model file; default 'none' "
        "just reports the dev-vs-eval comparison so you can decide",
    )
    args = ap.parse_args()

    if not DEFAULT_MODEL_PATH.exists():
        raise SystemExit(f"{DEFAULT_MODEL_PATH} missing -- run scripts/train_prosody.py first")
    blob = joblib.load(DEFAULT_MODEL_PATH)
    pipe, medians = blob["pipeline"], np.asarray(blob["feature_medians"], dtype=np.float32)
    print(f"loaded {DEFAULT_MODEL_PATH}  ({len(FEATURE_NAMES)} features)")

    cfg = load_config()
    seed = cfg["seed"]
    pcfg = preprocess_config(cfg)
    sr = pcfg.sample_rate
    man = resolve(cfg, "manifests")

    dev = _subset(load_manifest(man / "asvspoof19_la_dev.csv"), args.dev_per_class, seed)
    dev_y, dev_raw = _score(dev, pipe, medians, pcfg, sr)

    cals = {m: fit_calibrator(dev_raw, dev_y, method=m) for m in ("platt", "isotonic")}
    rows = [_report("dev (fit set)", dev_raw, dev_y, cals)]

    ev = _subset(load_manifest(man / "asvspoof19_la_eval.csv"), args.eval_per_class, seed)
    ev_y, ev_raw = _score(ev, pipe, medians, pcfg, sr)
    rows.append(_report("eval (unseen)", ev_raw, ev_y, cals))

    for r in rows:
        r.update(git_commit=git_commit())
    tables = resolve(cfg, "tables")
    pd.DataFrame(rows).to_csv(tables / "prosody_calibration.csv", index=False)
    print(f"\nprovenance -> {tables / 'prosody_calibration.csv'}")

    if args.ship == "none":
        print(
            "\n--ship not given: nothing written to the model file. Inspect the table above "
            "-- a calibrator that only helps on its own fit set should not ship. Re-run with "
            "--ship platt|isotonic once you've decided one actually improves eval."
        )
    else:
        blob["calibrator"] = cals[args.ship].to_dict()
        blob["calibration_method"] = args.ship
        joblib.dump(blob, DEFAULT_MODEL_PATH)
        print(f"saved fitted '{args.ship}' calibrator -> {DEFAULT_MODEL_PATH}")


if __name__ == "__main__":
    main()
