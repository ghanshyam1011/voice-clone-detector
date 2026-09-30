"""Fit a real probability calibration for the spoof signal, replacing the
ad-hoc logit-shift documented as "not real calibration (P3)" in
detect/scorer.py.

    python scripts/calibrate_cm.py --model aasist --pretrained          # report only
    python scripts/calibrate_cm.py --model aasist --pretrained --ship platt   # + activate

Fits both a Platt and an isotonic mapping from the model's raw
P(synthetic) logit to a calibrated probability, using ASVspoof19 LA dev as
the held-out fitting set (the model never trained on it). Reports
calibration quality (ECE, Brier) *before* (today's ad-hoc method) and
*after each fitted method* on dev, on eval (unknown attacks -- the honest
in-domain check), and on In-the-Wild if present (cross-dataset -- the
hardest, most honest check, per docs/evaluation_protocol.md).

Without --ship, nothing is written -- this prints the comparison and stops,
because a fit that helps on dev is not guaranteed to help where it matters
(see docs/calibration.md for a real example: a first Platt fit here cut
dev ECE by 6x but made eval and In-the-Wild ECE *worse*). Only pass --ship
once the table shows the method you're picking actually improves eval/ITW,
not just dev. Writes into the model's existing calibration JSON so
CMScorer picks it up automatically -> results/tables/cm_<run>_calibration.csv
"""

# ruff: noqa: E402
from __future__ import annotations

import json
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from voiceguard.config import load_config, preprocess_config, resolve
from voiceguard.data import load_manifest
from voiceguard.data.torch_dataset import CMDataset
from voiceguard.detect.scorer import _logit
from voiceguard.eval import (
    brier_score,
    expected_calibration_error,
    fit_calibrator,
    git_commit,
)
from voiceguard.models.neural import build_cm


def _subsample(df, k, seed):
    if not k:
        return df
    return pd.concat(
        [g.sample(min(k, len(g)), random_state=seed) for _, g in df.groupby("label")]
    ).reset_index(drop=True)


@torch.no_grad()
def _raw_logits(model, df, pcfg, device, bs, workers):
    """-> (labels: 1=spoof, logit of raw P(synthetic))."""
    ds = CMDataset(df, pcfg, None, train=False, seed=0)
    ld = DataLoader(ds, batch_size=bs, num_workers=workers)
    p_bona, labels = [], []
    for wav, y in tqdm(ld, desc="score", leave=False):
        p_bona.append(model.score_bonafide(wav.to(device)).float().cpu().numpy())
        labels.append((y != 0).numpy().astype(int))  # 1 = spoof
    p_bona = np.concatenate(p_bona)
    labels = np.concatenate(labels)
    raw_synth = np.clip(1.0 - p_bona, 1e-6, 1 - 1e-6)
    return labels, _logit(raw_synth)


def _report(name, logits, labels, old_op, old_temp, cals: dict):
    """Before (ad-hoc logit-shift, temperature=old_temp centred at old_op)
    vs each fitted calibrator in `cals`, ECE + Brier. Reports every method
    from the same scored data -- generalization differences between
    methods matter more here than any single number."""
    before = 1.0 / (1.0 + np.exp(-(logits - _logit(old_op)) / old_temp))
    row = {
        "split": name,
        "n": len(labels),
        "ece_before": expected_calibration_error(before, labels),
        "brier_before": brier_score(before, labels),
    }
    line = (
        f"{name:14s} n={row['n']:6d}  "
        f"before ECE {row['ece_before']:.3f} Brier {row['brier_before']:.3f}"
    )
    for method, cal in cals.items():
        after = cal.apply(logits)
        row[f"ece_{method}"] = expected_calibration_error(after, labels)
        row[f"brier_{method}"] = brier_score(after, labels)
        line += f"  |  {method} ECE {row[f'ece_{method}']:.3f} Brier {row[f'brier_{method}']:.3f}"
    print(line)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model", default="aasist", choices=["rawnet2", "aasist", "aasist-l", "ssl-aasist"]
    )
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument(
        "--ship",
        default="none",
        choices=["none", "platt", "isotonic"],
        help="which fitted method (if any) to actually write into the calibration JSON; "
        "default 'none' just reports the comparison so you can decide",
    )
    ap.add_argument("--limit-per-class", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config()
    seed = cfg["seed"]
    pcfg = preprocess_config(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_dir = resolve(cfg, "results").parent / "models"

    model = build_cm(args.model).to(device)
    if args.pretrained:
        name_map = {"aasist": "AASIST.pth", "aasist-l": "AASIST-L.pth"}
        wpath = models_dir / "pretrained" / name_map[args.model]
        model.backbone.load_state_dict(torch.load(wpath, map_location=device))
        calib_path = models_dir / "pretrained" / f"{args.model}_demo_calib.json"
        run = f"{args.model}-pretrained"
    else:
        ckpt = models_dir / "cm" / args.model / "best.pt"
        model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
        calib_path = ckpt.with_name("demo_calib.json")
        run = args.model
    model.eval()

    if not calib_path.exists():
        raise SystemExit(f"{calib_path} missing -- run the scorer once first to create it")
    old = json.loads(calib_path.read_text())
    old_op = old["operating_point"]
    old_temp = 2.0  # detect/scorer.py's _CALIB_TEMPERATURE, duplicated here deliberately:
    # this script reports the ad-hoc method's numbers for comparison, it must not import
    # scorer internals that could silently drift the "before" baseline out from under it.
    print(f"{run} <- {calib_path.name}  (old operating_point={old_op:.4f})")

    man = resolve(cfg, "manifests")
    dev_df = _subsample(load_manifest(man / "asvspoof19_la_dev.csv"), args.limit_per_class, seed)
    eval_df = _subsample(
        load_manifest(man / "asvspoof19_la_eval.csv"), args.limit_per_class, seed
    )

    dev_y, dev_x = _raw_logits(model, dev_df, pcfg, device, args.batch_size, args.workers)
    eval_y, eval_x = _raw_logits(model, eval_df, pcfg, device, args.batch_size, args.workers)

    cals = {m: fit_calibrator(dev_x, dev_y, method=m) for m in ("platt", "isotonic")}

    rows = [_report("dev (fit set)", dev_x, dev_y, old_op, old_temp, cals)]
    rows.append(_report("eval (unseen)", eval_x, eval_y, old_op, old_temp, cals))

    itw_path = man / "in_the_wild_eval.csv"
    if itw_path.exists():
        itw_df = _subsample(load_manifest(itw_path), args.limit_per_class, seed)
        itw_y, itw_x = _raw_logits(model, itw_df, pcfg, device, args.batch_size, args.workers)
        rows.append(_report("in-the-wild", itw_x, itw_y, old_op, old_temp, cals))
    else:
        print("(in_the_wild_eval.csv manifest not found -- skipping the cross-dataset check)")

    for r in rows:
        r.update(model=run, git_commit=git_commit())
    tables = resolve(cfg, "tables")
    pd.DataFrame(rows).to_csv(tables / f"cm_{run}_calibration.csv", index=False)
    print(f"\nprovenance -> {tables / f'cm_{run}_calibration.csv'}")

    if args.ship == "none":
        print(
            "\n--ship not given: nothing written to the calibration JSON. Inspect the table "
            "above -- a calibrator that helps on dev but hurts on eval/in-the-wild should not "
            "ship (see docs/calibration.md). Re-run with --ship platt|isotonic once you've "
            "decided one is actually an improvement on the splits that matter."
        )
    else:
        old["calibrator"] = cals[args.ship].to_dict()
        old["calibration_method"] = args.ship
        calib_path.write_text(json.dumps(old, indent=2))
        print(f"saved fitted '{args.ship}' calibrator -> {calib_path}")


if __name__ == "__main__":
    main()
