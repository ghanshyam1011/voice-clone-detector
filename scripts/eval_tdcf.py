"""min-tDCF alongside EER, using the ASVspoof 2019 organizers' own
reference implementation and their official baseline ASV system's scores
(voiceguard.eval.tdcf; see docs/tdcf.md).

    python scripts/eval_tdcf.py --model aasist --pretrained

-> results/tables/cm_<run>_tdcf.csv (dev + eval, provenance-stamped same as
every other result in this project).
"""

# ruff: noqa: E402
from __future__ import annotations

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
from voiceguard.eval import compute_eer, compute_min_tdcf, git_commit, load_asv_scores
from voiceguard.models.neural import build_cm


def _subsample(df, k, seed):
    if not k:
        return df
    return pd.concat(
        [g.sample(min(k, len(g)), random_state=seed) for _, g in df.groupby("label")]
    ).reset_index(drop=True)


@torch.no_grad()
def _score(model, df, pcfg, device, bs, workers):
    ds = CMDataset(df, pcfg, None, train=False, seed=0)
    ld = DataLoader(ds, batch_size=bs, num_workers=workers)
    scores, labels = [], []
    for wav, y in tqdm(ld, desc="score", leave=False):
        scores.append(model.score_bonafide(wav.to(device)).float().cpu().numpy())
        labels.append((y == 0).numpy().astype(int))  # 1 = bonafide
    return np.concatenate(labels), np.concatenate(scores)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model", default="aasist", choices=["rawnet2", "aasist", "aasist-l", "ssl-aasist"]
    )
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--limit-per-class", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config()
    seed = cfg["seed"]
    pcfg = preprocess_config(cfg)
    fp = pcfg.fingerprint()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_dir = resolve(cfg, "results").parent / "models"
    asv_dir = resolve(cfg, "asvspoof_la") / "ASVspoof2019_LA_asv_scores"

    model = build_cm(args.model).to(device)
    if args.pretrained:
        name_map = {"aasist": "AASIST.pth", "aasist-l": "AASIST-L.pth"}
        if args.model not in name_map:
            raise SystemExit(f"--pretrained not available for {args.model}")
        wpath = models_dir / "pretrained" / name_map[args.model]
        model.backbone.load_state_dict(torch.load(wpath, map_location=device))
        run = f"{args.model}-pretrained"
    else:
        ckpt = models_dir / "cm" / args.model / "best.pt"
        model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
        run = args.model
    model.eval()
    print(f"{run}  front_end={fp}")

    man = resolve(cfg, "manifests")
    rows = []
    splits = [
        ("dev_known_attacks", "asvspoof19_la_dev.csv", "ASVspoof2019.LA.asv.dev.gi.trl.scores.txt"),
        (
            "eval_unknown_attacks",
            "asvspoof19_la_eval.csv",
            "ASVspoof2019.LA.asv.eval.gi.trl.scores.txt",
        ),
    ]
    for split, cm_file, asv_file in splits:
        asv_path = asv_dir / asv_file
        if not asv_path.exists():
            print(f"skipping {split}: {asv_path} not found")
            continue
        df = _subsample(load_manifest(man / cm_file), args.limit_per_class, seed)
        labels, scores = _score(model, df, pcfg, device, args.batch_size, args.workers)
        bonafide_cm, spoof_cm = scores[labels == 1], scores[labels == 0]

        tar_asv, non_asv, spoof_asv = load_asv_scores(asv_path)
        r = compute_min_tdcf(bonafide_cm, spoof_cm, tar_asv, non_asv, spoof_asv)
        cm_eer, _ = compute_eer(labels, scores)  # cross-check against the project's own compute_eer

        print(
            f"{split:22s} min-tDCF={r.min_tdcf:.4f}  EER(cm)={cm_eer * 100:.2f}%  "
            f"EER(asv)={r.eer_asv * 100:.2f}%  n_cm={r.n_bonafide_cm + r.n_spoof_cm}"
        )
        rows.append(
            {
                "metric": "min_tdcf",
                "split": split,
                "value": r.min_tdcf,
                "eer_cm_pct": cm_eer * 100,
                "eer_asv_pct": r.eer_asv * 100,
                "pfa_asv": r.pfa_asv,
                "pmiss_asv": r.pmiss_asv,
                "pmiss_spoof_asv": r.pmiss_spoof_asv,
                "n_bonafide_cm": r.n_bonafide_cm,
                "n_spoof_cm": r.n_spoof_cm,
                "n_asv_trials": r.n_asv_trials,
                "model": run,
                "seed": seed,
                "front_end": fp,
                "git_commit": git_commit(),
            }
        )

    if not rows:
        raise SystemExit(
            "No ASV score files found -- see docs/tdcf.md. min-tDCF needs the ASVspoof 2019 LA "
            "organizers' own ASV score files (ASVspoof2019_LA_asv_scores/), distributed with the "
            "corpus alongside the CM protocol files."
        )
    tables = resolve(cfg, "tables")
    pd.DataFrame(rows).to_csv(tables / f"cm_{run}_tdcf.csv", index=False)
    print(f"\nsaved -> {tables / f'cm_{run}_tdcf.csv'}")


if __name__ == "__main__":
    main()
