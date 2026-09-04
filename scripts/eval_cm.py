"""Evaluate a trained countermeasure and write provenance-stamped tables,
same protocol as the baseline (docs/evaluation_protocol.md).

    python scripts/eval_cm.py --model aasist
    python scripts/eval_cm.py --model aasist --limit-per-class 3000   # faster

-> results/tables/cm_<name>_{in_domain,cross_dataset,silence_ablation}.csv
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from voiceguard.audio import load_wave, normalize_loudness
from voiceguard.config import load_config, preprocess_config, resolve
from voiceguard.data import load_manifest
from voiceguard.data.torch_dataset import NB_SAMP, CMDataset, _fix_length
from voiceguard.eval import harness
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
    scores, labels, attacks = [], [], list(df["attack_id"])
    for wav, y in tqdm(ld, desc="score", leave=False):
        scores.append(model.score_bonafide(wav.to(device)).float().cpu().numpy())
        labels.append((y == 0).numpy().astype(int))  # 1 = bonafide
    return np.concatenate(labels), np.concatenate(scores), np.asarray(attacks)


@torch.no_grad()
def _silence_ablation(model, cfg, pcfg, device, seed, n, top_db):
    df = load_manifest(resolve(cfg, "manifests") / "asvspoof19_la_dev.csv")
    df = pd.concat(
        [g.sample(min(n, len(g)), random_state=seed) for _, g in df.groupby("label")]
    ).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    S, L = [], []
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="silence", leave=False):
        try:
            raw = load_wave(row.path, pcfg.sample_rate)
            sil = harness.silence_only_waveform(raw, pcfg.sample_rate, top_db)
            sil = normalize_loudness(sil, pcfg)
            sil = _fix_length(sil, NB_SAMP, rng, random_crop=False)
            w = torch.from_numpy(np.ascontiguousarray(sil, np.float32)).unsqueeze(0).to(device)
            S.append(float(model.score_bonafide(w).item()))
            L.append(1 if row.label == "bonafide" else 0)
        except Exception:
            pass
    return np.asarray(L), np.asarray(S)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="aasist", choices=["rawnet2", "aasist", "aasist-l"])
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--limit-per-class", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    cfg = load_config()
    seed = cfg["seed"]
    pcfg = preprocess_config(cfg)
    fp = pcfg.fingerprint()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tables = resolve(cfg, "tables")

    ckpt = args.ckpt or (resolve(cfg, "results").parent / "models" / "cm" / args.model / "best.pt")
    state = torch.load(ckpt, map_location=device)
    model = build_cm(args.model).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    saved_eer = state.get("dev_eer", float("nan")) * 100
    print(f"{args.model} loaded from {ckpt} (dev EER at save {saved_eer:.2f}%)")

    man = resolve(cfg, "manifests")
    dev_df = _subsample(load_manifest(man / "asvspoof19_la_dev.csv"), args.limit_per_class, seed)
    eval_df = _subsample(load_manifest(man / "asvspoof19_la_eval.csv"), args.limit_per_class, seed)

    ld, sd, _ = _score(model, dev_df, pcfg, device, args.batch_size, args.workers)
    le, se, atk = _score(model, eval_df, pcfg, device, args.batch_size, args.workers)
    rows = harness.in_domain_rows(args.model, (ld, sd), (le, se), atk, seed=seed, front_end=fp)
    pd.DataFrame(rows).to_csv(tables / f"cm_{args.model}_in_domain.csv", index=False)

    itw = man / "in_the_wild_eval.csv"
    if itw.exists():
        w_df = _subsample(load_manifest(itw), args.limit_per_class or 3000, seed)
        lw, sw, _ = _score(model, w_df, pcfg, device, args.batch_size, args.workers)
        pd.DataFrame(
            [
                harness.cross_dataset_row(
                    args.model, lw, sw, dataset="in_the_wild", seed=seed, front_end=fp
                )
            ]
        ).to_csv(tables / f"cm_{args.model}_cross_dataset.csv", index=False)

    sl, ss = _silence_ablation(
        model,
        cfg,
        pcfg,
        device,
        seed,
        cfg["eval"]["quick_n_per_class"],
        cfg["eval"]["silence_ablation"]["top_db"],
    )
    pd.DataFrame(
        [
            harness.silence_ablation_row(
                args.model,
                sl,
                ss,
                seed=seed,
                front_end=fp,
                leak_threshold=cfg["eval"]["silence_ablation"]["leak_eer_threshold"],
            )
        ]
    ).to_csv(tables / f"cm_{args.model}_silence_ablation.csv", index=False)

    d = pd.read_csv(tables / f"cm_{args.model}_in_domain.csv")
    print(
        d[d.split.isin(["dev_known_attacks", "eval_unknown_attacks"])][
            ["model", "split", "value_pct"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
