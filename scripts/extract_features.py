"""Extract the 61-dim handcrafted features for a manifest, THROUGH the
shared anti-shortcut front-end.

Output: data/processed/handcrafted_<front-end-fingerprint>/<name>.npz
        with arrays  X, y (1=spoof), attack_id, speaker_id
The fingerprint in the directory name means features from two different
front-ends can never be silently mixed.

    python scripts/extract_features.py --manifest asvspoof19_la_dev
    python scripts/extract_features.py --all
    python scripts/extract_features.py --all --limit-per-class 400   # quick
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

from voiceguard.audio import preprocess_file
from voiceguard.config import load_config, preprocess_config, resolve
from voiceguard.data import load_manifest
from voiceguard.features import extract_handcrafted

ALL_MANIFESTS = [
    "asvspoof19_la_train",
    "asvspoof19_la_dev",
    "asvspoof19_la_eval",
    "in_the_wild_eval",
]


def _subsample(df, limit_per_class, seed):
    if not limit_per_class:
        return df
    parts = [
        g.sample(min(limit_per_class, len(g)), random_state=seed) for _, g in df.groupby("label")
    ]
    return pd.concat(parts).reset_index(drop=True)


def extract_one(name: str, cfg: dict, limit_per_class: int | None) -> None:
    pcfg = preprocess_config(cfg)
    manifests_dir = resolve(cfg, "manifests")
    out_dir = resolve(cfg, "processed") / f"handcrafted_{pcfg.fingerprint()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_manifest(manifests_dir / f"{name}.csv")
    df = _subsample(df, limit_per_class, cfg["seed"])

    X, y, attack, speaker, failed = [], [], [], [], 0
    for row in tqdm(df.itertuples(index=False), total=len(df), desc=name):
        try:
            wav = preprocess_file(row.path, pcfg)
            X.append(extract_handcrafted(wav, pcfg.sample_rate))
            y.append(1 if row.label == "spoof" else 0)
            attack.append(row.attack_id)
            speaker.append(row.speaker_id)
        except Exception:
            failed += 1

    dest = out_dir / f"{name}.npz"
    np.savez_compressed(
        dest,
        X=np.asarray(X, dtype=np.float32),
        y=np.asarray(y, dtype=np.int64),
        attack_id=np.asarray(attack),
        speaker_id=np.asarray(speaker),
        front_end=pcfg.fingerprint(),
    )
    print(f"  {name}: {len(X)} rows, {failed} failed -> {dest}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", help="single manifest name (no .csv)")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--limit-per-class", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config()
    names = ALL_MANIFESTS if args.all else [args.manifest]
    if not names or names == [None]:
        ap.error("pass --manifest NAME or --all")

    manifests_dir = resolve(cfg, "manifests")
    for name in names:
        if not (manifests_dir / f"{name}.csv").exists():
            print(f"  SKIP {name} (no manifest)")
            continue
        extract_one(name, cfg, args.limit_per_class)


if __name__ == "__main__":
    main()
