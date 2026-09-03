"""Extract the 61-dim handcrafted features for a manifest, THROUGH the
shared anti-shortcut front-end.

Output: data/processed/handcrafted_<front-end-fingerprint>/<name>.npz
        with arrays  X, y (1=spoof), attack_id, speaker_id
The fingerprint in the directory name means features from two different
front-ends can never be silently mixed.

Resumable: a checkpoint is written every few thousand clips, so a killed or
suspended run continues where it left off. A finished manifest is skipped
unless --force.

    python scripts/extract_features.py --manifest asvspoof19_la_dev
    python scripts/extract_features.py --all
    python scripts/extract_features.py --all --limit-per-class 400   # quick
    python scripts/extract_features.py --manifest asvspoof19_la_eval --force
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

CHECKPOINT_EVERY = 2000


def _subsample(df, limit_per_class, seed):
    if not limit_per_class:
        return df
    parts = [
        g.sample(min(limit_per_class, len(g)), random_state=seed) for _, g in df.groupby("label")
    ]
    return pd.concat(parts).reset_index(drop=True)


def _save(path, X, y, attack, speaker, fp, n_done):
    np.savez_compressed(
        path,
        X=np.asarray(X, dtype=np.float32),
        y=np.asarray(y, dtype=np.int64),
        attack_id=np.asarray(attack),
        speaker_id=np.asarray(speaker),
        front_end=fp,
        n_done=n_done,
    )


def extract_one(name: str, cfg: dict, limit_per_class: int | None, force: bool) -> None:
    pcfg = preprocess_config(cfg)
    fp = pcfg.fingerprint()
    out_dir = resolve(cfg, "processed") / f"handcrafted_{fp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / f"{name}.npz"
    ckpt = out_dir / f"{name}.ckpt.npz"

    if final.exists() and not force:
        print(f"  SKIP {name} (already done: {final}); use --force to redo")
        return

    df = load_manifest(resolve(cfg, "manifests") / f"{name}.csv")
    df = _subsample(df, limit_per_class, cfg["seed"])

    X, y, attack, speaker = [], [], [], []
    start = 0
    if ckpt.exists() and not force:
        d = np.load(ckpt, allow_pickle=True)
        if str(d["front_end"]) == fp and int(d["n_done"]) <= len(df):
            X, y = list(d["X"]), list(d["y"])
            attack, speaker = list(d["attack_id"]), list(d["speaker_id"])
            start = int(d["n_done"])
            print(f"  resume {name} from row {start}")

    failed = 0
    rows = list(df.itertuples(index=False))
    for i in tqdm(range(start, len(rows)), initial=start, total=len(rows), desc=name):
        row = rows[i]
        try:
            wav = preprocess_file(row.path, pcfg)
            X.append(extract_handcrafted(wav, pcfg.sample_rate))
            y.append(1 if row.label == "spoof" else 0)
            attack.append(row.attack_id)
            speaker.append(row.speaker_id)
        except Exception:
            failed += 1
        if (i + 1) % CHECKPOINT_EVERY == 0:
            _save(ckpt, X, y, attack, speaker, fp, i + 1)

    _save(final, X, y, attack, speaker, fp, len(rows))
    ckpt.unlink(missing_ok=True)
    print(f"  {name}: {len(X)} rows, {failed} failed -> {final}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", help="single manifest name (no .csv)")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--limit-per-class", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="redo even if the .npz exists")
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
        extract_one(name, cfg, args.limit_per_class, args.force)


if __name__ == "__main__":
    main()
