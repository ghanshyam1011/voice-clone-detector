"""Fit the prosody logistic-regression on an ASVspoof19-train subset.

    python scripts/train_prosody.py --per-class 1500

Extracts prosodic features (Praat/Parselmouth) through the anti-shortcut
front-end, fits StandardScaler + LogisticRegression, reports its standalone
dev EER, and saves models/prosody/prosody_lr.joblib (tracked, a few KB).
"""

# ruff: noqa: E402
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

from voiceguard.audio import load_wave, preprocess_wave
from voiceguard.config import load_config, preprocess_config, resolve
from voiceguard.data import load_manifest
from voiceguard.eval.metrics import compute_eer
from voiceguard.prosody.features import FEATURE_NAMES, prosody_features
from voiceguard.prosody.scorer import DEFAULT_MODEL_PATH


def _subset(df: pd.DataFrame, per_class: int, seed: int) -> pd.DataFrame:
    return pd.concat(
        [g.sample(min(per_class, len(g)), random_state=seed) for _, g in df.groupby("label")]
    ).reset_index(drop=True)


def _feats(df: pd.DataFrame, pcfg, sr: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for row in tqdm(df.itertuples(index=False), total=len(df)):
        w = preprocess_wave(load_wave(row.path, sr), pcfg)
        X.append(prosody_features(w, sr))
        y.append(0 if row.label == "bonafide" else 1)
    return np.asarray(X, dtype=np.float32), np.asarray(y)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=1500)
    ap.add_argument("--dev-per-class", type=int, default=800)
    args = ap.parse_args()

    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    cfg = load_config()
    seed = cfg["seed"]
    pcfg = preprocess_config(cfg)
    sr = pcfg.sample_rate
    man = resolve(cfg, "manifests")

    tr = _subset(load_manifest(man / "asvspoof19_la_train.csv"), args.per_class, seed)
    dv = _subset(load_manifest(man / "asvspoof19_la_dev.csv"), args.dev_per_class, seed)
    print(f"train {len(tr)}  dev {len(dv)}  ({len(FEATURE_NAMES)} features)")

    Xtr, ytr = _feats(tr, pcfg, sr)
    Xdv, ydv = _feats(dv, pcfg, sr)

    medians = np.nanmedian(Xtr, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    fill = lambda X: np.where(np.isfinite(X), X, medians)  # noqa: E731

    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5)),
        ]
    )
    pipe.fit(fill(Xtr), ytr)

    p_spoof = pipe.predict_proba(fill(Xdv))[:, 1]
    eer = compute_eer((ydv == 0).astype(int), 1.0 - p_spoof)[0]  # 1 = bonafide, higher = bonafide
    print(f"prosody-only dev EER: {eer * 100:.1f}%  (weak on its own — it is a fusion signal)")

    coefs = sorted(
        zip(FEATURE_NAMES, pipe.named_steps["lr"].coef_[0], strict=True),
        key=lambda kv: abs(kv[1]),
        reverse=True,
    )
    print("top features:", ", ".join(f"{n}({c:+.2f})" for n, c in coefs[:5]))

    DEFAULT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"pipeline": pipe, "feature_medians": medians.astype(np.float32), "dev_eer": float(eer)},
        DEFAULT_MODEL_PATH,
    )
    print(f"saved -> {DEFAULT_MODEL_PATH}")


if __name__ == "__main__":
    main()
