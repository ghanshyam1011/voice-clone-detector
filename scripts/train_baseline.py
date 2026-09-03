"""Train the frozen handcrafted GBM baseline on the re-extracted features.

    python scripts/train_baseline.py

Reads   data/processed/handcrafted_<fp>/asvspoof19_la_train.npz
Writes  baselines/artifacts/<fp>/{scaler,xgboost,lightgbm,catboost}.pkl
"""

from __future__ import annotations

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from voiceguard.config import load_config, preprocess_config, resolve
from voiceguard.models.baseline_gbm import train_baseline_ensemble


def main() -> None:
    cfg = load_config()
    fp = preprocess_config(cfg).fingerprint()
    feat_dir = resolve(cfg, "processed") / f"handcrafted_{fp}"
    train_npz = feat_dir / "asvspoof19_la_train.npz"
    if not train_npz.exists():
        raise SystemExit(
            f"{train_npz} not found -- run:\n"
            f"  python scripts/extract_features.py --manifest asvspoof19_la_train"
        )

    data = np.load(train_npz, allow_pickle=True)
    X, y = data["X"], data["y"]
    print(f"train: {X.shape}  bonafide={(y == 0).sum()}  spoof={(y == 1).sum()}  front_end={fp}")

    scaler = StandardScaler().fit(X)
    models = train_baseline_ensemble(scaler.transform(X), y, seed=cfg["seed"])

    out_dir = resolve(cfg, "baselines") / fp
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, out_dir / "scaler.pkl")
    for name, model in models.items():
        joblib.dump(model, out_dir / f"{name}.pkl")
    print(f"saved scaler + {', '.join(models)} -> {out_dir}")


if __name__ == "__main__":
    main()
