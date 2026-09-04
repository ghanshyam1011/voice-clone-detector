"""The "Generation 1" handcrafted-feature baseline: XGBoost / LightGBM /
CatBoost on the 61-dim vector, mean-probability ensemble.

Deliberately different from the old notebook 02:
  * no SMOTE  -- it inflated 2,580 real bonafide clips to 22,800 by
    interpolation; use class weights instead.
  * fixed protocol splits, shared anti-shortcut front-end.
  * this is a *baseline row*, never a serving component.
"""

from __future__ import annotations

import numpy as np

BASELINE_MODELS = ("xgboost", "lightgbm", "catboost")


def _class_weight(y: np.ndarray) -> float:
    """scale_pos_weight style ratio: n_negative / n_positive, with y==1 spoof."""
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    return neg / pos if pos else 1.0


def train_baseline_ensemble(X: np.ndarray, y: np.ndarray, *, seed: int = 42) -> dict:
    """y: 1 = spoof, 0 = bonafide (the stored label convention).
    Returns {name: fitted_estimator}. Caller scales X first."""
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    spw = _class_weight(y)
    models = {
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=spw,
            eval_metric="logloss",
            random_state=seed,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            class_weight="balanced",
            random_state=seed,
            verbose=-1,
        ),
        "catboost": CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            auto_class_weights="Balanced",
            random_seed=seed,
            verbose=0,
        ),
    }
    for m in models.values():
        m.fit(X, y)
    return models


def score_bonafide(model, X_scaled: np.ndarray) -> np.ndarray:
    """P(bonafide), higher = more bonafide-like. Reads model.classes_ rather
    than assuming column order."""
    proba = model.predict_proba(X_scaled)
    bonafide_col = list(model.classes_).index(0)  # 0 = bonafide
    return proba[:, bonafide_col]


def ensemble_score(models: dict, X_scaled: np.ndarray) -> np.ndarray:
    return np.mean([score_bonafide(m, X_scaled) for m in models.values()], axis=0)
