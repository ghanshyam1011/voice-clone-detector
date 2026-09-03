"""Run the full baseline evaluation and write provenance-stamped tables.

    python scripts/run_eval.py

Produces, under results/tables/:
    baseline_in_domain.csv     dev (known) / eval (unknown) / per-attack EER
    baseline_cross_dataset.csv In-the-Wild EER  (the honest number)
    baseline_silence_ablation.csv  silence-only EER  (must be ~50%)
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from voiceguard.audio import preprocess_file
from voiceguard.audio.preprocess import PreprocessConfig
from voiceguard.config import load_config, preprocess_config, resolve
from voiceguard.data import load_manifest
from voiceguard.eval import harness
from voiceguard.features import extract_handcrafted
from voiceguard.models.baseline_gbm import ensemble_score, score_bonafide

MODELS = ("xgboost", "lightgbm", "catboost")


def _load_npz(path):
    d = np.load(path, allow_pickle=True)
    return d["X"], d["y"], d["attack_id"]


def _bonafide_labels(y_spoof_is_1: np.ndarray) -> np.ndarray:
    return (y_spoof_is_1 == 0).astype(int)  # metrics want 1 = bonafide


def main() -> None:
    cfg = load_config()
    pcfg = preprocess_config(cfg)
    fp = pcfg.fingerprint()
    seed = cfg["seed"]

    feat_dir = resolve(cfg, "processed") / f"handcrafted_{fp}"
    art_dir = resolve(cfg, "baselines") / fp
    tables = resolve(cfg, "tables")
    tables.mkdir(parents=True, exist_ok=True)

    scaler = joblib.load(art_dir / "scaler.pkl")
    models = {m: joblib.load(art_dir / f"{m}.pkl") for m in MODELS}

    # ---------- in-domain: dev (known) vs eval (unknown) + per-attack ----------
    Xd, yd, _ = _load_npz(feat_dir / "asvspoof19_la_dev.npz")
    Xe, ye, atk_e = _load_npz(feat_dir / "asvspoof19_la_eval.npz")
    Xd_s, Xe_s = scaler.transform(Xd), scaler.transform(Xe)
    ld, le = _bonafide_labels(yd), _bonafide_labels(ye)

    rows = []
    for name, model in {**models, "ensemble": models}.items():
        sd = ensemble_score(model, Xd_s) if name == "ensemble" else score_bonafide(model, Xd_s)
        se = ensemble_score(model, Xe_s) if name == "ensemble" else score_bonafide(model, Xe_s)
        rows += harness.in_domain_rows(name, (ld, sd), (le, se), atk_e, seed=seed, front_end=fp)
    pd.DataFrame(rows).to_csv(tables / "baseline_in_domain.csv", index=False)
    print(f"  -> {tables / 'baseline_in_domain.csv'}")

    # ---------- cross-dataset: In-the-Wild ----------
    itw_npz = feat_dir / "in_the_wild_eval.npz"
    if itw_npz.exists():
        Xw, yw, _ = _load_npz(itw_npz)
        Xw_s = scaler.transform(Xw)
        lw = _bonafide_labels(yw)
        rows = []
        for name, model in {**models, "ensemble": models}.items():
            sw = ensemble_score(model, Xw_s) if name == "ensemble" else score_bonafide(model, Xw_s)
            rows.append(
                harness.cross_dataset_row(
                    name, lw, sw, dataset="in_the_wild", seed=seed, front_end=fp
                )
            )
        pd.DataFrame(rows).to_csv(tables / "baseline_cross_dataset.csv", index=False)
        print(f"  -> {tables / 'baseline_cross_dataset.csv'}")
    else:
        print("  SKIP cross-dataset (no in_the_wild_eval.npz)")

    # ---------- silence-only ablation ----------
    _run_silence_ablation(cfg, pcfg, fp, seed, scaler, models, tables)


def _run_silence_ablation(cfg, pcfg: PreprocessConfig, fp, seed, scaler, models, tables) -> None:
    """Re-preprocesses a balanced dev subset from raw, strips speech, extracts
    features, and scores. A model trained through the anti-shortcut front-end
    must land near 50% EER here."""
    n = cfg["eval"]["quick_n_per_class"]
    top_db = cfg["eval"]["silence_ablation"]["top_db"]
    leak_thr = cfg["eval"]["silence_ablation"]["leak_eer_threshold"]

    df = load_manifest(resolve(cfg, "manifests") / "asvspoof19_la_dev.csv")
    df = pd.concat(
        [g.sample(min(n, len(g)), random_state=seed) for _, g in df.groupby("label")]
    ).reset_index(drop=True)

    X, y = [], []
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="silence-ablation"):
        try:
            wav = preprocess_file(row.path, pcfg)
            sil = harness.silence_only_waveform(wav, pcfg.sample_rate, top_db)
            X.append(extract_handcrafted(sil, pcfg.sample_rate))
            y.append(1 if row.label == "spoof" else 0)
        except Exception:
            pass
    Xs = scaler.transform(np.asarray(X, dtype=np.float32))
    labels = (np.asarray(y) == 0).astype(int)

    rows = []
    for name, model in {**models, "ensemble": models}.items():
        s = ensemble_score(model, Xs) if name == "ensemble" else score_bonafide(model, Xs)
        rows.append(
            harness.silence_ablation_row(
                name, labels, s, seed=seed, front_end=fp, leak_threshold=leak_thr
            )
        )
    out = pd.DataFrame(rows)
    out.to_csv(tables / "baseline_silence_ablation.csv", index=False)
    print(f"  -> {tables / 'baseline_silence_ablation.csv'}")
    print(out[["model", "value_pct", "verdict"]].to_string(index=False))


if __name__ == "__main__":
    main()
