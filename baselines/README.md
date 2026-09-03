# Baselines

## Generation 1 — handcrafted features + gradient boosting

61-dim vector (40 MFCC + rolloff + ZCR + 12 chroma + 7 spectral-contrast, time-averaged)
→ `StandardScaler` → XGBoost / LightGBM / CatBoost → mean-probability ensemble.

**This is a comparison-table row, not a serving component.** Time-averaged MFCC
means discard the temporal and phase structure where vocoder artifacts live; on
In-the-Wild the old version scored ~50% EER (coin flip). It stays here as the
honest handcrafted baseline and as a fast sanity check on the data pipeline.

### Differences from the old notebook 02

| Old (`notebooks/02_ml_ensemble.ipynb`) | Now |
|---|---|
| `librosa.load()` with no trim/loudness → silence leak | shared anti-shortcut front-end |
| SMOTE: 2,580 real bonafide → 22,800 interpolated | class weights, no synthetic data |
| accuracy reported, threshold hand-picked | EER, DET, provenance-stamped |

### Reproduce

```bash
python scripts/build_manifests.py
python scripts/extract_features.py --manifest asvspoof19_la_train
python scripts/extract_features.py --manifest asvspoof19_la_dev
python scripts/extract_features.py --manifest asvspoof19_la_eval
python scripts/extract_features.py --manifest in_the_wild_eval
python scripts/train_baseline.py
python scripts/run_eval.py
```

Artifacts land in `baselines/artifacts/<front-end-fingerprint>/` (git-ignored).
Result tables land in `results/tables/baseline_*.csv`.
