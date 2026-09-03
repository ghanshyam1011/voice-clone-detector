# notebooks/ — exploratory record only

**Not part of the pipeline.** Nothing in `src/`, `scripts/` or `tests/`
imports from here. Outputs are stripped on commit (`nbstripout` pre-commit
hook + CI check). Ruff skips this directory.

These are the original exploration that led to the rebuild plan. They use
hard-coded Windows paths and the pre-P0 (leaked) feature pipeline. Keep
them for reference; do not treat their numbers as real — see
`results/tables/_superseded/` and `docs/evaluation_protocol.md`.

| Notebook | What it explored | Superseded by |
|---|---|---|
| `00_data_exploration` | ASVspoof LA class balance, waveforms | — |
| `01_feature_extraction` | the 61-dim handcrafted vector | `voiceguard.features.handcrafted` + `scripts/extract_features.py` |
| `02_ml_ensemble` | XGB/LGBM/Cat + SMOTE | `voiceguard.models.baseline_gbm` (no SMOTE) |
| `03_deep_learning` | wav2vec2-base fine-tune, CNN-BiLSTM, autoencoder | P1 (SSL + AASIST) |
| `04_streaming_pipeline` | offline two-gate file scorer | P2 (real streaming) |
| `05_shap_explainability` | SHAP on the ensemble | P7 model card / AASIST attention |

Real analysis notebooks added later go in a `notebooks/analysis/` subfolder
and cite a `front_end` fingerprint like everything else.
