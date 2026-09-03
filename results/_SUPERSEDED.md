# Stale artifacts in this directory (git-ignored, safe to delete)

Produced by the pre-P0 pipeline. Nothing in `src/voiceguard/` or `scripts/`
references them. Left on disk only so you can inspect them; delete whenever.

| Path | Why stale |
|---|---|
| `xgboost_model.pkl`, `lightgbm_model.pkl`, `catboost_model.pkl`, `scaler.pkl` | trained on silence-leaked features; replaced by `baselines/artifacts/<fp>/` |
| `xgboost_mic_tuned.pkl`, `mic_threshold.npy` | output of deleted `finetune_mic.py` — tuned on ~10 clips of one voice |
| `autoencoder.pth`, `tau.npy` | autoencoder branch dropped (43% detection @ 5% FPR) |
| `cnn_bilstm.pth` | trained on 2k clips, never EER-evaluated, no cross-dataset |
| `wav2vec2_finetuned/`, `wav2vec2_finetuned_backup/` | 27% dev EER; the 378 MB file that bloated `.git`. Model is cut. |
| `wav2vec2_threshold.npy` | threshold for the cut model |

`results/figures/*.png` are kept — they're the baseline exploration record.
`results/tables/baseline_*.csv` (after `scripts/run_eval.py`) are the real numbers.
