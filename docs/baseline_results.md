# Baseline results — Generation 1 (handcrafted + GBM)

**Front-end:** `640b0644ae` (trim top_db 30, pad 40 ms, EBU R128 −23 LUFS, −1 dBFS ceiling)
**Train:** ASVspoof 2019 LA train · **Model:** XGB/LGBM/CatBoost mean-probability ensemble · class weights, no SMOTE
**Sample:** 2,500 clips/class per split (subset — see note). Full-corpus re-run: `python scripts/extract_features.py --all --force`
**Metric:** EER %, score direction higher = bonafide. Raw rows: `results/tables/baseline_*.csv`.

## The headline

| Test set | Old pipeline (silence leak) | Fixed front-end |
|---|--:|--:|
| ASVspoof19 dev — known attacks A01–A06 | 1.83% | **7.6%** |
| ASVspoof19 eval — unknown attacks A07–A19 | 15.7% | **33.9%** |
| In-the-Wild — cross-dataset | ~50% | **58.7%** |
| **Silence-only ablation** (chance = 50%) | **13–15% — LEAK** | **50–59% — no leak** |

The old 1.83% was mostly the ASVspoof-2019 silence-duration artifact. With the shortcut removed, the handcrafted baseline is what it actually is: **weak** — a third of unknown attacks pass, and it is worse than a coin flip on real-world audio. That is the correct, honest starting point for P1.

## Per-attack EER, ASVspoof19 eval (ensemble)

| Attack | EER % | | Attack | EER % |
|---|--:|---|---|--:|
| A07 | 0.0 | | A13 | 60.3 |
| A16 | 0.6 | | A18 | 58.3 |
| A19 | 0.7 | | A12 | 54.8 |
| A08 | 5.5 | | A10 | 52.2 |
| A09 | 5.6 | | A11 | 37.2 |
| A14 | 26.9 | | A17 | 29.2 |
| A15 | 27.0 | | | |

Waveform-concatenation and older TTS (A07–A09, A16, A19) are easy; neural
TTS / VC (A10–A13, A18) defeat the handcrafted features completely.

## In-the-Wild — model disagreement

| Model | EER % |
|---|--:|
| XGBoost | 64.0 |
| LightGBM | 72.7 |
| CatBoost | 42.2 |
| Ensemble | 58.7 |

The 30-point spread across models on the same data means the handcrafted
features carry almost no transferable signal here — the "predictions" are
close to noise. Do not read the CatBoost 42% as encouraging.

## Note on the subset

2,500 clips/class keeps the re-measurement tractable on a laptop. EER
confidence intervals at this size are roughly ±1–2 pp for the in-domain
splits and wider for per-attack rows (~230 clips each). The direction and
magnitude of every finding above is not sensitive to that — but the
full-corpus numbers replace these before anything is published.
