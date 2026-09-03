# Superseded result tables — do not cite

These CSVs were produced before the anti-shortcut audio front-end existed.
The silence-only ablation (`silence_ablation.csv`) shows **13–15% EER on
non-speech-only input** — proof the models keyed on the ASVspoof-2019
silence-duration / loudness artifact, not synthesis.

Every number here is contaminated by that leak, including the headline
**1.83% dev EER** in `asvspoof19_la_eer.csv`. Kept only so the "before" is
on record.

The trustworthy replacements are `results/tables/baseline_*.csv`, produced
by `scripts/run_eval.py` after `scripts/extract_features.py` re-extracts
features through `voiceguard.audio.preprocess_*`. See
`docs/evaluation_protocol.md`.
