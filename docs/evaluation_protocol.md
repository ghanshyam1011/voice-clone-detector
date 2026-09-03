# Evaluation protocol (frozen)

Change this file only by team agreement. Every result we report follows it.

## 1. Splits — use the official protocol files, never a home-made split

| Split | Source | Role |
|---|---|---|
| `asvspoof19_la_train` | `ASVspoof2019.LA.cm.train.trn.txt` | training |
| `asvspoof19_la_dev` | `ASVspoof2019.LA.cm.dev.trl.txt` | validation / threshold selection. Known attacks **A01–A06** |
| `asvspoof19_la_eval` | `ASVspoof2019.LA.cm.eval.trl.txt` | in-domain test. Unknown attacks **A07–A19** |
| `in_the_wild_eval` | In-the-Wild `meta.csv` | **cross-dataset test only — never trained on, never used to pick a threshold** |

A home-made random split leaks speakers and attack types across partitions and inflates scores into meaninglessness. Manifests are built once by `scripts/build_manifests.py` from the protocol files.

## 2. The anti-shortcut front-end — mandatory, identical for both classes

`src/silence_ablation.py` (old) measured **13–15% EER using only the non-speech parts of dev clips** (chance = 50%). The old ensemble had learned the ASVspoof-2019 silence-duration / loudness artifact, not synthesis. Every feature vector and model trained before the fix is contaminated; those numbers are archived under `results/tables/_superseded/`.

Every feature extractor and every model path **must** load audio through `voiceguard.audio.preprocess_file` / `preprocess_wave`:

1. load → mono → 16 kHz (`soxr_hq`)
2. `librosa.effects.trim` (leading/trailing near-silence), then re-pad a fixed margin — **identical for bonafide and spoof**
3. loudness normalise to a fixed integrated loudness (EBU R128 via `pyloudnorm`, RMS fallback), then peak-limit
4. (P1) optional codec pass — when used, applied to **both** classes

The front-end settings are hashed into a 10-char `front_end` fingerprint that is stamped on every results row and into the processed-feature directory name. Numbers with different fingerprints are not comparable and must not share a table.

## 3. The silence-ablation gate — permanent

After any model change, run the silence-only ablation (`scripts/run_eval.py`). A model trained through the front-end above **must score EER within ~5 points of 50%** on silence-only input. Below 40% = shortcut leak = stop and fix the pipeline before trusting any other number. This gate stays in place for every new dataset added later.

## 4. What every result row records

`scripts/run_eval.py` writes these columns on every row (`voiceguard.eval.provenance.stamp`):

`train_set, test_set, model, seed, front_end, score_direction, git_commit, utc` + the metric.

No number goes in a slide, a README, or a paper without that provenance.

## 5. Metrics

- **EER** — headline. `voiceguard.eval.metrics.compute_eer`, unit-tested. Score convention: **higher = more bonafide**, label convention: **1 = bonafide, 0 = spoof**.
- **Per-attack EER** on the eval split — each generator's spoofs vs *all* bonafide, so a weak generator can't hide in the aggregate. Report **known (A01–A06)** and **unknown (A07–A19)** separately; unknown-attack EER predicts real-world behaviour.
- **min t-DCF / a-DCF** — added in P1 from the ASVspoof reference implementation.
- **Never** report accuracy as a headline — the classes are ~9:1.

## 6. Generalisation rule

At least one generator and (from P6) one language are always held out of training. The cross-dataset In-the-Wild number is the honest one; the in-domain eval number is the flattering one. Both appear in every comparison table, side by side.
