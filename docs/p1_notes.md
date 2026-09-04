# P1 — a detector that generalises

Goal: **In-the-Wild EER < 15%** with a documented method + an ablation table
attributing the gain (baseline is 58.7%). At least one generator and one
language always held out of training (docs/evaluation_protocol.md).

## What's wired (this increment)

| Piece | Module | Notes |
|---|---|---|
| RawBoost augmentation | `voiceguard.audio.augment` | vendored reference (`_vendor/rawboost_ref.py`); algos 1–8, default 5 (LnL→ISD series) |
| Codec pass | `voiceguard.audio.codec` | ffmpeg: G.711 µ-law, G.722, AMR-NB, Opus@8k — applied to **both classes** |
| RawNet2 | `voiceguard.models.neural` | vendored `clovaai/aasist` RawNet2Spoof (17.6M) — Tier-1 streaming candidate |
| AASIST | `voiceguard.models.neural` | vendored `clovaai/aasist` (0.30M) — the field reference backend |
| AASIST-L | `voiceguard.models.neural` | 0.085M — lightweight variant, another Tier-1 candidate |
| torch Dataset | `voiceguard.data.torch_dataset` | load → front-end → augment (train only) → tile-pad/crop to 64600 |
| Train loop | `scripts/train_cm.py` | Adam + cosine, AMP, class-weighted CE, per-epoch dev EER, best-checkpoint |
| Eval | `scripts/eval_cm.py` | dev/eval/per-attack + In-the-Wild + silence ablation, provenance-stamped |

Class convention: upstream uses class 1 = bonafide; the wrapper flips the
logit columns so **column 0 = bonafide** everywhere in this repo.

## Not yet done

- **SSL-AASIST** (wav2vec2 / XLS-R frozen front-end + AASIST backend) — this
  is the model that actually reaches <15% In-the-Wild. Needs an SSL feature
  cache (frozen backbone) + an AASIST variant that consumes `(B, T, D)`
  features instead of raw audio. Next.
- MUSAN additive noise + RIR reverb augmentation (needs the MUSAN download).
- The ablation grid (aug on/off, frozen vs fine-tune, front-end choice).
- min t-DCF alongside EER.

## Compute

RawNet2 / AASIST / AASIST-L train end-to-end on the 6 GB laptop GPU. Full
SSL fine-tuning does not — use the frozen-backbone + cached-features route,
or move SSL training to Kaggle (T4, 30 h/wk). **Open decision.**

## Run

```bash
python scripts/train_cm.py --model aasist --epochs 40 --batch-size 16
python scripts/eval_cm.py  --model aasist
```
Checkpoints -> `models/cm/<name>/best.pt` (git-ignored).
Numbers -> `results/tables/cm_<name>_*.csv`.
