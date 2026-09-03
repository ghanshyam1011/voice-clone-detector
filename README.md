# voiceguard — SIH26104

Real-time detection of synthetic / cloned speech in live calls.
Origin: Smart India Hackathon 2026 (AICTE Cyber Security Cell). Target: a
deployable voice-integrity layer, not a demo.

Full plan and roadmap: the rebuild plan (P0–P7). This repo is currently at **P0 —
foundations & trustworthy measurement**.

## Status

| Phase | State |
|---|---|
| P0 foundations + fix the silence leak + re-measure | **done** (full-corpus re-run pending on mains power) |
| P1 a detector that generalises (RawNet2 / AASIST / SSL-AASIST + augmentation) | next |
| P2 real streaming (VAD, sliding window, EMA, latency budget) | not started |
| P3 speaker verification + prosody branch + calibrated fusion | not started |
| P4 prevention / policy engine | not started |
| P5 privacy (feature-only logging, edge) + REST/gRPC + SDK | not started |
| P6 multilingual / Indic evaluation set | not started |
| P7 hardening, model card, release | not started |

## Setup

```bash
python -m venv myenv && myenv/Scripts/activate    # Windows; use source myenv/bin/activate elsewhere
pip install -e ".[baseline,dev]"
```

Python 3.11. `ffmpeg` on PATH is required for codec augmentation (P1+).

## Data

Not in git. Place the corpora at the paths in `config/default.yaml`:

- ASVspoof 2019 LA → `data/raw/LA/LA/`
- In-the-Wild → `data/external/in_the_wild/extracted/release_in_the_wild/`

Then build the manifests (the single source of truth for splits):

```bash
python scripts/build_manifests.py
```

## Results so far

The silence leak is fixed — silence-only EER went from **13–15%** (shortcut) to
**50–59%** (chance). With the shortcut gone the handcrafted baseline is honestly
weak: **7.6%** dev EER, **33.9%** on unknown attacks, **58.7%** on In-the-Wild.
The old 1.83% was mostly the artifact. Full breakdown:
[docs/baseline_results.md](docs/baseline_results.md).

## Reproduce the baseline

See [baselines/README.md](baselines/README.md). Evaluation rules are frozen in
[docs/evaluation_protocol.md](docs/evaluation_protocol.md) — read it before
quoting any number.

## Layout

```
src/voiceguard/     installable package
  audio/            the anti-shortcut front-end (load, trim, loudness, codec)
  data/             manifest build + load
  features/         handcrafted 61-dim (baseline only)
  models/           baseline GBM now; RawNet2/AASIST/SSL-AASIST in P1
  eval/             metrics (unit-tested), provenance, harness
config/             default.yaml — no path is hard-coded anywhere else
scripts/            thin CLIs: build_manifests, extract_features, train_baseline, run_eval
baselines/          frozen Gen-1 baseline + its artifacts (git-ignored)
tests/              metrics, audio front-end, manifests
notebooks/          exploratory record only — NOT imported, outputs stripped
legacy/             pre-refactor scripts kept for reference — NOT maintained
docs/               evaluation_protocol.md (frozen); more per phase
```
