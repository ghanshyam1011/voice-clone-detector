# voiceguard — SIH26104

Real-time detection of synthetic / cloned speech in live calls.
Origin: Smart India Hackathon 2026 (AICTE Cyber Security Cell). Target: a
deployable voice-integrity layer, not a demo.

Full plan and roadmap: the rebuild plan (P0–P7).

## Status

| Phase | State |
|---|---|
| P0 foundations + fix the silence leak + re-measure | **done** (full-corpus re-run pending on mains power) |
| P1 a detector that generalises | **in progress** — RawNet2 / AASIST / AASIST-L + RawBoost/codec wired; SSL-AASIST next |
| P2 real streaming (VAD, sliding window, EMA, latency budget) | partial — sliding window + EMA + session stream; no VAD / latency budget |
| P3 speaker verification + prosody branch + calibrated fusion | partial — WavLM speaker check + Praat prosody branch (26.3% EER) + rule-based fusion done; real calibration next |
| P4 prevention / policy engine | partial — scenario thresholds + reason codes + ALLOW/VERIFY/ESCALATE + simulated alert & acknowledge |
| P5 privacy (feature-only logging, edge) + REST/gRPC + SDK | partial — feature-only audit log + REST session API + headless client; no gRPC / edge |
| P6 multilingual / Indic evaluation set | partial — `scripts/make_indic_spoof.py` + `eval_indic.py`, 5-language MMS-TTS attacks, per-language recall/EER; real-speech pilot slice pending (`docs/indic_eval.md`) |
| P7 hardening, model card, release | not started |

For the SIH demo build (2–3 day vertical slice), see
[docs/project_status.md](docs/project_status.md) and
[docs/demo_runbook.md](docs/demo_runbook.md).

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

## Demo

Pretrained AASIST + WavLM speaker verification, inference only. See
[docs/demo.md](docs/demo.md) and the SIH script in
[docs/demo_runbook.md](docs/demo_runbook.md).

```bash
python scripts/demo.py --sample 8            # score random real + fake clips (CLI)
pip install -e ".[dl,serve]" && python scripts/serve_demo.py
#   http://127.0.0.1:8000  -> live-call console (spoof + speaker + context -> decision)
#   http://127.0.0.1:8000/detector  -> raw detector (file + live mic)
```

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
  detect/           windowed scorer + EMA risk timeline (demo + start of P2)
  speaker/          WavLM x-vector enrolment + per-session speaker check
  prosody/          Praat F0/jitter/shimmer/HNR/pause features + a tiny LR
  risk/             signal fusion + policy engine (ALLOW / VERIFY / ESCALATE)
  audit.py          feature-only audit log (no audio / transcript / voiceprint)
  serve/            demo web UI + operator console (FastAPI, localhost only)
  eval/             metrics (unit-tested), provenance, harness
config/             default.yaml — no path is hard-coded anywhere else
scripts/            thin CLIs: build_manifests, extract_features, train_*, run_eval, demo*, serve_demo
baselines/          frozen Gen-1 baseline + its artifacts (git-ignored)
tests/              metrics, audio front-end, manifests
notebooks/          exploratory record only — NOT imported, outputs stripped
legacy/             pre-refactor scripts kept for reference — NOT maintained
docs/               evaluation_protocol.md (frozen); more per phase
```
