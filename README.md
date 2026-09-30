# VoiceGuard

**A multi-signal voice-integrity layer for high-risk phone calls, and an honest
measurement of where synthetic-speech detection actually stands.**

Smart India Hackathon 2026, problem SIH26104 (AICTE Cyber Security Cell). The
brief asks for a real-time system that detects cloned/synthetic speech, verifies
the claimed speaker, measures behavioural anomalies, and recommends action before
a transaction — with a privacy-preserving design and Indian-language support.

This repository takes the position that **no single acoustic detector generalises
to unseen speech synthesis**, and builds around that fact rather than against it.

**New to this repo?** [`GETTING_STARTED.md`](GETTING_STARTED.md) walks through
getting it running on your own machine from a fresh clone — installing
libraries, downloading the dataset, running the live demo, and (optionally)
training — with every command spelled out. Everything below is the
research-oriented account of the work.

---

## 1. Motivation

Published anti-spoofing countermeasures report sub-1% Equal Error Rate (EER) on
ASVspoof 2019. Those numbers do not survive contact with real audio:

- Müller et al. (2022), *"Does Audio Deepfake Detection Generalize?"* — up to
  **1000% relative degradation** on found ("In-the-Wild") audio.
- We reproduce this. The strongest published model we tested (AASIST) drops from
  **~1% published EER to 35.8% EER** on In-the-Wild once evaluated honestly, and
  a current text-to-speech model (Meta MMS-TTS, English) is flagged only **8% of
  the time**.

A product that stakes its decision on one such detector fails silently the first
time it meets a generator it wasn't trained on. VoiceGuard instead fuses four
weak-but-partly-independent signals and keeps a human in the loop.

---

## 2. What has been done

![VoiceGuard multi-signal decision pipeline](docs/architecture.svg)

<sub>Figure style adapted from the pipeline diagram in Sun et al.,
[*AI-Synthesized Voice Detection Using Neural Vocoder Artifacts*](https://arxiv.org/abs/2304.13085)
(CVPRW 2023). The architecture shown is VoiceGuard's own.</sub>

> One anti-shortcut front-end feeds four weak, partly-independent signals
> (§2.2). `voiceguard.risk` fuses the signals that are present and a policy
> table turns the fused score into **ALLOW / VERIFY / ESCALATE** with reason
> codes (§2.3). Every decision is written to a feature-only audit log and shown
> in the operator console; a human acknowledges and acts. The subsections below
> take each stage in turn.

### 2.1 A trustworthy evaluation harness (the core methodological contribution)

The ASVspoof 2019 LA corpus contains a **silence-duration / loudness artifact**:
bonafide and spoof clips differ systematically in leading/trailing silence. A
model can score well by learning that artifact instead of synthesis.

We measured it: a classifier trained on our earlier pipeline scored **13–15% EER
using *only the non-speech regions* of dev clips** (chance = 50%). Its apparent
1.83% EER was mostly the artifact.

The fix (`voiceguard.audio`, [`docs/evaluation_protocol.md`](docs/evaluation_protocol.md)):

1. every audio path — features, training, inference — loads through one
   front-end: mono → 16 kHz → uniform silence trim + fixed re-pad →
   EBU-R128 loudness normalisation → peak limit, **identical for both classes**;
2. the front-end settings hash to a 10-char `front_end` fingerprint stamped on
   every result row; numbers with different fingerprints never share a table;
3. a **permanent silence-ablation gate**: after any model change, EER on
   silence-only input must be within ~5 points of 50%. Below 40% ⇒ leak ⇒ stop.

After the fix, silence-only EER is **55–61%** (GBM) / **41%** (AASIST) — the
models are keying on speech, not the artifact.

Splits come from the official protocol files only. **In-the-Wild is never trained
on and never used to pick a threshold.** Every table reports the flattering
in-domain number and the honest cross-dataset number side by side.

### 2.2 Detection signals

| Signal | Implementation | Standalone performance |
|---|---|---|
| **Synthetic-voice** | AASIST (vendored `clovaai/aasist`), published weights, run through the front-end | dev (known) **9.6%** · eval (unknown) **23.3%** · In-the-Wild **35.8%** EER |
| **Speaker consistency** | WavLM x-vector (`microsoft/wavlm-base-plus-sv`), per-session enrolment, cosine similarity → mismatch risk | same-speaker cosine ≈ 0.95+ on ≥ 6 s enrolment; Platt-calibrated, held-out speakers ECE 0.389→0.181 |
| **Prosody / behaviour** | Praat/Parselmouth features (F0 stats, jitter, shimmer, HNR, pause ratio, energy flux) + logistic regression on ASVspoof-train | **26.3%** dev EER — weak by design, a contributing signal |
| **Call context** | rules over caller-known / transfer amount / prior flags / channel | deterministic |

Also implemented and unit-tested but **not yet trained at scale**: RawNet2,
AASIST-L, SSL-AASIST (frozen wav2vec2 / XLS-R front-end + AASIST graph back-end);
RawBoost + telephony-codec augmentation; a resumable training loop
(`scripts/train_cm.py`). A 2-epoch local run confirms the code trains correctly
(dev EER 34.1% → 26.6%); full training is a compute problem, not a code problem
(see §5).

### 2.3 Fusion, policy, and the operator layer

- **`voiceguard.risk`** — weighted fusion over the *present* signals (weights
  renormalised when a signal is absent) with a lone-signal floor, then a policy
  table mapping the fused score to **ALLOW / VERIFY / ESCALATE** under three
  scenarios (routine call, transaction, privileged change), with plain-English
  reason codes and a recommendation. Deterministic, unit-tested. Thresholds are
  documented operating points, not calibrated.
- **`voiceguard.audit`** — feature-only JSONL log: scores, action, reasons,
  model version, front-end fingerprint, git commit, timestamp. It **rejects any
  key** that would carry audio, a transcript, or an embedding. Retention horizon
  + prune. This is the privacy posture.
- **`voiceguard.serve`** — a FastAPI session API (`POST /api/session`, `enroll`,
  `context`, `analyze`, `acknowledge`, `WS .../stream`, `GET .../audit`) and a
  single-page **operator console** (`/`): decision banner, fused-risk timeline,
  four signal meters, reason codes, editable call context, a simulated alert +
  acknowledge, and the live audit trail. `examples/rest_client.py` drives the
  whole flow headless. The same session API is also reachable over **gRPC**
  (`voiceguard.serve.grpc_service`, `scripts/serve_grpc.py`,
  `examples/grpc_client.py`) — one decision engine, two transports; see
  [`docs/grpc.md`](docs/grpc.md).

### 2.4 Per-language evaluation (IndicCall-Eval, pilot)

`scripts/make_indic_spoof.py` generates MMS-TTS attacks in **English, Hindi,
Tamil, Bengali, Marathi**; `scripts/eval_indic.py` scores them per language.

| language | detector flags the synthetic clip | mean prosody score |
|---|--:|--:|
| Marathi | **85%** | 63% |
| Hindi | 58% | 55% |
| Tamil | 55% | 73% |
| Bengali | 38% | 43% |
| English | **8%** | 47% |

The detector was trained **only on English** ASVspoof 2019, yet flags
Indian-language synthesis *more* often than English synthesis — non-English TTS
sits further from the "natural English" distribution the model learned, so its
artefacts trip the detector more. Uneven, measured, per-language — **not** a
"supports all Indian languages" claim. The consented real-speech slice (protocol
in [`docs/indic_eval.md`](docs/indic_eval.md)) is what turns recall into a
per-language EER; the team records it.

---

## 3. Key results

All numbers follow [`docs/evaluation_protocol.md`](docs/evaluation_protocol.md)
and carry provenance in `results/tables/`. Front-end fingerprint `640b0644ae`,
seed 42, trained on `asvspoof19_la_train`.

| System | dev EER (known A01–06) | eval EER (unknown A07–19) | **In-the-Wild EER** | silence-only EER |
|---|--:|--:|--:|--:|
| Handcrafted 61-dim + GBM ensemble (Gen-1 baseline) | 7.0% | 33.4% | 58.2% | 59.0% (≈ chance ✓) |
| **AASIST, published weights, our front-end** | 9.6% | **23.3%** | **35.8%** | 41.3% (≈ chance ✓) |

- AASIST's dev EER (9.6%, vs its published ~1%) is *worse* through our front-end
  because the front-end strips the silence artifact it had partly learned. The
  honest score is the higher one.
- In-the-Wild **58.2% → 35.8%** is the one improvement that matters (a
  cross-dataset test the model never saw). It is still far from deployable
  (target < 15%).
- Per-attack (AASIST eval): near-perfect on A08/A09 (< 2%), fails on A10/A12
  (44–52%) — unfamiliar neural vocoders. Full breakdown in
  `results/tables/cm_aasist-pretrained_in_domain.csv`.

**min-tDCF** (the ASVspoof community's own deployment-oriented metric — a
countermeasure scored in tandem with a fixed ASV system under an explicit
cost model, using the organizers' reference implementation and their
baseline ASV system's scores; 0 = perfect, 1 = no better than no
countermeasure at all — see [`docs/tdcf.md`](docs/tdcf.md)):

| System | dev min-tDCF | eval min-tDCF |
|---|--:|--:|
| **AASIST, published weights, our front-end** | 0.288 | **0.588** |

Same story as EER, from a different, independently-computed metric: usable
on known attacks, collapses on unseen ones. ASVspoof 2021 DF, a second
held-out cross-dataset test, is *not* built — the corpus needs a
registration step we haven't done yet (see `docs/tdcf.md`).

**The system-level result:** a modern TTS clip the acoustic model rates "real"
(risk 29/100) still reaches **ESCALATE** when the speaker-consistency check
(clone ≠ enrolled customer) and the call-context rules fire. That is the thesis,
demonstrated end to end.

---

## 4. Reproduce

```bash
python -m venv myenv
myenv/Scripts/activate                 # Windows; source myenv/bin/activate elsewhere
pip install -e ".[baseline,dl,dev]"    # + ",serve,prosody" for the console
```

Python 3.11. `ffmpeg` on `PATH` for the codec pass.

**Data** (not in git — place at the `config/default.yaml` paths):
`ASVspoof 2019 LA` → `data/raw/LA/LA/`, `In-the-Wild` →
`data/external/in_the_wild/extracted/release_in_the_wild/`, then:

```bash
python scripts/build_manifests.py      # manifests = the single source of truth
python scripts/train_baseline.py       # Gen-1 baseline
python scripts/run_eval.py             # baseline_*.csv (incl. the silence gate)
python scripts/eval_cm.py --model aasist --pretrained   # AASIST through the front-end
python scripts/eval_tdcf.py --model aasist --pretrained # cm_*_tdcf.csv (needs the LA asv_scores/ files)
python scripts/train_prosody.py        # models/prosody/prosody_lr.joblib (tracked, 2 KB)
```

**The console demo** ([`docs/demo_runbook.md`](docs/demo_runbook.md)):

```bash
pip install -e ".[dl,serve,prosody]"
python scripts/pick_demo_clips.py      # demo_audio/{enrol,benign,cloned}.wav from ASVspoof
python scripts/serve_demo.py           # http://127.0.0.1:8000  (console) + /detector
```

**Same session, over gRPC instead** ([`docs/grpc.md`](docs/grpc.md)):

```bash
pip install -e ".[dl,serve,prosody,grpc]"
python scripts/serve_grpc.py           # 127.0.0.1:50051, in a second terminal
python examples/grpc_client.py demo_audio/enrol.wav demo_audio/benign.wav demo_audio/cloned_1.wav
```

**Per-language pilot:**

```bash
pip install -e ".[research]"
python scripts/make_indic_spoof.py --per-lang 40
python scripts/eval_indic.py           # results/tables/indic_eval.csv
```

103 unit tests (`pytest -q`) — metrics, front-end, manifests, models, speaker,
prosody, risk fusion, audit, the session API (REST and gRPC), VAD,
calibration, min-tDCF.

---

## 5. Limitations and what is not done

- **The detector is not ours and not good enough.** All model numbers use
  published AASIST weights. 35.8% In-the-Wild EER is not deployable. Full
  training does not fit the development laptop (≈ 50 min/epoch, 30–40 epochs
  needed); it is set up to run on a Kaggle T4 — see
  [`docs/training_runbook.md`](docs/training_runbook.md) for the concrete
  setup and why AASIST, not SSL-AASIST, should be trained first.
- **Calibration measured, not assumed — and it only held up for one signal
  of three.** Speaker consistency ships a Platt calibrator verified to
  generalize to held-out speakers (ECE 0.389→0.181). Spoof and prosody
  calibrators were fit and measured the same way, and *made things worse*
  on unseen attacks / cross-dataset audio, so neither ships; both stay on
  their pre-existing ad-hoc operating points. See
  [`docs/calibration.md`](docs/calibration.md).
- **Latency.** The acoustic model still needs a 4.04 s window for a
  *confirmed* read — not built shorter. What is built: speech-onset VAD
  (latency measured from when the caller starts talking, not call-connect)
  and safety-capped provisional reads (a short/tiled-window score is marked
  `provisional` and capped at VERIFY, never a silent ESCALATE, until a full
  window confirms it). See [`docs/latency.md`](docs/latency.md).
- **Alerts are simulated** in the console; no real SMS/email/webhook sink.
- **IndicCall-Eval** has the synthetic side and the harness; the consented
  real-speech slice is pending.
- **gRPC + SDK** now ships (`voiceguard.serve.grpc_service`,
  `examples/grpc_client.py`) alongside REST/WS — same `SessionManager`,
  verified to produce identical decisions on the same clip. See
  [`docs/grpc.md`](docs/grpc.md).
- **Model card** now ships at
  [`models/pretrained/MODEL_CARD.md`](models/pretrained/MODEL_CARD.md).
  It documents a real gap while doing so: **this repository has no
  top-level LICENSE file yet** — the vendored AASIST/RawBoost code carries
  its own upstream terms, but this project's own code doesn't declare one.
  Not a decision this pass makes unilaterally.
- **Not started:** on-device/edge inference, MUSAN/RIR augmentation,
  ASVspoof 2021 DF as a second eval set (see [`docs/tdcf.md`](docs/tdcf.md)),
  real alert delivery (SMS/email/webhook — the console panel is simulated),
  auth/TLS hardening for the REST and gRPC servers (both currently bind to
  localhost only, by design, not yet a gap being closed).

Roadmap and phase status: [`docs/project_status.md`](docs/project_status.md).

---

## 6. Repository layout

```
src/voiceguard/
  audio/       anti-shortcut front-end (load, trim, loudness, codec) + augmentation;
               speech-onset VAD (vad.py)
  data/        manifest build + load (repo-relative, portable)
  features/    handcrafted 61-dim features (Gen-1 baseline only)
  models/      baseline GBM; RawNet2 / AASIST / AASIST-L / SSL-AASIST (vendored refs)
  eval/        EER / DET metrics, calibration (ECE/Brier/Platt/isotonic), min-tDCF
               (reference impl vendored in _vendor/), provenance stamping, harness
  detect/      windowed CMScorer + EMA risk timeline
  speaker/     WavLM x-vector enrolment + per-session consistency check (calibrated)
  prosody/     Praat features + logistic-regression scorer
  risk/        signal fusion + policy engine (ALLOW / VERIFY / ESCALATE)
  audit.py     feature-only audit log
  serve/       session REST/WS API + operator console (FastAPI, localhost only);
               grpc_service.py + proto/ for the gRPC transport (same SessionManager)
config/        default.yaml — the only place any path is written
examples/      rest_client.py + grpc_client.py — same flow, two transports
notebooks/     kaggle_train_cm.ipynb — runnable Kaggle trainer (item c); 00-05 are
               pre-P0 exploration, superseded, kept for reference only (see its README)
scripts/       build_manifests, train_baseline, train_cm, train_prosody, run_eval,
               eval_cm, eval_indic, make_indic_spoof, make_modern_spoof,
               finetune_modern, pick_demo_clips, prep_demo_call, serve_demo, serve_grpc,
               demo*, calibrate_cm, calibrate_prosody, calibrate_speaker, eval_tdcf
docs/          evaluation_protocol.md (frozen), baseline_results.md, project_status.md,
               demo_runbook.md, indic_eval.md, initial-review.md, calibration.md,
               latency.md, tdcf.md, grpc.md, training_runbook.md
models/        pretrained/ (+ MODEL_CARD.md) + prosody/ + speaker/ weights and
               calibrators, committed; everything else under models/ (training
               checkpoints) is gitignored
paper/         IEEE-style writeup (tex/pdf/docx) + slide deck
results/tables/ every metric, provenance-stamped, committed
tests/         103 unit tests
```

Vendored reference implementations (`src/voiceguard/models/_vendor/`) retain their
upstream licenses: AASIST/RawNet2 from `clovaai/aasist`, RawBoost from
`TakHemlata/RawBoost-antispoofing`.

---

## 7. References

- Jung et al., *AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal
  Graph Attention Networks*, ICASSP 2022.
- Müller et al., *Does Audio Deepfake Detection Generalize?*, Interspeech 2022
  (In-the-Wild dataset).
- Tak et al., *RawBoost: A Raw Data Boosting and Augmentation Method for Speech
  Anti-Spoofing*, ICASSP 2022.
- Chen et al., *WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack
  Speech Processing*, 2022.
- Wang et al., *ASVspoof 2019: A Large-Scale Public Database of Synthesized,
  Converted and Replayed Speech*.
- Pratap et al., *Scaling Speech Technology to 1,000+ Languages* (Meta MMS).
