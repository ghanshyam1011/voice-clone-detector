# SIH26104 product review and winning roadmap

**Audience:** Voice Clone Detector team  
**Date:** 3 September 2026  
**Scope:** Review of `SIH26104_Project_Blueprint.md`, `SIH26104_Reference_Pack.md`, and the current repository; focused research on requirements, evaluation, and product direction.

## Executive answer

Do not position the current prototype as the final product. It is a useful **offline ASVspoof-2019 baseline**, with a partially working microphone demo. SIH26104 requires a privacy-preserving, multilingual, real-time **voice-integrity security layer**: continuous risk scoring, prosody analysis, cross-session speaker consistency, contextual policy, alerts/workflows, and REST/gRPC integration.

The winning story is therefore:

> **VaaneeShield is an India-first, privacy-preserving voice firewall for live high-risk calls. It detects synthetic speech, verifies the claimed speaker, measures behavioural anomalies, and recommends verification before a transaction is approved.**

This is substantially stronger than “we classify a WAV as real/fake.”

## What SIH explicitly requires

The supplied reference pack says the judging target includes five deliverable groups:

1. Deep-learning acoustic/spectral analysis; prosody/behavioural analysis; cross-session speaker consistency.
2. Continuous, configurable risk scoring enriched with call and transaction context.
3. UI/SMS/email/in-app alerts and pre-transaction recommendations.
4. Privacy: minimal retention, edge inference, anonymisation or feature-only logs.
5. REST and gRPC APIs/SDKs, plus Indian-language and accent support.

This is the build contract. Challenge-response is an excellent demo extension, but is optional; it must not replace the required parts.

## Current repository: honest assessment

### What is already good

- The repository trains a classical MFCC/spectral-feature ensemble and has a fine-tuned Wav2Vec2 experiment.
- `src/evaluate.py` correctly moves beyond accuracy: it computes EER, separates known ASVspoof dev attacks from unknown evaluation attacks, and reports per-attack failures.
- `src/metrics.py` has sensible automated tests for EER/FAR/FRR.
- The measured ensemble result is **1.83% EER** on ASVspoof 2019 development attacks but **15.72% EER** on its unknown evaluation attacks. That gap is a valuable honest result, not a failure.

### Why it is not product-ready yet

| Area | Current state | Required correction |
|---|---|---|
| Live inference | `src/streaming.py` consumes disjoint 3-second microphone chunks. | Use VAD-gated 1.5-second sliding windows with a 300 ms hop, EMA smoothing, timestamps and measured p50/p95 latency. |
| Decision quality | Wav2Vec2 softmax is shown as “confidence”; a fixed threshold is tuned against a few local recordings. | Calibrate on held-out data; display **risk**, uncertainty and reason codes. Select thresholds by a documented fraud-vs-friction policy. |
| Generalisation | Training/evaluation is almost entirely ASVspoof 2019 LA. | Keep In-the-Wild, ASVspoof 2021 DF, RTCFake/telephony transformations, and Indian held-out data strictly evaluation-only. |
| Model | Handcrafted ensemble is suitable only as a transparent baseline. The Wav2Vec2 code starts from a general checkpoint and uses a small sample. | Build a robust anti-spoof model branch (RawNet2/AASIST or SSL-AASIST) and use channel augmentation. |
| Required prosody | MFCC/chroma/contrast are not a prosody/behaviour branch. | Extract eGeMAPS/Praat features: F0 contour, rhythm, pauses, jitter, shimmer and energy; train a small calibrated classifier and fuse it with CM score. |
| Impersonation | No speaker verification / historical consistency exists. | Add ECAPA-TDNN embeddings, consented enrolment, cosine similarity and spoof-aware fusion. |
| Product | No API, dashboard, workflow, privacy store, multilingual evidence or alerting layer. | Build the six product modules in the roadmap below. |
| Engineering | Absolute Windows paths, duplicated old feature code, and no product integration tests make demo deployment fragile. | Use `pathlib`, configuration files, a reproducible environment, and API-level smoke tests. |

Important: do **not** claim that a 1.83% EER means 98% real-world protection. Your own 15.72% unknown-attack EER already disproves that. The research literature finds sharp degradation outside lab data: the original In-the-Wild study reports performance degradation of up to 1000% on found audio. [Müller et al. (2022)](https://arxiv.org/abs/2203.16263).

## Recommended architecture

```text
Telephony / microphone / VoIP
  -> VAD + 16 kHz normalisation + 1.5 s rolling buffer
  -> Tier 1 edge scorer: lightweight spoof + prosody features
  -> risk timeline (EMA; score every 300 ms)
       -> Tier 2 async verifier: SSL-AASIST / AASIST on 5–10 s context
       -> speaker consistency: ECAPA embedding versus consented enrolment
       -> context: known number, call channel, transfer amount, prior flags
  -> calibrated fusion engine
  -> policy: allow / warn + call-back recommendation / supervisor escalation
  -> operator UI, audit event, REST API and bidirectional gRPC stream
```

The output is not “fake.” It is a clear, defensible decision such as:

> **HIGH RISK (0.91): synthetic-acoustic score high; enrolled-voice similarity inconsistent; new caller number; ₹75,000 transfer requested. Recommended action: pause approval and verify through the registered number.**

That makes prevention visible and usable while retaining a human decision maker.

## Build roadmap in priority order

### P0 — make the existing baseline credible (next 1–2 days)

1. Preserve the current EER CSV as a baseline; add the commit hash, dataset split, sample count, score direction and threshold policy to every results table.
2. Fix streaming to overlap windows, smooth scores, and log latency. Demonstrate the first risk score within two seconds and report p95 end-to-end latency.
3. Add a **silence-only leakage test**: remove speech regions, retrain the simple classifier, and verify that it is near chance. Equalise trimming, loudness and codec processing across classes.
4. Run the existing tests plus a synthetic streaming test for window/hop/EMA logic. Do not overwrite the user’s existing uncommitted work.

### P1 — build every required product component (next 3–5 days)

1. **Prosody branch:** use openSMILE eGeMAPS or Parselmouth; fit a small LightGBM/logistic model. Report its independent and fused EER/AUC.
2. **Speaker-consistency branch:** use pretrained ECAPA-TDNN. Store only a consented embedding; return similarity and confidence, never a claim that similarity alone proves identity.
3. **Risk engine:** fuse calibrated spoof, prosody, speaker and context scores. Define three scenarios: routine call, high-value transaction, privileged approval. Each has explicit operating thresholds.
4. **Privacy module:** default to a rolling in-memory buffer. Persist only event ID, model version, scores, time, and a salted/rotated feature representation; preserve raw audio only with explicit incident policy and retention expiry.
5. **Integration contract:** FastAPI REST for file/session events; protobuf bidirectional gRPC for audio frames; ship a small Python SDK and an OpenAPI/proto example.
6. **Operator console:** a web dashboard with live risk timeline, contributing signals, alert acknowledgement, recommended action and immutable-looking audit events. Simulate email/SMS rather than integrating real customer data for the hackathon.

### P2 — prove the India-first claim (next 3–5 days)

1. Produce a small, consented **IndicCall-Eval** benchmark—not a misleading training dataset. Start with Hindi, Marathi, Tamil and Bengali; balance gender/age/accent where feasible.
2. Collect bona-fide speech in three channels: clean mic, handset/VoIP, and speaker-to-mic replay. Create synthetic material only from team members who signed consent, and label generator/channel/language. Never use non-consented voice samples.
3. Keep an untouched test partition. Report every metric by language, channel and attack family; show confidence intervals where sample size allows. State that it is a pilot dataset.
4. Fine-tune/augment using ASVspoof 2019 + 2021 DF + legally permitted data, but do not train on In-the-Wild. The official ASVspoof 2021 release includes logical-access, physical-access and DF data plus scoring resources. [ASVspoof 2021](https://www.asvspoof.org/index2021.html)

### P3 — improve the anti-spoof model only after P0–P2 work

- Compare current ensemble, RawNet2/AASIST, and SSL-AASIST/XLS-R using one fixed evaluation harness.
- Use RawBoost-like/channel augmentations: telephony codec, clipping, reverberation, noise, resampling and noise suppression.
- Evaluate clean, codec-transformed, unseen-generator, real-world, speaker-disjoint and each IndicCall-Eval slice.
- Prefer the smallest model that meets your measured latency and worst-slice target. A huge model without a product path will not win the demo.

## Evaluation scorecard for slides and jury questions

Report these, with no invented figures:

| Claim | Evidence you must collect |
|---|---|
| Detection quality | EER, ROC-AUC, FAR/FRR at each selected policy threshold, confusion matrix. |
| Generalisation | Separate EER/AUC on ASVspoof unknown attacks, ASVspoof 2021 DF, In-the-Wild, telephony transforms and IndicCall-Eval. |
| Equity | Per-language/accent/gender slice results and sample counts; say “insufficient data” where true. |
| Real time | Time to first score, p50/p95 audio-to-score latency, CPU/RAM/device. |
| Prevention | Scripted test calls showing low/medium/high policy outcome and acknowledgement/call-back workflow. |
| Privacy | Architecture proof: audio stays in edge buffer, expiry test, feature-only audit record shown in UI. |
| Integration | REST upload/session call, gRPC audio stream, and SDK snippet exercised end-to-end. |

ASVspoof itself makes clear that current challenge data includes diverse attacks and the 2021 DF release provides a more realistic deepfake test than 2019 LA alone. [ASVspoof 2021](https://www.asvspoof.org/index2021.html). ASVspoof 5 further adds a standalone countermeasure track and a spoofing-aware speaker-verification task, supporting the decision to combine spoof detection and speaker consistency. [ASVspoof 5 evaluation plan](https://www.asvspoof.org/file/ASVspoof5___Evaluation_Plan_Phase2.pdf).

## What to say—and not say—in the pitch

**Say:** “We deliberately measure unknown attacks, codecs and Indian-language pilot data. The model’s confidence is a decision-support risk score, not proof.”

**Say:** “We minimise retention: edge scoring and feature-only audit events are the default.” India’s DPDP Act establishes consent and data-minimisation/purpose-related obligations; obtain a legal review before making a compliance certification claim. [Digital Personal Data Protection Act, 2023](https://www.meity.gov.in/writereaddata/files/Digital%20Personal%20Data%20Protection%20Act%202023.pdf)

**Do not say:** “100% deepfake detection,” “works for all Indian languages,” “DPDP compliant,” or “prevents fraud” unless you have scope-specific evidence. Say “pilot,” “evaluated on,” and “recommended verification.”

**Do not lead with:** SHAP plots or a model architecture diagram. Lead with a 45-second bank-call scenario: live call -> risk grows -> transaction warning -> call-back recommendation -> audit entry.

## Evidence and research conclusions

- Generalisation is the central technical risk, not just an academic caveat. The In-the-Wild data contains 37.9 hours for 58 public figures and was designed specifically to test beyond lab benchmarks. [Fraunhofer AISEC dataset page](https://deepfake-demo.aisec.fraunhofer.de/in_the_wild)
- Current real-time communication channels change the signal through codecs and processing. RTCFake was introduced in 2026 precisely because offline detectors struggle with these platform transformations; it reports roughly 600 hours across conferencing/social platforms. [RTCFake paper](https://arxiv.org/abs/2604.23742)
- India-specific multilingual evaluation is a credible differentiator. IndicSynth offers 4,000+ hours of synthetic audio across 12 Indian languages, but its research-only CC BY-NC license must be respected and it should not be presented as telephony realism. [IndicSynth dataset](https://huggingface.co/datasets/vdivyasharma/IndicSynth)
- A generic detection model is not automatically fair across speakers or languages. Treat slice reporting, consent and small-sample limitations as product requirements, not footnotes.

## Definition of done for a winning hackathon demo

The demo is done when it can run a scripted live call, display a stable risk timeline within two seconds, distinguish a benign call from a consented synthetic/replay scenario, show speaker-consistency and prosody contributions, recommend verification before a mock transfer, create a feature-only audit event, and expose the same session through REST/gRPC—all with measured metrics and stated limits.

## Research limits

The official SIH language in the supplied pack is a structured restatement, not a verified official PDF quotation. Before submission, download and archive the official problem-statement PDF and reconcile wording. The current repository’s measured EER numbers were read from its local CSV; they have not been independently re-run in this review.
