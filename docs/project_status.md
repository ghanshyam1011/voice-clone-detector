# Project status — voiceguard / SIH26104

_Snapshot: 2026-09-08 (rev 2). Honest read of where the build is against what the problem statement asks for._

## The gap in one paragraph

SIH26104 asks for a **voice-integrity security layer for live high-risk calls**:
synthetic-speech detection **plus** prosody/behaviour analysis **plus** cross-session
speaker consistency, fused into a **continuous, context-aware risk score**, wired to
**alerts and a pre-transaction recommendation**, under a **privacy/edge** model, over
**REST + gRPC** with **Indian-language** evidence. A working vertical slice of that
layer now exists: **four signals** (spoof + prosody + speaker + context) → one
ALLOW/VERIFY/ESCALATE decision with reason codes → a simulated alert the agent
acknowledges → a feature-only audit trail, all in an operator console and over REST.
gRPC/SDK, the Indian-language set, real alert delivery, real calibration and a
trained-by-us model are designed but not built.

See [`docs/initial-review.md`](initial-review.md) for the full requirement breakdown
and [`docs/demo_runbook.md`](demo_runbook.md) for the demo script.

---

## What is actually done

| # | Piece | State | Note |
|---|---|---|---|
| 1 | **Honest evaluation harness** | ✅ solid | Anti-shortcut audio front-end (silence-leak fixed), manifest-driven splits, provenance stamped on every result row. Strongest asset — the numbers can be trusted. |
| 2 | **Generation-1 baseline** | ✅ done | Handcrafted + GBM. Honestly weak: 7.6 % dev / 33.9 % unknown-attack / 58.7 % In-the-Wild EER. Comparison row only. |
| 3 | **4 model architectures, coded + unit-tested** | ✅ code done | RawNet2, AASIST, AASIST-L, SSL-AASIST. Wired end to end. **None trained yet.** |
| 4 | **Augmentation** | ✅ code done | RawBoost + telephony codec. MUSAN / RIR not added. |
| 5 | **Detection signal in the product** | 🟡 partial | Published AASIST through our front-end: **35.8 % In-the-Wild EER** (baseline 58.7 %). Target < 15 %; uses upstream weights, not ours. |
| 6 | **Speaker-consistency branch** | ✅ built | `voiceguard.speaker` — WavLM x-vector, per-session enrolment, cosine → mismatch risk. No training. Same-speaker cosine ~0.95+; needs a clean 6 s enrolment. Operating point tuned, not calibrated. |
| 7 | **Risk fusion + policy engine** | ✅ built | `voiceguard.risk` — weighted fuse of spoof + speaker + context, lone-signal floor, 3 scenarios → ALLOW / VERIFY / ESCALATE with reason codes. Deterministic, unit-tested. Weights/thresholds are documented operating points. |
| 8 | **Feature-only audit log** | ✅ built | `voiceguard.audit` — JSONL of scores / action / reasons / provenance; **rejects** any audio / transcript / embedding key; retention + prune. This is the privacy story. |
| 9 | **Prosody / behaviour branch** | ✅ built | `voiceguard.prosody` — Praat F0/jitter/shimmer/HNR/pause features + a logistic regression (`scripts/train_prosody.py`, tracked 1.9 KB model). **26.3 % dev EER standalone** — weak by design, a contributing signal. Wired as the 4th fused signal. |
| 10 | **Session REST/WS API + operator console** | ✅ built | `voiceguard.serve` — `POST /api/session`, `.../enroll`, `.../context`, `.../analyze`, `.../acknowledge`, `WS .../stream`, `GET .../audit`; a `/` console with the decision banner, fused-risk timeline, four contributing-signal meters, reason codes, editable call context, a simulated **alert panel + acknowledge**, and the live audit trail (`ALLOW → ESCALATE → ACKNOWLEDGED`). Raw detector at `/detector`. `examples/rest_client.py` drives the whole flow headless. |
| 11 | **Windowed scoring + risk timeline** | 🟡 partial | Sliding 4 s window + EMA. No VAD, no latency budget, window slower than the < 2 s target. |

---

## What is not started

Every item below is an explicit SIH26104 deliverable with **zero code** today.

| # | Piece | Deliverable group | Rough effort |
|---|---|---|---|
| c | **A trained model** — one Kaggle run (AASIST or SSL-AASIST) to replace the upstream weights with ours; target < 15 % In-the-Wild EER | 1 — required | training days + wiring |
| d | **gRPC audio stream + Python SDK** (REST + `examples/rest_client.py` exist) | 5 — required | 2 days |
| e | **Indian-language evaluation** — pipeline built (`scripts/make_indic_spoof.py` + `eval_indic.py`, 5 languages of MMS-TTS attacks, per-language recall/EER, `docs/indic_eval.md`). **Remaining:** the consented real-speech slice — team records it | 5 — required | ~1 day (team recordings) |
| f | **Real calibration** — replace the logit-shift + tuned speaker/prosody/context operating points with calibration on held-out data | supports 2 | 0.5–1 day |
| g | **Real alert delivery** — the console panel is simulated; wire an actual SMS/email/webhook sink | 3 — required | ~1 day |
| h | **min t-DCF** alongside EER; ASVspoof 2021 DF as a second eval set | supports evidence | 0.5 day |
| i | **Edge / on-device inference path**; VAD + sub-2 s Tier-1 latency | 2, 4 — required | 2–3 days |
| j | **Model card + release hardening** | 7 | 1 day |

---

## The blocker: no training compute

Local training on the RTX 4050 is **~50–57 min per epoch**; a real run is 30–40 epochs
= days per model, and the machine can only do one job at a time. A 2-epoch test run
proved the training code is correct (dev EER 34 % → 27 %) but that is as far as the
laptop goes.

**Nothing that improves the model can proceed until training moves off the laptop.**
Options: Kaggle (T4, 30 GPU-h/week, free), Colab, or a rented cloud GPU. Kaggle is
the recommended path — one notebook, dataset uploaded once.

---

## What's next, in order

The vertical slice — four signals + fusion + policy + audit + alert + console + REST —
is built and tested (68 tests). Remaining, highest value first:

1. **Rehearse the demo** with real audio — [`docs/demo_runbook.md`](demo_runbook.md).
   Pick and pre-check the spoof clip on `/detector`; time the run.
2. **One Kaggle training run** (c) — running in parallel; swap the weights in only if
   `scripts/eval_cm.py` beats 35.8 % In-the-Wild.
3. **Real calibration** (f) once there's a stable model to calibrate against.
4. **A couple more scenario recordings** and the architecture slide for the cut items.

**Still cut for the hackathon** (show the design, don't build): gRPC + SDK, full
IndicCall-Eval (do a 4-language team-voice pilot instead), real alert delivery,
MUSAN/RIR, ablation grid, ASVspoof 2021 DF eval, edge inference, model card.

---

## Definition of done (hackathon demo)

From the review: runs a scripted live call, stable risk timeline within 2 s,
separates a benign call from a consented synthetic/replay scenario, shows the
speaker-consistency and prosody contributions, recommends verification before a
mock transfer, writes a feature-only audit event, and exposes the same session
over REST — all with measured metrics and stated limits.

**Met now (6 of 7 clauses):** benign vs synthetic/replay separation, speaker-consistency
**and prosody** contributions, verification recommendation before a mock transfer, an
alert the agent acknowledges, feature-only audit event, same session over REST.
**Not met:** stable read within 2 s (the window is 4 s); and "measured metrics" only
partly — spoof EER 33.9 %/35.8 % and prosody EER 26.3 % are measured, but the
speaker / context / fusion weights are operating points, not calibrated.
