# Project status — voiceguard / SIH26104

_Snapshot: 2026-09-25 (rev 7). Re-audited against the actual repo state — git
log, `results/tables/`, `models/`, test suite, lint — not carried forward by
assumption. Previous snapshot (2026-09-08, rev 2) was written for the
hackathon demo; the demo happened 2026-09-09. This one is written for the
stated goal beyond it: **industry-ready, not hackathon-ready.** Later
revisions, same push: item i (VAD + latency) moved from not-started to
partial — see [`docs/latency.md`](latency.md); item f (calibration) moved to
partial (1 of 3 signals) — see [`docs/calibration.md`](calibration.md); item
h (min-tDCF half) moved from not-started to done — see
[`docs/tdcf.md`](tdcf.md). Next day: item d (gRPC + SDK) moved from
not-started to done — see [`docs/grpc.md`](grpc.md); item j (model card
half) moved from not-started to done —
see [`models/pretrained/MODEL_CARD.md`](../models/pretrained/MODEL_CARD.md)._

## The gap in one paragraph

SIH26104 asks for a **voice-integrity security layer for live high-risk calls**:
synthetic-speech detection **plus** prosody/behaviour analysis **plus** cross-session
speaker consistency, fused into a **continuous, context-aware risk score**, wired to
**alerts and a pre-transaction recommendation**, under a **privacy/edge** model, over
**REST + gRPC** with **Indian-language** evidence. A working vertical slice of that
layer exists and is demoed: **four signals** (spoof + prosody + speaker + context) → one
ALLOW/VERIFY/ESCALATE decision with reason codes → a simulated alert the agent
acknowledges → a feature-only audit trail, all in an operator console and over REST.
A research paper and a presentation deck now exist too, and real calibration and
speech-onset VAD/latency safety are partly built (measured and shipped where they
generalize, honestly not shipped where they don't). The session API is now also
reachable over gRPC, verified to produce identical decisions to REST on the same
clip, and a model card documents what's actually measured (and what isn't). The
consented Indian-language real-speech set, real alert delivery, a sub-2 s confirmed
read, and a trained-by-us model are still designed but not built — see the table
below.

See [`docs/initial-review.md`](initial-review.md) for the full requirement breakdown,
[`docs/demo_runbook.md`](demo_runbook.md) for the demo script, and
[`paper/README.md`](../paper/README.md) for the paper/deck.

---

## What is actually done

| # | Piece | State | Note |
|---|---|---|---|
| 1 | **Honest evaluation harness** | ✅ solid | Anti-shortcut audio front-end (silence-leak fixed), manifest-driven splits, provenance stamped on every result row. Strongest asset — the numbers can be trusted. |
| 2 | **Generation-1 baseline** | ✅ done | Handcrafted + GBM. Honestly weak: 7.0 % dev / 33.4 % unknown-attack / 58.2 % In-the-Wild EER. Comparison row only. |
| 3 | **4 model architectures, coded + unit-tested** | ✅ code done | RawNet2, AASIST, AASIST-L, SSL-AASIST. Wired end to end. **None trained to convergence yet** — see blocker below. |
| 4 | **Augmentation** | ✅ code done | RawBoost + telephony codec. MUSAN / RIR not added. |
| 5 | **Detection signal in the product** | 🟡 partial | Published AASIST through our front-end: **35.8 % In-the-Wild EER** (baseline 58.2 %). Target < 15 %; uses upstream weights, not ours. |
| 6 | **Speaker-consistency branch** | ✅ built | `voiceguard.speaker` — WavLM x-vector, per-session enrolment, cosine → mismatch risk. No training. Same-speaker cosine ~0.95+; needs a clean 6 s enrolment. **Now calibrated** — see #17. |
| 7 | **Risk fusion + policy engine** | ✅ built | `voiceguard.risk` — weighted fuse of spoof + speaker + context, lone-signal floor, 3 scenarios → ALLOW / VERIFY / ESCALATE with reason codes. Deterministic, unit-tested. Fusion weights and policy thresholds remain documented operating points (not calibrated — no labeled session-level data exists to calibrate them against). |
| 8 | **Feature-only audit log** | ✅ built | `voiceguard.audit` — JSONL of scores / action / reasons / provenance; **rejects** any audio / transcript / embedding key; retention + prune. This is the privacy story. |
| 9 | **Prosody / behaviour branch** | ✅ built | `voiceguard.prosody` — Praat F0/jitter/shimmer/HNR/pause features + a logistic regression (`scripts/train_prosody.py`, tracked 1.9 KB model). **26.3 % dev EER standalone** — weak by design, a contributing signal. Wired as the 4th fused signal. Calibration attempted, not shipped — see #17. |
| 10 | **Session REST/WS API + operator console** | ✅ built | `voiceguard.serve` — `POST /api/session`, `.../enroll`, `.../context`, `.../analyze`, `.../acknowledge`, `WS .../stream`, `GET .../audit`; a `/` console with the decision banner, fused-risk timeline, four contributing-signal meters, reason codes, editable call context, a simulated **alert panel + acknowledge**, and the live audit trail (`ALLOW → ESCALATE → ACKNOWLEDGED`). Raw detector at `/detector`. `examples/rest_client.py` drives the whole flow headless. |
| 11 | **Windowed scoring + risk timeline** | 🟡 partial | Sliding 4 s window + EMA. Latency budget/VAD — see #16. Window itself is still slower than the < 2 s target. |
| 12 | **IndicCall-Eval pilot (synthetic side)** | ✅ built | `scripts/make_indic_spoof.py` + `eval_indic.py`, 5 languages of MMS-TTS attacks, per-language synthetic recall (85 % Marathi → 7.5 % English). No bona fide multilingual slice yet — recall only, not a true EER. |
| 13 | **Research paper (IEEE format)** | ✅ new — not committed | `paper/voiceguard_ieee.tex`, grounded entirely in this repo's own measured numbers, 13 verified citations; updated across two passes — a new Results subsection + table for min-tDCF and corrected Limitations/Future-Work text (both previously described VAD and min-tDCF as unbuilt), then a further pass correcting System Interfaces and Future Work to describe gRPC and the model card as shipped rather than future work. **`.docx`/`.pdf` are now stale against the `.tex`** — no LaTeX/pandoc is installed on this machine to rebuild them; recompile via Overleaf (`paper/README.md`) before sharing. Draft — needs real author names/affiliations and a double-blind pass before any submission. See `paper/README.md` for venue guidance. |
| 14 | **Presentation deck** | ✅ new — not committed | `paper/voiceguard_presentation.pptx` (+ `.pdf`), 14 slides, condensed from the paper. |
| 15 | **103 unit tests, ruff clean** | ✅ verified today | `pytest -q` → 103 passed. `ruff check .` → clean. |
| 16 | **Speech-onset VAD + safety-capped provisional reads** | 🟡 partial — new today | `voiceguard.audio.vad.speech_onset_s`, `Signal.provisional` / `FusedRisk.has_provisional`, `policy.decide()` caps ESCALATE→VERIFY on a still-building read. Reports `speech_onset_s`/`first_confirmed_read_s` in the API and on the console. Verified live: a 1.5 s clip that fuses to 0.84 returns VERIFY, not ESCALATE. Paper's Limitations/Future-Work bullets, which still described VAD as entirely unbuilt, corrected today (see #18's note — done in the same pass). **Not done:** a true sub-2 s *confirmed* acoustic read (needs a second short-window model — AASIST's input is fixed at 4.04 s) and edge/on-device inference. See [`docs/latency.md`](latency.md). |
| 17 | **Real calibration — measured on all three signals, shipped on one** | 🟡 partial — new today | `voiceguard.eval.calibration` (Platt/isotonic fitting + ECE/Brier, 9 tests) plus `scripts/calibrate_{cm,prosody,speaker}.py`. **Spoof and prosody:** fit cleanly on ASVspoof dev, then measurably *worsen* calibration on eval (unseen attacks) and, for spoof, on In-the-Wild too — a real, doubly-confirmed (Platt + isotonic) negative result, not shipped. **Speaker:** a Platt fit on real ASVspoof speaker identities cuts held-out (unseen-speaker) ECE from 0.39 to 0.18 — genuinely generalizes, shipped and active by default. Paper updated to match (worked-example benign-call numbers changed: speaker 0→44, fused 10→22; the escalate case is unchanged, ~100→99 rounding only). The presentation deck only ever showed the escalate-case numbers, which are unaffected — nothing to update there. See [`docs/calibration.md`](calibration.md). |
| 18 | **min-tDCF alongside EER** | ✅ done — new 2026-09-24 | `voiceguard.eval.tdcf` wraps the ASVspoof 2019 organizers' own reference implementation (vendored verbatim, `eval/_vendor/`) and their official baseline ASV system's scores, distributed with the LA corpus — not this project's own WavLM signal, so the number stays comparable to every other reported min-tDCF. `scripts/eval_tdcf.py --model aasist --pretrained` → dev **0.288**, eval **0.588** (`results/tables/cm_aasist-pretrained_tdcf.csv`); `eer_cm` cross-checked against the independently-computed 9.6 %/23.3 % dev/eval EER — matches. Same story as EER: usable on known attacks, collapses on unseen ones. 8 new tests, incl. one against real AASIST scores + real ASV scores end to end. Paper updated to match: new Results subsection + table, and the Limitations/Future-Work bullets that used to call this future work now describe what shipped. **Not done:** ASVspoof 2021 DF as a second held-out cross-dataset test — corpus needs a registration step, not present locally. See [`docs/tdcf.md`](tdcf.md). |
| 19 | **gRPC transport for the session API** | ✅ done — new today | `voiceguard.serve.grpc_service` (`VoiceGuardServicer`) reuses the exact same `SessionManager` REST/WS uses — both now call a shared `build_session_stack()` (refactored out of `app.py` into `session.py` for this). `voiceguard.proto` mirrors the REST session routes 1:1, incl. the bidi-streaming `StreamAudio` RPC matching `WS .../stream`. `scripts/serve_grpc.py` + `examples/grpc_client.py` (the "Python SDK" deliverable — a thin, direct usage example, same bar `rest_client.py` set for REST, not a separately versioned package). 5 new tests, incl. one that runs the *same clip* through both REST and gRPC and asserts identical `action`/`fused` — not just "both work," but "both are the same engine." Caught and fixed a real proto3 default-value bug pre-ship: `CreateSessionRequest.caller_known` needed `optional bool`, not plain `bool` (see `docs/grpc.md`). **Not done:** gRPC parity for the raw non-session detector path (`/api/score`, `/api/stream`) — out of scope, that path is the demo's "it's real-time" proof, not a product surface. Paper's System Interfaces and Future Work sections updated to match. See [`docs/grpc.md`](grpc.md). |
| 20 | **Model card** | ✅ done — new today | [`models/pretrained/MODEL_CARD.md`](../models/pretrained/MODEL_CARD.md), Mitchell et al. format (model details / intended use / factors / metrics / evaluation & training data / quantitative analyses / ethical considerations / caveats), every number sourced from a `results/tables/` CSV already in this repo — no new claims, just organized ones. Surfaced a real gap while writing it: **this repository has no top-level LICENSE file** (the vendored AASIST/RawBoost code has its own upstream terms; this project's own code declares none). Not fixed here — picking a license has real IP implications for an SIH submission and isn't this pass's call to make unilaterally. "Release hardening" (the other half of item j) was scoped down to just the model card — auth/TLS/rate-limiting for REST and gRPC were treated as a separate, not-yet-scoped item, not silently bundled in. |

---

## What is not started

Every item below is an explicit SIH26104 deliverable with **zero code** today.
Unchanged since the last snapshot — none of this has moved in the last two weeks.

| # | Piece | Deliverable group | Rough effort | Blocked on |
|---|---|---|---|---|
| c | **A trained model** — one Kaggle run to replace the upstream weights with ours; target < 15 % In-the-Wild EER. **Recommendation: AASIST first, not SSL-AASIST** — see [`docs/training_runbook.md`](training_runbook.md) for the full setup + reasoning (SSL-AASIST lacks a feature cache today, so training it now would waste weekly GPU quota), or upload [`notebooks/kaggle_train_cm.ipynb`](../notebooks/kaggle_train_cm.ipynb) directly and run it top to bottom | 1 — required | training days + wiring | **you** — needs the Kaggle run started/finished; only the old 2-epoch local sanity checkpoint exists (`models/cm/aasist/`, dev EER 34→27 %, proves the loop works, not a real result) |
| d | ~~gRPC audio stream + Python SDK~~ **done 2026-09-25** — `voiceguard.serve.grpc_service` + `examples/grpc_client.py`, verified to agree with REST on the same clip (see `docs/grpc.md`) | 5 — required | done | — |
| e | **Indian-language evaluation, real-speech slice** — synthetic side is done; need 3–5 consented speakers/language reading a few lines each | 5 — required | ~1 day of team recording | **you/team** — needs actual people recording |
| f | ~~Real calibration~~ **done 2026-09-24, 1 of 3 signals** — speaker is calibrated and shipped; spoof and prosody were measured and found not to generalize, so neither ships (see `docs/calibration.md`). Remaining: fusion weights and policy thresholds are still uncalibrated operating points (no labeled session-level data exists yet to calibrate them against) | supports 2 | done for what's feasible now | nothing left that's unblocked — the remainder needs labeled data this project doesn't have |
| g | **Real alert delivery** — the console panel is simulated; wire an actual SMS/email/webhook sink | 3 — required | ~1 day | **you** — needs a provider account (Twilio/SMTP/etc.) and credentials |
| h | ~~min t-DCF alongside EER~~ **done 2026-09-24** — dev 0.288 / eval 0.588, using the ASVspoof 2019 organizers' reference implementation + baseline ASV scores (see `docs/tdcf.md`). Remaining: **ASVspoof 2021 DF** as a second held-out cross-dataset eval set | supports evidence | 0.5 day for the rest | **you** — DF corpus needs a registration step with the organizers before download |
| i | ~~VAD + safety-capped provisional reads~~ **done 2026-09-24** — remaining: **edge/on-device inference**; a true sub-2 s *confirmed* acoustic read (needs a second short-window model) | 2, 4 — required | 2–3 days for the rest | nothing — pure code, I can start |
| j | ~~Model card~~ **done 2026-09-25** (`models/pretrained/MODEL_CARD.md`, see `docs/project_status.md` #20). Remaining, narrower than "release hardening" implied: **pick a LICENSE** (blocks anything downstream that depends on the project's own IP terms, not just this repo's polish); auth/TLS/rate-limiting for the REST and gRPC servers is a separate, larger item, not scoped here | 7 | license: minutes, once decided; hardening: unscoped | **you** — license choice; hardening needs a decision on whether it's in scope at all before effort is estimated |

---

## The blocker: no training compute

Local training on the RTX 4050 is **~50–57 min per epoch**; a real run is 30–40 epochs
= days per model, and the machine can only do one job at a time. A 2-epoch test run
(2026-09-05) proved the training code is correct (dev EER 34 % → 27 %) but that is as
far as the laptop goes, and nothing further has been trained locally since.

**Nothing that improves the model (item c) can proceed until a Kaggle run happens.**
Everything else in the "not started" table with "nothing — I can start" in the
Blocked-on column does **not** depend on this and can be built in parallel.

---

## What's next, in priority order (industry-readiness lens)

The hackathon demo's definition of done is met (see below, unchanged). Past that,
in order of what most changes whether this is a real product:

1. **Item c — the trained model.** Everything else is downstream of having a
   detector that's actually ours. Highest leverage, but it's on you to kick off
   (or hand me Kaggle access / a checkpoint to wire in).
2. ~~Item f — real calibration.~~ **Done 2026-09-24, honestly** (1 of 3 signals
   shipped, 2 measured and correctly *not* shipped — see `docs/calibration.md`).
   Worth revisiting once item c lands: a trained-by-us model has a different
   raw-score distribution, so spoof calibration is worth re-measuring against
   it rather than assumed to fail the same way.
3. ~~Item i — VAD + latency.~~ **Half done 2026-09-24** (speech-onset VAD +
   safety-capped provisional reads, see `docs/latency.md`); the remaining half
   (a genuinely sub-2 s *confirmed* read, and edge/on-device inference) needs
   a second short-window model — bigger scope.
4. ~~Item h — min t-DCF.~~ **Done 2026-09-24** (dev 0.288 / eval 0.588, see
   `docs/tdcf.md`) — the metric the ASVspoof community actually trusts over
   raw EER, now reported alongside it. Remaining half, **ASVspoof 2021 DF**
   as a second held-out cross-dataset test, is blocked on a registration
   step with the organizers (**you**).
5. ~~Item d — gRPC + SDK.~~ **Done 2026-09-25** (`docs/grpc.md`) — the same
   `SessionManager` REST/WS already used, reached a second way; verified
   against REST on the same clip, not just "it responds."
6. ~~Item j — model card.~~ **Done 2026-09-25** in the narrow sense (the
   card itself, `models/pretrained/MODEL_CARD.md`). Surfaced that this repo
   has no LICENSE file yet — **your call**, not mine to pick.
7. **Item e / g — need you.** Team recordings and a real alert-provider account
   are the two items no amount of my time moves without your input.
8. **Item d's raw-detector half, if it matters to you** — gRPC parity for
   `/api/score`/`/api/stream` (the non-session, "it's real-time" demo path)
   was deliberately left out as out of scope for the actual deliverable;
   say if you want it too.

---

## Definition of done (hackathon demo) — met, historical record

From the original review: runs a scripted live call, stable risk timeline within 2 s,
separates a benign call from a consented synthetic/replay scenario, shows the
speaker-consistency and prosody contributions, recommends verification before a
mock transfer, writes a feature-only audit event, and exposes the same session
over REST — all with measured metrics and stated limits.

**Met (6 of 7 clauses):** benign vs synthetic/replay separation, speaker-consistency
**and prosody** contributions, verification recommendation before a mock transfer, an
alert the agent acknowledges, feature-only audit event, same session over REST.
**Not met:** stable read within 2 s (the window is 4 s; item i's speech-onset VAD and
provisional-read safety cap help the honesty of what's reported, not the 4 s window
itself); and "measured metrics" only partly — spoof EER 23.3 %/35.8 % and prosody EER
26.3 % are measured, and as of item f the speaker signal now has a real, held-out-
verified calibration too, but fusion weights and policy thresholds remain uncalibrated
operating points (no labeled session-level data exists to calibrate them against).

---

## Loose end: uncommitted work

Everything from this session and the next day's follow-up — the paper/deck, items
i, f, h, d and j, this doc's own updates, and a new Kaggle runbook for item c
(`docs/training_runbook.md` + `notebooks/kaggle_train_cm.ipynb`, guidance and a
runnable notebook only — no training has actually run) — is
on disk but **not yet committed**. Committing from an assistant session adds a
Claude co-author trailer, which was deliberately stripped from this repo's
history before, so this is left for you to run directly. Six logical commits,
in the order the work happened:

```powershell
git add paper/ examples/rest_client.py
git commit -m "paper: IEEE draft + presentation deck; rest_client: ruff line-length fix"

git add src/voiceguard/audio/vad.py src/voiceguard/risk/signals.py src/voiceguard/risk/fusion.py src/voiceguard/risk/policy.py src/voiceguard/audio/__init__.py src/voiceguard/serve/static/console.html tests/test_vad.py tests/test_risk.py tests/test_session.py docs/latency.md
git commit -m "P2: speech-onset VAD + safety-capped provisional reads"

git add src/voiceguard/eval/calibration.py src/voiceguard/detect/scorer.py src/voiceguard/prosody/scorer.py src/voiceguard/speaker/verify.py scripts/calibrate_cm.py scripts/calibrate_prosody.py scripts/calibrate_speaker.py tests/test_calibration.py tests/test_detect.py tests/test_prosody.py tests/test_speaker.py results/tables/*_calibration.csv models/speaker/ .gitignore docs/calibration.md
git commit -m "P3: real calibration -- measured on all three signals, shipped on one (speaker)"

git add src/voiceguard/eval/__init__.py src/voiceguard/eval/tdcf.py src/voiceguard/eval/_vendor/ tests/test_tdcf.py scripts/eval_tdcf.py results/tables/cm_aasist-pretrained_tdcf.csv docs/tdcf.md
git commit -m "P4: min-tDCF alongside EER, via the ASVspoof 2019 organizers' reference impl"

git add pyproject.toml src/voiceguard/serve/session.py src/voiceguard/serve/app.py src/voiceguard/serve/__init__.py src/voiceguard/serve/grpc_service.py src/voiceguard/serve/proto/ scripts/serve_grpc.py examples/grpc_client.py tests/test_grpc.py docs/grpc.md models/pretrained/MODEL_CARD.md
git commit -m "P5: gRPC transport for the session API (same SessionManager as REST/WS) + model card"

git add README.md docs/project_status.md docs/training_runbook.md notebooks/kaggle_train_cm.ipynb notebooks/README.md GETTING_STARTED.md
git commit -m "status: rev 7 -- calibration + latency + min-tDCF + gRPC + model card; README synced; Kaggle training runbook + notebook; a layman's getting-started guide"

git push
```

**Why some files moved commits from earlier revisions of this list, and why
a couple of "P2"/"P4" commits will carry a stray line or two from later
work:** several files are touched by more than one of today's/yesterday's
features (`session.py` and `app.py` by both VAD and the gRPC refactor;
`pyproject.toml` by both the tdcf ruff-exclude and the gRPC deps;
`eval/__init__.py` by both calibration's and tdcf's exports; `README.md` by
all of them). `git add` stages a file's **current, full** on-disk content,
not a diff against "when the feature was conceptually done" -- so a shared
file can only be cleanly split across commits with `git add -p`, which
isn't used here to keep this runnable as a straight copy-paste. Instead,
each shared file above is placed in the **last** commit that touches it
(e.g. `session.py`/`app.py` moved from P2 to P5; `README.md` moved out of
P4/P5 into the final commit). The practical effect: the commit whose
message describes a file's *original* feature may not literally contain
every line of that feature for shared files, and `eval/__init__.py`
specifically is import-broken if P3 or P4 is checked out in isolation
without the other (P3 needs its own calibration exports from
`eval/__init__.py`'s current content, which also already carries P4's tdcf
exports). None of this matters for the tree you end up with -- all six
commits land before you push -- only for `git bisect` on an old commit,
which this project hasn't needed so far.

paper/ already carries the calibration-, min-tDCF/VAD- and gRPC/model-card-
driven edits (made in place, incl. a new Results subsection + table for
min-tDCF and corrected Limitations/Future-Work text) -- **PDF/DOCX are now
stale against the `.tex`** and need a recompile (e.g. via Overleaf per
`paper/README.md`); no LaTeX/pandoc is installed on this machine to do it
here.
