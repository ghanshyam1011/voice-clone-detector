# VoiceGuard: viva notes

What to say, in what order, and how to answer the hard questions. Written in plain English so you can say it
aloud. Every number is from this repository (`results/tables/`, `docs/`); the file that proves it is named next
to it, so if an examiner asks "where does that come from?" you can open it.

**How to use this.** Read sections 1-3 until you can say them without looking (about 20 minutes). Skim 4-9
for the detail behind each claim. Rehearse the Q&A in section 13 out loud. Print section 15 (the numbers) and
keep it in front of you.

**Contents**

1. The pitch: 30 seconds, 2 minutes, 5 minutes
2. The problem, in plain words
3. What we claim: the thesis and the contributions (and what we do *not* claim)
4. The system, end to end (one call, then each part)
5. How we evaluate, and why you can trust the numbers
6. The results, as a story
7. The lesson from real audio (the WhatsApp voice note)
8. Limitations, said candidly
9. What is next
10. Engineering quality, privacy and ethics
11. What we built versus what we used
12. Mistakes we found and fixed (examiners like these)
13. The question bank (50 questions with model answers)
14. Glossary
15. The numbers on one page
16. The demo script and what to do if it breaks
17. What not to say
18. Checklist for the night before

---

## 1. The pitch

### 30 seconds

"Free AI tools can now copy anyone's voice from a few seconds of audio, and fraudsters use that to phone banks
and pretend to be a customer. We built **VoiceGuard**, a decision layer that listens to a live call and tells the
agent, in plain words, whether to trust the caller: *real customer*, *double-check*, or *stop*. It never bets on
one detector. It combines a voice-cloning detector, a match against the customer's registered voice, the
recording quality, and the call details such as an unknown number and the amount, and a person makes the final
decision. Our main contribution is not a new neural network; it is an **honest way of measuring**, a **system
around the detector**, and a clear record of **where it fails**."

### 2 minutes

Say the 30 seconds, then:

"Three things make this different. **One: we do not trust benchmark numbers.** The standard dataset, ASVspoof 2019,
has a hidden shortcut: a model can score well by listening to the silent parts of the clips instead of the voice.
Our first classifier scored about 14 % error on *silence alone*, where guessing would be 50 %. We built a fair
audio front-end and a permanent silence test. After that, a published detector that reports about 1 % error in
the lab shows **9.6 % on known attacks, 23.3 % on unseen attacks and 35.8 % on real-world audio**. **Two: because
no single detector generalises, we fuse four kinds of evidence**, and the fused system is right about 90 % of the
time on speakers it never saw, across 19 awkward situations like rooms, echo, clipping and phone codecs.
**Three: we publish where it fails.** Against modern open-source cloning tools the detector alone sees only 2-3 %
of them; the voice match does most of the catching, and about a quarter of the best clean-line clones still get
through. When a friend's real WhatsApp voice note was flagged as fake, we found a shortcut inside our own model,
fixed it, and reported what the fix cost."

### 5 minutes (use this as your opening if they say "tell us about your project")

1. **The problem** (30 s): voice cloning fraud; why voice can no longer prove identity.
2. **The gap** (45 s): published detectors look near-perfect in the lab; real audio is different. Figure: ≈1 % →
   36 %. Müller et al. report up to 1000 % relative degradation on real-world audio; we reproduced it.
3. **Our answer** (60 s): four signals, one decision, a human in the loop. Walk the architecture picture left to
   right: input gate → fair front-end → detector, voice match, recording style, call details → fusion → policy →
   ALLOW / VERIFY / ESCALATE with reasons.
4. **How we know it works** (60 s): the silence test, frozen protocol, provenance on every number, In-the-Wild
   never used for training or thresholds; the 19-situation stress test with situations held out of fitting.
5. **Where it fails** (45 s): modern clones, real-world audio, simulated (not recorded) situations, alerts are
   simulated, no authentication yet.
6. **What we would do next** (20 s): train the detector on modern cloning tools and real-world codec audio; collect
   consented real calls and Indian-language speech.
7. **Demo** (offer it): three calls, real customer, fake voice, the honest hard case.

---

## 2. The problem, in plain words

**Voice cloning.** Modern text-to-speech (TTS) and voice-conversion (VC) systems can imitate a person from a few
seconds of their speech, taken from a social-media clip or an earlier call. A fraudster types a script or speaks
and the system outputs it in the victim's voice.

**Why it matters for banks.** Call centres and banks use voice as a soft proof of identity: confirming a transfer,
adding a payee, resetting a password. If a cloned voice can pass, the agent approves a payment they should not.

**What SIH26104 asks for** (paraphrased; check the official wording before quoting): a real-time system that
(1) detects cloned or synthetic speech, (2) checks the claimed speaker is consistent, (3) measures behavioural
anomalies, (4) combines them into a continuous, context-aware risk score, (5) recommends an action before a
transaction, (6) respects privacy (minimal retention, feature-only logs), and (7) works across Indian languages,
through REST and gRPC.

**Why detection is hard.**
- A detector learns the *artefacts* of the generators it was trained on. A new generator has different
  artefacts, and the detector fails silently.
- Real calls are noisy, compressed, echoey and clipped, none of which the clean benchmark audio is.
- The benchmark itself can have shortcuts (we found one).

**Two kinds of mistake, with different costs.** A *false accept* (a clone passes) means fraud. A *false alarm* (a
real customer is challenged) means friction and lost trust. A good system makes both visible and lets the
operator choose the balance per situation (our "scenarios").

---

## 3. What we claim

### The thesis (say it exactly)

> **No single acoustic detector generalises to unseen speech synthesis, so a product should fuse several partly
> independent signals, say how sure it is, and keep a human in the loop.**

### The four contributions

1. **An evaluation we can trust.** We showed the standard benchmark can be passed with a shortcut (silence
   duration and loudness), built an anti-shortcut front-end that treats real and fake identically, stamped every
   result with a fingerprint and git commit, and added a permanent silence-ablation gate. Cross-dataset
   (In-the-Wild) audio is never trained on or used to choose a threshold.
2. **An honest re-measurement of a published detector.** AASIST, reported at about 1 % error by its authors on
   its own benchmark: 9.6 % known attacks, 23.3 % unseen attacks, 35.8 % In-the-Wild through our front-end;
   min-tDCF 0.288 / 0.588.
3. **A decision system around the detector.** Four signals, a learned fusion, a policy with three scenarios,
   reason codes, a feature-only audit log, a console, and one engine over REST, WebSocket and gRPC.
4. **A measured account of robustness and failure.** A 19-situation stress evaluation with situations held out of
   fitting, a detector fine-tuned for noise and rooms, modern-clone tests, an input-quality gate, and a real-world
   failure (a WhatsApp voice note) traced to its cause and fixed, with the cost reported.

### What we do **not** claim (volunteer these; it builds credibility)

- We did **not** invent a new anti-spoofing architecture. The detector is the published AASIST, fine-tuned.
- We are **not the first** to notice a silence shortcut in ASVspoof. It is documented in the literature
  (Müller et al., 2021, *"Speech is Silver, Silence is Golden: What do ASVspoof-trained Models Really Learn?"*;
  check the exact citation before quoting it). Our contribution is making it a **permanent, reproducible gate**
  and showing its effect on our pipeline. *(The paper draft does not cite this yet; add it.)*
- We do **not** claim to detect every deepfake, to be "DPDP compliant", to support "all Indian languages", or to
  be deployable as is.
- We did **not** train a detector from scratch (laptop GPU: about 50 minutes per epoch; a real run needs days).

---

## 4. The system, end to end

### 4.1 One call, as the system sees it

1. **t = 0 s.** The call connects. The customer's voice is already on file (enrolled earlier from at least 3 s of
   clear speech; about 6 s is best). The operator picks the *kind* of call: routine, payment, or sensitive change.
2. **0-1 s of speech.** The input gate has too little speech to judge. The answer is **WAITING**: not "allow", not an
   alarm. (Call context still counts: an unknown number asking for ₹75,000 is raised even before anyone has spoken.)
3. **About 1-2 s of speech.** The first reading appears, marked **provisional**: it can raise a *double-check* but
   can never say *stop* by itself.
4. **About 4 s of speech.** The detector has a full 4.04 s window; the first **confirmed** reading.
5. **Every 0.5 s after that,** the system re-scores the latest audio (a rolling 12 s buffer), blends each new
   reading with the previous ones (smoothing), and updates the answer and the reasons.
6. **The answer** is one of ALLOW, VERIFY ("ask a knowledge question"), ESCALATE ("pause approval, verify through the
   registered number"), with reason codes in plain English.
7. **The agent acts** and presses "I've seen this"; the audit log records the sequence, for example
   `ALLOW → ESCALATE → ACKNOWLEDGED`, with scores and reasons, **never audio**.

### 4.2 The pipeline, stage by stage

```
live audio → [input gate] → [fair front-end] → ┬ voice detector (AASIST)         ┐
                                               ├ voice match (WavLM voiceprint)   ├→ [fusion] → [policy] → answer
                                               ├ recording style (descriptors)    │             (+ reasons)
                                               └ call details (rules)            ┘
                                                                                       ↓
                                                             console · REST · WebSocket · gRPC · audit log
```

#### Input gate (`src/voiceguard/audio/quality.py`)

*What it does.* Before any model sees the audio, it repairs what can be repaired and refuses to guess when there is
nothing to judge: NaN or Inf samples are replaced; audio scaled in the wrong range is rescaled; a DC offset is
removed; very quiet speech is raised; speech is found with **Silero VAD** (a small neural voice activity detector,
batched for speed). Under 1 s of speech → **WAITING**. 1-2 s → provisional read. 2 s and a full window → normal.
Enrolment is refused below 3 s of speech.

*Why it exists.* We audited the first version with awkward audio and found it **crashed** on an empty array or NaN, said
**ALLOW** to a single sample, and said **ESCALATE** to a muted line, white noise or hold music. A product receives all of
those. After the gate every one of them returns WAITING or a normal read (`docs/input_quality.md`).

#### Fair front-end (`src/voiceguard/audio/preprocess.py`, frozen by team agreement)

Mono → 16 kHz → trim leading and trailing silence and re-pad a fixed margin → loudness-normalise (EBU R128,
−23 LUFS) → peak-limit (−1 dBFS). **Identical for real and fake audio.** Its settings hash to a 10-character
fingerprint (`640b0644ae`) stamped on every result, so numbers made with different front-ends can never share a
table. This is what closed the silence shortcut.

#### Voice detector (`src/voiceguard/detect/scorer.py`, `models/neural.py`, vendored AASIST)

*What AASIST is, in one breath.* A neural network that takes the **raw waveform** (64,600 samples = 4.04 s at
16 kHz), runs it through a learnable filter bank, then uses **graph attention** over spectral and temporal nodes
to find the localised artefacts that synthesis leaves behind, and outputs a real-versus-synthetic score. It has
about 0.3 million parameters (Jung et al., ICASSP 2022).

*What we did to it.* We did not train it from scratch. We **fine-tuned the published weights** on ASVspoof 2019 LA
**train only**, applying degradations to *both* classes identically (so "is it noisy?" can never be a shortcut):
background voices, broadband noise, telephone codecs, RawBoost, then room reverberation, speakerphone, narrow band,
clipping, packet loss and hum. Music and burst noise were deliberately **never** trained on, to stay a true
"unseen situation" check. The checkpoint was chosen on dev EER with a clean-audio budget.

*How it is read.* The latest 10 s are cut into overlapping 4 s windows (stride 1 s), each scored and smoothed, so
one noisy window cannot decide the read. The score is a number between 0 and 1 (higher = more synthetic) with a
calibration shift anchored at the dev equal-error point (0.2387 for the shipped weights).

#### Voice match (`src/voiceguard/speaker/`)

*What it does.* The customer's registered voice becomes an **embedding** (a vector, a "voiceprint") from
`microsoft/wavlm-base-plus-sv`, a WavLM model with an **x-vector** head fine-tuned for speaker verification. Each
later 6 s window of the call is embedded and compared with the voiceprint by **cosine similarity** (1 = same
direction). Same speaker: about 0.95 or above on clean audio. A different person: about 0.4-0.7. The similarity
becomes a *mismatch risk* through a **Platt calibrator** that we fitted on real ASVspoof speaker identities and
verified on held-out speakers (calibration error 0.389 → 0.181).

*Why it matters most.* It is the one check that does not depend on knowing the cloning tool. A clone has to sound
like the customer closely enough to pass this too.

*Privacy.* The voiceprint lives in memory for the session only and is dropped when the session ends.

#### Prosody (`src/voiceguard/prosody/`), built, shown, not a decision input

Praat features (F0 statistics, jitter, shimmer, harmonics-to-noise ratio, pause ratio, energy flux) with a logistic
regression: **26.3 % dev EER on its own**, and about **46 % (chance) on the system trials**. We kept the code and the
optional meter but removed it from the decision, because adding a signal at chance only adds noise. Say this: it
shows we *measure and drop* what does not help.

#### Recording style (`src/voiceguard/audio/channel.py`)

Ten cheap descriptors of *how the recording sounds*, eight of which the decision uses: dynamic range
(`dyn_db`), noise floor relative to speech (`floor_rel_db`), high-frequency share (`hf_ratio`), spectral flatness
(`flatness`), loudness-envelope autocorrelation (`env_acf`, high with echo, low with steady noise), crest factor
(`crest_db`) and share of samples at the peak (`flat_top_db`, clipping), and mains-hum prominence (`hum_db`). The
absolute level is **not allowed** as an input, because it moves with microphone gain and would make the same call
"noisy" or "clean" at will.

*Their job, stated carefully:* they tell the system **how far to trust the detector on this line** (a noisy room
raises genuine callers' detector scores; a clipped microphone makes a real voice read as synthetic). They are
**not** evidence of cloning by themselves; that distinction is the lesson in section 7.

#### Call details (`src/voiceguard/risk/context.py`)

A small, explicit rules table: unknown number (+0.35), transfer amount (₹5,000 +0.15, ₹25,000 +0.30,
₹100,000 +0.45), earlier flags on this caller (+0.20 each, capped), an unrecognised VoIP call (+0.10). Context
**never decides alone**; it turns "the voice sounds a bit off" into "pause this ₹75,000 transfer".

#### Fusion (`src/voiceguard/risk/learned.py`, parameters in `models/fusion/fusion.json`)

Two small heads, fitted by `scripts/fit_fusion.py`, then combined as **independent alarms**:

- **Clone head** ("is this audio synthetic?"): `p_clone = sigmoid(offset + gain(recording style) × logit(detector score))`.
  The recording style sets the *gain* (how much the detector is believed on this line). The gain is positive by
  construction, so the probability always rises with the detector's score, and **recording style can never raise
  an alarm on its own**.
- **Impostor head** ("is this a different person?"): a logistic model on the voice-match mismatch risk and the
  recording style.
- **Combination:** `p_acoustic = 1 − (1 − p_clone)(1 − p_impostor)`; call context is a third independent alarm
  (weighted 0.8). Either alarm raises the risk; neither dilutes the other.
- **Operating points** (`t_warn`, `t_esc`) are chosen on the training speakers only: warn = best balanced accuracy,
  escalate = the lowest score that keeps genuine ESCALATE under 3 %. They are mapped onto the policy's anchors so
  the console's meter keeps its meaning.

*Why learned and not hand-set?* A hand-set weighted mean treats a 0.3 detector score the same on a quiet line and
in a noisy room. Fitting the two heads on trials replayed under many situations fixed that (balanced accuracy
83.9 % → 90.2 % on unseen speakers).

#### Policy (`src/voiceguard/risk/policy.py`)

Three scenarios, each with a (warn, escalate) threshold on the 0-1 risk scale:

| Scenario | Warn → VERIFY | Escalate → ESCALATE | Meaning |
|---|--:|--:|---|
| routine | 0.55 | 0.75 | ordinary support call |
| transaction | 0.40 | 0.65 | a payment is on the table |
| privileged | 0.30 | 0.55 | password reset, new payee, higher limit |

Higher stakes need less evidence. Two safety rules: a **provisional** reading (little speech, or a window not yet
full) is capped at **VERIFY**, never ESCALATE; and **WAITING** is a fourth state, "nothing to judge", never ALLOW and
never an alarm. The output is a **recommendation for a human**, never an automatic block.

#### Smoothing and streaming (`src/voiceguard/serve/session.py`)

Each signal has an exponential moving average (new = (1 − a) × reading + a × previous; a = 0.5 for the detector,
0.45 for the voice match) and the fused score has one too (a = 0.6). The live path keeps a **rolling 12 s buffer**,
re-scores every **0.5 s**, and always scores the *latest* audio (if scoring is slow, old audio is skipped, so the
verdict never lags further behind). The server reports `heard_s` (how much of the call the answer covers).

#### Audit log (`src/voiceguard/audit.py`)

A JSON-lines file of: time, session, scenario, scores, action, reasons, model version, front-end fingerprint, git
commit. It **rejects at write time** any key that could carry audio, a transcript or an embedding, enforced in code,
not just policy. It has a retention horizon and a prune.

#### Interfaces

- **REST** (`POST /api/session`, `/enroll`, `/context`, `/analyze`, `/acknowledge`, `GET /audit`) and **WebSocket**
  (`/stream`) in `serve/app.py` (FastAPI).
- **gRPC** (`serve/grpc_service.py`, `proto/voiceguard.proto`), including a bidirectional streaming RPC.
- All of them drive **the same** `SessionManager`, built once in `build_session_stack()`. A test sends the identical
  clip through REST and gRPC and asserts the **same** action and score.
- **Operator console** (`serve/static/console.html`): a single page written for people who have never heard of
  voice cloning: one answer in plain words, the four checks behind it, what to do now, a scorecard, and the
  technical detail folded away.

### 4.3 Where to find each thing in the code (so you can open it live)

| Topic | File |
|---|---|
| Fair front-end | `src/voiceguard/audio/preprocess.py` |
| Input gate, WAITING | `src/voiceguard/audio/quality.py` |
| Degradations (the "situations") | `src/voiceguard/audio/degrade.py` |
| Recording-style descriptors | `src/voiceguard/audio/channel.py` |
| Detector wrapper | `src/voiceguard/detect/scorer.py`, `models/neural.py` |
| Voice match | `src/voiceguard/speaker/embed.py`, `verify.py` |
| Fusion and heads | `src/voiceguard/risk/learned.py`, `scripts/fit_fusion.py` |
| Policy, scenarios, context | `src/voiceguard/risk/policy.py`, `context.py` |
| Session engine, streaming | `src/voiceguard/serve/session.py` |
| REST / WebSocket / console | `src/voiceguard/serve/app.py`, `static/console.html` |
| gRPC | `src/voiceguard/serve/grpc_service.py` |
| Audit log | `src/voiceguard/audit.py` |
| Metrics, trials, calibration | `src/voiceguard/eval/` |
| Detector fine-tuning | `scripts/finetune_robust.py` |
| Stress evaluation | `scripts/build_system_trials.py`, `eval_stress.py` |

---

## 5. How we evaluate, and why you can trust the numbers

### 5.1 The data

| Dataset | What it is | Size (our manifests) | Our use |
|---|---|---|---|
| **ASVspoof 2019 LA train** | English studio speech (VCTK) with 6 spoofing systems | 25,380 clips (2,580 real, 22,800 fake) | training only |
| **ASVspoof 2019 LA dev** | same, **known** attacks A01-A06 | 24,844 clips | choosing checkpoints and thresholds |
| **ASVspoof 2019 LA eval** | **unseen** attacks A07-A19, 67 speakers | 71,237 clips (7,355 real, 63,882 fake) | in-domain test |
| **In-the-Wild** | found audio of public figures (Müller et al.) | 31,779 clips (19,963 real, 11,816 fake) | **cross-dataset test only; never trained on, never used to pick a threshold** |
| Modern clones (made here) | kNN-VC, SpeechT5 + x-vector, MMS-TTS | 648 clips | testing |
| IndicCall-Eval (pilot) | MMS-TTS attacks in 5 languages | 40 per language | per-language probe |

Splits come only from the **official protocol files**; a home-made random split leaks speakers and attack types.
The classes are about 9 : 1 fake to real, so we never headline accuracy.

### 5.2 The metrics (be able to define each)

- **EER (equal error rate).** The error rate at the threshold where *false accepts* (a fake passes) equal *false
  rejects* (a real call is blocked). Lower is better; independent of any threshold; the field's standard.
- **min t-DCF.** The ASVspoof challenges' own deployment metric: the countermeasure scored **together with a fixed
  speaker-verification system** under an explicit cost model (letting a spoof through costs more). 0 = perfect,
  1 = no better than no countermeasure. We used the organisers' own reference code and their baseline ASV scores
  so it is comparable with everyone else's.
- **Balanced accuracy.** The mean of "genuine callers accepted" and "attacks flagged". In our system trials six of
  eight calls are attacks, so a system that flags everything scores 75 % plain accuracy; balanced accuracy removes
  that trap.
- **ECE (expected calibration error), Brier score.** Do predicted probabilities match reality? A prediction of 0.9
  should be right about 90 % of the time.
- **Per-attack EER.** Each generator against all real audio, so a weak generator cannot hide inside an average.

### 5.3 The protocol rules (`docs/evaluation_protocol.md`, frozen)

1. Official splits only.
2. The fair front-end is mandatory and identical for both classes; its fingerprint is stamped on every result.
3. **Silence-ablation gate, permanent.** After any model change, score the *silent parts only*. Guessing is 50 %;
   within about 5 points of 50 % passes; **below 40 % means a shortcut, stop**.
4. Every result row records: train set, test set, model, seed, front-end fingerprint, score direction, git commit,
   timestamp.
5. At least one generator is always held out. The In-the-Wild number (honest) and the in-domain number (flattering)
   always appear **side by side**.

### 5.4 The system-level trials

A *trial* is a call against a claimed speaker: **genuine** (the enrolled speaker's own held-out clips → correct
answer ALLOW), **clone** (spoof clips targeting that speaker → flag), **impostor** (a different real person → flag).
Each call is about 10 s (3 clips) and is replayed under the same degradations for every class. Enrolment audio and
test audio never overlap.

We report three different questions, because one number blurs them:

- **Situation never seen** (leave-situations-out): fit on other speakers *and* other situations (related ones held
  out together, e.g. reverb + speakerphone + reverb-then-codec). *This is the number to plan with.*
- **Situation covered, speaker new** (4-fold cross-validation grouped by speaker): the normal operating regime.
- **Untouched dev**: ten more speakers, never used to fit or choose anything.

### 5.5 Why you can trust it (say these)

- A frozen protocol, a fingerprinted front-end and a git commit on every number.
- A permanent test (silence-only) that would catch the shortcut that fooled our first classifier.
- A hard rule: In-the-Wild never trained on or thresholded on.
- **Negative results are kept**: calibrators that overfit, a boosted-tree head with no gain, a second detector that
  moved the operating point without separating classes, and prosody at chance are all documented and *not
  shipped*.
- 268 automated tests; `pytest -q` and `ruff check .` pass.

---

## 6. The results, as a story

### Chapter 1: the first classifier looked great, and was cheating

Handcrafted features (61 numbers: 40 MFCC, spectral roll-off, zero-crossing rate, 12 chroma, 7 contrast) and a
gradient-boosting ensemble (XGBoost, LightGBM, CatBoost) scored **1.83 % EER** on ASVspoof dev. We then fed it
**only the non-speech parts** of dev clips (where it should know nothing, 50 % = chance) and it scored
**13-15 % EER**. It had learned the corpus's silence-duration and loudness artefact. After the fair front-end:
dev **7.0 %**, unseen attacks **33.4 %**, In-the-Wild **58.2 %**, and silence-only back at **about 50-61 %**
(`docs/baseline_results.md`, `results/tables/baseline_*.csv`). *This is the honest start.*

### Chapter 2: a published detector, measured honestly

AASIST through the fair front-end (published weights):

| | EER |
|---|--:|
| dev (known attacks) | **9.6 %** (about 1 % reported by its authors) |
| eval (unseen attacks) | **23.3 %** |
| In-the-Wild | **35.8 %** |
| silence-only | 41 % (passes the gate) |
| min-tDCF dev / eval | 0.288 / **0.588** |

Per attack, it is near perfect on A08/A09 (1.6 % / 0.7 %) and fails on A10 / A12 (52.1 % / 43.7 %), unfamiliar neural
vocoders. A single aggregate hides a 0.7-52 % spread. *Its dev number is worse than published because the front-end
removes the artefact it had partly learned; the honest score is the higher one.*

### Chapter 3: calibration, measured, and mostly not shipped

We fitted Platt and isotonic calibrators for all three learned signals and measured calibration error on data they
had never seen. **Spoof and prosody:** both methods looked perfect on the fit data (isotonic ≈ 0) and got *worse* than
doing nothing on unseen attacks and In-the-Wild (spoof ECE 0.105 → 0.116 on eval, 0.136 → 0.191 on In-the-Wild) → **not
shipped**. **Speaker:** held-out ECE 0.389 → 0.181 with Platt → **shipped**. The lesson: *calibration inherits the same
generalisation problem as detection, so measure it before you trust it* (`docs/calibration.md`).

### Chapter 4: background voices break the detector; fixing it

With three other people talking at 5 dB, the pretrained detector's dev EER went from 7.6 % to **48.2 % (chance)**.
Fine-tuning on ASVspoof train only, with class-independent noise, codecs and RawBoost: clean **3.4 %**, 5 dB babble
**27.0 %**. A second round (rooms, speakerphone, narrow band, clipping, packet loss, hum, applied in the order a real call
reaches the model) took 5 dB babble to **13.2 %** and the room/line mix from **21.0 % to 11.6 %**, with music and burst
noise (never trained on) at **4.4 %**. Gates still pass: dev **4.03 %**, eval **20.15 %**, In-the-Wild **36.18 %**
(unchanged: this teaches robustness to distortion, not new synthesis), silence-only **44.75 %**.

### Chapter 5: one detector is not a system: fusion

On speakers the model never saw (19 situations, 4,157 reads, 4-fold CV grouped by speaker):

| System | Balanced accuracy | Genuine accepted | Clones caught | Impostors caught |
|---|--:|--:|--:|--:|
| Detector alone, one threshold | 73.6 % | 84.3 % | 86.1 % | **16.3 %** |
| Hand-set weighted mean | 83.9 % | 84.0 % | 85.6 % | 80.0 % |
| **Learned, trust-weighted fusion (shipped)** | **90.2 %** | 89.0 % | 92.1 % | 89.8 % |

The detector alone catches almost no *impostors* (a different real human is not synthetic), which is why the voice
match is a separate, necessary signal. On **10 untouched dev speakers**: accuracy **92.4 %**, balanced **91.8 %**, genuine
accepted 90.5 %, clones caught 97.6 %, impostors 83.9 %, genuine ESCALATE 1.3 %, EER 7.9 %.

### Chapter 6: situations nobody planned for

We replayed calls under 14 further situations: reverb, speakerphone, narrow band, clipping, very quiet speech, packet
loss, mains hum, music, noise bursts, three phone codecs (AMR, Opus 8 kb/s, G.722), a noisy room followed by a mobile
codec, and an echoey room followed by one.

- **Before** (a system fitted only on the five standard conditions): balanced accuracy **78.6 %**, genuine callers
  accepted **63.3 %**, genuine ESCALATE **22.7 %**; with a **clipped microphone only 8 %** of genuine callers got through.
  Phone codecs on their own were fine (balanced accuracy 93-96 %), so *telephony is the case it was built for, and it holds*.
- **After** (robust detector + cause-naming descriptors + learned fusion), with the situation held out of fitting:
  **86.6 %** balanced, genuine accepted 79.2 %. On untouched speakers where the situation is covered: **91.3 %**
  (14 situations), **91.8 %** (all 19).
- **Weakest cases, said out loud:** a noisy room then a mobile codec (genuine accepted **60 %** on dev, 47.5 % if the
  situation was never in the fit), a clipped microphone never seen before (**8 %** accepted), speakerphone 75 %.
  **All degradations are simulated**, not recordings.

The descriptors that fixed it each name one cause: `env_acf` (reverberation), `crest_db` and `flat_top_db` (clipping),
`hum_db` (mains hum). `env_acf` alone did most of the work for echo.

### Chapter 7: modern cloning tools

We built clones with **kNN-VC** (open-source zero-shot voice conversion: WavLM features, nearest-neighbour matching of
the speaker's reference frames, a HiFi-GAN vocoder; here driven by MMS-TTS speech), **SpeechT5 + an x-vector** (a weak,
laptop-class clone) and plain **MMS-TTS** (nobody's voice). The code was vetted, downloads approved and weights loaded
as plain tensors (`docs/modern_attacks.md`).

- The **detector alone sees only 2-3 %** of kNN-VC and SpeechT5 reads (a score above 0.5). Only the older VITS-style
  MMS-TTS draws it in (40-48 %).
- The **fused system flags** kNN-VC clones on **72 % of clean-line calls**, **99 %** with background voices at 10 dB and
  **87 %** on a G.711 phone line; SpeechT5 **50-60 %**; MMS-TTS **100 %** (`results/tables/modern_by_attack_gain_*.csv`).
- A sweep of 80 kNN-VC calls with 12-107 s of the customer's voice as reference: **more reference audio did not help
  the attacker, but a clone that sounded closer to the customer got through more often** (similarity ≥ 0.92 passed
  roughly one time in five). *That sweep used the previous clone head and is to be re-run; say so.*
- Our own demo "cloned with a free tool" clip is **one of the hard ones**: speaker similarity 0.93, and it is let through
  with a note (an early warning that eases off). We show it on purpose.

### Chapter 8: what happens when there is nothing to judge

Before the gate: crashes, ALLOW for one sample, ESCALATE for silence. After: WAITING and normal reads for all of them
(`docs/input_quality.md`).

### Chapter 9: Indian languages (a pilot, not a claim)

MMS-TTS attacks in five languages (40 clips each) scored by the published detector: synthetic recall **Marathi 85 %,
Hindi 58 %, Tamil 55 %, Bengali 38 %, English 8 %**. The detector was trained only on English yet flags Indian-language
synthesis *more* than English synthesis, because non-English speech sits further from the natural English it learned.
That is **uneven, measured, per-language**: not "supports all Indian languages". A consented real-speech slice is the
missing half.

### Chapter 10: speed

One assessment of 10 s of audio on an RTX 4050 laptop GPU: about **160 ms** (spoof read 117 ms, speaker 36 ms, channel
features 3 ms) when measured on 2026-10-14. The input gate and the extra descriptors added later cost roughly another
160 ms under load, so say "a few hundred milliseconds per update", re-scored every 0.5 s. A **confirmed** acoustic read
still needs about **4 s of speech** (the detector's fixed window); earlier reads are provisional and capped at VERIFY.
We did *not* build a sub-2 s confirmed read.

---

## 7. The lesson from real audio (the WhatsApp voice note)

**What happened.** The project owner played a friend's WhatsApp voice note (Opus, 16 kHz, about 18 kbps) into the
console and it said *Stop: this may be a fake voice*. Then their own live microphone voice was flagged too.

**Two different causes, which the first console showed as one:**

1. **The wrong voice was on file.** The console compares every caller with the voice on file, and it was the *sample
   customer's*. A real friend is correctly "not the customer". The wording ("fake") was wrong. *Fixed:* a Stop now says
   "this isn't the customer on file" when that is the evidence; the console has a one-click "Try it with your own voice"
   that puts *your* voice on file first.
2. **A real flaw in our model.** With the friend's own voice on file, the "synthetic" check still read **0.98** while
   the detector itself was barely worried (**0.27**; studio speech **0.005**). We took the clone head apart term by term
   (log-odds): the detector contributed **−0.29**, the **bare recording-style terms +2.81**, interactions −0.04, baseline
   +1.39 → +3.87 → 0.98. The alarm came from *how the note was recorded* (low loudness-envelope correlation 0.29 against
   0.68 for a studio call, a very clean noise floor, little high-frequency energy), exactly what noise suppression plus a
   low-bitrate codec does.

**Why the model had learned that.** In ASVspoof every real clip is studio audio and every attack is vocoder output, so
"clean floor, little high frequency, jumpy envelope" separates the classes *there*. A head that may alarm on those
descriptors alone learns **the style of its training data, not synthesis**. The same shortcut had made our modern-clone
numbers look better than they were: our own notes had said the kNN-VC catch came from "the vocoder's clean output, not
cloning". That was the warning.

**The fix.** The clone head is now `p = sigmoid(offset + gain(style) × logit(detector))` with a positive gain: recording
style decides *how much the detector is believed*, never speaks for it. Two tests pin it (monotone in the detector for any
recording style; a clean, band-limited, noise-suppressed-looking recording cannot call a quiet detector synthetic). An
"interactions only" linear variant was tried and **rejected**: its free-sign products kept a hidden form of the shortcut.

**What it cost, said plainly** (`results/tables/clone_head_variants.csv`):

| | before | now |
|---|--:|--:|
| Untouched dev, balanced accuracy | 92.7 % | 91.8 % |
| Situations never seen, balanced | 89.2 % | 86.6 % |
| Clean-line kNN-VC clones flagged | 100 % | 72 % |
| Attacks reaching ESCALATE (Stop) | 86 % | 54 % |

**What remains.** The friend's own voice on file now gives **Double-check** (risk 0.68), not **Stop**, and not yet "looks like
the real customer": the detector itself still reads 0.27-0.39 on this Opus note. Likely contributors (one recording cannot
separate them): Opus at about 18 kbps (training saw Opus only at 8 kb/s), the phone's noise suppression, conversational
speech in a language the detector never saw, one speaker and one microphone. The honest fix is a detector trained for this
world (Opus 16-24 kb/s, LibriSpeech real speech, the modern generators with one family held out) plus **real recordings
as a held-out check**. The In-the-Wild EER of 36 % is the same weakness at scale.

**Why tell the examiners this.** It shows (a) a real-world test found what benchmarks could not, (b) we traced the cause
instead of tuning a threshold, (c) we reported the cost of the fix, and (d) we know what is still open.

---

## 8. Limitations, said candidly

Learn these as a list. Volunteering them is a strength; being found out is not.

1. **The detector is weak on audio from outside the lab** (36 % In-the-Wild EER; 20 % on unseen ASVspoof attacks). It is
   one input to a fused decision, never a gate.
2. **Modern clones:** the detector alone sees 2-3 %. The voice match does the catching; a clone that matches the enrolled
   voice as well as the real person does would pass (about one clean-line kNN-VC clone in four did). A fine-tuned or
   commercial cloning model was **not tested**.
3. **Every degradation is simulated** (reverb, hum, clipping, packet loss, codecs on studio speech), not recordings of
   real rooms, cars or phone networks. Real-world audio (a WhatsApp note) already exposed a weakness.
4. **The weakest measured cases:** a noisy room then a mobile codec (60 % of genuine callers accepted on dev), a clipped
   microphone never seen before.
5. **Sample sizes are small.** 48 + 10 speakers for system trials; a dev cell is 20 genuine and 60 attack reads, so one call
   moves it 5 points. The checkpoint was chosen on dev clips, so the dev row is slightly optimistic for the detector.
6. **Replay attacks (physical access) were not evaluated**; only logical-access synthesis and conversion. Speakerphone is a
   simulation, not a true replay test.
7. **Adversarial robustness was not tested** (an attacker who knows the fusion, adds noise on purpose, or optimises against
   the detector).
8. **Fairness is unmeasured** across accents, ages, genders, voice disorders and languages. Treat it as *unknown*, not benign.
9. **Indian languages:** a small synthetic pilot (40 clips per language); no consented real-speech set yet.
10. **Not deployment-hardened:** alerts are simulated (no SMS or e-mail), REST and gRPC bind to localhost with no
    authentication or TLS, and the repository has **no LICENSE file** yet (an owner decision).
11. **Latency:** a confirmed acoustic read needs about 4 s of speech; there is no sub-2 s confirmed read and no on-device
    (edge) inference.
12. **We did not train a detector from scratch**; the laptop cannot (about 50 min/epoch, 30-40 epochs needed). A Kaggle
    runbook and notebook exist.

---

## 9. What is next (in priority order)

1. **Detector v3:** fine-tune on modern generators (kNN-VC and SpeechT5 in training, MMS-TTS held out), Opus 16-24 kb/s and
   other messaging codecs, and LibriSpeech as extra real speech; then re-fit the fusion and re-run every table (about 3-4 h of
   GPU).
2. **Real recordings as a held-out test:** consented voice notes and calls from several people, phones and apps. Never
   used for training or thresholds.
3. **Consented Indian-language real speech** (3-5 speakers per language) so recall becomes a real EER.
4. **Whole-call evaluation** (tooling exists, not yet run): how often a genuine caller is flagged *at some point* in a long call,
   and how soon a clone is caught; decide whether a confirmed Stop should be sticky.
5. **Hardening:** authentication, TLS, rate limits, real alert delivery, a licence.
6. **Edge inference** and a shorter-window model for a faster confirmed read.

---

## 10. Engineering quality, privacy and ethics

**Reproducibility.** One `config/default.yaml` is the only place a path is written. Manifests are built once from the
official protocol files. `pip install -e ".[...]"`, then documented commands regenerate every table
(`README.md` §4, `GETTING_STARTED.md`). The CI checks are `ruff check .` and `ruff format --check .`.

**Tests: 268**, covering metrics, the front-end, manifests, models, speaker, prosody, risk and fusion (and the fitting of
it), audit, the session API over REST **and** gRPC, VAD, the input gate, calibration, min-tDCF, channel features, degradations,
trial design, the fine-tune data pipeline, the live buffer and the console's routes.

**Provenance.** Every row in `results/tables/` carries train set, test set, model, seed, front-end fingerprint, score direction,
git commit and time.

**Privacy.** Audio is processed in memory. The audit log stores scores, reasons and provenance only and **refuses** any field
that could carry audio, a transcript or an embedding. The voiceprint is held for the session only. The console says plainly
that no recording is kept. We **do not claim legal compliance** with India's DPDP Act: that needs a legal review.

**Ethics.** A "how synthetic does this voice sound" score is dual-use; it could be misused to challenge a genuine speaker
(a voice disorder, an accent, a bad line). Hence: decision support, reason codes, a human acting on every alert, and a model card
(`models/pretrained/MODEL_CARD.md`, Mitchell et al. format) that states intended use, out-of-scope uses and unmeasured
factors.

**Consent and data.** We used established research corpora (ASVspoof 2019, In-the-Wild, LibriSpeech) under their terms and
collected no new human speech for evaluation; the Indian-language pilot is synthetic **because** consent was not yet collected.
MMS-TTS is CC-BY-NC, so it is used for evaluation only.

---

## 11. What we built versus what we used

Be precise; it is the first thing a sharp examiner probes.

| Component | Source | What *we* did |
|---|---|---|
| AASIST architecture and published weights | clovaai/aasist (vendored, own licence) | wrapped it, **fine-tuned** it (robustness), calibrated, integrated |
| RawBoost augmentation | TakHemlata/RawBoost-antispoofing (vendored) | used it in fine-tuning |
| WavLM x-vector speaker model | Microsoft `wavlm-base-plus-sv` (no licence stated on its page: an audit item) | enrolment, scoring, **Platt calibration**, a repair for a library-version bug |
| Silero VAD | snakers4/silero-vad | batched it, built the **input gate** around it |
| min-tDCF reference code and ASV scores | ASVspoof 2019 organisers (vendored) | wired and verified it |
| kNN-VC, SpeechT5, MMS-TTS | open-source models | vetted, built the **clone-generation and trial harness** |
| ASVspoof 2019 LA, In-the-Wild, LibriSpeech | public datasets | manifests, splits, protocol |
| **Fair front-end, silence gate, provenance, frozen protocol** | **ours** | the core methodological work |
| **Degradation toolkit, stress and system trials, leave-situations-out validation** | **ours** | |
| **Recording-style descriptors, trust-weighted learned fusion, policy, context rules** | **ours** | |
| **Session engine, live buffer, REST / WebSocket / gRPC, audit log, operator console** | **ours** | |
| Handcrafted-feature baseline, prosody branch | ours (standard libraries) | measured and honestly demoted |

**If asked "did you use AI tools?"** Answer truthfully and in your own words. This repository was developed with an AI
coding assistant's help, and the paper draft already has a TODO for the disclosure wording
(`paper/README.md`). Your college or the event may have a policy. Whatever you say, you must be able to explain every number
and every design decision here; that is what these notes are for. A good framing is: *"We used AI assistance for
implementation and documentation, we ran and verified every experiment, and we can explain each part."* Practise it so it is
true.

---

## 12. Mistakes we found and fixed

Examiners like candour. Each is a one-sentence story.

1. **The silence shortcut.** Our first classifier's 1.83 % EER was mostly an artefact (13-15 % EER on silence alone). Fixed with
   the fair front-end and a permanent gate.
2. **Calibrators that overfit.** Platt and isotonic made calibration *worse* on unseen attacks; we shipped only the speaker one.
3. **A fusion fitted on five conditions failed on fourteen others** (clipped microphone: 8 % of genuine callers accepted). Fixed
   with a robust detector and cause-naming descriptors, and measured with situations held out.
4. **REST and gRPC disagreed.** The same clip fused to 0.77 over REST and 0.95 over gRPC because the spoof read was computed
   differently per transport. Fixed with one shared `spoof_read()` and a test that asserts identical answers.
5. **A proto3 default bug.** An unset `caller_known` silently became `false`; caught by a test and fixed with `optional bool`.
6. **No input gate.** The engine crashed on NaN and shouted ESCALATE at silence. Fixed with the gate and a WAITING state.
7. **A verdict that lagged the call.** Feeding and scoring in one loop made the answer fall further behind during a clip
   (about 1 s growing to 3 s over 8 s). Fixed with a rolling buffer that always scores the latest audio.
8. **A clone head that read recording style** and called a real voice note fake (section 7). Fixed, and the cost reported.
9. **A "caught 100 % of clones" claim that was too good.** The first modern-clone result was partly the same shortcut; the
   honest figure is about 72 % on a clean line.

---

## 13. The question bank

Short model answers. Put them in your own words. "If pushed" lines are follow-ups.

### The idea and the problem

**Q1. In one sentence, what does your project do?**
It listens to a live call and tells the agent, in plain words, whether to trust the caller (real customer, double-check, stop),
by combining a voice-cloning detector, a voice match against the customer's registered voice, the recording quality and the call
details.

**Q2. Why not just use one good detector?**
Because none generalises. The published AASIST drops from about 1 % error in the lab to 35.8 % on real-world audio, and the
detector sees only 2-3 % of modern open-source clones. A system must not stake a transaction on a signal that fails silently.

**Q3. What is novel here?**
Not a new network. (1) A reproducible, fingerprinted evaluation with a permanent silence gate, (2) a system that fuses four
signals with a trust-weighted rule where recording style can only scale the detector, (3) a measured, published account of
failures (stress situations, modern clones, a real-world voice note) with the cost of each fix. *If pushed: we do not claim to be
first to see the silence artefact; it is in the literature (Müller et al., 2021).*

**Q4. Who is the user?**
A contact-centre or bank agent (and their supervisor). The output is a recommendation to a human, with reasons, not an automatic
block.

**Q5. Why is a human in the loop?**
Because errors are costly in both directions and the detector is weak on real audio. A false alarm annoys a real customer; a miss
allows fraud. The policy lets the operator choose stricter or looser thresholds by scenario, and every alert is acknowledged and
logged.

### Data and evaluation

**Q6. What is ASVspoof 2019 LA?**
A public benchmark of English studio speech (VCTK) with bona fide clips and spoofed clips from systems labelled A01-A19.
Logical access means synthesised or converted speech (not replay). Training and dev use the six known attacks A01-A06; the
evaluation set uses A07-A19, mostly systems never seen in training.

**Q7. What is In-the-Wild and why is it special in your project?**
A found-audio set of public figures (Müller et al., Interspeech 2022). We use it only as a final cross-dataset check: never for
training and never for choosing a threshold, so it stays an honest measure.

**Q8. Define EER.**
The error rate at the threshold where the false-accept rate equals the false-reject rate. It is threshold-free, so it measures
whether the scores separate the classes, regardless of where you set the cut-off.

**Q9. Why not report accuracy?**
Because the classes are imbalanced (about 9 fake to 1 real), so a model that always says "fake" scores about 90 %. In our system
trials six of eight calls are attacks, so flag-everything scores 75 %. We use EER, min-tDCF and balanced accuracy.

**Q10. What is min-tDCF?**
The ASVspoof challenges' deployment metric: the cost of the countermeasure working *together with* a speaker-verification system,
with a cost model where letting a spoof through costs more. 0 is perfect, 1 is no better than no countermeasure. Ours: dev 0.288,
eval 0.588, computed with the organisers' own code and ASV scores.

**Q11. Explain the silence shortcut and how you found it.**
In ASVspoof 2019 the real and fake clips differ in how much silence surrounds the speech and how loud they are. A model can use
that. We tested by giving the classifier only the non-speech parts of dev clips: it scored 13-15 % EER where chance is 50 %, so
part of its 1.83 % in-domain score was the artefact. We built a front-end that trims, re-pads and loudness-normalises both classes
identically and a gate that must read about 50 % on silence-only input.

**Q12. Why is your AASIST dev EER (9.6 %) worse than the published ~1 %?**
Because our front-end removes the artefact the published model had partly learned. The higher number is the honest one.

**Q13. How do you avoid overfitting to dev?**
Official splits, thresholds chosen on training speakers only, cross-validation grouped by speaker, an untouched dev set of ten
speakers, a leave-situations-out test, and In-the-Wild never used. We also say the dev row is slightly optimistic for the detector
because checkpoints were chosen on dev clips.

**Q14. What is "leave-situations-out" and why does it matter?**
We fit the fusion on other speakers *and* other situations, holding related situations out together (for example reverb,
speakerphone and reverb-then-codec). It estimates behaviour in a situation nobody planned for. That number (86.6 % balanced)
is the one to plan with, not the 91.8 % on situations the system was fitted on.

**Q15. How big are your test sets? Are the results statistically solid?**
System trials use 48 + 10 speakers; a dev cell is 20 genuine and 60 attack reads, so a single call moves it five points, and the
pooled rows are the ones to quote. We state it as a limitation; 200+ speakers would tighten every number.

### The detector

**Q16. How does AASIST work?**
It takes the raw waveform, applies a learnable filter bank, then a residual encoder that produces a frequency-by-time feature
map. Graph attention networks treat the spectral and temporal positions as graph nodes, so the model can combine localised
artefacts across frequency bands and time segments, and a read-out layer outputs real-versus-synthetic. About 0.3 million
parameters.

**Q17. What did you change in it?**
Nothing in the architecture. We fine-tuned the published weights on ASVspoof train only with class-independent degradations (noise,
babble, codecs, RawBoost, then rooms, speakerphone, narrow band, clipping, packet loss, hum), selected the checkpoint on dev EER
with a clean-audio budget, and calibrated it.

**Q18. Why does the detector need 4 seconds?**
The model's input is fixed at 64,600 samples (4.04 s); its positional parameters are sized for it. Shorter audio is tiled to fill it,
which is out of distribution, so we mark such reads provisional and cap them at VERIFY. A true sub-2 s confirmed read would need a
second, genuinely short-window model; we did not build it.

**Q19. What is RawBoost?**
A waveform-level augmentation (Tak et al., ICASSP 2022) that adds linear and non-linear convolutive noise, impulsive noise and
stationary noise so a detector does not depend on a clean channel.

**Q20. Why is the detector nearly blind to kNN-VC and SpeechT5?**
It was trained on 2019 attack systems. Modern generators leave different artefacts, so only 2-3 % of reads reach a score of 0.5.
The next step is training on those generators with one family held out.

### The voice match

**Q21. How do you verify the speaker?**
We embed the registered voice with a WavLM x-vector model and compare each later 6 s window with it by cosine similarity. Same
speaker is about 0.95 or above on clean audio; a different person about 0.4-0.7. A Platt calibrator turns similarity into a
mismatch risk.

**Q22. Why WavLM and not ECAPA-TDNN?**
It ships as a Hugging Face model with no new dependency and gave good separation. ECAPA models are available locally but a
head-to-head comparison was not run; that is listed as future work.

**Q23. What is an x-vector? What is cosine similarity?**
An x-vector is a fixed-length embedding of a speaker built from frame-level features pooled over time. Cosine similarity is the
cosine of the angle between two vectors: 1 means the same direction, 0 unrelated.

**Q24. What are the limits of speaker verification here?**
It is the weaker head (a different real person is caught about 84-90 % of the time). It needs about 6 s of clean enrolment,
changes across channels and illness, and a clone that matches the voice closely passes it. We never claim it proves identity.

### Calibration and fusion

**Q25. What is calibration, and why did you not ship a spoof calibrator?**
Calibration makes a score behave like a probability. We fitted Platt and isotonic calibrators for the detector, measured on unseen
attacks and In-the-Wild, and both made things worse (they overfit the dev distribution). So we shipped no spoof calibrator. Speaker
calibration held up on unseen speakers (ECE 0.389 → 0.181), so that one ships.

**Q26. Explain ECE and Platt scaling.**
ECE bins predictions by confidence and averages the gap between predicted and actual frequency. Platt scaling fits a logistic
regression from a score to a probability. Isotonic fits a monotone step function.

**Q27. How does the fusion work?**
Two logistic-style heads: a clone head on the detector score and an impostor head on the voice-match risk, both aware of the recording
style, combined as independent alarms `1 − (1 − p_clone)(1 − p_impostor)`; call context is a third alarm. Thresholds are mapped onto the
policy's scale.

**Q28. What is a "trust-weighted" clone head, and why did you change it?**
It computes `sigmoid(offset + gain × logit(detector))` where the gain depends on the recording style and is always positive. The
recording style says how far to believe the detector but cannot raise an alarm by itself. We changed it because a real WhatsApp voice
note was called synthetic purely from its recording style (+2.81 log-odds from style, −0.29 from the detector).

**Q29. What did the change cost?**
Balanced accuracy on untouched dev 92.7 → 91.8 %, unseen-situation 89.2 → 86.6 %, clean-line kNN-VC flagged 100 → about 72 %, attacks
reaching ESCALATE 86 → 54 %. We report it because the old numbers were partly a property of the recordings.

**Q30. Why logistic heads and not a deep network?**
Little data (a few thousand reads), a need for interpretability and monotonicity, and numpy-only inference. A boosted-tree head was built
and tested and gave no gain once the descriptors existed, so the simpler model ships.

**Q31. Why is prosody not used?**
We built it (F0, jitter, shimmer, HNR, pauses): 26.3 % EER alone and about chance on the system trials. A signal at chance only adds noise,
so it is shown optionally but is not a decision input. Dropping it is a measured decision.

### The system

**Q32. What are ALLOW, VERIFY and ESCALATE?**
ALLOW: no red flags. VERIFY: ask a knowledge-based security question. ESCALATE: pause approval and verify through the registered number.
There is also WAITING: not enough clear speech yet; never an allow, never an alarm.

**Q33. Why three scenarios?**
The same evidence deserves different action in different situations: a routine question, a payment, a password reset. Higher-stakes
scenarios use lower thresholds (0.55/0.75, 0.40/0.65, 0.30/0.55).

**Q34. What is a "provisional" read?**
A reading built from too little speech or a not-yet-full window. It can raise a VERIFY but can never ESCALATE by itself; once enough audio
arrives it is confirmed automatically.

**Q35. Is it real-time?**
It re-scores every 0.5 s on a rolling 12 s buffer, a few hundred milliseconds of compute per update on a laptop GPU. The first reading appears
after about a second of speech (provisional) and the first confirmed reading after about 4 s. We state that this is not the "sub-2 s confirmed
read" the brief hoped for.

**Q36. How do REST and gRPC relate?**
Both call the same `SessionManager`. A test sends the same clip through each and asserts identical action and score.

**Q37. How do you handle silence, noise or garbage audio?**
The input gate. It repairs NaN or wrongly scaled audio, finds speech with Silero VAD, and returns WAITING when there is too little. We
found the first version crashed or shouted ESCALATE at silence, so this was a measured fix.

**Q38. How does the console help a non-expert?**
One answer in plain words, four checks (voice match, computer-generated clues, call details, audio clarity), what to do now, a meter, and a
scorecard on the sample calls (Correct, Caught, Partly caught, Missed, False alarm). Technical detail is folded away.

### Robustness and attacks

**Q39. What situations did you test?**
Nineteen: clean, background voices (10 and 5 dB), broadband noise, G.711, reverb, speakerphone, narrow band, clipping, very quiet speech,
packet loss, mains hum, music, noise bursts, AMR, Opus 8k, G.722, a noisy room then AMR, an echoey room then AMR. All simulated.

**Q40. Which were worst and why?**
A noisy room then a mobile codec (60 % of genuine callers accepted on dev) and a clipped microphone never seen in fitting (8 %). Both damage the
audio evidence itself, so the detector reads genuine speech as synthetic.

**Q41. Did you test replay attacks or adversarial attacks?**
No. Replay (physical access) and adversarial robustness are not evaluated; speakerphone is a simulation, not a replay test. They are in the
limitations.

**Q42. How do you know a clone that sounds exactly like the customer would pass?**
We measured it: in 80 kNN-VC clone calls the clones that scored speaker similarity 0.92 or more got through about one time in five, and our demo
clone at 0.93 passes. The system flags *imperfect* clones; a perfect one gives both heads nothing to find. That is why the human and the call
context remain.

**Q43. What about other languages?**
A synthetic pilot in five languages shows uneven recall (8-85 %). It proves language robustness must be measured, not that it exists.

### Privacy, ethics, deployment

**Q44. How is privacy handled?**
Audio is processed in memory; the audit log keeps scores, reasons and provenance only and rejects any field that could hold audio, a
transcript or an embedding. The voiceprint exists only for the session. We do not claim legal compliance.

**Q45. What are the ethical risks?**
Dual use (challenging a genuine speaker with an unusual voice or a bad line) and unmeasured fairness across accents, ages and languages. We
mitigate with reason codes, a human decision, a model card and an explicit "unknown, not benign" statement.

**Q46. How would you deploy this in a bank?**
Integrate via SIP or WebRTC audio taps into the gRPC stream; add authentication and TLS; run inference near the call (edge or private cloud);
collect consented real calls to re-fit thresholds on the real channel; monitor false-alarm rates; keep the human workflow. We built the
engine, interfaces and measurements; hardening is open.

**Q47. How does it scale?**
It does not yet. One process serialises GPU work with a lock and sessions are in memory with a one-hour timeout. Scaling would need
batching, multiple workers and a session store; not attempted.

### Process and teamwork

**Q48. What was the hardest problem?**
Knowing which numbers to believe. The silence shortcut, the calibrators that overfit and the recording-style shortcut each made a number look
better than it was. Each time we built a test that exposed it and reported the honest figure.

**Q49. What would you do differently?**
Collect real recordings earlier, and test with a real messaging-app voice note on day one. Our evaluation was synthetic for too long.

**Q50. What did you personally contribute?**
*(Fill in your own words: for example the evaluation protocol and front-end, the fusion and policy, the session engine and API, the console,
the experiments and documentation. Be specific, and be able to open the file.)*

---

## 14. Glossary

| Term | Plain meaning |
|---|---|
| **Voice cloning** | making a computer speak in a particular person's voice, from a few seconds of their speech |
| **TTS / VC** | text-to-speech (typed words become speech) / voice conversion (one person's speech becomes another's voice) |
| **Spoof / bona fide** | fake / real audio in anti-spoofing jargon |
| **Countermeasure (CM)** | a detector of fake audio |
| **ASV** | automatic speaker verification: is this the claimed person? |
| **EER** | equal error rate, the error where false accepts equal false rejects (lower is better) |
| **min t-DCF** | tandem cost of a countermeasure plus an ASV system under a cost model (0 best, 1 useless) |
| **FAR / FRR** | false-accept rate (fake passes) / false-reject rate (real blocked) |
| **Balanced accuracy** | mean of "real accepted" and "fakes flagged" |
| **ECE** | expected calibration error: how far confidence is from reality |
| **Platt scaling / isotonic** | two ways to map a score to a calibrated probability |
| **Logit** | `log(p / (1 − p))`, the log-odds of a probability |
| **Fusion** | combining several signals into one decision |
| **Cross-dataset** | testing on a different dataset than training (In-the-Wild) |
| **Shortcut** | a cue (such as silence length) that predicts the label in the dataset but has nothing to do with the task |
| **Front-end** | the fixed audio preparation (16 kHz, trim, loudness) applied before every model |
| **Silence ablation** | testing a model on only the silent parts of clips; it should be near chance |
| **VAD** | voice activity detection, finding where speech is |
| **EMA** | exponential moving average, smoothing a stream of readings |
| **Embedding / voiceprint** | a vector that represents a voice; x-vector is one kind |
| **Cosine similarity** | angle-based closeness of two embeddings (1 = same direction) |
| **AASIST** | an anti-spoofing network using graph attention over spectral and temporal nodes |
| **WavLM** | a self-supervised speech model from Microsoft; we use a version fine-tuned for speaker verification |
| **kNN-VC** | zero-shot voice conversion that replaces each speech frame with the nearest frames from a short reference recording |
| **RawBoost** | waveform augmentation for anti-spoofing |
| **Codec (G.711, AMR, Opus)** | the compression a phone call or voice message goes through |
| **Provisional read** | a reading from too little audio; can warn, cannot stop |
| **WAITING** | no verdict, nothing usable to judge |
| **LUFS** | a loudness unit (EBU R128 normalises to -23 LUFS) |
| **gRPC / REST / WebSocket** | three ways programs talk to the engine |
| **Provenance** | the record of exactly what produced a number (data, model, seed, commit) |

---

## 15. The numbers on one page

**Problem gap.** Published AASIST about 1 % → ours through the fair front-end: dev **9.6 %**, eval **23.3 %**, In-the-Wild **35.8 %**; min-tDCF
**0.288 / 0.588**. Handcrafted baseline: **7.0 / 33.4 / 58.2 %**. Silence-only: first pipeline **13-15 %** (shortcut), after fix **about 50 %**
(AASIST 41 %, fine-tuned 44.75 %).

**Detector (fine-tuned, `AASIST_robust2`).** dev **4.03 %**, eval **20.15 %**, In-the-Wild **36.18 %**. 5 dB babble: **48.2 → 27.0 %** (round 1),
**17.8 → 13.2 %** (round 2). Clean **3.4 → 3.8 %**. Room/line mix **21.0 → 11.6 %**. Never-trained music/burst **5.2 → 4.4 %**. Operating point 0.2387.

**Voice match.** Same speaker cosine about 0.95; calibration ECE **0.389 → 0.181** (held-out speakers). Enrolment needs ≥ 3 s speech (6 s best).

**Fusion (shipped, trust-weighted).** Unseen speakers, 19 situations: balanced **90.2 %** (detector alone 73.6 %, hand-set 83.9 %), genuine accepted 89.0 %,
clones 92.1 %, impostors 89.8 %, genuine ESCALATE 3.1 %, EER 9.8 %. Untouched dev (10 speakers): accuracy **92.4 %**, balanced **91.8 %**, genuine
accepted 90.5 %, clones 97.6 %, impostors 83.9 %, genuine ESCALATE 1.3 %, EER 7.9 %.

**Stress (14 situations).** Before: **78.6 %** balanced, genuine accepted **63.3 %**, genuine ESCALATE 22.7 %, clipped mic **8 %**. After, situation never seen:
**86.6 %** (genuine 79.2 %). After, untouched speakers: **91.3 %** (genuine 89.6 %). Weakest on dev: noisy room + codec **60 %** genuine accepted, clip 70 %,
speakerphone 75 %.

**Modern clones (shipped head, single reads).** kNN-VC flagged: clean **72 %**, 10 dB voices **99 %**, G.711 **87 %**. SpeechT5 **50-60 %**. MMS-TTS **100 %**.
Detector alone sees **2-3 %** of kNN-VC and SpeechT5. Previous head: 100 % kNN-VC (partly recording style). Demo clone similarity **0.928**.

**Indian-language pilot (published detector, 40 clips each).** Marathi **85 %**, Hindi **58 %**, Tamil **55 %**, Bengali **38 %**, English **8 %**.

**Speed.** About **160 ms** per assessment of 10 s (RTX 4050 laptop GPU, measured before the gate and extra descriptors, which add about 160 ms under load);
re-scored every **0.5 s**; first confirmed read after about **4 s** of speech.

**Policy.** routine **0.55 / 0.75**, transaction **0.40 / 0.65**, privileged **0.30 / 0.55** (warn / escalate). Context: unknown number +0.35; amounts
₹5k +0.15, ₹25k +0.30, ₹100k +0.45.

**Engineering.** **268** tests; ruff clean; every result stamped with front-end fingerprint `640b0644ae` and a git commit; seed 42.

**Data.** ASVspoof LA train 25,380 / dev 24,844 / eval 71,237 clips; In-the-Wild 31,779 clips; modern clones 648 clips.

---

## 16. The demo script and what to do if it breaks

**Setup (before the viva):** `python scripts/serve_demo.py`, wait for `warmed up in NN s` (about 40 s), open `http://127.0.0.1:8000`. The page opens with the
sample customer's voice already on file. Do a dry run once, timed (about 3 minutes). Allow the microphone *before* the session if you will use it.

1. **A real customer.** Play *The real customer calls*. Green answer, meter near zero, four green checks, scorecard *Correct*.
   Say: "Four checks, one answer, the reasons are right there."
2. **A fake voice.** Play *A computer-generated voice* (or *Another cloned voice*). Within seconds: red *Stop*, red meter, the voice does not match and the
   audio has signs of synthesis. Press **I've seen this**: "The agent acknowledges, and that is logged."
3. **The honest hard case.** Play *A voice cloned with a free tool*. It is the clone made with an open-source tool; the system warns early, then eases off
   and the scorecard says *Partly caught*. Say: "We show this on purpose: it is the kind of clone that can slip through, and the console says so."
4. **Yourself.** *Try it with your own voice*: talk for 8 s, it learns your voice, keep talking, it reads *Looks like the real customer*. (The browser microphone
   is requested raw, without noise suppression.)
5. **Open "More detail"** to show the risk-over-time chart and the activity log, and **"How it works"** for the plain-English glossary.

**If something breaks.**
- *Page says the engine is not responding:* the server is still loading; wait for `warmed up`.
- *No sample calls:* `python scripts/pick_demo_clips.py` (and `make_demo_clone.py`).
- *Microphone fails:* skip it. Everything works from files; the microphone is an extra.
- *A call shows Listening… not an answer:* the audio had no usable speech; that is deliberate (never ALLOW for a call nobody heard).
- *A real voice is flagged:* explain section 7, then press **Copy these details** and show the numbers. This is a strength, not an embarrassment.
- *Total failure:* show the three screenshots and the figures in `docs/poster_figures/`, and talk through the architecture picture.

---

## 17. What not to say

- "It detects all deepfakes" / "99 % accurate" / "100 %". The detector alone is 36 % EER on real-world audio.
- "Works for all Indian languages." It is a synthetic pilot with uneven results.
- "DPDP compliant" or "prevents fraud." Neither is established.
- "We invented the silence-shortcut finding." It is known; we built the gate.
- "We trained our own model." We fine-tuned a published one.
- "Real-time, under 2 seconds." A confirmed read needs about 4 s of speech; reads earlier than that are provisional.
- "Tested on real calls." The situations are simulated; one real voice note was tested and exposed a weakness.
- The old numbers: 100 % kNN-VC catch, 92.7 % dev balanced accuracy, 89.2 % unseen. They belong to the linear head that read recording style.
- Anything you cannot explain. If you do not know, say so and say how you would find out.

---

## 18. Checklist for the night before

- [ ] I can say the 30-second and 2-minute pitches without notes.
- [ ] I can draw the pipeline (gate → front-end → four evidence streams → fusion → policy → answer) on a whiteboard.
- [ ] I can define EER, min-tDCF, balanced accuracy, calibration, fusion, provisional read, WAITING.
- [ ] I can tell the silence-shortcut story, the stress-test story and the WhatsApp story in under a minute each.
- [ ] I know the five headline numbers: about 1 % → 36 %; 13-15 % silence; 73.6 → 90.2 %; 78.6 → 86.6 %; 72 % of clean-line clones flagged.
- [ ] I can list the limitations without being asked.
- [ ] I know which parts we built and which we used, and what *I* did.
- [ ] I have run the demo once, end to end, today, with the microphone allowed.
- [ ] I have the three console screenshots and the seven figures on my laptop as a backup.
- [ ] I have printed section 15.
