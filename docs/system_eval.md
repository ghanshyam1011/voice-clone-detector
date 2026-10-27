# System-level evaluation (fused decision, with noise)

Every earlier number in this repo scored **one signal** on **one clip**. This
document scores the thing the product actually ships: the fused
ALLOW / VERIFY / ESCALATE decision on a *call*, against a *claimed speaker*,
including calls with **background voices, room noise and telephone codecs**.

Reproduce: `scripts/build_system_trials.py` → `scripts/fit_fusion.py`
(tables in `results/tables/system_trials_v2_{eval,dev}.csv`).

## Trial design

For each target speaker (official ASVspoof 2019 LA protocol files only):

| Call type | Audio | Correct decision |
|---|---|---|
| **genuine** | held-out bona fide clips of the enrolled speaker | ALLOW |
| **clone** | spoof clips targeting that speaker (A07–A19 on `eval`) | VERIFY / ESCALATE |
| **impostor** | bona fide clips of a *different* human | VERIFY / ESCALATE |

Each call (3 clips, ≈ 10 s) is replayed under five conditions, identical for
genuine and attack calls so "is it noisy?" cannot be a shortcut:
`clean`, `babble10` / `babble5` (three other speakers talking, 10 / 5 dB SNR),
`pink10` (broadband noise), `ulaw` (G.711 telephone codec).
Totals: 48 `eval` speakers × 8 calls × 5 conditions = **1,920 reads**; 10 `dev`
speakers = **400 reads**. Call context is held neutral so the number reflects
the acoustic signals + fusion only (real context only adds risk to attacks).

## What was wrong, and the fixes

| Problem | Evidence | Fix |
|---|---|---|
| Detector falls apart with background voices | Pretrained AASIST dev EER **7.6 %** clean → **48.2 %** at 5 dB babble (chance) | `scripts/finetune_robust.py`: fine-tune on ASVspoof **train** only with class-independent babble / pink noise / codec / RawBoost → **3.4 %** clean, **27.0 %** babble-5 dB |
| One threshold cannot serve clean *and* noisy audio | Genuine calls' mean spoof risk 0.00 clean → 0.27 under babble; clones stay ≈ 0.65 | `risk/learned.py`: two heads (clone, impostor) that also see channel descriptors (`audio/channel.py`) |
| A weighted mean mixes two different jobs | Spoof catches clones (EER 4–15 %); speaker catches impostors (3–18 %) but *not* clones (25–47 %) | Independent-alarm combination `1-(1-p_clone)(1-p_impostor)`; context is a third independent alarm |
| Prosody adds noise | ≈ 46 % EER (chance) on these trials | Excluded from the decision and, by default, not computed (`--prosody` shows its meter) |
| REST, gRPC and streaming computed the spoof score differently | Same clip: fused 0.77 (REST) vs 0.95 (gRPC); raw last window bypassed the front-end | One `SessionManager.spoof_read()`: front-end → sliding 4 s windows over the latest 10 s → smoothed |

## Results

`acc` = share of calls decided correctly (flag = anything but ALLOW).
`ESCgen` / `ESCatk` = ESCALATE rate on genuine / attack calls.

Read `acc` with its prior in mind: **six of every eight calls are attacks** (4 clone, 2
impostor, against 2 genuine), so a system that flagged every call would score 75 %, and
`acc` weights catching attacks three times as heavily as letting genuine callers through.
The balanced figure (mean of genuine-accepted and attack-flagged) for the shipped fusion
below is 89.7 %: (90.8 % + (4 x 90.8 % + 2 x 84.2 %) / 6) / 2. `docs/stress_eval.md` reports
balanced accuracy first for the same reason.

**Cross-validated on `eval` — 4 folds grouped by speaker, every read scored by a
model that never saw that person (pooled out-of-fold).**

| System | acc | genuine OK | clone caught | impostor caught |
|---|--:|--:|--:|--:|
| Spoof score alone, one threshold | 69.6 % | 88.1 % | 89.3 % | 11.9 % |
| Previous hand-set weighted mean, one threshold | 85.4 % | 82.1 % | 89.6 % | 80.2 % |
| **Learned channel-aware fusion (shipped)** | **89.2 %** | **90.8 %** | **90.8 %** | **84.2 %** |

| condition | acc | genuine OK | clone | impostor | ESCgen | ESCatk |
|---|--:|--:|--:|--:|--:|--:|
| clean | 85.7 % | 100.0 % | 80.7 % | 81.2 % | 0.0 % | 68.1 % |
| babble 10 dB | 90.4 % | 83.3 % | 98.4 % | 81.2 % | 4.2 % | 83.7 % |
| babble 5 dB | 89.8 % | 74.0 % | 99.0 % | 87.5 % | 9.4 % | 86.8 % |
| pink noise 10 dB | 92.2 % | 97.9 % | 92.7 % | 85.4 % | 0.0 % | 82.3 % |
| G.711 phone | 87.8 % | 99.0 % | 83.3 % | 85.4 % | 0.0 % | 66.7 % |

**`dev` — never used for fitting or thresholds (different speakers, known
attacks): 91.8 % accuracy**, EER 8.0 %; eval OOF EER 10.0 %.
Pooled over all 2,320 reads ≈ **89.7 %**.

Previously (clean 8 s calls, pretrained detector, hand-set fusion, 25 speakers):
**64 %** accuracy, only **40 %** of clones flagged.

## Update 2026-10-23: the shipped system now covers 19 situations

Everything above is five conditions, and those five were also what the fusion was fitted
on. [`docs/stress_eval.md`](stress_eval.md) adds fourteen more (room reverberation,
speakerphone, narrow band, clipping, low level, packet loss, mains hum, music, bursts, three
more phone codecs and two compound cases), shows the system above losing 20 or more points
in several of them (a clipped microphone: 8 % of genuine callers accepted), and records the
fix: a detector trained on those situations (`AASIST_robust2`) and four more channel
descriptors that say *why* a call is degraded. What ships now, on all 19 situations
(4 folds grouped by speaker, every read scored by a model that never saw that person):

| System | acc | balanced | genuine OK | clone caught | impostor caught | ESCgen |
|---|--:|--:|--:|--:|--:|--:|
| Spoof score alone, one threshold | 68.2 % | 73.6 % | 84.3 % | 86.1 % | 16.3 % | 0.0 % |
| Hand-set weighted mean, one threshold | 83.8 % | 83.9 % | 84.0 % | 85.6 % | 80.0 % | 0.0 % |
| Linear clone head (shipped until 2026-10-23, see below) | 93.1 % | 92.9 % | 92.4 % | 95.6 % | 88.9 % | 3.4 % |
| **Learned channel-aware fusion, trust-weighted clone head (shipped)** | **90.8 %** | **90.2 %** | **89.0 %** | **92.1 %** | **89.8 %** | **3.1 %** |

**Why the shipped row is lower than the one above it.** On 2026-10-23 a real WhatsApp voice
note was called synthetic at 0.98 while the detector itself was barely worried. The linear clone
head could alarm on recording style (a clean noise floor, little high-frequency energy, a jumpy
loudness envelope) alone, which separates ASVspoof's studio audio from its vocoder attacks but
is a property of the data, not of cloning. It was replaced by a head in which the channel only
sets how far the detector is believed (`docs/real_world_audio.md`). The drop is the shortcut
being removed: three clone-head variants on the same reads are in
`results/tables/clone_head_variants.csv`.

On the untouched `dev` speakers (never used to fit or choose anything), all 19
situations: accuracy **92.4 %**, balanced **91.8 %**, genuine accepted 90.5 %, clones
caught 97.6 %, impostors 83.9 %, genuine ESCALATE 1.3 %, EER 7.9 % (linear head: 92.3 %,
92.7 %, 93.4 %, 95.7 %, 84.5 %, 2.6 %, 7.5 %).

The five conditions of this page, with the shipped model (out of fold):

| condition | acc | genuine OK | clone | impostor | ESCgen | ESCatk |
|---|--:|--:|--:|--:|--:|--:|
| clean | 84.9 % | 100.0 % | 79.7 % | 80.2 % | 0.0 % | 41.7 % |
| babble 10 dB | 91.1 % | 88.5 % | 96.9 % | 82.3 % | 3.1 % | 44.4 % |
| babble 5 dB | 90.6 % | 79.2 % | 97.9 % | 87.5 % | 5.2 % | 44.4 % |
| pink noise 10 dB | 95.1 % | 95.8 % | 97.4 % | 89.6 % | 1.0 % | 67.4 % |
| G.711 phone | 86.7 % | 100.0 % | 82.8 % | 81.2 % | 0.0 % | 39.9 % |

Clean-audio clone recall is 79.7 %, back at the first fusion's 80.7 % (the linear head had
reached 90.6 % partly by reading recording style), and 5 dB background voices still lift genuine
callers accepted from 74 % to 79 %. Fewer attack reads reach ESCALATE (54 % against 86 %): a
Stop now needs the detector or the voice check to be very sure. The honest comparison for
situations nobody planned for is in `stress_eval.md`: balanced accuracy 78.6 % before, 86.6 %
after, on situations held out of the fit.

## Honest limits

- **Heavy background voices are still a weak spot.** At 5 dB babble 78 % of genuine
  calls are accepted (70 % on the 20 `dev` reads); the rest get VERIFY — "ask a knowledge
  question" — and 10 % reach ESCALATE. A noisy room followed by a mobile codec is worse
  (60 % on `dev` with the shipped head). Both heads degrade in babble, so this is limited by
  the signals, not by the fusion.
- **Impostors are the weaker class**: a different human is caught 85-89 % of the time,
  the speaker model's limit. Clone recall on clean audio is 90.6 %; per attack, the unseen
  vocoders (A10, A12 …) are the misses.
- **Synthetic degradation, studio source.** Noise and babble are mixed in from
  ASVspoof speakers; this is not a recording of real phone calls. The attacks are
  2019-era; modern voice conversion and TTS are measured separately in
  [`modern_attacks.md`](modern_attacks.md) (caught, but by the speaker head, not the
  detector). The demo-clip check below is a single example, not a
  held-out set.
- **Small speaker pool** (48 + 10). Expect a couple of points of noise.
- The detector's checkpoint was chosen on `dev` clips, so the `dev` row is
  slightly optimistic for the *detector*; the `eval` out-of-fold row is the
  clean estimate.
- Thresholds: WARN maximises balanced accuracy; ESCALATE is the lowest score
  keeping genuine ESCALATE ≤ 3 % on the training speakers.

## End-to-end sanity (demo clips through the real session stack)

| clip | spoof | speaker | fused | action |
|---|--:|--:|--:|---|
| benign | 0.00 | 0.44 | 0.06 | ALLOW |
| cloned_1 | 0.91 | 0.92 | 1.00 | ESCALATE |
| cloned_2 | 0.97 | 0.59 | 1.00 | ESCALATE |
| modern_tts (rated "real" by the acoustic model) | 0.28 | 0.99 | 1.00 | ESCALATE |
| benign + unknown caller + ₹75,000 | 0.00 | 0.44 | 0.55 | VERIFY |

The modern-TTS row is the thesis: the acoustic model alone misses it; the
speaker-consistency head catches it.

## What was tried and not shipped

| Experiment | Result | Verdict |
|---|---|---|
| Second detector (upstream AASIST) as an extra clone-head input, `fit_fusion.py --alt` | Accuracy 89.2 → 91.0 %, but balanced accuracy 89.7 → 90.1 %, EER 10.0 → 10.4 %, genuine accepted 90.8 → 88.3 %, +≈ 115 ms/call | It moves the operating point, it does not separate classes better. **Not shipped** (code and tests kept: `clone_signals: 2`). |
| Stronger / weaker regularisation of the heads (C = 0.05 … 3) | 87.6 – 88.1 % | Fusion is at its plateau. |
| Flat "spoof OR speaker" threshold rule | 83 – 87 % | Worse than channel-aware heads. |

Note on the metric: the trial mix is 25 % genuine / 75 % attack, so plain
accuracy rewards leaning toward "flag". Compare **balanced accuracy** and **EER**
across experiments, as above.

## Where the next points of accuracy must come from

Remaining errors (eval, out-of-fold): clone recall is lowest on A17 (76 %), A12
(83 %), A14 (86 %) — and only 50–77 % of those even on *clean* audio; missed
impostors have a mean speaker risk of 0.63 (caught: 0.87); genuine false flags are
almost all background-voice calls and cluster on a few speakers.

1. **A stronger detector (largest lever).** Train SSL-AASIST (wav2vec2/XLS-R front
   end; already coded and unit-tested) with the same noise augmentation, on a
   GPU with more than the 6 GB laptop card, and add a *modern* TTS / voice-conversion
   training and test set. Self-supervised features are the published route to
   generalising to unseen vocoders (A12/A17-type attacks).
2. **A stronger speaker model.** The WavLM-base x-vector is the weak head
   (impostor recall 84 %). Try ECAPA-TDNN / WavLM-large, and enrol from more or
   cleaner audio; add speech-only gating (VAD) before embedding so babble does
   not enter the embedding.
3. **Speech enhancement / source separation** in front of both models for
   background-voice calls (the dominant failure mode).
4. **Real data.** Replace synthetic mixing with recorded phone calls (consented),
   incl. Indian-language speech, so thresholds are fitted on the real channel.
5. **More speakers.** 58 is enough to compare options, not to quote a third
   decimal; 200+ would tighten every number here.

## Performance (latency)

One assessment of 10 s of audio (RTX 4050 laptop GPU, warm):

| | before | now |
|---|--:|--:|
| spoof read (front-end + sliding windows) | 143 ms | 117 ms |
| speaker score | 71 ms | 36 ms |
| prosody | 166 ms | off (not a decision input) |
| channel features | – | 3 ms |
| **full `assess()`** | **396 ms** | **159 ms** |

Prosody is now computed only when it can affect the decision (hand-set fallback
fusion) or on request (`serve_demo.py --prosody`, `serve_grpc.py --prosody`).
The test suite also runs 2× faster (86 s → 39 s).
