# Real-world audio: a real voice note called fake

Written 2026-10-23, the day it happened. Every number below comes from a file in
`results/tables/` or from a command in this repository; the one real recording is described by its
measurements only (it is a private voice note and is not part of the repository).

## What happened

The project owner played a friend's WhatsApp voice note into the console as the caller. The console
said **Stop: this may be a fake voice**. The note is Opus in an OGG container, 16 kHz, mono,
14.6 s, about 18 kbps.

That answer was wrong, for two different reasons that the old console showed as one.

### Cause 1: the wrong person was on file

The console opens with the *sample customer's* voice on file. A friend's voice does not match it
(speaker similarity 0.40 against 0.94-0.99 for a genuine caller), so "this is not the customer on
file" is the correct finding. Calling that person a **fake voice** was not. Fixed in the console:

- a Stop now says what the evidence says: *Stop: this isn't the customer on file* when the voice
  does not match and the audio is not clearly synthetic, *Stop: this may be a fake voice* when it is;
- the "voice on file" step comes first under **Use your own audio**, with a one-line explanation, and a
  tip appears the first time a call is uploaded while the sample customer is still on file.

### Cause 2: a real false alarm in the "computer-generated voice" check

With the friend's *own* voice on file (their first 6 s, tested on the rest of the note) the clone
check still read **0.98**. The detector itself was barely worried (raw score 0.27, studio speech
0.005). Term by term, the clone head's decision was:

| contribution to the clone log-odds | friend's voice note | studio sample call | cloned sample (older tool) |
|---|--:|--:|--:|
| the detector's score | -0.29 | -2.63 | +1.32 |
| bare channel descriptors (recording style) | **+2.81** | -4.01 | -4.71 |
| detector x channel | -0.04 | -7.74 | +6.40 |
| intercept | +1.39 | +1.39 | +1.39 |
| **log-odds / probability** | **+3.87 / 0.98** | -12.99 / 0.00 | +4.39 / 0.99 |

(`head_terms.py`-style decomposition of the head shipped until 2026-10-23.) For the real note the
alarm came from the bare channel terms: mostly a low loudness-envelope correlation (0.29 against 0.68
for the studio call), then a very clean noise floor (-30 dB against -13 dB) and little high-frequency
energy. That is what noise suppression plus a low-bitrate codec does to a recording.

**Why the head learned that.** In the ASVspoof data every genuine clip is studio audio and the
attacks are vocoder output, so "clean floor, little high-frequency energy, jumpy envelope" separates the
classes there. A head that may alarm on those descriptors alone learns the style of its training data,
not synthesis. The same behaviour made the modern clone tests look better than they were:
`docs/modern_attacks.md` already said the kNN-VC catch came from "the vocoder's clean output, a
property of this vocoder, not of cloning". It was a warning that the shortcut was in the model.

### The same day: the live microphone

The owner's own voice on the live microphone was also called fake. Same two causes, one of them
worse:

1. **The sample customer was still on file**, so a live voice (anyone's but the sample's) can only
   ever be "not the customer". The console now has a one-click **Try it with your own voice**:
   eight seconds of talking put *your* voice on file, then it listens live. The "voice on file" bar
   always says whose voice it is (the sample customer, yours, or a recording), and a tip appears the
   first time a call is played against the sample customer.
2. **The browser's default microphone processing.** `getUserMedia({audio: true})` switches on echo
   cancellation, noise suppression and automatic gain control, which reshape a voice much as a
   messaging app does. The console now asks for the raw microphone (all three off), both for putting a
   voice on file and for listening, so the two match and the audio is nearer what the detector was
   trained on. A pinned test (`tests/test_serve.py`) stops anyone switching them back on unnoticed.

Not verified on the owner's real microphone from here (the stand-in microphone used in testing only
exercises the code path). If a real voice is still flagged, **More detail, Copy these details** copies
the numbers (similarity, detector score, evidence levels, recording descriptors; no audio) so the cause
can be read off instead of guessed.

## The fix: a trust-weighted clone check

`risk/learned.py::_GainHead`: `p = sigmoid(offset + gain(channel) x logit(detector))`, with
`gain = exp(w0 + w . channel)`. The recording style decides **how far to believe the detector** (a noisy
room earns a smaller gain, a clean line a larger one) and can never speak for it. The gain is positive
by construction, so the probability rises with the detector's score for every channel, and a quiet
detector stays quiet. Two unit tests pin that (`tests/test_learned.py`,
`tests/test_fit_fusion.py`): monotone in the detector for any channel, and a clean, band-limited,
noise-suppressed-looking channel cannot call a quiet detector synthetic.

An "interactions only" linear head (no bare channel terms, free-sign products) was tried first and
rejected: its products can still go negative, so it kept a hidden form of the same shortcut (the real
note still read 0.76 with a quiet detector).

### What the fix does to the real note

| setup | before | after |
|---|---|---|
| the wrong person on file (sample customer) | Stop, "fake voice" | Stop: this isn't the customer on file |
| the friend's own voice on file | **Stop**, risk 0.99 | **Double-check**, risk 0.68 |

### What it costs

| clone head | evaluation | balanced | genuine accepted | clones / impostors flagged | genuine ESCALATE | attacks ESCALATE |
|---|---|--:|--:|--:|--:|--:|
| linear (until 2026-10-23) | speaker-held-out (OOF) | 92.9 % | 92.4 % | 95.6 % / 88.9 % | 3.4 % | 85.7 % |
| linear (until 2026-10-23) | situation-held-out | 89.2 % | 83.7 % | 94.6 % / 94.8 % | 9.2 % | 86.5 % |
| linear (until 2026-10-23) | untouched dev | 92.7 % | 93.4 % | 95.7 % / 84.5 % | 2.6 % | 85.4 % |
| linear, products only (rejected) | speaker-held-out (OOF) | 90.5 % | 88.3 % | 93.8 % / 90.7 % | 2.8 % | 55.8 % |
| linear, products only (rejected) | situation-held-out | 87.1 % | 79.2 % | 95.4 % / 94.5 % | 9.2 % | 62.1 % |
| linear, products only (rejected) | untouched dev | 91.9 % | 90.3 % | 97.9 % / 84.7 % | 1.3 % | 74.4 % |
| **trust-weighted (shipped)** | speaker-held-out (OOF) | 90.2 % | 89.0 % | 92.1 % / 89.8 % | 3.1 % | 53.9 % |
| **trust-weighted (shipped)** | situation-held-out | 86.6 % | 79.2 % | 93.7 % / 94.6 % | 8.6 % | 61.0 % |
| **trust-weighted (shipped)** | untouched dev | 91.8 % | 90.5 % | 97.6 % / 83.9 % | 1.3 % | 72.6 % |

Same 4,157 reads (48 eval speakers, 19 situations) and 1,520 untouched dev reads, same 3 % cap on genuine ESCALATE, `results/tables/clone_head_variants.csv`. The unseen-situation row is the one to plan with.

Modern clones (`results/tables/modern_by_attack_*.csv`; single reads of ~10 s calls):

| kNN-VC clone calls, flagged | clean | babble 10 dB | G.711 |
|---|--:|--:|--:|
| linear head, dev (20 calls) | 100 % | 100 % | 100 % |
| linear head, eval (48 calls) | 100 % | 100 % | 100 % |
| **trust-weighted head (shipped)**, dev (20 calls) | 75 % | 100 % | 95 % |
| **trust-weighted head (shipped)**, eval (48 calls) | 71 % | 98 % | 83 % |

The honest reading: the earlier modern-clone numbers were partly the shortcut. kNN-VC clones are still
flagged most of the time, mainly by the speaker check, but no longer reach **Stop** on recording style
alone. The sample "cloned with a free tool" call is one of the calls this head lets through.

## What is not fixed

The answer for the real note is **Double-check**, not "looks like the real customer". The detector
itself reads 0.27-0.39 on this note (studio speech 0.005), above its own operating point (0.239). The
likely contributors, which one recording cannot separate:

- Opus at about 18 kbps (the detector saw Opus only at 8 kbps in training);
- the phone's own noise suppression and automatic gain;
- conversational speech in a language the detector never saw (ASVspoof is English read speech);
- one speaker and one microphone.

The In-the-Wild equal error rate of 36 % (`models/pretrained/MODEL_CARD.md`) is the same weakness
measured at scale. A synthetic-voice detector this weak on real-world audio cannot be a gate.

## What measures it, and what would fix it

1. **Measure it across speakers** (added 2026-10-23): the `opus16` and `opus24` situations
   (`audio/degrade.py::MESSAGING_CONDITIONS`, ffmpeg `libopus` in speech mode) replay the same genuine,
   clone and impostor calls as WhatsApp-style voice notes. Result below.
2. **Real recordings are the honest test.** A folder of real voice notes and calls from several people,
   never used for training or thresholds (the way In-the-Wild is held out), would tell us the false-alarm
   rate on audio that is not simulated. Not collected yet.
3. **Train the detector on this world**: Opus at 16-24 kbps, LibriSpeech (real speech from other
   microphones, downloaded, not yet extracted) as extra genuine audio, and the modern generators as
   spoof audio with one family held out. `finetune_robust.py` already supports the data flags; the Opus
   augmentation is the missing piece. About 3-4 hours of GPU time.

## Testing with a real person yourself

1. Open the console, then **Use your own audio**.
2. **1. Put a real person's voice on file**: upload one of their voice notes (6 s of clear speech or more).
3. **2. Play their call**: upload a *different* recording of the same person.

Playing someone's voice note while the sample customer is on file asks "is this the sample customer?",
and the answer is no.
