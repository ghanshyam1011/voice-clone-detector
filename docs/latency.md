# Latency: speech-onset VAD + provisional reads

_Added 2026-09-24, against the SIH26104 requirement "stable read within 2 s"
(unmet as of the 2026-09-08 status snapshot)._

## The constraint this works within

AASIST (and the CM backbone generally) takes a **fixed-length input**
(`NB_SAMP` = 64,600 samples = 4.04 s at 16 kHz) — the model's positional
parameters are sized for it. A window shorter than that is tiled/repeated to
fill the input (`CMScorer.score_waveform`, already existed before this
change); a score from a tiled window is a real model output, but it's out
of the distribution the model was trained on and its reliability is
unmeasured. **A true sub-2 s confirmed acoustic read needs a second,
genuinely short-window model — not built here.** What follows closes two
narrower, honest gaps instead.

## 1. Speech-onset detection (`voiceguard.audio.vad`)

`speech_onset_s(wave, sr)` reuses the exact same `librosa.effects.trim` call
as the anti-shortcut front-end's `trim_silence` (docs/evaluation_protocol.md)
— "is this speech" is judged identically everywhere in the codebase, no
second differently-tuned detector. Returns the first sample (in seconds)
where sustained non-silence begins, or `None` if the whole clip so far is
below threshold.

This matters because a call's first few seconds are often dead air (the
caller finds the phone, a greeting plays) that the caller isn't "read" for.
Measuring latency from **speech onset**, not from **call connect**, is the
honest way to report against a "sub-2 s" target.

## 2. Provisional reads, safety-capped

Every streaming/analysis call already tiles short audio to produce *a*
spoof score (pre-existing behaviour). What was missing: that score wasn't
distinguished from a full, real 4.04 s window — so on the raw stream
(`/api/session/{id}/stream`) it was silently excluded from the audit log
while `warming`, but **still visible on the console banner**, including as
a possible ESCALATE with no record of it. `/api/session/{id}/analyze` had
no equivalent gate at all: a short upload could drive a real, audit-logged
ESCALATE off a tiled-audio read.

Now:

- `Signal.provisional: bool` marks a signal still built from a short/tiled
  window (`session.py` sets this from the existing `warming` flag; the
  `/analyze` route now computes it too, from clip length vs. `NB_SAMP`).
- `fuse()` sets `FusedRisk.has_provisional` if any *present* signal is
  provisional.
- `policy.decide()` caps the action at **VERIFY** when `has_provisional` is
  true and the raw score would have said ESCALATE, with a reason code
  ("Early read — escalation available once the read is confirmed") so the
  operator understands why. It auto-upgrades to ESCALATE the moment the
  window fills, with no separate mechanism needed — the next `assess()`
  call simply isn't provisional anymore.
- Verified live: a 1.5 s clip that would fuse to 0.84 (well over the
  transaction-scenario ESCALATE threshold of 0.65) returns `VERIFY`, not
  `ESCALATE`, with `provisional: true`.

## 3. What the API/console now reports

`assess()`'s response gained three fields: `provisional` (bool),
`speech_onset_s` (float | null), `first_confirmed_read_s` (float | null —
set once a non-provisional read lands). The console shows a line under the
decision banner: an amber "Early read in progress — confirming…" during
the provisional phase, then "Confirmed read N.Ns after the caller started
talking" once it clears.

## 4. What the input gate and the situation descriptors cost (2026-10-23)

Two things were added in front of and beside the models, so their price is stated here
rather than discovered later. Measured on the development laptop (RTX 4050 laptop GPU,
i5-13420H) with the GPU busy with another job at the time, so read these as upper-ish
figures for one 10 s window:

| step | cost per assessment |
|---|--:|
| input-quality gate (`audio/quality.py`: Silero speech check in 16 batched segments, level and clipping checks) | ≈ 120 ms |
| channel descriptors (`audio/channel.py`, ten of them) | ≈ 63 ms (was ≈ 20 ms with five) |
| of which mains-hum prominence / loudness-envelope autocorrelation | 24 ms / 16 ms |
| unchanged: detector windows, speaker model, fusion | as in section 1-2 |

The WebSocket path re-scores every 0.5 s, so the added ≈ 160 ms stays inside one hop on
this machine, but it is no longer a rounding error: on a slower CPU, or with the GPU
shared, these two steps are the first place to look. Both only read audio already in
memory; the hum and envelope descriptors could be computed on a 4 s tail instead of 10 s
if a deployment needs the milliseconds back (not done, and not measured for accuracy).
The Silero check was made cheaper by batching 16 independent segments instead of stepping
frame by frame (about 16x fewer sequential model calls; speech seconds agree with the
library's own timestamps to within 0.1 s, `tests/test_quality.py`).

## What this is *not*

- Not a sub-2 s **confirmed** acoustic read — that's still ~4 s after
  speech onset, honestly reported as such.
- Not edge/on-device inference — a separate, larger item.
- Not a change to the anti-shortcut front-end (`preprocess.py`) itself —
  that file is frozen by team agreement (docs/evaluation_protocol.md) and
  wasn't touched. One related observation, noted for whoever next touches
  it: `trim_silence` doesn't shrink an all-digital-zero clip (a librosa
  quirk — an all-zero signal has no reference level to trim against), so
  `speech_onset_s` guards for that explicitly the way `normalize_loudness`
  already does. Not believed to affect any existing result — real audio is
  never exactly zero — flagged here rather than silently patched into a
  fingerprinted module.
