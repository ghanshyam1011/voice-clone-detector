# Input quality: what the system does when the audio is not a clean call

_Added 2026-10-22. Found by auditing the shipped session engine with awkward
audio; fixed in `voiceguard.audio.quality` and `SessionManager.assess`._

Every earlier number in this repo was measured on well-formed speech. A product
also receives muted lines, hold music, a recording 60 dB too quiet, a decoder that
emits NaN, a client that forgets to scale int16 samples, and a stream that has
been silent for ten seconds. The question is not "is the detector accurate" but
"what does the system say when there is nothing to judge".

## What it did before (audit of the shipped engine)

Six seconds of each input, enrolled caller, learned fusion, `SessionManager.assess`:

| Input | Before | Why that is wrong |
|---|---|---|
| empty array | **crash** (`RuntimeError`) | – |
| NaN / Inf in the audio | **crash** (`ParameterError`) | one bad sample kills the read |
| 1 sample | **ALLOW**, fused 0.00 | clears a call nobody has heard |
| 50 ms – 1 s of speech | **ESCALATE**, fused 1.00 | a 0.3 s clip tiled to fill the window is not evidence |
| digital silence | **ESCALATE** 1.00 | a muted line raises the loudest alarm |
| white noise, pure tone | **ESCALATE** 1.00 | line hum / DTMF / hold tones raise it too |
| genuine speech + DC offset 0.3 | **ESCALATE** 1.00 | the channel descriptors extrapolate |
| genuine speech scaled ×32767 (int16) | **ESCALATE** 1.00 | same |
| genuine speech 60 dB too quiet | **ESCALATE** 0.93 | the frozen front-end skips loudness normalisation below its −70 LUFS gate, so the detector sees near-silence and returns a constant 0.63 |

The common cause: the learned fusion conditions on channel descriptors and
extrapolates wildly outside the range it was fitted on, and nothing stood between
the transports and the models to say "this is not speech".

## What it does now

`voiceguard.audio.quality.analyze` runs first, inside `assess`, so REST, WebSocket
and gRPC all get it:

1. **Sanitise** – NaN/Inf → 0 (a stream that is mostly NaN is refused as corrupt);
   samples in 16-bit range are rescaled; a DC offset above 0.005 is removed.
   Audio that needs none of this is returned **bit for bit unchanged**, so every
   earlier measurement still describes the system.
2. **Find the speech** – Silero-VAD, a small neural detector (white noise, a tone
   and a muted line contain 0 s of speech; speech 60 dB too quiet is still found).
   Without `silero-vad` installed it falls back to an energy / spectral-flatness /
   syllable-modulation heuristic (reported as `method: "energy"`).
3. **Level** – speech below −50 dBFS is raised to −30 dBFS *before* the frozen
   front-end sees it (the front-end itself is untouched, by team agreement).
4. **Verdict** – `usable`, a plain-English `reason`, `speech_s`, `level_dbfs`,
   `clipped_frac`, `notes` (every fix applied). Returned in every response as
   `quality` (REST/WS JSON, gRPC `InputQuality`).

| Speech found | Result |
|---|---|
| < 0.25 s | `WAITING` – "no speech detected (silence, noise or music only)" |
| 0.25 – 1.0 s | `WAITING` – "only 0.6 s of speech so far (need 1.0 s)" |
| 1.0 – 2.0 s | read, but **provisional**: raises concern (VERIFY), never ESCALATE on its own |
| ≥ 2.0 s (and a full model window) | normal confirmed read |

### `WAITING` – a fourth action

`ALLOW` means "checked and cleared"; `ESCALATE` means "pause the transaction". A
call with nothing to judge is neither, so there is a fourth state:

* `action: "WAITING"`, recommendation "No verdict yet – not enough clear speech to
  judge. Keep the caller talking."
* **Not ALLOW** – the agent is never told a call was cleared that nobody heard.
* **Not an alarm** – no alert card, nothing on the risk timeline, **no audit event**.
* **Context still counts.** An unknown caller asking for ₹75,000 is a fact whether
  or not anyone has spoken: the decision is made on context alone and the reasons
  say so ("No usable speech yet (…): judged on context alone") – VERIFY, never
  ESCALATE (a read with no audio is provisional by definition).
* Session state (smoothing, last action, audit trail) is left exactly as it was, so
  a pause in the call does not blunt the next real read.

### Enrolment is gated too

`SessionManager.enroll` previously stored a voiceprint of *anything*. A voiceprint
of silence or noise silently makes every later speaker check meaningless, so it now
raises unless the clip holds ≥ 3 s of speech (REST 422, gRPC `INVALID_ARGUMENT`).
The response reports `speech_seconds`; `short` means under 4 s of speech (accepted,
but a weak voiceprint; ~6 s of clean speech is best).

### Streams

`PcmStream` decodes streamed float32 PCM: a chunk that ends mid-sample carries the
odd bytes into the next chunk instead of misaligning the stream, NaN/Inf become
silence, and absurd client-declared sample rates are ignored.

## After (same inputs)

| Input | After |
|---|---|
| empty, 1 sample, 50 ms, 0.3 s, 1.0 s of speech | `WAITING`, no crash |
| digital silence, −80 dB noise, white noise, pure tone | `WAITING` |
| speech ×0.001 (−60 dB) | ALLOW, same scores as the clean control (spoof 0.00, speaker 0.43) |
| speech + DC offset | ALLOW, identical to the control (fused 0.06) |
| NaN / Inf samples inside real speech | ALLOW, identical to the control |
| speech ×32767 (int16 range) | ALLOW, identical to the control |
| 10 minutes of speech | ALLOW, 3 s |

Tests: `tests/test_quality.py` (23: sanitising, the segmenter, parity with Silero's
own segmenter, real speech through distortions), `tests/test_session.py` and
`tests/test_grpc.py` (the real gate through REST and gRPC: silence → `WAITING`
and nothing logged, context still counts, enrolling silence refused, NaN audio
never crashes, a garbage gRPC stream survives).

## Cost

Silero is recurrent and is normally driven one 32 ms frame at a time from Python
(~1 ms of overhead per frame: ~0.3 s for a 10 s window). The window is instead cut
into up to 16 independent segments run in lockstep as one batch (~16× fewer
sequential calls); speech seconds agree with Silero's own `get_speech_timestamps`
to within 0.3 s on the demo clips (a test). Cost: the recurrent state restarts at
each cut, which does not matter for "how many seconds of speech are in this window".
Measured on a laptop that was running on battery (CPU pinned at its 2.1 GHz base
clock, about 10× slower than the AC figures quoted in `docs/system_eval.md`), so
treat absolute milliseconds in this document as pessimistic.

## What this does not do

* It is a **gate, not a classifier of fakes**. Speech that passes the gate is judged
  exactly as before; nothing here makes the detector better at synthetic speech.
* Silero is trained on real speech and noise; it can be wrong at the margins
  (whispers, very distorted speech, speech buried in loud music). A false "no speech"
  produces `WAITING` – the safe direction – never ALLOW.
* *Background voices* (babble) are speech, so they pass the gate by design; their
  effect on the detectors is the separate, documented weak spot in
  `docs/system_eval.md`.
* Real hold music has **not** been tested (no recordings here); silence, white
  noise and a pure tone are rejected, and speech with music mixed under it is
  covered by the stress set (`docs/stress_eval.md`).
