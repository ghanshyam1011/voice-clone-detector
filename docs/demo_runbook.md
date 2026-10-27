# SIH26104 demo runbook: the console in three minutes

The story: a bank agent is on a call and has to decide whether the caller is the real
customer. VoiceGuard listens while the caller talks and answers in plain words: **looks like the
real customer**, **double-check this caller**, or **stop**. Lead with that, not with the model
architecture.

The console is written for people who have never heard of voice cloning. Everything on the page is
in plain language, and the technical numbers are folded away under "More detail".

---

## Setup (before you present)

```bash
pip install -e ".[dl,serve,prosody]"   # once
python scripts/pick_demo_clips.py      # once: the sample calls, from the data on disk
python scripts/make_demo_clone.py      # once, optional: the "cloned with a free tool" sample
python scripts/serve_demo.py           # loads the models, warms them up, then serves
```

`serve_demo.py` prints `warmed up in NN s` when it is ready (about 40 s on a laptop GPU). Wait for
it: the first call after a cold start is otherwise slow. Open <http://127.0.0.1:8000>. The page
opens ready to play: the **sample customer's voice is already on file**. No microphone is needed.

Dry-run the whole script once. The calls play in real time, so the whole demo is about three
minutes.

## The script

**0:00 What is this?** Point at the card at the top: "Software can now copy anyone's voice.
Fraudsters phone the bank pretending to be the customer. VoiceGuard listens to the call while it
happens and tells the person who answered whether to trust the caller."

**0:20 A real customer.** Press **Play** on *The real customer calls*. The answer panel fills in as
the call plays: **Looks like the real customer**, risk in the green, four green checks. When the
call ends a small card says *Correct*.

- "It knows this voice because the customer's real voice is on file. It compares every caller with it."

**0:50 A fake voice.** Press **Play** on *A computer-generated voice*. Within a few seconds the answer
turns to **Stop** and the meter goes red. Under "How VoiceGuard decided" two checks are red: the
voice does not match the one on file, and the audio has strong signs of a computer-generated voice.

- "Four checks, one answer, and the reasons are right there."

**1:20 A cloned voice, and the agent acts.** Press **Play** on *Another cloned voice*. Stop appears about
four seconds in. Click **I've seen this** in the *What to do now* box.

- "The agent acknowledges the warning, and that is logged too." (More detail, then "What happened",
  shows the trail; it holds scores and reasons, never audio.)

**2:00 The hard one.** Press **Play** on *A voice cloned with a free tool*. This is the clone made
with an open-source tool from about ten seconds of the customer's speech, and it is the honest
hard case: it sounds close to the real voice.

- Say what the card says: "It can be missed." Whatever the answer is, read the small card at the end
  (*Caught*, *Partly caught* or *Missed*) aloud. On this sample it first warned, then eased off as the call
  went on, and the console says exactly that instead of showing a clean green tick.
- Do not cherry-pick: the clip was made by a fixed rule before anyone looked at the result
  (`scripts/make_demo_clone.py`), and docs/modern_attacks.md has the measured catch rate for this
  kind of clone.

**2:40 Try a real person: yourself.** Press **Start** on *Try it with your own voice*. Say anything for
eight seconds; VoiceGuard puts *your* voice on file, then listens live. Keep talking and it should read
"Looks like the real customer". Then let someone else speak, or play a fake voice near the microphone,
and watch it react.

- The microphone is asked for raw audio (no browser noise suppression or gain control), which is closer
  to what the detector was trained on. Allow the microphone in the address bar when asked.
- To test a friend's voice note instead, open **Use your own audio**: **1.** put their voice on file
  (upload one of their recordings), **2.** play their call (upload a different recording of the same
  person). VoiceGuard compares the caller with the voice on file: a friend's voice played while the
  *sample customer* is on file is correctly answered "This isn't the customer on file".
- If a real voice is flagged, open **More detail** and press **Copy these details**: it copies the
  numbers (no audio) so the cause can be found. Messaging-app voice notes are a known weak spot
  (docs/real_world_audio.md).

**3:00 Close.** "Decision support, not a gatekeeper: a person stays in charge." Open **How it works**
for the plain-English glossary.

---

## If something misbehaves

- **The page says the engine isn't responding.** The server is still loading; wait for
  `warmed up in NN s`.
- **No sample calls.** Run `python scripts/pick_demo_clips.py` (and `make_demo_clone.py`); the page lists
  only the clips that exist on disk.
- **The microphone misbehaves.** Skip it. Everything above works from files; the microphone is an
  extra under *Use your own audio*.
- **A verdict shows LISTENING instead of an answer.** The audio held no usable speech (silence,
  noise, under a second of speech). That is deliberate: it never says "real customer" about a call
  nobody has heard, and never raises an alarm on a muted line.
- **Short clip, amber answer with a red meter.** VoiceGuard never says "Stop" on less than about
  four seconds of audio; the console says so under the meter.

## Honest answers to expect

- **"Does it catch every clone?"** No. A clone made with a free tool from a few seconds of speech was
  flagged in most of our tests but not all, and the closer it sounds to the real voice the more often it
  gets through (docs/modern_attacks.md has the numbers). A fine-tuned or commercial cloning model was not
  tested. That is why it advises a person.
- **"Why did it say a real person was fake?"** Two different reasons, and the console now keeps them
  apart. *The voice does not match the one on file*: the answer says "isn't the customer on file", not
  "fake". *The recording looks processed*: messaging-app voice notes (WhatsApp, for example) are
  compressed and noise-suppressed, and the detector was trained on studio audio. A measured example
  and what was changed are in docs/real_world_audio.md; real-world audio is the biggest open
  accuracy item.
- **"Is 35.8 % In-the-Wild EER good?"** No, and we say so. That is the published AASIST through our
  honest front-end, down from the handcrafted baseline's 58.7 %, through an evaluation pipeline that
  *proves* the number is not inflated by the ASVspoof silence artifact. The fused system (detector,
  voice match, call details, channel awareness) is evaluated separately on calls with background
  voices, phone codecs and 17 other situations; limits are in docs/stress_eval.md.
- **"Is the risk score a probability?"** No. It is a relative score at documented operating points
  (docs/system_eval.md). The speaker signal has a fitted calibrator that holds up on unseen speakers;
  the detector's score was calibrated and measured to get worse on unseen attacks, so it ships no
  calibrator (docs/calibration.md).
- **"Indian languages / gRPC / real alerts?"** gRPC is built and verified against REST
  (docs/grpc.md): same engine, second transport. Indian-language evaluation is a synthetic-attack
  pilot only (docs/indic_eval.md). The alert in the console is simulated: nothing is sent.
