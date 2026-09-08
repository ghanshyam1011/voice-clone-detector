# SIH26104 demo runbook — the 45-second bank call

The story: a bank agent is on a call. The caller asks for a large transfer.
VoiceGuard runs on the audio and turns three signals — **synthetic-voice
detection**, **speaker consistency**, and **call context** — into one decision
the agent can act on.

Lead with this scenario. Do **not** open with the model architecture or a SHAP plot.

---

## Setup (before you present)

```bash
pip install -e ".[dl,serve,prosody]"   # once
python scripts/train_prosody.py        # once — fits the prosody model (~4 min); or use the tracked one
python scripts/serve_demo.py           # loads AASIST + WavLM + prosody, opens http://127.0.0.1:8000
```

Startup should print `prosody branch active`. The console is at `/`; the raw
detector (file + live mic) is at `/detector`.

## Get the demo clips — use your own voice

Academic ASVspoof clips don't land with judges. Record a real person and clone them.

**1. Record two clips of yourself** (Windows **Voice Recorder**, or Audacity):

| Clip | Say something like | Length |
|---|---|---|
| `enrol` | "Hi, I'm calling about my account — I want to check my recent transactions and confirm my address on file." | ~10 s |
| `benign` | "Can you tell me my current balance, and when my next payment is due?" | ~6 s |

**2. Clone your voice** for the attack line. Free options:
- **ElevenLabs** free tier — Instant Voice Clone from ~1 min of your audio, then generate:
  *"I need to transfer seventy-five thousand rupees to a new beneficiary right now, it's urgent."*
- Or any TTS site (TTSMaker, Play.ht) if you don't need it to sound like *you*.

Save it as `clone.mp3` / `clone.wav`.

**3. Prep all three** — trims, levels, and (for the two call clips) adds a landline codec
so they sound like an actual phone call:

```bash
python scripts/prep_demo_call.py my_enrol.m4a  --as enrol           # keep enrol clean
python scripts/prep_demo_call.py my_benign.m4a --as benign  --phone
python scripts/prep_demo_call.py clone.mp3     --as cloned  --phone --check
```

`--check` prints the detector score — `cloned` must read **SYNTHETIC**. Files land in
`demo_audio/`.

_Quick fallback with no recording:_ `python scripts/pick_demo_clips.py` pulls clips
straight from the ASVspoof data on disk (verified: benign → ALLOW, cloned → ESCALATE).

Dry-run the whole script once, timed.

---

## The script

**0:00 — frame it.** "A bank agent is on a call. The caller wants to move ₹75,000.
Here's what the agent sees."

**0:05 — set the scene.** On the console:
- Scenario → **Payment / transfer on the call**
- Leave "caller's number is in known contacts" **unchecked** later, but for now check it
- Transfer amount → leave `0` for now
- Click **Enrol caller's voice** → it records 6 s → speak, or play `enrol.wav` into the mic.
  Pill turns to **enrolled**.

**0:15 — the benign stretch.** Drop `benign.wav` onto **Call audio**.
- All three signal bars sit low. Decision banner: **ALLOW** — "Proceed. Keep monitoring."
- "So far, nothing unusual. Real voice, matches the enrolled caller, ordinary call."

**0:25 — it turns.** Drop `cloned.wav`.
- **Synthetic voice** and/or **Prosody / behaviour** bars jump, **Speaker mismatch**
  rises (different voice from the enrolled one). Decision → **VERIFY CALLER**.
- "The voice reads as synthetic — both the acoustic model and the prosody check —
  and it doesn't match who enrolled."

**0:32 — context makes it serious.** On the console:
- Uncheck "caller's number is in known contacts"
- Type `75000` in the transfer amount (the last clip re-scores automatically)
- Fused risk climbs, decision → **ESCALATE**, and the red **Alert** panel appears:
  *simulated SMS to supervisor · email to fraud-ops · in-app banner*.
- Read the recommendation aloud: **"Pause approval and verify the caller through the
  registered number."**
- Point at **Why**: four signals — synthetic voice, unnatural prosody, speaker
  mismatch, unrecognised caller + ₹75,000 transfer.

**0:40 — the agent acts.** Click **Acknowledge** on the Alert panel.
- "The agent acknowledges — and that acknowledgement is logged too."

**0:44 — the audit trail.** Point at the **Audit trail** panel — `ALLOW → ESCALATE →
ACKNOWLEDGED`.
- "Every decision and action is logged — scores, reasons, model version, timestamp.
  **No audio, no transcript, no voiceprint.** An investigator gets the decision trail,
  not a recording of the customer."

**0:45 — done.** "Four signals, one decision, a logged action, human stays in control."

---

## If the microphone misbehaves

The enrol step needs the mic. If it fails:
- Use `/detector` → it has a **Microphone** device picker; select the right input, test
  that the "your mic" bar moves, then come back.
- Or skip live enrolment: there's no file-upload enrol in this build, so for a pure
  file demo, **enrol from a played clip** — play `enrol.wav` through the speakers into
  the mic during the 6-second capture. Rehearse this.
- The rest of the demo (benign / cloned / context) is **all file-based** — no mic needed.

## Honest answers to expect from judges

- **"Is 35.8 % In-the-Wild EER good?"** — No, and we say so. It's down from the
  handcrafted baseline's 58.7 %, using published AASIST weights through an evaluation
  pipeline that *proves* the number isn't inflated by the ASVspoof silence artifact —
  something most submissions can't show. Roadmap: SSL frontend + fine-tuning to push
  under 15 %.
- **"Is the risk score a probability?"** — No. It's a relative measure with documented
  operating points. Real calibration is next work.
- **"Speaker check accuracy?"** — WavLM x-vectors; same-speaker cosine ~0.95+, needs a
  clean 6 s enrolment. Not a biometric claim — one signal among four.
- **"Prosody accuracy?"** — a logistic regression on F0/jitter/shimmer/HNR/pause
  features, **26.3 % dev EER on its own** — weak, deliberately: it is a contributing
  signal, and it catches clips the acoustic model misses. It is the noisiest of the
  four; a single reading is not a verdict.
- **"Indian languages / gRPC / real alerts?"** — Indian-language eval and gRPC are
  designed, not built (show the architecture slide). Alerts are simulated in the
  console — no real SMS/email is sent.
