# Demo

Three entry points, all run the **published AASIST** model (inference only,
GPU) via `voiceguard.detect`. First run calibrates on a dev sample (~20 s,
cached to `models/pretrained/aasist_demo_calib.json`).

## 1. Score audio files

```bash
python scripts/demo.py clip1.wav clip2.flac
python scripts/demo.py --sample 6            # 6 real + 6 fake random ASVspoof clips
python scripts/demo.py --sample 6 --wild     # ...from In-the-Wild (harder, honest)
```

Prints per file: **VERDICT** (GENUINE / UNCERTAIN / SYNTHETIC), a 0-100 risk
score, and — for clips longer than the 4 s window — a per-window risk
timeline and "crossed synthetic at ~Xs".

## 2. Live microphone

```bash
python scripts/demo_mic.py
```

Rolling ~4 s window, scored every 1 s, EMA-smoothed risk bar that updates in
place (green / yellow / red). Ctrl+C to stop.

Demo flow: talk normally (bar low, green) -> play a cloned/synthetic clip
through the speakers (bar climbs, flips to SYNTHETIC).

## 3. Web UI + operator console

```bash
pip install -e ".[dl,serve]"     # fastapi + uvicorn + websockets (+ torch/transformers from [dl])
python scripts/serve_demo.py      # loads AASIST + WavLM, opens http://127.0.0.1:8000
```

- **`/` — Live-call console** (the SIH demo). Enrol a caller's voice, then
  play or stream the call: synthetic-voice + speaker-consistency + call-context
  fuse into one ALLOW / VERIFY / ESCALATE decision with reason codes and a
  feature-only audit trail. Full script in [`demo_runbook.md`](demo_runbook.md).
- **`/detector` — raw detector**, two tabs:
  - *Analyze a clip* — drag in audio, get the verdict + risk-over-time chart.
  - *Live microphone* — rolling risk meter + device picker.

`--model aasist-l` for the lighter backbone, `--ours` for a local checkpoint,
`--port` / `--no-open` as needed. **Localhost only — not a hardened service.**
Routes: `GET /api/info`, `POST /api/score`, `WS /api/stream`, and the session
layer `POST /api/session`, `.../enroll`, `.../context`, `.../analyze`,
`WS .../stream`, `GET .../audit`.

For a live flip on stage, routing the fake clip into the mic digitally
(e.g. VB-CABLE) is far more reliable than speaker -> mic playback.

## Talking points (be honest)

- On **known synthesis methods** (ASVspoof dev distribution): ~87% correct,
  dev EER 7.5%.
- On **novel / in-the-wild deepfakes**: weaker — ~60%, In-the-Wild EER ~36%.
  This gap *is* the research problem the project targets (see the rebuild
  plan) — and it's already a big step from the handcrafted baseline's ~58%
  In-the-Wild EER.
- The risk score is a **logit-shift** of an overconfident softmax, not real
  calibration (that's P3). Present it as "relative risk", not a probability.
- The 4 s window means the first confident read is ~4 s in; sub-2 s latency
  needs a shorter-context Tier-1 model (P2).
