# Demo

Two entry points, both run the **published AASIST** model (inference only,
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
