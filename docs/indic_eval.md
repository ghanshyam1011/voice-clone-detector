# IndicCall-Eval — pilot protocol

SIH26104 asks for Indian-language and accent support. The right way to
show it is a **small, consented pilot** measured per language — not a
scraped or repurposed training set presented as telephony realism.

## What this repo produces now

`scripts/make_indic_spoof.py` generates synthetic-speech attacks with
**MMS-TTS (VITS)** in **English, Hindi, Tamil, Bengali, Marathi** — short
bank-call lines in native script, half of each pushed through a random
telephony codec (G.711 / G.722 / AMR-NB / Opus). Output:
`data/indic_eval/spoof/<lang>/`.

`scripts/eval_indic.py` scores every clip with the synthetic-voice
detector and the prosody branch and reports, **per language**:

- **synthetic recall** — fraction of the generated attacks flagged at the
  console's "synthetic" threshold (0.65)
- **mean spoof risk**, **mean prosody score**
- a proper **EER** *only* where real clips are present

```bash
python scripts/make_indic_spoof.py --per-lang 40
python scripts/eval_indic.py
cat results/tables/indic_eval.csv
```

## Pilot result (2026-09-08, upstream AASIST, 40 MMS-TTS clips / language)

| language | synthetic recall | mean spoof risk | mean prosody |
|---|--:|--:|--:|
| Marathi | **85 %** | 78 % | 63 % |
| Hindi | 58 % | 64 % | 55 % |
| Tamil | 55 % | 65 % | 73 % |
| Bengali | 38 % | 58 % | 43 % |
| English | **8 %** | 32 % | 47 % |

The finding worth stating out loud: the detector was trained **only on
English** ASVspoof 2019, yet it flags Indian-language MMS-TTS *better* than
English MMS-TTS (8 % recall on English, 55–85 % on Hindi/Tamil/Marathi).
Non-English synthesis sits further from the English "natural speech" the
model learned, so its artefacts trip the detector more often. Prosody
adds a partly independent signal (highest on Tamil, 73 %).

This is uneven, measured, per-language robustness — not a "supports all
Indian languages" claim. The real-speech slice below is what turns recall
into a defensible EER.

## The real-speech slice (the team records this)

Do **not** download someone else's voice data. Collect a pilot:

1. **3–5 speakers per language**, mix of gender where possible. Each reads
   4–6 short neutral sentences (~30 s total). Consent recorded.
2. Capture in **two channels**: clean phone mic, and speaker-to-mic replay
   (to have a replay-attack sample too).
3. Trim + level with `python scripts/prep_demo_call.py <file> --as <name> --phone`.
4. Save to `data/indic_eval/real/<lang>/` (wav, 16 kHz). `eval_indic.py`
   then reports a real-vs-synthetic **EER** per language.

State clearly in the write-up: **pilot dataset, small n, not telephony-
representative**. Report every metric by language and channel; say
"insufficient data" where a slice is too small.

## Honest framing for the pitch

- The synthetic attacks are **MMS-TTS**, one open TTS family. A production
  claim needs several generators (commercial voice-clone services included)
  and a held-out test partition.
- The detector was trained on **English** ASVspoof 2019. Any cross-lingual
  robustness it shows is transfer, not training — call it that.
- `data/indic_eval/` is git-ignored (audio); regenerate on the demo machine.
