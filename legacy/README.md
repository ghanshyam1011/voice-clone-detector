# legacy/ — pre-refactor scripts, kept for reference only

**Not imported. Not tested. Not maintained. Not on any current code path.**
Ruff and pytest skip this directory. These files have hard-coded Windows
paths and other issues; do not run them as-is.

Kept because they contain ideas worth carrying into later phases:

| File | Salvage for | Phase |
|---|---|---|
| `streaming.py` | cautionary example — disjoint 3 s chunks, no VAD, no sliding window, no smoothing. The rebuild does the opposite. | P2 |
| `train_wav2vec2.py` | the "also train on your own mic audio so real mic input isn't OOD" idea; the honest holdout split | P1 |
| `calibrate_wav2vec2.py` | the calibration methodology — EER-based threshold on a held-out set, **never tuned on your own voice**, two operating points (balanced / high-security) | P3 |
| `calibrate.py` | mic-recording prompts — a starting point for the consented-enrolment tool | P3 |

The Wav2Vec2 model these targeted (facebook/wav2vec2-base, 27% dev EER) is
cut. P1 starts from an SSL anti-spoofing model + AASIST head + augmentation.

Delete this directory whenever the team is done referencing it.
