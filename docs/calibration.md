# Calibration: what "real calibration on held-out data" actually found

_Added 2026-09-24, against project_status.md item f ("replace the logit-shift
+ tuned speaker/prosody/context operating points with calibration on
held-out data")._

## Why this matters

A ranking metric (EER, AUC) tells you a detector separates two classes. It
says nothing about whether the number it emits — "0.9" — means "90% likely"
in any real sense. Every risk score this system fuses is read by a human,
and by the policy engine's fixed thresholds, as a probability. Before this
work, none of the three learned signals had ever had that measured:

| Signal | What existed before | What it actually was |
|---|---|---|
| Spoof (AASIST) | a logit shift, temperature=2.0, centred at the EER operating point | explicitly documented as "not real calibration (P3)" |
| Prosody (LR) | raw `predict_proba` | never checked for calibration quality at all |
| Speaker (WavLM) | linear interpolation between two hand-picked cosine thresholds (0.88 / 0.55) | "a documented operating point, not calibrated" |

`voiceguard.eval.calibration` (9 unit tests) now provides real machinery for
this: Platt and isotonic fitting, plus Expected Calibration Error (ECE) and
Brier score to actually measure the result — on data the fit never saw.

## Finding 1 (the important one): naive calibration does not generalize

`scripts/calibrate_cm.py` fit both Platt and isotonic calibrators for the
spoof signal on ASVspoof19 LA dev (held out from AASIST's own training) and
measured ECE/Brier on dev itself, on eval (unseen attacks), and on
In-the-Wild (cross-dataset) — the same three-way split every other result
in this project reports.

| Split | ECE before | ECE Platt | ECE isotonic |
|---|--:|--:|--:|
| dev (fit set) | 0.074 | **0.012** | **~0.000** |
| eval (unseen attacks) | 0.105 | 0.116 | 0.114 |
| In-the-Wild (cross-dataset) | 0.136 | 0.191 | 0.195 |

Both methods look dramatically better than the old ad-hoc logit-shift **on
the data they were fit on** — isotonic's dev ECE is essentially zero, a
textbook overfitting signature. On eval and In-the-Wild, both methods are
**worse** than doing nothing. This isn't one method failing to a quirk of
its shape; Platt and isotonic fail in the same direction, by similar
margins, which points at the real cause: the model's raw-score distribution
shifts between known and unseen attacks (the same generalization gap the
rest of this project measures everywhere else), so a mapping fit on the
known-attack distribution actively mis-corrects on the unseen one.

**Decision: no spoof calibrator ships.** The ad-hoc logit-shift stays the
default — it is not "real calibration," but it is measurably more robust
across the distribution shift than a naive fit is. Reproduce with
`python scripts/calibrate_cm.py --model aasist --pretrained` (report-only
by default; `--ship platt|isotonic` would activate one, and is not
recommended by this data).

## Finding 2: prosody — same pattern, smaller

`scripts/calibrate_prosody.py` fit Platt and isotonic on dev
(1600 clips), checked on eval (1600 clips, never seen by the LR or the
calibrator).

| Split | ECE before | ECE Platt | ECE isotonic |
|---|--:|--:|--:|
| dev (fit set) | 0.069 | 0.056 | **~0.000** |
| eval (unseen attacks) | 0.069 | 0.074 | 0.101 |

Isotonic shows the identical overfitting signature as the spoof signal
(dev ECE collapses to ~0, eval gets clearly worse — +46% relative). Platt
is closer to a wash: eval ECE ticks up slightly (0.069→0.074) while eval
Brier ticks down slightly (0.220→0.217) — small, inconsistent movement at
n=1600, not a real improvement. **Decision: no prosody calibrator ships.**
The raw `predict_proba` output stays as-is. Reproduce with
`python scripts/calibrate_prosody.py`.

## Finding 3: speaker — the one that actually worked

`scripts/calibrate_speaker.py` built genuine/impostor trials from ASVspoof's
real speaker IDs (bona fide clips only — spoof clips don't have a stable
"real speaker" the same way), pooling train+dev (40 speakers, 480 clips),
and split by **speaker** (28 fit / 12 held-out, no speaker in both) so the
held-out number is a genuine generalization check, not just a fresh sample
from the same speakers.

| Split | ECE before | ECE Platt | ECE isotonic |
|---|--:|--:|--:|
| fit speakers (n=512 trials) | 0.326 | 0.166 | 0.000 |
| held-out speakers (n=184 trials, never fit on) | 0.389 | **0.181** | 0.077 |

Unlike spoof and prosody, this **generalizes**: both methods are a large,
real improvement on speakers the calibrator never saw — Platt cuts held-out
ECE from 0.389 to 0.181 (more than half), isotonic cuts it further to
0.077. The old method (linear interpolation between two hand-picked cosine
thresholds, 0.88/0.55) was never fit to any data at all, so it losing to
almost anything fit on real trials is expected; what matters is that the
fitted mapping holds up on speakers it never saw.

**Decision: ship Platt, not isotonic**, despite isotonic's better raw
number. Isotonic's fit-set ECE is suspiciously close to zero — the same
overfitting signature seen in the spoof/prosody findings above — and its
fit→held-out gap (0.000→0.077) is the full size of its held-out error,
i.e. none of its apparent quality is confirmed stable. Platt's gap
(0.166→0.181) is much smaller, its held-out number is a more trustworthy
estimate of real generalization, and simple logistic-regression
calibration of verification scores is the standard approach in the speaker
verification field (not a preference invented for this project). At n=184
held-out trials (46 genuine), isotonic's extra flexibility is not something
this sample size can confirm; Platt's is. Reproduce with
`python scripts/calibrate_speaker.py --ship platt`; `SpeakerVerifier` now
loads it automatically from `models/speaker/speaker_calibration.json`.

## What this means for the product

One of three signals now has real, held-out-verified calibration: the
speaker-consistency check went from an uncalibrated hand-picked
interpolation to a Platt mapping that measurably generalizes to unseen
speakers. The other two do not ship a fitted calibrator, and that is not a
gap papered over — it's the honest result of actually measuring it, which
had never been done before this work.

The headline for spoof and prosody is not "we couldn't calibrate them," it's
**calibration inherits the same generalization problem as detection, and
naively fitting one on the data available today can make things worse in
exactly the cases that matter (unseen attacks, cross-dataset audio) — so it
has to be measured before it's trusted, not assumed from a good dev-set
number.** That pattern showing up identically in two independent signals,
while the third (a differently-shaped problem — stable verification
geometry, not an adversarial generator distribution) calibrates cleanly, is
itself informative: it's more evidence for the project's central thesis
that the failure mode here is specifically about generalizing to *unseen
generators*, not a generic "these models are just uncalibrated" story.
Consistent with the silence-shortcut result in
`docs/evaluation_protocol.md`: a number that looks good on the split you
fit on is not the number that matters.

## What would actually fix this (not built here)

- **More, and more diverse, fitting data.** A calibrator fit across dev
  *and* eval (still never touching In-the-Wild, per the protocol's rule
  that ITW is never used to pick anything) would at least see both known-
  and unseen-attack score distributions. Untested here — would need a
  documented protocol change, not a unilateral one.
- **Per-generator or domain-aware calibration** — recognizing that "the
  score distribution shifts with the generator" and calibrating
  conditionally, rather than with one global mapping. Real complexity, not
  attempted here.
- **Calibrating against a deployment-representative base rate.** All of the
  above uses the evaluation protocol's ~1:1 class balance; a real deployment
  sees far fewer spoofed calls than genuine ones. Recalibrating for that
  needs a labeled sample drawn from an actual deployment, which does not
  exist yet — flagged, not solved.
