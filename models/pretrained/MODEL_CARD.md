# Model Card: VoiceGuard synthetic-voice detector (AASIST, published weights)

_Added 2026-09-25, against `docs/project_status.md` item j. Format follows
Mitchell et al., "Model Cards for Model Reporting" (FAT* 2019), scoped to
what this project actually measured -- every number below is reproducible
from `results/tables/` and traceable to a git commit._

## Model details

- **What it is.** AASIST (Audio Anti-Spoofing using Integrated
  Spectro-Temporal Graph Attention Networks; Jung et al., ICASSP 2022),
  **published upstream weights** (`clovaai/aasist`, `AASIST.pth`,
  0.30M params), run through this project's own anti-shortcut audio
  front-end (`voiceguard.audio`, `docs/evaluation_protocol.md`). This
  project did not train these weights -- see "Training data" below.
- **Where it sits in the product.** This is one of four fused signals
  (synthetic-voice, speaker consistency, prosody/behaviour, call context;
  `voiceguard.risk`). The numbers in this card describe the acoustic
  signal **standalone**; the system-level decision (`ALLOW`/`VERIFY`/
  `ESCALATE`) also depends on the other three signals and the call
  scenario. Deploying this score alone, without fusion, is not how this
  project recommends using it.
- **Version / provenance.** Every result below carries a front-end
  fingerprint and a git commit in its source CSV (`results/tables/`), per
  this project's provenance-stamping convention -- there is no
  unstamped number anywhere in this card.
- **License.** The vendored AASIST code (`src/voiceguard/models/_vendor/`)
  and its published weights carry their own upstream terms (see
  `src/voiceguard/models/_vendor/__init__.py`) and are not owned by this
  project. **This repository does not yet have its own top-level LICENSE
  file** -- an open item, not a decision this card makes.

## Intended use

- **Primary intended use.** A decision-support signal inside a live-call
  fraud/social-engineering defense system, feeding a fused risk score that
  a human operator (or a policy engine) acts on -- not an automated,
  unreviewed block/allow gate.
- **Primary intended users.** A contact-centre or high-risk-call operator
  console, as built in `voiceguard.serve`; developers integrating the
  session REST/WS/gRPC API (`docs/grpc.md`).
- **Out-of-scope uses.** Fully automated call-blocking without human
  review; any use as a standalone/sole signal (see "sits in the product"
  above); any legal, biometric-identity, or forensic determination of who
  a speaker is (this is a real/synthetic classifier and a same/different
  speaker check, not an identity system); use on audio outside the 16 kHz
  telephony-like domain this was evaluated on without re-evaluation.

## Factors

Measured factors affecting performance (all in "Quantitative analyses"
below):

- **Known vs. unseen attack generator** -- the single largest factor.
  Near-perfect on generators close to AASIST's original training
  distribution (A08/A09: <2% EER), and it fails on unfamiliar neural
  TTS/voice-conversion systems (A10/A12: 44-52% EER) -- within one
  in-domain test partition.
- **Known vs. cross-dataset audio** -- eval-partition unseen attacks
  (23.3% EER) vs. a fully independent corpus, In-the-Wild (35.8% EER):
  performance degrades further off the ASVspoof distribution entirely.
- **Modern commercial TTS specifically** -- the worked example in
  `README.md` shows a modern cloned-voice clip scored "real" (risk 29/100)
  by this signal alone; this is the central motivating failure case for
  fusing in the other three signals rather than shipping this signal
  alone.
- **Not measured:** performance by speaker demographic (age, gender,
  accent), by language other than the IndicCall-Eval synthetic pilot (see
  below), or under an adversarial attacker aware of the fusion logic
  (`docs/project_status.md`, "not started" item).

## Metrics

- **EER** (Equal Error Rate) -- headline metric, ASVspoof-community
  standard, at the threshold where false-accept = false-reject.
- **min-tDCF** -- deployment-oriented tandem cost, computed with the
  ASVspoof 2019 organizers' own reference implementation and their fixed
  baseline ASV system (`docs/tdcf.md`), not accuracy (misleading at this
  corpus's ~9:1 bona fide:spoof class balance).
- **Silence-only ablation EER** -- this project's own permanent honesty
  gate (`docs/evaluation_protocol.md`): score non-speech audio alone;
  chance is 50%, below 40% means the model learned a corpus artifact, not
  synthesis. Every number in this card passed that gate.

## Evaluation data

- **ASVspoof 2019 LA** (Wang et al.) -- dev (known-attack) and eval
  (unseen-attack) partitions, official protocol splits, speaker- and
  attack-disjoint from training by construction.
- **In-the-Wild** (Muller et al., Interspeech 2022) -- fully independent
  corpus, used **only** for a final cross-dataset check, never for model
  selection or threshold tuning (`docs/evaluation_protocol.md`).
- **IndicCall-Eval pilot** -- this project's own small, synthetic-only,
  5-language probe (`docs/indic_eval.md`); recall against a fixed
  threshold, not a true EER (no bona fide multilingual speech collected
  yet).
- ASVspoof 2021 DF is **not** part of this card's evaluation -- the corpus
  requires a registration step not yet completed (`docs/tdcf.md`).

## Training data

**None -- these are the published upstream weights, unmodified.** This
project's own training loop (`scripts/train_cm.py`, RawBoost + telephony
codec augmentation, resumable) is implemented and unit-tested but has not
been run to convergence (local GPU is a 6 GB laptop part; a real run is
30-40 epochs over days). See `docs/project_status.md` item c. The
upstream weights' own training data is ASVspoof 2019 LA train, per Jung
et al. -- this project did not curate, augment, or filter it.

## Quantitative analyses

**EER** (`results/tables/cm_aasist-pretrained_in_domain.csv`,
`results/tables/cm_aasist-pretrained_cross_dataset.csv`):

| Split | EER |
|---|--:|
| Dev (known attacks) | 9.6% |
| Eval (unseen attacks) | 23.3% |
| In-the-Wild (cross-dataset) | 35.8% |
| *Gen-1 baseline, for scale (handcrafted + GBM)* | *7.0% / 33.4% / 58.2%* |

**min-tDCF** (`results/tables/cm_aasist-pretrained_tdcf.csv`,
`docs/tdcf.md`):

| Split | min-tDCF |
|---|--:|
| Dev | 0.288 |
| Eval | 0.588 |

**Per-attack EER, eval partition** (unseen attacks; full table in
`paper/voiceguard_ieee.tex`, Table II): near-perfect on A08 (1.6%) / A09
(0.7%); worst on A10 (52.1%) / A12 (43.7%). Range across 13 attack types:
0.7% to 52.1% -- a single aggregate EER hides this spread.

**Silence-only ablation** (`results/tables/cm_aasist-pretrained_silence_ablation.csv`):
41.25% EER on non-speech audio alone (chance = 50%, gate fails below
40%) -- passes. Before the front-end fix, the equivalent test on an
earlier pipeline reached 13-15% EER (i.e. a shortcut, not detection);
that fix and this permanent gate are this project's single most
consequential finding (`docs/evaluation_protocol.md`).

**Calibration** (`docs/calibration.md`, `results/tables/cm_aasist-pretrained_calibration.csv`):
measured, not assumed. A Platt fit was tried and **rejected** for this
signal: it improved calibration error on the data it was fit on but
*worsened* it on unseen attacks and cross-dataset audio (a textbook
overfitting signature) -- the pre-existing ad-hoc logit-shift ships
instead, as the more robust-under-shift of the two, not because it is
"real" calibration.

## Ethical considerations

- **Dual use.** A system that scores "how synthetic does this voice
  sound" is dual-use by nature -- the same signal that flags a cloned
  voice in a fraud attempt could be misapplied to surveil or challenge a
  genuine speaker (e.g. someone with a voice disorder, a non-native
  accent, or a low-quality connection producing an atypical acoustic
  signature). This is why the product design routes the score to a human
  operator with reason codes and a recommendation, not an automated block
  (`voiceguard.risk.policy`).
- **Privacy.** The audit log this signal feeds is feature-only by
  construction -- `voiceguard.audit.AuditLog` rejects any key that would
  carry audio, a transcript, or an embedding, enforced in code, not just
  policy (`voiceguard.audit`, `_FORBIDDEN` keys).
- **Consent.** Evaluation audio is drawn from ASVspoof 2019 LA and
  In-the-Wild, both established research corpora with their own
  publication consent processes; this project collected no new human
  speech for evaluating this signal specifically (IndicCall-Eval's pilot
  is synthetic-only for the same reason -- see `docs/indic_eval.md`).
- **Disparate performance.** Language coverage beyond English is
  evaluated only via a small synthetic pilot (IndicCall-Eval); real
  disparate-impact analysis across languages, accents, or channel quality
  has not been done and should be treated as unknown, not assumed benign.

## Caveats and recommendations

- **35.8% cross-dataset EER is not a deployable figure on its own** --
  this is the central, explicitly stated limitation across this project's
  docs (`README.md` §5, `docs/project_status.md` item c). Treat this
  signal as one input to a fused decision, never as a sole gate.
- **Do not retune thresholds against In-the-Wild.** It is held out by
  protocol specifically so it stays a trustworthy final check
  (`docs/evaluation_protocol.md`); tuning against it would silently
  reintroduce the kind of leakage the silence-ablation gate exists to
  catch in a different form.
- **Re-run the silence-ablation gate after any front-end or model
  change.** It is cheap and it is the one check that would have caught
  this project's own biggest measurement error early.
- **A trained-by-us model would change several numbers in this card** --
  see `docs/project_status.md` item c. Until that happens, every number
  here describes upstream weights evaluated honestly, not a component
  this project built from scratch.
