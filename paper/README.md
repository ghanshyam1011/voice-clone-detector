# VoiceGuard — paper + presentation drafts

## Presentation

[`voiceguard_presentation.pptx`](voiceguard_presentation.pptx) (+ a matching
[`.pdf`](voiceguard_presentation.pdf)) is a 14-slide pitch/overview deck for
presenting the project — condensed from the paper below, same numbers, no
paragraph reading required. Built with `pptxgenjs`, exported and every slide
visually inspected via PowerPoint itself before delivery. Edit the `.pptx`
directly in PowerPoint; there's no separate script to maintain unless you
want to regenerate it from scratch.

Slide 1's presenter line ("Ghanshyam Kumavat · Team · Your Institution
Name") is a placeholder — update it before presenting, same as the paper's
author block below.

## Research paper

[`voiceguard_ieee.tex`](voiceguard_ieee.tex) is a full first-draft research
paper in standard IEEE two-column conference format (`IEEEtran`,
`conference` option), built entirely from this repo's own measured
results, docs, and git history. Nothing in it is invented: every number
traces to a file under `results/tables/`, `docs/`, or a command re-run
during drafting (see the "Provenance" section below).

Three files, same content, different jobs:

| File | Use it for |
|---|---|
| [`voiceguard_ieee.tex`](voiceguard_ieee.tex) | the **canonical source** — final edits and the actual conference submission (LaTeX is what nearly every IEEE venue expects) |
| [`voiceguard_ieee.docx`](voiceguard_ieee.docx) | reading/editing in Word, sharing with teammates or a mentor who doesn't use LaTeX, tracked-changes review |
| [`voiceguard_ieee.pdf`](voiceguard_ieee.pdf) | quick read, printing, or attaching somewhere a PDF is expected |

The `.docx` and `.pdf` were built independently of the `.tex` (docx-js +
Word's own PDF export, since this machine has no LaTeX installed) — same
wording and numbers, a plain block-diagram image in place of the `.tex`'s
native TikZ figure, and the equation in Section III-F rendered as text
instead of typeset math. If you edit one file going forward, the other
two will drift — treat `voiceguard_ieee.tex` as the one to keep in sync
for anything you intend to submit.

## How to compile

No LaTeX is installed on this machine, so the easiest path is Overleaf
(free, no install):

1. Go to [overleaf.com](https://overleaf.com) → **New Project** → **Blank
   Project**.
2. Delete the placeholder `main.tex` and upload / paste in
   `voiceguard_ieee.tex`.
3. Click **Recompile**. `tikz`, `booktabs`, `cite`, `hyperref` etc. are
   all present on Overleaf by default — nothing else to install.

If you do install a local TeX distribution (TeX Live or MiKTeX) later,
`pdflatex voiceguard_ieee.tex` (run twice, so cross-references settle) is
enough — the bibliography is a manual `thebibliography` block, so no
`bibtex`/`biber` pass is needed.

## What you must fill in before submitting anywhere

Search the `.tex` file for `TODO` — there are three:

1. **Author block** (`\author{...}` near the top). I filled in
   `Ghanshyam Kumavat` as a best guess from your git commit identity
   (`kumavatghanshyam10@gmail.com`) — confirm the spelling, add every
   teammate, and put in real institution/department names and e-mails.
2. **Double-blind anonymization.** Most IEEE conferences (and all the
   "bigger" ones) review double-blind. The version above is a
   *camera-ready-style* draft with real names — before submitting to a
   double-blind venue, make a second copy with:
   - the author block replaced by placeholder text (check the venue's
     own instructions for the exact expected format),
   - the Acknowledgment section removed or deferred to camera-ready,
   - any identifying link (e.g. this GitHub repo, if you cite it) swapped
     for an anonymized mirror such as an
     [anonymous.4open.science](https://anonymous.4open.science) link.
3. **AI-assistance disclosure wording.** The Acknowledgment section
   currently discloses that an AI assistant helped draft the text and
   search the literature. IEEE's own policy (and most venues' as of
   2024–2026) requires disclosure like this and **prohibits listing an
   AI as an author** — which this draft correctly does not do — but the
   exact required wording/placement differs by venue (some want it in
   the acknowledgments, some in a separate ethics/cover-letter field, a
   few restrict AI use in the review process itself). Check the specific
   call for papers before submitting, every time — this is easy to get
   wrong and can get a paper desk-rejected.

## Provenance — where every number came from

| Table/claim | Source |
|---|---|
| Silence-only ablation (Table II) | `results/tables/baseline_silence_ablation.csv`, `results/tables/cm_aasist-pretrained_silence_ablation.csv` |
| Main detection table (Table III) | `results/tables/baseline_in_domain.csv`, `baseline_cross_dataset.csv`, `cm_aasist-pretrained_in_domain.csv`, `cm_aasist-pretrained_cross_dataset.csv` |
| Per-attack EER (Table IV) | `results/tables/cm_aasist-pretrained_in_domain.csv` |
| Prosody 26.3% dev EER | `models/prosody/prosody_lr.joblib` (`dev_eer` field) |
| Worked example (Table VI) | freshly re-run during drafting: `python examples/rest_client.py demo_audio/enrol.wav demo_audio/benign.wav demo_audio/modern_tts.wav` against the live server |
| IndicCall-Eval (Table VII) | `results/tables/indic_eval.csv` |
| 2-epoch sanity run | `results/tables/cm_aasist_trainlog.csv` |
| Lessons-learned section | this project's own git history / session memory (WavLM `parametrizations` bug, short-clip speaker gate, per-signal EMA, `datasets` library breakage, etc.) |
| Related-work citations | verified live (title/authors/venue) via web search while drafting — not from training-data recall; still worth a final citation-format pass before submission (see below) |

If you re-run any script and the numbers move, update the corresponding
table and prose together — the paper's own honesty rule.

## Strengthening it further

The paper is honest about what's *not* done yet (Section VIII of the
`.tex`, mirroring `docs/project_status.md`). Each of these directly
strengthens a resubmission, roughly in order of impact:

1. **A model you actually trained** beating 35.8% ITW EER (the Kaggle
   run) — this is the single biggest lever; right now every number is a
   re-evaluation of someone else's published weights.
2. **Calibration** of the fusion weights/policy thresholds on held-out
   labeled data, replacing the current "documented operating point"
   framing.
3. **A consented multilingual bona fide slice**, turning Table VII's
   recall numbers into real per-language EERs.
4. **min t-DCF/a-DCF** alongside EER (the field's preferred deployment
   metric), and ASVspoof 2021 DF as a second cross-dataset test.
5. A wider related-work section — 13 references is a solid, fully
   verified core, but a full IEEE paper commonly cites 25–40; add more
   as you read around the calibration/adaptive-attacker work above.

## Where this honestly fits, venue-wise

Read this plainly: the paper's strongest, most defensible contribution
is the **evaluation methodology** (the silence-shortcut finding + the
anti-shortcut front-end + honest generalization numbers) and the
**system design** (multi-signal fusion, policy engine, privacy-preserving
audit). It is *not* a new state-of-the-art detector — 35.8% cross-dataset
EER on a model you didn't train is not a competitive number to lead
with. Framing matters more than venue prestige here.

- **Realistic now, as a first submission:** a workshop paper at the
  **ASVspoof/SASV workshop** (co-located with Interspeech — this
  community specifically values generalization/evaluation-methodology
  findings like the silence-shortcut result), a **regional IEEE
  conference** (IEEE INDICON, TENCON, SPCOM, ANTS, or your local IEEE
  Section's student conference), or post it to **arXiv** first to
  establish priority and get a citable link while you shop it around.
- **A good match once item 1–2 above are done:** **IEEE Access**
  (open-access journal, broad scope, values thorough empirical/systems
  work, realistic turnaround) or an applied security/telephony-fraud
  venue.
- **A real stretch, but not unreasonable for the methodology angle
  alone:** the main **INTERSPEECH** or **ICASSP** tracks, or **IEEE
  Trans. on Information Forensics and Security** — competitive, and
  much more winnable once the trained-model, calibration, and
  adaptive-attacker items are in.

Submitting to several venues in parallel is fine (nothing here is under
review anywhere yet), but tailor the framing and length to each call for
papers rather than sending the identical PDF everywhere.
