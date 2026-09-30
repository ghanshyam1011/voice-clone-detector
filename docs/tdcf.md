# min-tDCF, and the ASVspoof 2021 DF gap

`docs/evaluation_protocol.md` (frozen) already named this as planned: "min
t-DCF / a-DCF -- added in P1 from the ASVspoof reference implementation."
This is that addition. It does not modify the frozen protocol file.

## Why EER alone isn't enough

EER treats a false accept (a spoof let through) and a false reject (a
genuine user blocked) as equally bad, and ignores how likely each is to
happen in practice. min-tDCF is the ASVspoof challenges' own metric:
it scores a countermeasure (CM) in **tandem with a fixed automatic speaker
verification (ASV) system**, under an explicit cost model where letting a
spoof through costs more than annoying a genuine caller. It's the number
the community actually ranks submissions on, and the number a reviewer
familiar with this field will look for first.

## What we used

- **The organizers' own reference implementation**
  (`voiceguard.eval._vendor.asvspoof19_tdcf`), vendored verbatim from
  `eval_metrics.py`, the official ASVspoof 2019 evaluation script (see
  `_vendor/__init__.py` for exact provenance). Not reimplemented from a
  paper's equations -- this metric has enough subtlety (a two-system tandem
  cost, a fixed ASV operating point, sign-sensitive cost weights) that using
  the community's own tested code is the only way to be confident the
  number means what everyone else's min-tDCF means.
- **The organizers' own baseline ASV system's scores**
  (`data/raw/LA/LA/ASVspoof2019_LA_asv_scores/`), distributed with the
  ASVspoof 2019 LA corpus alongside the CM protocol files -- not this
  project's WavLM speaker-consistency signal. Swapping in our own ASV would
  produce a number that isn't comparable to anyone else's min-tDCF, which
  defeats the purpose of reporting a standardized metric at all. Our
  speaker-consistency work is evaluated on its own terms in
  `docs/calibration.md`.
- Cost model: the challenge's fixed defaults (`Pspoof=0.05`,
  `Ptar=(1-Pspoof)*0.99`, `Pnon=(1-Pspoof)*0.01`, `Cmiss_asv=Cmiss_cm=1`,
  `Cfa_asv=Cfa_cm=10`) -- not a tunable knob; changing it would make results
  incomparable with every other ASVspoof 2019 submission.

Run it: `python scripts/eval_tdcf.py --model aasist --pretrained` ->
`results/tables/cm_<run>_tdcf.csv`, provenance-stamped like every other
result in this project (front-end fingerprint, git commit, seed).

## The ASVspoof 2021 DF gap -- not built, and why

`docs/project_status.md` item h also asks for ASVspoof 2021 DF as a second
held-out cross-dataset test, alongside In-the-Wild. **This is not done.**
The corpus is not present on this machine (`data/` has no `*2021*` path),
and unlike ASVspoof 2019 LA or In-the-Wild, obtaining it requires a
registration step with the organizers before download (it's distributed
through Zenodo with an access request, and the audio set alone is
tens of GB). Writing manifest-loading code against a protocol format (the keys package
centers on a `trial_metadata.txt` file, confirmed by name but not by exact
column layout -- and it's a different, richer format than 2019's simpler
protocol files, since DF adds compression/codec metadata) without the
actual files to test it against would risk exactly the kind of untested,
unverified code this project has otherwise avoided -- so it wasn't written
blind.

**What's needed to close this:** download the ASVspoof 2021 DF evaluation
set + keys (registration required, see the ASVspoof 2021 organizers'
resources), place it under `data/raw/DF2021/` (or wherever's convenient --
nothing is hard-coded yet), and this becomes a same-shaped follow-up to
`scripts/eval_cm.py`'s existing cross-dataset scoring path once the
protocol format is confirmed against real files.
