"""Vendored reference implementations. Do not edit -- keep faithful to
upstream so results are directly comparable to other ASVspoof submissions.
Linting/formatting is disabled for this directory (see pyproject.toml).

| file                 | source                                                                | licence |
|----------------------|------------------------------------------------------------------------|---------|
| asvspoof19_tdcf.py   | `eval_metrics.py`, the ASVspoof 2019 organizers' official min-tDCF     | none upstream -- see note |
|                      | evaluation script, widely mirrored (e.g. github.com/nesl/asvspoof2019, |         |
|                      | github.com/yzyouzhang/AIR-ASVspoof). Fetched 2026-09-24.                |         |

Note on licence: the upstream file carries no SPDX/LICENSE header in any of
the mirrors it was checked against; it is the challenge organizers' own
evaluation script, distributed for exactly this purpose (scoring spoofing
countermeasures) and reproduced verbatim here for that same purpose. We do
not claim a licence for it we cannot verify -- if that changes, this note
should change with it.

`voiceguard.eval.tdcf` wraps this with our own driver (score-file loading,
default cost model, provenance stamping) rather than vendoring the
organizers' driver script, which uses an `np.float` alias numpy removed in
2.0 and would not run on this project's pinned numpy.
"""
