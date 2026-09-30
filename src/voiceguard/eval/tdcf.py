"""min-tDCF: the ASVspoof community's deployment-oriented metric, reported
alongside EER (docs/evaluation_protocol.md already names this "added in P1
from the ASVspoof reference implementation").

Unlike EER, min-tDCF scores a countermeasure (CM) in tandem with a fixed
automatic speaker verification (ASV) system, under an explicit cost model
(false-accepting a spoof is costlier than false-rejecting a genuine user).
It is the metric the ASVspoof challenges themselves rank submissions on.

We use the ASVspoof 2019 organizers' own reference implementation
(`voiceguard.eval._vendor.asvspoof19_tdcf`, vendored verbatim) and their
official baseline ASV system's scores, distributed with the corpus at
`data/raw/LA/LA/ASVspoof2019_LA_asv_scores/` -- NOT this project's own
WavLM speaker-consistency signal (`voiceguard.speaker`), which is a
different, product-side component with its own evaluation
(`scripts/calibrate_speaker.py`, `docs/calibration.md`). Using the
organizers' fixed ASV system is what makes a min-tDCF number comparable
across different papers' countermeasures in the first place; swapping in
our own ASV would produce a number nobody else's min-tDCF is comparable to.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from voiceguard.eval._vendor import asvspoof19_tdcf as _ref

# The ASVspoof 2019 challenge's official cost model, from the organizers'
# evaluate_tDCF_asvspoof19.py. Not a tunable knob: changing it would make
# results incomparable with every other ASVspoof 2019 submission, which is
# the entire point of reporting this metric.
_PSPOOF = 0.05
DEFAULT_COST_MODEL = {
    "Pspoof": _PSPOOF,  # prior probability of a spoofing attack
    "Ptar": (1 - _PSPOOF) * 0.99,  # prior probability of a target (genuine) speaker
    "Pnon": (1 - _PSPOOF) * 0.01,  # prior probability of a nontarget (zero-effort impostor)
    "Cmiss_asv": 1,  # cost of ASV falsely rejecting a target speaker
    "Cfa_asv": 10,  # cost of ASV falsely accepting a nontarget speaker
    "Cmiss_cm": 1,  # cost of CM falsely rejecting a target speaker's genuine speech
    "Cfa_cm": 10,  # cost of CM falsely passing a spoof through to ASV
}


@dataclass
class MinTDCFResult:
    min_tdcf: float
    eer_cm: float  # countermeasure EER (matches voiceguard.eval.compute_eer on the same scores)
    eer_asv: float  # the fixed ASV system's own EER, for context
    pfa_asv: float
    pmiss_asv: float
    pmiss_spoof_asv: float
    n_bonafide_cm: int
    n_spoof_cm: int
    n_asv_trials: int
    cost_model: dict


def load_asv_scores(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse an ASVspoof2019 LA organizers' ASV score file:
    ``<source> <target|nontarget|spoof> <score>`` per line, whitespace
    separated (e.g. ``ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.dev.gi.trl.scores.txt``).
    Returns ``(target_scores, nontarget_scores, spoof_scores)``."""
    data = np.genfromtxt(str(path), dtype=str)
    if data.ndim != 2 or data.shape[1] != 3:
        raise ValueError(f"{path}: expected 3 whitespace-separated columns per line")
    keys = data[:, 1]
    scores = data[:, 2].astype(np.float64)
    return scores[keys == "target"], scores[keys == "nontarget"], scores[keys == "spoof"]


def compute_min_tdcf(
    bonafide_cm: np.ndarray,
    spoof_cm: np.ndarray,
    tar_asv: np.ndarray,
    non_asv: np.ndarray,
    spoof_asv: np.ndarray,
    cost_model: dict | None = None,
) -> MinTDCFResult:
    """min-tDCF for a countermeasure (``bonafide_cm``/``spoof_cm``: our CM's
    raw scores on its own eval trials, higher = more bonafide-like) run in
    tandem with a fixed ASV system (``tar_asv``/``non_asv``/``spoof_asv``:
    that ASV system's own scores on its own trial list -- see
    :func:`load_asv_scores`). The ASV system is fixed at its own EER
    operating point, per the ASVspoof 2019 protocol.

    Raises ``ValueError`` (not the vendored code's ``SystemExit``) on
    invalid input -- NaN/Inf scores, scores that look like hard decisions
    rather than soft scores, or a degenerate cost model."""
    cost_model = cost_model or DEFAULT_COST_MODEL
    bonafide_cm = np.asarray(bonafide_cm, dtype=np.float64)
    spoof_cm = np.asarray(spoof_cm, dtype=np.float64)
    tar_asv = np.asarray(tar_asv, dtype=np.float64)
    non_asv = np.asarray(non_asv, dtype=np.float64)
    spoof_asv = np.asarray(spoof_asv, dtype=np.float64)

    eer_asv, asv_threshold = _ref.compute_eer(tar_asv, non_asv)
    eer_cm = _ref.compute_eer(bonafide_cm, spoof_cm)[0]
    pfa_asv, pmiss_asv, pmiss_spoof_asv = _ref.obtain_asv_error_rates(
        tar_asv, non_asv, spoof_asv, asv_threshold
    )
    try:
        tdcf_curve, _ = _ref.compute_tDCF(
            bonafide_cm, spoof_cm, pfa_asv, pmiss_asv, pmiss_spoof_asv, cost_model, False
        )
    except SystemExit as exc:  # the vendored function calls sys.exit() on bad input
        raise ValueError(f"min-tDCF computation rejected the input: {exc}") from exc

    return MinTDCFResult(
        min_tdcf=float(np.min(tdcf_curve)),
        eer_cm=float(eer_cm),
        eer_asv=float(eer_asv),
        pfa_asv=float(pfa_asv),
        pmiss_asv=float(pmiss_asv),
        pmiss_spoof_asv=float(pmiss_spoof_asv),
        n_bonafide_cm=int(bonafide_cm.size),
        n_spoof_cm=int(spoof_cm.size),
        n_asv_trials=int(tar_asv.size + non_asv.size + spoof_asv.size),
        cost_model=dict(cost_model),
    )
