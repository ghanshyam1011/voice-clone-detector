"""min-tDCF: the vendored ASVspoof 2019 reference implementation, our
wrapper around it, and (where the organizers' ASV score files are present)
an end-to-end check against real data."""

from pathlib import Path

import numpy as np
import pytest

from voiceguard.eval.metrics import compute_eer
from voiceguard.eval.tdcf import DEFAULT_COST_MODEL, compute_min_tdcf, load_asv_scores

_ASV_DIR = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "raw"
    / "LA"
    / "LA"
    / "ASVspoof2019_LA_asv_scores"
)
_DEV_ASV = _ASV_DIR / "ASVspoof2019.LA.asv.dev.gi.trl.scores.txt"
needs_asv_scores = pytest.mark.skipif(
    not _DEV_ASV.exists(), reason="ASVspoof2019 LA organizers' ASV score files absent"
)
_CM_CKPT = Path(__file__).resolve().parents[1] / "models" / "pretrained" / "AASIST.pth"
needs_weights = pytest.mark.skipif(not _CM_CKPT.exists(), reason="pretrained AASIST weights absent")


def _synthetic_asv_trials(seed=0, n=2000):
    rng = np.random.default_rng(seed)
    tar = rng.normal(1.0, 0.3, n)
    non = rng.normal(-1.0, 0.3, n)
    spoof = rng.normal(0.0, 0.5, n)
    return tar, non, spoof


def test_cost_model_priors_sum_to_one():
    m = DEFAULT_COST_MODEL
    assert m["Ptar"] + m["Pnon"] + m["Pspoof"] == pytest.approx(1.0)


def _perfectly_separable_cm(seed, n=500):
    # near-constant but with real variance -- the vendored code rejects truly
    # constant scores (2 unique values) as "not soft scores, binary decisions"
    rng = np.random.default_rng(seed)
    bonafide_cm = 10.0 + rng.normal(0, 0.01, n)
    spoof_cm = -10.0 + rng.normal(0, 0.01, n)
    return bonafide_cm, spoof_cm


def test_perfect_cm_scores_low_min_tdcf():
    tar, non, spoof = _synthetic_asv_trials()
    bonafide_cm, spoof_cm = _perfectly_separable_cm(seed=10)
    r = compute_min_tdcf(bonafide_cm, spoof_cm, tar, non, spoof)
    assert r.eer_cm == pytest.approx(0.0, abs=1e-6)
    assert 0.0 <= r.min_tdcf < 0.05  # near the floor set by the fixed ASV system's own errors


def test_uninformative_cm_scores_high_min_tdcf():
    rng = np.random.default_rng(1)
    tar, non, spoof = _synthetic_asv_trials(seed=1)
    bonafide_cm = rng.random(500)  # CM score carries no information at all
    spoof_cm = rng.random(500)
    perfect_bona, perfect_spoof = _perfectly_separable_cm(seed=11)
    perfect = compute_min_tdcf(perfect_bona, perfect_spoof, tar, non, spoof)
    useless = compute_min_tdcf(bonafide_cm, spoof_cm, tar, non, spoof)
    assert useless.min_tdcf > perfect.min_tdcf


def test_min_tdcf_eer_cm_matches_our_own_eer_implementation():
    """Cross-check: the vendored EER computation and voiceguard.eval.metrics'
    independent implementation must agree on the same scores."""
    rng = np.random.default_rng(2)
    bonafide_cm = rng.normal(0.6, 0.2, 2000)
    spoof_cm = rng.normal(0.3, 0.2, 2000)
    tar, non, spoof = _synthetic_asv_trials(seed=2)

    r = compute_min_tdcf(bonafide_cm, spoof_cm, tar, non, spoof)
    labels = np.concatenate([np.ones(2000), np.zeros(2000)])
    scores = np.concatenate([bonafide_cm, spoof_cm])
    our_eer, _ = compute_eer(labels, scores)
    assert r.eer_cm == pytest.approx(our_eer, abs=1e-6)


def test_raises_value_error_not_system_exit_on_bad_input():
    tar, non, spoof = _synthetic_asv_trials(seed=3)
    with pytest.raises(ValueError):
        compute_min_tdcf(np.array([np.nan, 1.0]), np.array([0.5, 0.2]), tar, non, spoof)


@needs_asv_scores
def test_load_asv_scores_real_file():
    tar, non, spoof = load_asv_scores(_DEV_ASV)
    assert tar.size > 0 and non.size > 0 and spoof.size > 0
    assert tar.size + non.size + spoof.size == 29548  # ASVspoof2019 LA dev.gi trial count
    assert np.all(np.isfinite(tar)) and np.all(np.isfinite(non)) and np.all(np.isfinite(spoof))


@needs_asv_scores
def test_dev_asv_system_eer_is_plausible():
    """Sanity check on the organizers' own baseline ASV system, independent
    of anything this project built. Uses voiceguard.eval.metrics.compute_eer
    (labels, scores) -- NOT the vendored asvspoof19_tdcf.compute_eer(target,
    nontarget), a different function of the same name; two independent EER
    implementations should still agree (see the cross-check test above)."""
    tar, non, spoof = load_asv_scores(_DEV_ASV)
    labels = np.concatenate([np.ones(tar.size), np.zeros(non.size)])
    scores = np.concatenate([tar, non])
    eer, _ = compute_eer(labels, scores)
    assert 0.0 < eer < 0.2  # the published baseline ASV EER is a few percent


@needs_asv_scores
@needs_weights
def test_end_to_end_min_tdcf_on_real_aasist_scores():
    """The full path: real AASIST CM scores x the organizers' real ASV
    scores -> a plausible min-tDCF, same shape as every other honestly
    measured number in this project."""
    import torch

    from voiceguard.audio import load_wave, preprocess_wave
    from voiceguard.config import load_config, preprocess_config, resolve
    from voiceguard.data import load_manifest
    from voiceguard.models.neural import build_cm

    cfg = load_config()
    pcfg = preprocess_config(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_cm("aasist").to(device)
    model.backbone.load_state_dict(torch.load(_CM_CKPT, map_location=device))
    model.eval()

    man = resolve(cfg, "manifests") / "asvspoof19_la_dev.csv"
    df = load_manifest(man)
    df = (
        df.groupby("label", as_index=False, group_keys=False)[df.columns]
        .apply(lambda g: g.sample(min(60, len(g)), random_state=0))
        .reset_index(drop=True)
    )

    from voiceguard.data.torch_dataset import NB_SAMP, _fix_length

    rng = np.random.default_rng(0)
    scores, labels = [], []
    with torch.no_grad():
        for row in df.itertuples(index=False):
            y = preprocess_wave(load_wave(row.path, pcfg.sample_rate), pcfg)
            y = _fix_length(y, NB_SAMP, rng, random_crop=False)
            w = torch.from_numpy(np.ascontiguousarray(y, np.float32)).unsqueeze(0).to(device)
            scores.append(float(model.score_bonafide(w).item()))
            labels.append(1 if row.label == "bonafide" else 0)
    scores, labels = np.array(scores), np.array(labels)

    tar, non, spoof = load_asv_scores(_DEV_ASV)
    r = compute_min_tdcf(scores[labels == 1], scores[labels == 0], tar, non, spoof)
    assert 0.0 <= r.min_tdcf <= 1.5  # normalized t-DCF; >1 means "worse than no CM at all"
    assert r.n_bonafide_cm + r.n_spoof_cm == len(df)
