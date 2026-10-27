"""The fusion fit is the last step before a decision rule ships, so its plumbing is
tested on synthetic trials where the right answer is known: clones carry a high spoof
score, impostors a high speaker risk, and a noisy channel inflates *both* classes'
scores (so a channel-blind threshold fails and the channel-aware heads must not)."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from voiceguard.risk.learned import CHANNEL, LearnedFusion

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "fit_fusion.py"
_spec = importlib.util.spec_from_file_location("fit_fusion", _PATH)
ff = importlib.util.module_from_spec(_spec)
sys.modules["fit_fusion"] = ff
_spec.loader.exec_module(ff)


def _trials(n_speakers=24, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_speakers):
        for kind in ("genuine", "genuine", "clone", "clone", "impostor", "impostor"):
            for cond, noisy in (("clean", 0.0), ("babble5", 1.0)):
                spoof = {"genuine": 0.05, "clone": 0.7, "impostor": 0.05}[kind]
                spk = {"genuine": 0.35, "clone": 0.4, "impostor": 0.85}[kind]
                spoof = np.clip(spoof + 0.25 * noisy + rng.normal(0, 0.06), 0.001, 0.999)
                spk = np.clip(spk + 0.1 * noisy + rng.normal(0, 0.05), 0.001, 0.999)
                rows.append(
                    dict(
                        speaker=f"LA_{s:04d}",
                        kind=kind,
                        cond=cond,
                        spoof=spoof,
                        speaker_risk=spk,
                        prosody=0.5,
                        dyn_db=45 - 20 * noisy + rng.normal(0, 1),
                        floor_db=-60 + 25 * noisy + rng.normal(0, 1),
                        level_db=-30 + rng.normal(0, 1),
                        hf_ratio=0.01,
                        flatness=0.005 + 0.05 * noisy,
                    )
                )
    d = pd.DataFrame(rows)
    d["floor_rel_db"] = d["floor_db"] - d["level_db"]
    return d


def test_fit_produces_a_valid_loadable_model_with_channel_range():
    d = _trials()
    spec = ff._fit_all(d, C=0.3, fpr_cap=0.05)
    assert spec["channel"] == list(CHANNEL)
    assert set(spec["channel_range"]) == set(CHANNEL)
    for lo, hi in spec["channel_range"].values():
        assert lo <= hi
    lf = LearnedFusion(json.loads(json.dumps(spec)))  # survives a JSON round trip
    assert 0.0 < lf.t_warn < lf.t_esc <= 0.999


def test_a_fit_can_condition_on_extra_descriptors_and_the_model_remembers_which():
    d = _trials()
    rng = np.random.default_rng(5)
    d["crest_db"] = 16 - 8 * (d.cond == "babble5") + rng.normal(0, 1, len(d))
    names = (*CHANNEL, "crest_db")
    spec = ff._fit_all(d, C=0.3, fpr_cap=0.05, names=names)
    assert spec["channel"] == list(names) and set(spec["channel_range"]) == set(names)
    lf = LearnedFusion(json.loads(json.dumps(spec)))
    assert lf.channel == names
    p = ff._p_attack(spec, d)  # scores through the model's own channel list
    assert (
        np.isfinite(p).all()
        and p[(d.kind != "genuine").values].mean() > p[(d.kind == "genuine").values].mean()
    )


def test_the_fitted_heads_separate_attacks_from_genuine_in_both_channels():
    d = _trials()
    spec = ff._fit_all(d, C=0.3, fpr_cap=0.05)
    p = ff._p_attack(spec, d)
    y = (d.kind != "genuine").values
    flagged = p >= spec["t_warn"]
    for cond in ("clean", "babble5"):  # the noisy channel must not turn into false alarms
        m = (d.cond == cond).values
        assert flagged[m & ~y].mean() < 0.2 and flagged[m & y].mean() > 0.8, cond


def test_a_table_built_before_the_gain_invariant_feature_is_converted_on_load(
    tmp_path, monkeypatch
):
    d = _trials(4).drop(columns=["floor_rel_db"])
    d.to_csv(tmp_path / "system_trials_old_eval.csv", index=False)
    monkeypatch.setattr(ff, "TAB", tmp_path)
    loaded = ff._load("eval", ["_old"])
    assert np.allclose(loaded["floor_rel_db"], loaded["floor_db"] - loaded["level_db"])


def test_exported_boosted_trees_reproduce_scikit_learns_probabilities():
    d = _trials()
    clamp = ff._clamp_of(d, CHANNEL)
    clf, X = ff._fit_gbm(d, "spoof", "clone", clamp, CHANNEL)
    head = ff._export_gbm(clf)
    from voiceguard.risk.learned import _head_from_dict

    walker = _head_from_dict(json.loads(json.dumps(head)))  # through json, as shipped
    rng = np.random.default_rng(0)
    points = np.vstack([X, X[rng.integers(0, len(X), 200)] + rng.normal(0, 0.5, (200, X.shape[1]))])
    ours = np.array([walker.prob(x) for x in points])
    assert np.allclose(ours, clf.predict_proba(points)[:, 1], atol=1e-6)


def test_a_boosted_fit_separates_the_classes_in_both_channels_and_is_monotone_in_the_signal():
    d = _trials()
    spec = ff._fit_all(d, C=0.3, fpr_cap=0.05, names=CHANNEL, head="gbm")
    assert spec["head"] == "gbm" and spec["clone"]["type"] == "gbm"
    lf = LearnedFusion(json.loads(json.dumps(spec)))
    assert 0.0 < lf.t_warn < lf.t_esc <= 0.999
    p = ff._p_attack(spec, d)
    y = (d.kind != "genuine").values
    flagged = p >= spec["t_warn"]
    for cond in ("clean", "babble5"):
        m = (d.cond == cond).values
        assert flagged[m & ~y].mean() < 0.25 and flagged[m & y].mean() > 0.75, cond
    ch = {k: d[k].median() for k in CHANNEL}
    clone_p = [lf.probs(s, 0.4, ch)[1] for s in np.linspace(0.02, 0.98, 25)]
    assert all(b >= a - 1e-12 for a, b in zip(clone_p, clone_p[1:], strict=False))


def test_the_fit_records_its_detector_and_refuses_to_mix_detectors(tmp_path):
    a = _trials(2).assign(detector="aasist (r2)", op_point=0.2)
    assert ff._detector_of(a, a) == {"name": "aasist (r2)", "operating_point": 0.2}
    assert ff._detector_of(_trials(2), None) is None  # tables from before the record
    with pytest.raises(SystemExit, match="different detectors"):
        ff._detector_of(a, a.assign(detector="aasist (r1)"))
    with pytest.raises(SystemExit, match="unrecorded"):
        ff._detector_of(a, _trials(2))  # a recorded table with an unrecorded one


def test_unjudged_reads_are_excluded_from_fitting():
    d = _trials(4)
    d.loc[d.index[:5], "spoof"] = np.nan  # the gate declined these
    kept = ff._answered(d)
    assert len(kept) == len(d) - 5 and kept.spoof.notna().all()


def test_thresholds_respect_the_genuine_escalate_budget():
    rng = np.random.default_rng(1)
    p = np.r_[rng.beta(2, 8, 500), rng.beta(8, 2, 500)]  # genuine low, attacks high
    y = np.r_[np.zeros(500), np.ones(500)].astype(int)
    t_warn, t_esc = ff._thresholds(y, p, fpr_cap=0.03)
    assert t_warn <= t_esc
    assert (p[y == 0] >= t_esc).mean() <= 0.03 + 1e-9


@pytest.mark.parametrize("held", ["reverberant", "distorted"])
def test_situation_groups_name_real_degradations(held):
    from voiceguard.audio.degrade import STRESS_CONDITIONS

    assert set(ff.SITUATION_GROUPS[held]) <= set(STRESS_CONDITIONS)


def test_a_trust_weighted_clone_head_separates_the_classes_and_cannot_alarm_on_the_channel():
    d = _trials()
    spec = ff._fit_all(d, C=0.3, fpr_cap=0.05, clone_form="gain")
    assert spec["clone"]["type"] == "gain"
    lf = LearnedFusion(json.loads(json.dumps(spec)))
    p = ff._p_attack(spec, d)
    y = (d.kind != "genuine").values
    flagged = p >= spec["t_warn"]
    for cond in ("clean", "babble5"):
        m = (d.cond == cond).values
        assert flagged[m & ~y].mean() < 0.25 and flagged[m & y].mean() > 0.75, cond
    # at either extreme of what recordings looked like, more suspicion never lowers the alarm ...
    lo, hi = ({k: getattr(d[k], f)() for k in CHANNEL} for f in ("min", "max"))
    for ch in (lo, hi):
        ps = [lf.probs(s, 0.4, ch)[1] for s in np.linspace(0.02, 0.98, 25)]
        assert all(b >= a - 1e-12 for a, b in zip(ps, ps[1:], strict=False))
        # ... and a quiet detector stays below the alarm line however the recording looks
        assert lf.probs(0.03, 0.4, ch)[1] < 0.5


def test_a_clone_head_without_bare_channel_terms_is_recorded_as_such_and_loads():
    d = _trials()
    spec = ff._fit_all(d, C=0.3, fpr_cap=0.05, clone_form="interactions")
    assert spec["clone"]["main_effects"] is False and spec["impostor"]["main_effects"] is True
    assert len(spec["clone"]["coef"]) == 1 + len(CHANNEL)
    lf = LearnedFusion(json.loads(json.dumps(spec)))
    assert np.isfinite(ff._p_attack(spec, d)).all() and 0.0 < lf.t_warn < lf.t_esc <= 0.999
