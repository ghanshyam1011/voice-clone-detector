import math

import numpy as np
import pytest

from voiceguard.risk.learned import (
    ALLOWED_CHANNEL,
    CHANNEL,
    LearnedFusion,
    combine_with_context,
    head_features,
)

CH = {"dyn_db": 30.0, "floor_rel_db": -27.0, "hf_ratio": 0.1, "flatness": 0.1}
N = 1 + 2 * len(CHANNEL)


def _spec(clone_w=3.0, imp_w=3.0, t_warn=0.5, t_esc=0.8):
    def head(w):
        coef = [0.0] * N
        coef[0] = w  # only logit(signal) matters
        return dict(mean=[0.0] * N, std=[1.0] * N, coef=coef, intercept=-0.0)

    return dict(
        clone=head(clone_w),
        impostor=head(imp_w),
        t_warn=t_warn,
        t_esc=t_esc,
        channel=list(CHANNEL),  # a spec without this is a legacy (gain-dependent) model
    )


def test_head_feature_vector_shape_and_logit():
    x = head_features(0.5, CH)
    assert x.shape == (N,) and x[0] == pytest.approx(0.0)
    assert head_features(0.9, CH)[0] == pytest.approx(math.log(9), rel=1e-6)


def test_either_alarm_raises_attack_probability():
    lf = LearnedFusion(_spec())
    quiet, _, _ = lf.probs(0.05, 0.05, CH)
    clone, pc, _ = lf.probs(0.95, 0.05, CH)
    imp, _, pi = lf.probs(0.05, 0.95, CH)
    assert clone > quiet + 0.5 and imp > quiet + 0.5 and pc > 0.9 and pi > 0.9


def test_missing_signal_contributes_no_alarm():
    lf = LearnedFusion(_spec())
    p, pc, pi = lf.probs(0.95, None, CH)
    assert pi == 0.0 and p == pytest.approx(pc)


def test_risk_map_is_monotonic_and_hits_policy_anchors():
    lf = LearnedFusion(_spec(t_warn=0.4, t_esc=0.7))
    assert lf.risk(0.4) == pytest.approx(0.55) and lf.risk(0.7) == pytest.approx(0.75)
    xs = np.linspace(0, 1, 50)
    ys = [lf.risk(x) for x in xs]
    assert all(b >= a for a, b in zip(ys, ys[1:], strict=False)) and ys[0] == 0 and ys[-1] == 1


def test_context_never_dilutes_and_can_only_add_risk():
    assert combine_with_context(0.6, 0.0) == pytest.approx(0.6)
    assert combine_with_context(0.0, 0.5) == pytest.approx(0.4)
    assert combine_with_context(0.6, 0.5) > 0.6
    assert combine_with_context(1.0, 1.0) == pytest.approx(1.0)


def test_load_returns_none_when_no_model(tmp_path):
    assert LearnedFusion.load(tmp_path / "missing.json") is None


def test_two_detector_clone_head_features_and_requirement():
    assert head_features([0.2, 0.8], CH).shape == (2 + len(CHANNEL) + 2 * len(CHANNEL),)

    n2 = 2 + len(CHANNEL) + 2 * len(CHANNEL)
    spec = _spec()
    spec["clone"] = dict(
        mean=[0.0] * n2, std=[1.0] * n2, coef=[2.0, 2.0] + [0.0] * (n2 - 2), intercept=0.0
    )
    spec["clone_signals"] = 2
    lf = LearnedFusion(spec)
    with pytest.raises(ValueError):
        lf.probs(0.9, 0.1, CH)  # second detector score is mandatory for this model
    hi, lo = lf.probs(0.9, 0.1, CH, spoof_alt=0.9)[1], lf.probs(0.1, 0.1, CH, spoof_alt=0.1)[1]
    assert hi > 0.9 > 0.5 > lo


def _channel_sensitive_spec():
    """A head whose logit moves strongly with dyn_db, so extrapolation is visible."""
    spec = _spec()
    for h in ("clone", "impostor"):
        spec[h]["coef"][1] = 0.05  # dyn_db slope: sigmoid mid-range at the edge of the range
        spec[h]["intercept"] = 0.79
    return spec


def test_channel_features_far_outside_the_fitted_range_are_clamped():
    spec = _channel_sensitive_spec()
    spec["channel_range"] = {k: [CH[k] - 5, CH[k] + 5] for k in CHANNEL}
    clamped, free = (
        LearnedFusion(spec),
        LearnedFusion({k: v for k, v in spec.items() if k != "channel_range"}),
    )
    weird = {**CH, "dyn_db": CH["dyn_db"] + 200.0}  # a channel no training call had
    edge = {**CH, "dyn_db": CH["dyn_db"] + 5.0}
    assert clamped.probs(0.3, 0.3, weird)[0] == pytest.approx(clamped.probs(0.3, 0.3, edge)[0])
    assert (
        free.probs(0.3, 0.3, weird)[0] > free.probs(0.3, 0.3, edge)[0] + 0.1
    )  # runaway without it


def test_a_fusion_model_fitted_on_other_channel_features_is_refused():
    spec = _spec()
    spec["channel"] = ["dyn_db", "floor_db", "hf_ratio", "flatness"]  # the old, gain-dependent set
    with pytest.raises(ValueError, match="refit"):
        LearnedFusion(spec)
    spec["channel"] = list(CHANNEL)
    LearnedFusion(spec)  # the current set is accepted


def test_the_allowed_descriptors_are_exactly_the_gain_invariant_ones_the_audio_module_makes():
    from voiceguard.audio.channel import FEATURE_NAMES

    assert set(ALLOWED_CHANNEL) == set(FEATURE_NAMES) - {"level_db"}
    assert set(CHANNEL) <= set(ALLOWED_CHANNEL)


def test_a_fusion_knows_which_detector_it_was_fitted_for():
    spec = _spec()
    legacy = LearnedFusion(spec)
    assert "older model" in legacy.detector_mismatch("aasist (AASIST_robust)", 0.1316)
    spec["detector"] = {"name": "aasist (AASIST_robust)", "operating_point": 0.1316}
    lf = LearnedFusion(spec)
    assert lf.detector_mismatch("aasist (AASIST_robust)", 0.13160004) is None
    assert "AASIST_robust2" in lf.detector_mismatch("aasist (AASIST_robust2)", 0.1316)
    assert "calibration moved" in lf.detector_mismatch("aasist (AASIST_robust)", 0.14)


def _stump(feature, threshold, lo, hi):
    """x[feature] <= threshold -> leaf ``lo``, else leaf ``hi`` (node 0 splits; 1, 2 are leaves)."""
    return dict(
        feature=[feature, -1, -1],
        threshold=[threshold, 0.0, 0.0],
        left=[1, 0, 0],
        right=[2, 0, 0],
        value=[0.0, lo, hi],
    )


def _tree_spec(clone_trees, imp_trees=(), **extra):
    return dict(
        clone=dict(type="gbm", baseline=-1.0, trees=list(clone_trees)),
        impostor=dict(type="gbm", baseline=-9.0, trees=list(imp_trees)),
        t_warn=0.5,
        t_esc=0.8,
        channel=list(CHANNEL),
        head="gbm",
        **extra,
    )


def test_a_boosted_tree_head_walks_its_trees_and_sums_the_leaves():
    # feature 0 is logit(spoof); the clone head is baseline -1, plus +3 / -2 by which side it falls
    lf = LearnedFusion(_tree_spec([_stump(0, 0.0, -2.0, 3.0)]))
    assert lf.probs(0.9, 0.5, CH)[1] == pytest.approx(1 / (1 + math.exp(-(-1.0 + 3.0))))
    assert lf.probs(0.1, 0.5, CH)[1] == pytest.approx(1 / (1 + math.exp(-(-1.0 - 2.0))))
    assert lf.probs(0.9, 0.5, CH)[2] < 1e-3  # the impostor head has no trees: baseline only
    two = LearnedFusion(_tree_spec([_stump(0, 0.0, -2.0, 3.0), _stump(0, 0.0, -0.5, 0.5)]))
    assert two.probs(0.9, 0.5, CH)[1] == pytest.approx(1 / (1 + math.exp(-(-1.0 + 3.0 + 0.5))))


def test_a_tree_head_sees_the_channel_features_after_the_signal_and_clamps_them():
    # feature 2 = the second channel descriptor (floor_rel_db); the split must see the clamped value
    spec = _tree_spec([_stump(2, -27.0, -4.0, 4.0)])
    spec["channel_range"] = {k: [CH[k] - 5, CH[k] + 5] for k in CHANNEL}
    lf = LearnedFusion(spec)
    below = lf.probs(0.5, 0.5, {**CH, "floor_rel_db": -200.0})[1]  # clamped to -32 -> left leaf
    above = lf.probs(0.5, 0.5, {**CH, "floor_rel_db": +200.0})[1]  # clamped to -22 -> right leaf
    assert below < 0.01 and above > 0.9


def test_a_tree_head_takes_one_detector_score_not_two():
    spec = _tree_spec([_stump(0, 0.0, -2.0, 3.0)], clone_signals=2)
    with pytest.raises(ValueError, match="single detector score"):
        LearnedFusion(spec).probs(0.9, 0.5, CH, spoof_alt=0.9)


def test_the_absolute_level_can_never_be_a_model_input():
    spec = _spec()
    spec["channel"] = [*CHANNEL, "level_db"]
    with pytest.raises(ValueError, match="level_db"):
        LearnedFusion(spec)


def test_a_model_names_its_own_channel_features_and_uses_them():
    names = [*CHANNEL, "crest_db", "flat_top_db"]
    n = 1 + 2 * len(names)
    ch = {**CH, "crest_db": 16.0, "flat_top_db": -43.0}

    def head(coef, intercept=0.0):
        return dict(mean=[0.0] * n, std=[1.0] * n, coef=coef, intercept=intercept)

    flat = [0.0] * n
    crest = list(flat)
    crest[1 + names.index("crest_db")] = 0.3  # only the crest factor moves this head
    spec = dict(
        clone=head(crest, intercept=-4.5),  # 6 dB -> p~0.06, 26 dB -> p~0.96
        impostor=head(flat),
        t_warn=0.5,
        t_esc=0.8,
        channel=names,
        channel_range={k: [ch[k] - 20, ch[k] + 20] for k in names},
    )
    lf = LearnedFusion(spec)
    assert lf.channel == tuple(names)
    low, high = {**ch, "crest_db": 6.0}, {**ch, "crest_db": 26.0}
    assert lf.probs(0.5, 0.5, high)[1] > lf.probs(0.5, 0.5, low)[1] + 0.5
    assert head_features(0.5, ch, names=names).shape == (n,)
    with pytest.raises(KeyError):
        lf.probs(0.5, 0.5, CH)  # a call described without the descriptors this model needs


def _gain_spec(w0=0.0, w=None, offset=-1.0):
    n = len(CHANNEL)
    spec = _spec()
    spec["clone"] = dict(
        type="gain",
        mean=[0.0] * n,
        std=[1.0] * n,
        w0=w0,
        w=[0.0] * n if w is None else w,
        offset=offset,
    )
    return spec


def test_a_trust_weighted_head_is_monotone_in_the_detector_whatever_the_channel():
    # large channel weights of mixed sign: the gain may be tiny or huge, but never negative
    lf = LearnedFusion(_gain_spec(w0=0.3, w=[2.0, -3.0, 1.5, -0.5]))
    rng = np.random.default_rng(0)
    for _ in range(100):
        ch = dict(zip(CHANNEL, rng.uniform(-8, 8, len(CHANNEL)), strict=True))
        ps = [lf.probs(s, None, ch)[1] for s in (0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99)]
        assert ps == sorted(ps)


def test_a_recording_style_cannot_call_a_voice_synthetic_when_the_detector_is_quiet():
    # the failure this head exists for: a clean, band-limited, noise-suppressed-looking
    # recording (a messaging-app voice note) with an unremarkable detector score
    lf = LearnedFusion(_gain_spec(w0=0.0, w=[0.0, -2.0, 0.0, -2.0], offset=-1.5))
    clean_looking = dict(dyn_db=30.0, floor_rel_db=-60.0, hf_ratio=0.0, flatness=0.0)
    assert lf.probs(0.05, None, clean_looking)[1] < 0.2 < lf.probs(0.9, None, clean_looking)[1]


def test_a_trust_weighted_head_sees_the_clamped_channel_and_survives_json():
    import json

    spec = _gain_spec(w0=0.0, w=[0.0, 3.0, 0.0, 0.0], offset=0.0)
    spec["channel_range"] = {k: [CH[k] - 5, CH[k] + 5] for k in CHANNEL}
    lf = LearnedFusion(json.loads(json.dumps(spec)))
    # a suspicious detector score is believed more on a cleaner floor, within the fitted range ...
    far_below = lf.probs(0.9, None, {**CH, "floor_rel_db": -500.0})[1]
    at_edge = lf.probs(0.9, None, {**CH, "floor_rel_db": CH["floor_rel_db"] - 5})[1]
    assert far_below == pytest.approx(at_edge)  # ... and an extreme value counts as the edge


def test_without_main_effects_the_head_sees_the_signal_and_its_products_only():
    assert head_features(0.5, CH).shape == (1 + 2 * len(CHANNEL),)
    x = head_features(0.9, CH, main_effects=False)
    assert x.shape == (1 + len(CHANNEL),)
    assert x[1:] == pytest.approx(math.log(9) * np.array([CH[k] for k in CHANNEL]))


def test_a_linear_head_without_main_effects_is_read_back_as_one():
    spec = _spec()
    n = 1 + len(CHANNEL)
    spec["clone"] = dict(
        mean=[0.0] * n,
        std=[1.0] * n,
        coef=[1.0] + [0.0] * (n - 1),
        intercept=0.0,
        main_effects=False,
    )
    lf = LearnedFusion(spec)  # a 5-feature head would break if the bare channel terms were added
    assert lf.probs(0.9, None, CH)[1] == pytest.approx(1 / (1 + math.exp(-math.log(9))))


def test_levels_word_a_heads_finding_against_the_models_own_thresholds():
    lf = LearnedFusion(_spec(t_warn=0.6, t_esc=0.9))
    assert [lf.level(p) for p in (0.0, 0.29, 0.3, 0.59, 0.6, 0.89, 0.9, 1.0)] == [
        "none",
        "none",
        "some",
        "some",
        "likely",
        "likely",
        "strong",
        "strong",
    ]


def test_a_head_that_is_nearly_certain_is_worded_strong_even_when_escalation_is_stricter():
    lf = LearnedFusion(_spec(t_warn=0.64, t_esc=0.999))  # the shipped shape: escalate needs ~1.0
    assert lf.level(0.80) == "likely" and lf.level(0.94) == "likely"
    assert lf.level(0.95) == "strong" and lf.level(0.99) == "strong"
