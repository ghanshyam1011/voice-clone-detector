"""Fusion + policy + context rules."""

from voiceguard.risk import CallContext, Signal, decide, fuse
from voiceguard.risk.fusion import DEFAULT_WEIGHTS


def _sig(name, value, detail=""):
    return Signal(name, name.title(), value, DEFAULT_WEIGHTS[name], detail)


def test_absent_signals_are_dropped_and_weights_renormalise():
    # only spoof present -> fused == spoof value
    f = fuse([_sig("spoof", 0.8), _sig("speaker", None), _sig("context", None)])
    assert f.score == 0.8
    assert [s.name for s in f.present] == ["spoof"]


def test_no_signals_is_zero_risk():
    assert fuse([_sig("spoof", None), _sig("speaker", None)]).score == 0.0


def test_weighted_mean_of_present_signals():
    f = fuse([_sig("spoof", 0.6), _sig("speaker", 0.2)])
    # (0.6*0.45 + 0.2*0.35) / (0.45+0.35)
    assert abs(f.score - (0.6 * 0.45 + 0.2 * 0.35) / 0.80) < 1e-6


def test_lone_high_signal_floors_the_fused_score():
    sigs = [_sig("spoof", 0.95), _sig("speaker", 0.05), _sig("context", 0.0)]
    plain_mean = sum(s.value * s.weight for s in sigs) / sum(s.weight for s in sigs)
    f = fuse(sigs)
    assert f.score > plain_mean  # floored above the weighted mean
    assert f.score >= 0.65
    assert "alone" in f.note


def test_scenario_thresholds_map_to_actions():
    mid = fuse([_sig("spoof", 0.5)])  # score 0.5
    assert decide(mid, "routine").action == "ALLOW"  # warn 0.55
    assert decide(mid, "transaction").action == "VERIFY"  # warn 0.40, esc 0.65
    assert decide(mid, "privileged").action == "VERIFY"  # warn 0.30, esc 0.55

    hi = fuse([_sig("spoof", 0.8)])  # score 0.8
    assert decide(hi, "routine").action == "ESCALATE"  # esc 0.75
    assert decide(hi, "privileged").action == "ESCALATE"

    lo = fuse([_sig("spoof", 0.35)])  # score 0.35
    assert decide(lo, "transaction").action == "ALLOW"  # warn 0.40
    assert decide(lo, "privileged").action == "VERIFY"  # warn 0.30


def test_reason_codes_include_elevated_signals_and_context():
    f = fuse([_sig("spoof", 0.78, "score 0.78"), _sig("speaker", 0.62, "similarity 0.60 (low)")])
    d = decide(f, "transaction", ["Caller number not recognised"])
    joined = " | ".join(d.reasons)
    assert "Spoof" in joined and "Speaker" in joined
    assert "Caller number not recognised" in joined
    assert d.recommendation


def test_context_rules():
    r, reasons = CallContext(caller_known=False, amount=75_000).assess()
    assert r > 0  # unknown caller + large transfer
    assert any("not recognised" in x for x in reasons)
    assert any("75,000" in x for x in reasons)

    r0, reasons0 = CallContext().assess()
    assert r0 == 0.0 and reasons0 == []

    r_cap, _ = CallContext(caller_known=False, amount=1_000_000, prior_flags=5).assess()
    assert r_cap == 1.0
