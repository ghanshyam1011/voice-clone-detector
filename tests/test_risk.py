"""Fusion + policy + context rules."""

from voiceguard.risk import CallContext, Signal, decide, fuse
from voiceguard.risk.fusion import DEFAULT_WEIGHTS


def _sig(name, value, detail="", provisional=False):
    return Signal(name, name.title(), value, DEFAULT_WEIGHTS[name], detail, provisional)


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


def test_provisional_signal_caps_escalate_to_verify():
    # this would ESCALATE (score 0.8 >= 0.75) if the spoof read were confirmed
    confirmed = fuse([_sig("spoof", 0.8)])
    assert decide(confirmed, "routine").action == "ESCALATE"

    # the same score, but the window wasn't full yet -> capped at VERIFY, not
    # silently dropped -- a fast-but-uncertain read still surfaces concern
    early = fuse([_sig("spoof", 0.8, provisional=True)])
    assert early.has_provisional is True
    d = decide(early, "routine")
    assert d.action == "VERIFY"
    assert any("confirm" in r.lower() for r in d.reasons)


def test_provisional_does_not_change_allow_or_verify_outcomes():
    # provisional only ever caps DOWN from ESCALATE; a score that was never
    # going to escalate is unaffected
    low = fuse([_sig("spoof", 0.2, provisional=True)])
    assert decide(low, "routine").action == "ALLOW"

    mid = fuse([_sig("spoof", 0.6, provisional=True)])  # routine: warn 0.55, esc 0.75
    assert decide(mid, "routine").action == "VERIFY"


def test_has_provisional_reflects_only_present_signals():
    # an absent (None) signal marked provisional doesn't count -- only
    # present signals matter
    f = fuse([_sig("spoof", 0.5), _sig("speaker", None, provisional=True)])
    assert f.has_provisional is False

    f2 = fuse([_sig("spoof", 0.5, provisional=True), _sig("speaker", 0.1)])
    assert f2.has_provisional is True


def test_context_rules():
    r, reasons = CallContext(caller_known=False, amount=75_000).assess()
    assert r > 0  # unknown caller + large transfer
    assert any("not recognised" in x for x in reasons)
    assert any("75,000" in x for x in reasons)

    r0, reasons0 = CallContext().assess()
    assert r0 == 0.0 and reasons0 == []

    r_cap, _ = CallContext(caller_known=False, amount=1_000_000, prior_flags=5).assess()
    assert r_cap == 1.0
