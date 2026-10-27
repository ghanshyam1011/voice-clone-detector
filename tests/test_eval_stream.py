"""The call-level policy arithmetic is tested on hand-made traces whose answer is known:
a genuine call with one spike, a clone whose score climbs, and a muted line."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "eval_stream.py"
_spec = importlib.util.spec_from_file_location("eval_stream", _PATH)
es = importlib.util.module_from_spec(_spec)
sys.modules["eval_stream"] = es
_spec.loader.exec_module(es)


def _call(kind, fused, call, *, status="ok", provisional=False, step=0.5):
    t = np.arange(1, len(fused) + 1) * step
    return pd.DataFrame(
        dict(
            speaker="LA_0001",
            call=call,
            cond="clean",
            kind=kind,
            t=t,
            fused=fused,
            action=["ESCALATE" if f >= 0.75 else "VERIFY" if f >= 0.55 else "ALLOW" for f in fused],
            status=status,
            provisional=provisional,
        )
    )


def _traces():
    spike = np.full(40, 0.2)
    spike[19] = 0.8  # one update above the escalate threshold, at t = 10 s
    climb = np.r_[np.full(10, 0.2), np.linspace(0.3, 0.95, 30)]
    return pd.concat(
        [_call("genuine", spike, 0), _call("clone", climb, 1), _call("impostor", climb, 2)]
    )


def test_persistence_requires_k_consecutive_updates():
    x = np.array([0, 1, 0, 1, 1, 1, 0], bool)
    assert es._persist(x, 1).tolist() == x.tolist()
    assert es._persist(x, 3).tolist() == [False, False, False, False, False, True, False]


def test_a_single_spike_flags_a_genuine_call_live_but_not_once_it_must_persist():
    d = _traces()
    live = es.per_call(d, None).set_index("kind")
    assert live.loc["genuine", "first_flag"] == 10.0 and live.loc["genuine", "first_esc"] == 10.0
    persistent = es.per_call(d, 2).set_index("kind")
    assert np.isnan(persistent.loc["genuine", "first_flag"])  # one update is not evidence


def test_persistence_delays_the_catch_of_a_real_clone_by_the_updates_it_waits_for():
    d = _traces()
    first = {k: es.per_call(d, k).set_index("kind").loc["clone", "first_flag"] for k in (1, 4)}
    assert first[4] == first[1] + 3 * 0.5  # three more 0.5 s updates above the threshold


def test_recomputed_policy_k1_matches_the_logged_live_action():
    d = _traces()
    live, k1 = es.per_call(d, None), es.per_call(d, 1)
    pd.testing.assert_series_equal(live.first_flag, k1.first_flag)
    pd.testing.assert_series_equal(live.first_esc, k1.first_esc)


def test_a_provisional_read_can_flag_but_never_escalate_and_a_muted_line_never_flags():
    d = pd.concat(
        [
            _call("genuine", np.full(10, 0.9), 0, provisional=True),
            _call("genuine", np.full(10, 0.9), 1, status="no_speech"),
        ]
    )
    c = es.per_call(d, 1).set_index("call")
    assert c.loc[0, "first_flag"] == 0.5 and np.isnan(c.loc[0, "first_esc"])
    assert np.isnan(c.loc[1, "first_flag"])


def test_summary_reports_call_level_rates_and_time_to_detect():
    s = es.summarise(es.per_call(_traces(), None))
    assert s["n_gen"] == 1 and s["n_clone"] == 1 and s["n_imp"] == 1
    assert s["FA@5"] == 0.0 and s["FA@10"] == 1.0 and s["FA@end"] == 1.0
    assert s["clone_det"] == 1.0 and 5 < s["clone_TTD"] < 14
