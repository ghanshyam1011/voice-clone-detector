"""The trial design every system-level number rests on: which clips enrol, which are the
genuine / clone / impostor calls, and which are background voices. A leak here (the
same clip on both sides, an impostor who is the target, background voices from the
target) would inflate every result built on it, so it is tested directly."""

import numpy as np
import pandas as pd
import pytest

from voiceguard.eval.trials import (
    CALL_CLIPS,
    CLONE_CALLS,
    ENROL_CLIPS,
    choose_speakers,
    eligible_speakers,
    plan_speaker,
    window_stats,
)


def _manifest(n_speakers=8, bona_each=30, spoof_each=20):
    rows = []
    for s in range(n_speakers):
        spk = f"LA_{1000 + s}"
        for i in range(bona_each):
            rows.append(("bonafide", "bonafide", f"b/{spk}_{i}.flac", spk))
        for i in range(spoof_each):
            rows.append(("spoof", f"A{7 + i % 3:02d}", f"s/{spk}_{i}.flac", spk))
    df = pd.DataFrame(rows, columns=["label", "attack_id", "path", "speaker_id"])
    return df[df.label == "bonafide"], df[df.label == "spoof"]


def test_eligibility_needs_enough_bona_fide_and_spoof_clips():
    bona, spoof = _manifest()
    assert len(eligible_speakers(bona, spoof)) == 8
    few_bona, _ = _manifest(bona_each=ENROL_CLIPS + 2 * CALL_CLIPS - 1)
    assert eligible_speakers(few_bona, spoof) == []
    _, few_spoof = _manifest(spoof_each=CLONE_CALLS * CALL_CLIPS - 1)
    assert eligible_speakers(bona, few_spoof) == []


def test_choose_speakers_is_deterministic_and_zero_means_all():
    ok = [f"LA_{i}" for i in range(20)]
    assert choose_speakers(ok, 0, 7) == ok
    assert choose_speakers(ok, 5, 7) == choose_speakers(ok, 5, 7)
    assert choose_speakers(ok, 5, 7) != choose_speakers(ok, 5, 8)
    assert len(choose_speakers(ok, 99, 7)) == 20


def test_no_clip_is_shared_between_enrolment_and_any_call():
    bona, spoof = _manifest()
    ok = eligible_speakers(bona, spoof)
    for s in ok:
        plan = plan_speaker(bona, spoof, ok, s, seed=7)
        call_clips = [p for _, _, ps in plan.trials for p in ps]
        assert set(plan.enrol).isdisjoint(call_clips)
        genuine = [p for k, _, ps in plan.trials if k == "genuine" for p in ps]
        assert len(genuine) == len(set(genuine))  # the two genuine calls do not overlap


def test_each_kind_holds_the_right_audio():
    bona, spoof = _manifest()
    ok = eligible_speakers(bona, spoof)
    s = ok[0]
    plan = plan_speaker(bona, spoof, ok, s, seed=7)
    kinds = [k for k, _, _ in plan.trials]
    assert kinds.count("genuine") == 2 and kinds.count("clone") == CLONE_CALLS
    assert kinds.count("impostor") == 2
    for kind, attack, paths in plan.trials:
        assert len(paths) == CALL_CLIPS
        speakers = {p.split("/")[-1].rsplit("_", 1)[0] for p in paths}
        if kind == "genuine":
            assert (
                speakers == {s} and attack == "bonafide" and all(p.startswith("b/") for p in paths)
            )
        elif kind == "clone":
            assert (
                speakers == {s}
                and attack.startswith("A")
                and all(p.startswith("s/") for p in paths)
            )
        else:  # impostor: a different real human
            assert (
                s not in speakers and len(speakers) == 1 and all(p.startswith("b/") for p in paths)
            )


def test_background_voices_never_come_from_the_target_or_an_impostor():
    bona, spoof = _manifest()
    ok = eligible_speakers(bona, spoof)
    s = ok[0]
    plan = plan_speaker(bona, spoof, ok, s, seed=7)
    impostors = {
        p.split("/")[-1].rsplit("_", 1)[0]
        for k, _, ps in plan.trials
        if k == "impostor"
        for p in ps
    }
    pool_speakers = {p.split("/")[-1].rsplit("_", 1)[0] for p in plan.pool}
    assert s not in pool_speakers and pool_speakers.isdisjoint(impostors)


def test_plans_are_reproducible_per_speaker_and_independent_of_run_order():
    bona, spoof = _manifest()
    ok = eligible_speakers(bona, spoof)
    first = [plan_speaker(bona, spoof, ok, s, seed=7) for s in ok]
    second = [plan_speaker(bona, spoof, ok, s, seed=7) for s in reversed(ok)][::-1]
    for a, b in zip(first, second, strict=True):
        assert a.enrol == b.enrol and a.trials == b.trials
        # the same stream also draws the degradations afterwards: it must start identical
        assert a.rng.random() == b.rng.random()


def test_window_stats_aggregators():
    st = window_stats([0.1, 0.9, 0.5])
    assert st["win_n"] == 3 and st["win_mean"] == pytest.approx(0.5)
    assert st["win_med"] == 0.5 and st["win_max"] == 0.9 and st["win_min"] == 0.1
    assert st["win_last"] == 0.5
    nothing = window_stats(None)
    assert all(np.isnan(v) for v in nothing.values())
