"""System-level trial design, shared by every evaluation that replays "calls".

For each target speaker (official ASVspoof 2019 LA protocol files only):

    genuine   held-out bona fide clips of the enrolled speaker      -> should be ALLOWed
    clone     spoof clips targeting that speaker                    -> should be flagged
    impostor  bona fide clips of a *different* human                -> should be flagged

A call is ``CALL_CLIPS`` clips (~10 s); the session is enrolled from ``ENROL_CLIPS``
other clips. ``plan_speaker`` only chooses *which* clips; loading and degrading the
audio is the caller's job, using the plan's own random stream so that a resumed or
re-ordered run draws exactly the same trials (and the same degradations) as an
uninterrupted one.

This logic used to live inside ``scripts/build_system_trials.py``; it is unchanged
(the random-number draws happen in the same order), so tables built before the
extraction are reproduced exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

CALL_CLIPS = 3
ENROL_CLIPS = 4
GENUINE_CALLS = 2
CLONE_CALLS = 4
IMPOSTOR_CALLS = 2


@dataclass
class SpeakerPlan:
    speaker: str
    enrol: list[str]  # manifest paths of the enrolment clips
    trials: list[tuple[str, str, list[str]]]  # (kind, attack_id, manifest paths of the call)
    pool: np.ndarray  # background-voice clips: never the speaker, never an impostor
    rng: np.random.Generator  # this speaker's stream; draws the trials, then the degradations


def eligible_speakers(bona: pd.DataFrame, spoof: pd.DataFrame) -> list[str]:
    need = ENROL_CLIPS + 2 * CALL_CLIPS
    return sorted(
        s
        for s, g in bona.groupby("speaker_id")
        if len(g) >= need and (spoof.speaker_id == s).sum() >= CLONE_CALLS * CALL_CLIPS
    )


def choose_speakers(eligible: list[str], n: int, seed: int) -> list[str]:
    """``n`` = 0 means all eligible speakers."""
    if not n:
        return list(eligible)
    rng = np.random.default_rng(seed)
    return list(rng.choice(eligible, size=min(n, len(eligible)), replace=False))


def plan_speaker(
    bona: pd.DataFrame, spoof: pd.DataFrame, eligible: list[str], speaker: str, seed: int
) -> SpeakerPlan:
    rng = np.random.default_rng([seed, int(speaker.split("_")[-1])])
    b = bona[bona.speaker_id == speaker].sample(frac=1, random_state=int(rng.integers(1e9)))
    enrol = b.path.iloc[:ENROL_CLIPS].tolist()
    rest = b.path.iloc[ENROL_CLIPS:].tolist()
    gen = [rest[i * CALL_CLIPS : (i + 1) * CALL_CLIPS] for i in range(GENUINE_CALLS)]
    sp = spoof[spoof.speaker_id == speaker].sample(frac=1, random_state=int(rng.integers(1e9)))
    clones = [sp.iloc[i * CALL_CLIPS : (i + 1) * CALL_CLIPS] for i in range(CLONE_CALLS)]
    others = rng.choice([o for o in eligible if o != speaker], size=IMPOSTOR_CALLS, replace=False)
    imp = [
        bona[bona.speaker_id == o]
        .sample(CALL_CLIPS, random_state=int(rng.integers(1e9)))
        .path.tolist()
        for o in others
    ]
    pool = bona[~bona.speaker_id.isin([speaker, *others])].path.values
    trials = (
        [("genuine", "bonafide", p) for p in gen]
        + [("clone", c.attack_id.iloc[0], c.path.tolist()) for c in clones]
        + [("impostor", "bonafide", p) for p in imp]
    )
    return SpeakerPlan(speaker, enrol, trials, pool, rng)


def window_stats(windows) -> dict:
    """Aggregators of the per-window spoof risks behind one read, cached so that how a call's
    windows are combined (the shipped read is an EMA that weights the last few) can be
    compared on identical audio: ``win_mean`` / ``win_med`` / ``win_max`` / ``win_min`` /
    ``win_last`` and the window count. A read with no windows (nothing judged) -> NaNs."""
    w = np.asarray(windows if windows else [], dtype=float)
    if w.size == 0:
        return dict.fromkeys(
            ["win_n", "win_mean", "win_med", "win_max", "win_min", "win_last"], np.nan
        )
    return dict(
        win_n=int(w.size),
        win_mean=float(w.mean()),
        win_med=float(np.median(w)),
        win_max=float(w.max()),
        win_min=float(w.min()),
        win_last=float(w[-1]),
    )
