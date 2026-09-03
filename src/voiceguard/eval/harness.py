"""One evaluation harness.

Merges what used to be three separate scripts (evaluate.py,
cross_dataset_eval.py, silence_ablation.py). These functions take already
computed *scores* (P(bonafide), higher = more bonafide) and *labels*
(1 = bonafide, 0 = spoof) and return provenance-stamped rows. Model
loading, feature extraction and scoring live in scripts/run_eval.py so
this module stays pure and testable.
"""

from __future__ import annotations

import numpy as np

from voiceguard.eval.metrics import compute_eer
from voiceguard.eval.provenance import stamp

# ASVspoof 2019 LA: attacks seen in train/dev vs only in eval
KNOWN_ATTACKS = {"A01", "A02", "A03", "A04", "A05", "A06"}


def eer_overall_and_by_group(
    labels: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray | None = None,
) -> dict[str, float]:
    """{'overall': eer, '<group>': eer, ...}. Each per-group EER is computed
    against that group's spoof clips plus *all* bonafide clips, so a single
    weak generator can't hide inside an aggregate."""
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    out = {"overall": compute_eer(labels, scores)[0]}
    if groups is None:
        return out

    groups = np.asarray(groups)
    bonafide = labels == 1
    for g in sorted(x for x in np.unique(groups) if x not in ("bonafide", "unknown")):
        mask = bonafide | (groups == g)
        if len(np.unique(labels[mask])) == 2:
            out[str(g)] = compute_eer(labels[mask], scores[mask])[0]
    return out


def silence_only_waveform(y: np.ndarray, sr: int = 16000, top_db: int = 25) -> np.ndarray:
    """Everything librosa does NOT consider speech, concatenated. Used to
    prove a model is not keying on non-speech cues: a model trained through
    the anti-shortcut front-end must score ~50% EER on these."""
    import librosa

    y = np.asarray(y, dtype=np.float32)
    speech = librosa.effects.split(y, top_db=top_db)
    keep = np.ones(len(y), dtype=bool)
    for start, end in speech:
        keep[start:end] = False
    silence = y[keep]
    floor = int(sr * 0.1)
    if silence.size < floor:  # keep the sample set label-balanced
        silence = np.pad(silence, (0, floor - silence.size))
    return silence


def in_domain_rows(
    model: str,
    dev: tuple[np.ndarray, np.ndarray],
    eval_: tuple[np.ndarray, np.ndarray],
    eval_attacks: np.ndarray,
    *,
    seed: int,
    front_end: str,
) -> list[dict]:
    """dev = (labels, scores) for the dev split (known attacks A01-A06);
    eval_ = (labels, scores) for the eval split (unknown attacks A07-A19)."""
    dev_labels, dev_scores = dev
    eval_labels, eval_scores = eval_

    rows = [
        stamp(
            {
                "metric": "eer",
                "split": "dev_known_attacks",
                "value_pct": compute_eer(dev_labels, dev_scores)[0] * 100,
            },
            train_set="asvspoof19_la_train",
            test_set="asvspoof19_la_dev",
            model=model,
            seed=seed,
            front_end=front_end,
        ),
        stamp(
            {
                "metric": "eer",
                "split": "eval_unknown_attacks",
                "value_pct": compute_eer(eval_labels, eval_scores)[0] * 100,
            },
            train_set="asvspoof19_la_train",
            test_set="asvspoof19_la_eval",
            model=model,
            seed=seed,
            front_end=front_end,
        ),
    ]
    per_attack = eer_overall_and_by_group(eval_labels, eval_scores, eval_attacks)
    for attack, eer in per_attack.items():
        if attack == "overall":
            continue
        rows.append(
            stamp(
                {"metric": "eer", "split": f"eval_attack_{attack}", "value_pct": eer * 100},
                train_set="asvspoof19_la_train",
                test_set="asvspoof19_la_eval",
                model=model,
                seed=seed,
                front_end=front_end,
            )
        )
    return rows


def cross_dataset_row(
    model: str,
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    dataset: str,
    seed: int,
    front_end: str,
) -> dict:
    return stamp(
        {
            "metric": "eer",
            "split": "cross_dataset",
            "value_pct": compute_eer(labels, scores)[0] * 100,
            "n": int(len(labels)),
        },
        train_set="asvspoof19_la_train",
        test_set=dataset,
        model=model,
        seed=seed,
        front_end=front_end,
    )


def silence_ablation_row(
    model: str,
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    seed: int,
    front_end: str,
    leak_threshold: float = 0.40,
) -> dict:
    eer = compute_eer(labels, scores)[0]
    return stamp(
        {
            "metric": "eer",
            "split": "silence_only",
            "value_pct": eer * 100,
            "n": int(len(labels)),
            "verdict": "LEAK" if eer < leak_threshold else "no leak (~chance)",
        },
        train_set="asvspoof19_la_train",
        test_set="asvspoof19_la_dev_silence_only",
        model=model,
        seed=seed,
        front_end=front_end,
    )
