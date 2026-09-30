"""Fit a real probability calibration for the speaker-consistency signal,
replacing the hand-picked linear interpolation between two cosine
thresholds documented as "a documented operating point, not calibrated" in
speaker/verify.py.

    python scripts/calibrate_speaker.py                  # report only
    python scripts/calibrate_speaker.py --ship platt     # + activate

There is no off-the-shelf speaker-verification trial list for this corpus,
so this script builds one: ASVspoof19 LA's bona fide clips carry real
speaker IDs (the spoof clips don't have a stable "real speaker" in the
same sense, so this uses bona fide only). Genuine trials pair two clips
from the same speaker; impostor trials pair clips from different
speakers. The fit and evaluation speaker sets are disjoint (no speaker
appears in both), the way a verification trial list is supposed to be
built -- otherwise the reported calibration quality would be optimistic.

Reports ECE/Brier before (today's linear interpolation) vs after each
fitted method (Platt, isotonic) on held-out (never-fit-on) speakers.
Nothing is written without --ship. Saves the chosen calibrator to
models/speaker/speaker_calibration.json.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import itertools
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from voiceguard.audio import load_wave, preprocess_wave
from voiceguard.config import REPO_ROOT, load_config, preprocess_config, resolve
from voiceguard.data import load_manifest
from voiceguard.eval import brier_score, expected_calibration_error, fit_calibrator, git_commit
from voiceguard.speaker.embed import SAMPLE_RATE, SpeakerEmbedder, cosine
from voiceguard.speaker.verify import _MATCH_SIM, _MISMATCH_SIM

OUT_PATH = REPO_ROOT / "models" / "speaker" / "speaker_calibration.json"


def _bonafide_speakers(cfg, seed, clips_per_speaker):
    man = resolve(cfg, "manifests")
    frames = []
    for name in ("asvspoof19_la_train.csv", "asvspoof19_la_dev.csv"):
        df = load_manifest(man / name)
        frames.append(df[df.label == "bonafide"])
    df = pd.concat(frames, ignore_index=True)
    # cap clips per speaker so pairing stays tractable
    df = (
        df.groupby("speaker_id", as_index=False, group_keys=False)[df.columns]
        .apply(lambda g: g.sample(min(clips_per_speaker, len(g)), random_state=seed))
        .reset_index(drop=True)
    )
    return df


def _embed_all(df, pcfg, embedder):
    embs = {}
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="embed", leave=False):
        w = preprocess_wave(load_wave(row.path, SAMPLE_RATE), pcfg)
        if len(w) < 2.5 * SAMPLE_RATE:  # mirrors SpeakerVerifier's own reliability floor
            continue
        embs[row.path] = (row.speaker_id, embedder.embed(w))
    return embs


def _build_trials(embs, rng, max_impostor):
    by_speaker: dict[str, list] = {}
    for spk, e in embs.values():
        by_speaker.setdefault(spk, []).append(e)

    genuine = []  # cosine similarities, same speaker
    for vecs in by_speaker.values():
        for a, b in itertools.combinations(vecs, 2):
            genuine.append(cosine(a, b))

    speakers = list(by_speaker)
    impostor = []
    pairs_seen = set()
    attempts = 0
    target = min(max_impostor, len(genuine) * 3)
    while len(impostor) < target and attempts < target * 20:
        attempts += 1
        s1, s2 = rng.choice(speakers, size=2, replace=False)
        key = tuple(sorted((s1, s2))) + (len(pairs_seen),)  # allow repeats across speaker pairs
        if key in pairs_seen:
            continue
        pairs_seen.add(key)
        a = by_speaker[s1][rng.integers(len(by_speaker[s1]))]
        b = by_speaker[s2][rng.integers(len(by_speaker[s2]))]
        impostor.append(cosine(a, b))

    return np.array(genuine), np.array(impostor)


def _report(name, sims, labels, old_match, old_mismatch, cals: dict):
    span = old_match - old_mismatch
    before = np.clip((old_match - sims) / span, 0.0, 1.0)
    row = {
        "split": name,
        "n": len(labels),
        "ece_before": expected_calibration_error(before, labels),
        "brier_before": brier_score(before, labels),
    }
    line = (
        f"{name:20s} n={row['n']:5d}  "
        f"before ECE {row['ece_before']:.3f} Brier {row['brier_before']:.3f}"
    )
    for method, cal in cals.items():
        after = cal.apply(1.0 - sims)  # calibrator was fit on 1-similarity, see main()
        row[f"ece_{method}"] = expected_calibration_error(after, labels)
        row[f"brier_{method}"] = brier_score(after, labels)
        line += f"  |  {method} ECE {row[f'ece_{method}']:.3f} Brier {row[f'brier_{method}']:.3f}"
    print(line)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips-per-speaker", type=int, default=12)
    ap.add_argument("--max-impostor", type=int, default=4000)
    ap.add_argument("--fit-speaker-frac", type=float, default=0.7)
    ap.add_argument(
        "--ship",
        default="none",
        choices=["none", "platt", "isotonic"],
        help="which fitted method (if any) to write into the calibration file; default "
        "'none' just reports the held-out-speaker comparison so you can decide",
    )
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config()
    seed = args.seed if args.seed is not None else cfg["seed"]
    pcfg = preprocess_config(cfg)
    rng = np.random.default_rng(seed)

    df = _bonafide_speakers(cfg, seed, args.clips_per_speaker)
    speakers = sorted(df.speaker_id.unique())
    rng.shuffle(speakers)
    n_fit = max(2, int(round(len(speakers) * args.fit_speaker_frac)))
    fit_speakers, test_speakers = set(speakers[:n_fit]), set(speakers[n_fit:])
    print(f"{len(df)} bona fide clips, {len(speakers)} speakers "
          f"-> {len(fit_speakers)} fit / {len(test_speakers)} held-out (speaker-disjoint)")

    embedder = SpeakerEmbedder()
    fit_df = df[df.speaker_id.isin(fit_speakers)]
    test_df = df[df.speaker_id.isin(test_speakers)]
    fit_embs = _embed_all(fit_df, pcfg, embedder)
    test_embs = _embed_all(test_df, pcfg, embedder)

    fit_gen, fit_imp = _build_trials(fit_embs, rng, args.max_impostor)
    test_gen, test_imp = _build_trials(test_embs, rng, args.max_impostor)
    print(f"fit trials: {len(fit_gen)} genuine / {len(fit_imp)} impostor")
    print(f"held-out trials: {len(test_gen)} genuine / {len(test_imp)} impostor")

    # 1 = impostor (different speaker = the "risk" class), higher raw = more suspicious
    fit_sim = np.concatenate([fit_gen, fit_imp])
    fit_y = np.concatenate([np.zeros(len(fit_gen)), np.ones(len(fit_imp))]).astype(int)
    cals = {m: fit_calibrator(1.0 - fit_sim, fit_y, method=m) for m in ("platt", "isotonic")}

    rows = [_report("fit speakers", fit_sim, fit_y, _MATCH_SIM, _MISMATCH_SIM, cals)]
    test_sim = np.concatenate([test_gen, test_imp])
    test_y = np.concatenate([np.zeros(len(test_gen)), np.ones(len(test_imp))]).astype(int)
    rows.append(_report("held-out speakers", test_sim, test_y, _MATCH_SIM, _MISMATCH_SIM, cals))

    for r in rows:
        r.update(git_commit=git_commit())
    tables = resolve(cfg, "tables")
    pd.DataFrame(rows).to_csv(tables / "speaker_calibration.csv", index=False)
    print(f"\nprovenance -> {tables / 'speaker_calibration.csv'}")

    if args.ship == "none":
        print(
            "\n--ship not given: nothing written to the calibration file. Inspect the table "
            "above -- a calibrator that only helps on its own fit speakers should not ship. "
            "Re-run with --ship platt|isotonic once you've decided one actually improves the "
            "held-out speakers."
        )
    else:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(
            json.dumps({"calibrator": cals[args.ship].to_dict(), "method": args.ship}, indent=2)
        )
        print(f"saved fitted '{args.ship}' calibrator -> {OUT_PATH}")


if __name__ == "__main__":
    main()
