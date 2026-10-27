"""Cache the raw per-signal reads for a large, degraded system-level trial set.

    python scripts/build_system_trials.py --split eval
    python scripts/build_system_trials.py --split dev

Same trial design as eval_system.py (genuine / clone / impostor against an
enrolled speaker, official ASVspoof 2019 LA protocol files only), but every
call is replayed under several acoustic conditions so robustness is
measured, not assumed:

    clean     as-is
    babble10  other people talking in the background, 10 dB SNR
    babble5   same, 5 dB SNR (a noisy room)
    pink10    broadband room/line noise, 10 dB SNR
    ulaw      G.711 telephone codec

Genuine and attack calls get *identical* degradation, so a detector cannot
win by keying on "is it noisy". The raw signals (spoof / speaker / prosody
risk) are cached to results/tables/system_trials_<split>.csv; fusion is then
fit and scored offline by scripts/fit_fusion.py without re-running models.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from voiceguard.audio import load_wave
from voiceguard.audio.channel import RECENT_SECONDS, channel_features
from voiceguard.audio.codec import codec_roundtrip
from voiceguard.audio.degrade import STRESS_CONDITIONS, apply_condition
from voiceguard.config import REPO_ROOT, load_config, resolve
from voiceguard.data import load_manifest
from voiceguard.eval import git_commit
from voiceguard.eval.trials import (
    choose_speakers,
    eligible_speakers,
    plan_speaker,
    window_stats,
)
from voiceguard.serve.session import build_session_stack

SR = 16000
CONDITIONS = ("clean", "babble10", "babble5", "pink10", "ulaw")


def _cat(paths):
    return np.concatenate([load_wave(REPO_ROOT / p, SR) for p in paths])


def _rms(y):
    return float(np.sqrt(np.mean(np.square(y)) + 1e-12))


def _mix(y, noise, snr_db):
    noise = np.resize(noise, len(y))
    g = _rms(y) / (_rms(noise) * 10 ** (snr_db / 20))
    out = y + g * noise
    peak = float(np.max(np.abs(out)))
    return (out / peak * 0.95).astype(np.float32) if peak > 0.99 else out.astype(np.float32)


def _pink(n, rng):
    f = np.fft.rfftfreq(n)
    f[0] = f[1]
    spec = (rng.standard_normal(len(f)) + 1j * rng.standard_normal(len(f))) / np.sqrt(f)
    x = np.fft.irfft(spec, n)
    return (x / (np.std(x) + 1e-9)).astype(np.float32)


def degrade(y, cond, babble_pool, rng):
    if cond == "clean":
        return y
    if cond.startswith("babble"):
        paths = rng.choice(babble_pool, size=3, replace=False)
        return _mix(y, _cat(list(paths)), float(cond[6:]))
    if cond == "pink10":
        return _mix(y, _pink(len(y), rng), 10.0)
    if cond == "ulaw":
        return codec_roundtrip(y, SR, "g711_ulaw").astype(np.float32)

    # every other situation (reverb, more codecs, level, loss, ...) is defined once, in
    # voiceguard.audio.degrade, so training augmentation and this evaluation share it
    def voices(rng_, _n):
        return _cat(list(rng_.choice(babble_pool, size=3, replace=False)))

    return apply_condition(cond, y, rng, babble=voices)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="eval", choices=["dev", "eval"])
    ap.add_argument("--speakers", type=int, default=0, help="0 = all eligible")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--weights", default=None, help="a .pth under models/pretrained/")
    ap.add_argument(
        "--alt", action="store_true", help="also record the upstream (pretrained) AASIST read"
    )
    ap.add_argument("--resume", action="store_true", help="skip speakers already in the table")
    ap.add_argument("--tag", default="", help="suffix for the output table, e.g. robust")
    ap.add_argument(
        "--conds",
        default=",".join(CONDITIONS),
        help="comma-separated situations (see voiceguard.audio.degrade); default = the five "
        "used to fit the fusion. 'stress' = the 14 situations the shipped system never saw",
    )
    args = ap.parse_args()
    conds = STRESS_CONDITIONS if args.conds == "stress" else tuple(args.conds.split(","))
    os.environ.setdefault("OMP_NUM_THREADS", "4")

    cfg = load_config()
    df = load_manifest(resolve(cfg, "manifests") / f"asvspoof19_la_{args.split}.csv")
    bona = df[df.label == "bonafide"]
    spoof = df[df.label == "spoof"]
    ok = eligible_speakers(bona, spoof)
    spk = choose_speakers(ok, args.speakers, args.seed)
    print(f"{args.split}: {len(spk)} speakers", flush=True)

    # reads are independent (ema=False), so evaluation runs must not append thousands of
    # decisions to the product's real audit trail
    audit_path = Path(tempfile.gettempdir()) / "voiceguard_trials_audit.jsonl"
    mgr = build_session_stack(
        "aasist", pretrained=True, weights=args.weights, audit_path=audit_path
    )
    alt = None
    if args.alt:
        from voiceguard.detect import build_scorer

        alt = build_scorer("aasist", pretrained=True)
    out = REPO_ROOT / "results" / "tables" / f"system_trials{args.tag}_{args.split}.csv"
    rows = []
    done = set()
    if args.resume and out.exists():
        prev = pd.read_csv(out)
        rows = prev.to_dict("records")
        done = set(prev.speaker.unique())
        print(f"resuming: {len(done)} speakers already done", flush=True)
    for s in tqdm(spk, desc="speakers"):
        if s in done:
            continue
        # per-speaker stream: a resumed run draws the same trials as an uninterrupted one
        plan = plan_speaker(bona, spoof, ok, s, args.seed)
        enrol_wave = _cat(plan.enrol)
        # enrol once: with ema=False every read is independent of the last, so one session
        # per speaker gives the same numbers as one per read, without re-embedding the
        # enrolment clips ~100 times
        sess = mgr.create("routine")
        mgr.enroll(sess.session_id, enrol_wave)
        for ti, (kind, att, paths) in enumerate(plan.trials):
            base = _cat(paths)
            for cond in conds:
                y = degrade(base, cond, plan.pool, plan.rng)
                r = mgr.assess(sess.session_id, y, ema=False)
                rows.append(
                    dict(
                        split=args.split,
                        speaker=s,
                        trial=ti,
                        kind=kind,
                        attack=att,
                        cond=cond,
                        spoof=r["spoof"],
                        speaker_risk=r["speaker"],
                        prosody=r["prosody"],
                        similarity=r.get("similarity"),
                        spoof_alt=None
                        if alt is None
                        else float(
                            alt.score_waveform(y[-int(RECENT_SECONDS * SR) :], hop_s=1.0).risk
                        ),
                        speech_s=r["quality"]["speech_s"],
                        # which detector (and calibration) produced this read: a fusion fitted
                        # on these tables is only valid for the same pair
                        detector=mgr.scorer.name,
                        op_point=round(mgr.scorer.operating_point, 6),
                        **window_stats(r.get("spoof_windows")),
                        **channel_features(y),
                    )
                )
        mgr.end(sess.session_id)
        pd.DataFrame(rows).to_csv(out, index=False)

    print(f"wrote {out}  ({len(rows)} reads, {git_commit()})", flush=True)


if __name__ == "__main__":
    main()
