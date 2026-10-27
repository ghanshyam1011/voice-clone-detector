"""Is the speaker head the weak link, and would a different speaker model fix it?

    python scripts/eval_speaker_models.py --split eval --speakers 0
    python scripts/eval_speaker_models.py --report                # re-print from the cached table

Replays the system trials (``voiceguard.eval.trials``: genuine / clone / impostor
calls against an enrolled speaker, under the five standard conditions, with the same
random streams as ``scripts/build_system_trials.py``) but runs ONLY the speaker
models, so many speakers and models are affordable. For every read it caches the
cosine similarity to the enrolment under:

    wavlm       microsoft/wavlm-base-plus-sv x-vector  (what ships today)
    ecapa       speechbrain/spkrec-ecapa-voxceleb      (trained with noise/babble/reverb)
    *_vad       the same, fed only the speech frames found by the VAD
    both        mean of the two models' z-scored similarities

and reports EER (genuine vs impostor, genuine vs clone) per condition. The similarity
scale differs between models, so ``both`` uses z-scores fitted on the genuine /
impostor reads of the *other* speakers (grouped cross-validation) -- no read is scored
with statistics that include its own speaker.

Needs ``pip install "speechbrain==1.0.3"`` (newer speechbrain wants a newer torch).
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from voiceguard.audio import load_wave
from voiceguard.audio.degrade import apply_condition
from voiceguard.audio.quality import analyze
from voiceguard.config import REPO_ROOT, load_config, resolve
from voiceguard.data import load_manifest
from voiceguard.eval import compute_eer
from voiceguard.eval.trials import choose_speakers, eligible_speakers, plan_speaker
from voiceguard.speaker import SpeakerEmbedder

SR = 16000
CONDS = ("clean", "babble10", "babble5", "pink10", "ulaw")
TAB = REPO_ROOT / "results" / "tables"
WINDOW_S = 6.0  # the speaker window the product uses


def _cat(paths):
    return np.concatenate([load_wave(REPO_ROOT / p, SR) for p in paths])


class Ecapa:
    def __init__(self):
        import torch
        from speechbrain.inference.speaker import EncoderClassifier

        self.torch = torch
        # a local copy (fetched once; checkpoints verified to be plain tensors) -- never the
        # hub name, which would let speechbrain fetch whatever the repo holds today
        local = str(REPO_ROOT / "models" / "external" / "spkrec-ecapa-voxceleb")
        self.m = EncoderClassifier.from_hparams(
            source=local,
            savedir=local,
            run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        )

    def embed(self, y: np.ndarray) -> np.ndarray:
        with self.torch.no_grad():
            e = self.m.encode_batch(self.torch.from_numpy(y).float().unsqueeze(0))
        e = e.squeeze().cpu().numpy()
        return e / (np.linalg.norm(e) + 1e-9)


def _cos(a, b) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _speech_only(y: np.ndarray) -> np.ndarray:
    """Concatenate the VAD's speech segments (the whole clip if none were found)."""
    _, _, segs = analyze(y, SR)
    if not segs:
        return y
    out = np.concatenate([y[int(a * SR) : int(b * SR)] for a, b in segs])
    return out if len(out) > SR else y


def build(args) -> pd.DataFrame:
    cfg = load_config()
    df = load_manifest(resolve(cfg, "manifests") / f"asvspoof19_la_{args.split}.csv")
    bona, spoof = df[df.label == "bonafide"], df[df.label == "spoof"]
    ok = eligible_speakers(bona, spoof)
    spk = choose_speakers(ok, args.speakers, args.seed)
    print(f"{args.split}: {len(spk)} speakers", flush=True)

    wavlm, ecapa = SpeakerEmbedder(), Ecapa()
    models = {"wavlm": wavlm.embed, "ecapa": ecapa.embed}
    out_path = TAB / f"speaker_models_{args.split}.csv"
    rows, done = [], set()
    if args.resume and out_path.exists():
        prev = pd.read_csv(out_path)
        rows, done = prev.to_dict("records"), set(prev.speaker.unique())

    for s in tqdm(spk, desc="speakers"):
        if s in done:
            continue
        plan = plan_speaker(bona, spoof, ok, s, args.seed)
        enrol = _cat(plan.enrol)
        ref = {k: f(enrol) for k, f in models.items()}
        ref.update({f"{k}_vad": f(_speech_only(enrol)) for k, f in models.items()})
        for ti, (kind, att, paths) in enumerate(plan.trials):
            base = _cat(paths)
            for cond in CONDS:
                babble = lambda r, n, pool=plan.pool: _cat(  # noqa: E731
                    list(r.choice(pool, size=3, replace=False))
                )
                y = apply_condition(cond, base, plan.rng, babble=babble)
                w = y[-int(WINDOW_S * SR) :]
                row = dict(split=args.split, speaker=s, trial=ti, kind=kind, attack=att, cond=cond)
                for k, f in models.items():
                    row[f"sim_{k}"] = _cos(f(w), ref[k])
                    row[f"sim_{k}_vad"] = _cos(f(_speech_only(w)), ref[f"{k}_vad"])
                rows.append(row)
        pd.DataFrame(rows).to_csv(out_path, index=False)
    print("wrote", out_path)
    return pd.DataFrame(rows)


def _eer(pos: np.ndarray, neg: np.ndarray) -> float:
    """EER where higher score = more likely the enrolled speaker (genuine=pos)."""
    if len(pos) < 2 or len(neg) < 2:
        return float("nan")
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    return compute_eer(y, np.r_[pos, neg])[0]


def report(d: pd.DataFrame) -> None:
    # z-score each model on the OTHER speakers' genuine+impostor reads (grouped folds)
    cols = [c for c in d.columns if c.startswith("sim_")]
    spk = d.speaker.unique()
    folds = np.array_split(np.random.default_rng(0).permutation(spk), 4)
    for c in cols:
        z = np.zeros(len(d))
        for f in folds:
            te = d.speaker.isin(f)
            ref = d[~te & d.kind.isin(["genuine", "impostor"])][c]
            z[te.values] = (d.loc[te, c] - ref.mean()) / (ref.std() + 1e-9)
        d[f"z_{c}"] = z
    for v in ("", "_vad"):
        d[f"sim_both{v}"] = (d[f"z_sim_wavlm{v}"] + d[f"z_sim_ecapa{v}"]) / 2

    systems = [
        "sim_wavlm",
        "sim_ecapa",
        "sim_both",
        "sim_wavlm_vad",
        "sim_ecapa_vad",
        "sim_both_vad",
    ]
    for against in ("impostor", "clone"):
        print(f"\nEER (%), enrolled speaker vs {against.upper()}   (lower is better)")
        t = []
        for cond in ("ALL", *CONDS):
            g = d if cond == "ALL" else d[d.cond == cond]
            gen = g[g.kind == "genuine"]
            neg = g[g.kind == against]
            t.append(
                {"cond": cond}
                | {s.replace("sim_", ""): 100 * _eer(gen[s].values, neg[s].values) for s in systems}
            )
        print(pd.DataFrame(t).round(1).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="eval", choices=["dev", "eval"])
    ap.add_argument("--speakers", type=int, default=0, help="0 = all eligible")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--report", action="store_true", help="only re-print from the cached table")
    args = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    path = TAB / f"speaker_models_{args.split}.csv"
    d = pd.read_csv(path) if args.report and path.exists() else build(args)
    report(d)


if __name__ == "__main__":
    main()
