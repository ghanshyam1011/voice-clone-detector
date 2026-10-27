"""Fine-tune AASIST to be robust to the situations real calls are recorded in.

    python scripts/finetune_robust.py --epochs 5                  # v1: background voices / noise
    python scripts/finetune_robust.py --aug v2 --init AASIST_robust.pth --out AASIST_robust2.pth \
        --epochs 4 --modern data/modern/manifest.csv --modern-train speecht5x,mms,knnvc \
        --extra-bona data/manifests/librispeech_dev-clean.csv

Problem this fixes: the published AASIST was trained on clean studio audio, so a *genuine*
caller scores as synthetic as soon as the recording is not clean (measured: mean spoof risk of
genuine calls 0.00 clean -> 0.27 at 5 dB babble -> 0.71 with a clipped microphone).

Training data: ASVspoof 2019 LA **train** only (dev/eval/In-the-Wild stay held out), plus, if
given, modern synthetic speech and a second real-speech corpus. Every clip, both classes alike,
is degraded with probability (class-independent, so degradation cannot become a shortcut):

    v1   babble 35% (3 other speakers, SNR U(0,20)), pink noise 15%, codec 30%, RawBoost 50%
    v2   babble 25%, pink 10%, codec 30%, RawBoost 50%, plus ONE room/line situation from
         voiceguard.audio.degrade with the weights in ``V2_SITUATIONS`` (reverb, speakerphone,
         narrow-band, clipping, packet loss, hum). ``music`` and ``burst`` noise are never
         trained on: they are the held-out *unseen situation* check.

Modern generators / bona fide domains
  ``--modern`` adds spoof clips from scripts/make_modern_clones.py (``--modern-train`` generators
  in training; the others are held out), ``--extra-bona`` adds bona fide clips from another corpus.
  ``--modern-frac`` / ``--extra-frac`` set what share of each class they make up.

Selection: the checkpoint with the best mean dev EER over ``--select`` (default clean + babble
5 dB; v2 adds a mix of trained situations, a mix of the held-out ones, and held-out modern
generators), subject to a clean-EER budget so clean performance is not sacrificed. Writes the
best backbone to models/pretrained/<--out> and a per-epoch log to results/tables/.
"""

# ruff: noqa: E402
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from voiceguard.audio import load_wave, preprocess_wave
from voiceguard.audio.augment import AugmentConfig, augment_waveform
from voiceguard.audio.degrade import apply_condition, mix_at_snr, pink_noise
from voiceguard.config import REPO_ROOT, load_config, preprocess_config, resolve
from voiceguard.data import load_manifest
from voiceguard.data.torch_dataset import CMDataset, _fix_length
from voiceguard.eval.metrics import compute_eer
from voiceguard.models.neural import build_cm

torch.set_num_threads(2)
SR = 16000
PRETRAINED = REPO_ROOT / "models" / "pretrained" / "AASIST.pth"

# P(apply this situation to a clip) in v2. music / burst are deliberately absent.
V2_SITUATIONS = {
    "reverb": 0.15,
    "speakerphone": 0.06,
    "narrow": 0.06,
    "clip": 0.10,  # the worst case measured (docs/stress_eval.md): genuine speech reads as a clone
    "loss": 0.08,
    "hum": 0.05,
}
HELD_OUT_SITUATIONS = ("music", "burst")


class NoisyCMDataset(CMDataset):
    """CMDataset + class-independent additive background voices / noise / room situations."""

    def __init__(
        self,
        *a,
        babble_pool=None,
        babble_prob=0.35,
        pink_prob=0.15,
        situations=None,
        fixed=None,
        serving_order=False,
        **k,
    ):
        super().__init__(*a, **k)
        # False (v1): front-end first, then degrade. True (v2): degrade -> codec -> front-end,
        # the order a real call reaches the model in (the line damages the audio, then our
        # trim + loudness normalisation runs), so the model is trained on what it will see
        self.serving_order = serving_order
        self.pool = babble_pool or []
        self.babble_prob, self.pink_prob = babble_prob, pink_prob
        self.situations = situations or {}  # name -> probability (training only)
        # eval only: ("babble", snr) | ("pink", snr) | ("situation", name) | ("mix", [names])
        self.fixed = fixed

    def _voices(self, rng, n):
        picks = [self.pool[int(rng.integers(len(self.pool)))] for _ in range(3)]
        return np.sum(picks, axis=0)

    def __getitem__(self, idx):
        row = self.rows.iloc[idx]
        aug_on = self.train and self.augment is not None and self.augment.enabled
        rng = (
            np.random.default_rng()
            if aug_on
            else np.random.default_rng(self.seed * 1_000_003 + idx)
        )
        if aug_on:
            np.random.seed()
        y = load_wave(row["path"], self.pcfg.sample_rate)
        if not self.serving_order:
            y = preprocess_wave(y, self.pcfg)

        kind, snr, situation = None, 0.0, None
        if self.fixed is not None:
            how, arg = self.fixed
            if how in ("babble", "pink"):
                kind, snr = how, float(arg)
            elif how == "situation":
                situation = arg
            else:  # "mix"
                situation = arg[int(rng.integers(len(arg)))]
        elif aug_on:
            u = rng.random()
            if u < self.babble_prob:
                kind, snr = "babble", float(rng.uniform(0, 20))
            elif u < self.babble_prob + self.pink_prob:
                kind, snr = "pink", float(rng.uniform(5, 25))
            for name, p in self.situations.items():  # at most one room/line situation
                if rng.random() < p:
                    situation = name
                    break
        if kind == "babble" and self.pool:
            y = mix_at_snr(y, self._voices(rng, len(y)), snr)
        elif kind == "pink":
            y = mix_at_snr(y, pink_noise(len(y), rng), snr)
        if situation is not None:
            y = apply_condition(situation, y, rng, babble=lambda r, n: self._voices(r, n))

        if aug_on:
            y = augment_waveform(y, self.pcfg.sample_rate, self.augment, rng)
        if self.serving_order:
            y = preprocess_wave(y, self.pcfg)
        y = _fix_length(y, self.nb_samp, rng, random_crop=self.train)
        label = 1 if row["label"] == "spoof" else 0
        return torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32)), label


class Cached:
    """A deterministic evaluation set, rendered once. Its clips are the same every epoch, so
    the (CPU-heavy) degradation pipeline only has to run on them the first time."""

    def __init__(self, loader):
        self.loader, self.batches = loader, None

    def __iter__(self):
        if self.batches is None:
            self.batches = list(self.loader)
        return iter(self.batches)


@torch.no_grad()
def _eer(model, loader, device) -> float:
    model.eval()
    s, y = [], []
    for wav, lab in loader:
        s.append(model.score_bonafide(wav.to(device)).float().cpu().numpy())
        y.append((lab == 0).numpy().astype(int))
    return compute_eer(np.concatenate(y), np.concatenate(s))[0]


def _take(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    return df.sample(min(n, len(df)), random_state=seed)


def _extra(path: str, label: str, split_filter=None) -> pd.DataFrame:
    d = load_manifest(REPO_ROOT / path if not os.path.isabs(path) else path)
    d = d[d.label == label]
    return d if split_filter is None else d[split_filter(d)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aug", default="v1", choices=["v1", "v2"])
    ap.add_argument(
        "--init", default="AASIST.pth", help="starting backbone under models/pretrained/"
    )
    ap.add_argument("--out", default="AASIST_robust.pth", help="best backbone is written here")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--per-class", type=int, default=3000)
    ap.add_argument("--dev-n", type=int, default=500, help="dev clips per class for selection")
    ap.add_argument("--max-clean-eer", type=float, default=0.14)
    ap.add_argument("--modern", default=None, help="manifest from scripts/make_modern_clones.py")
    ap.add_argument("--modern-train", default="", help="comma list of generators trained on")
    ap.add_argument("--modern-frac", type=float, default=0.3, help="share of the spoof class")
    ap.add_argument("--extra-bona", default=None, help="manifest of another bona fide corpus")
    ap.add_argument("--extra-frac", type=float, default=0.3, help="share of the bona fide class")
    ap.add_argument("--log", default=None, help="log table name (default finetune_<out>_log.csv)")
    ap.add_argument("--fresh", action="store_true", help="ignore last.pt, start over")
    ap.add_argument(
        "--workers",
        type=int,
        default=0,
        help="data-loading processes; the codec/RawBoost augmentation is CPU-bound (~0.2 s per "
        "clip), so 3-4 keeps the GPU fed",
    )
    args = ap.parse_args()

    cfg = load_config()
    seed = cfg["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pcfg = preprocess_config(cfg)
    man = resolve(cfg, "manifests")
    v2 = args.aug == "v2"

    tr = load_manifest(man / "asvspoof19_la_train.csv")
    spoof = _take(tr[tr.label == "spoof"], args.per_class, seed)
    bona = _take(tr[tr.label == "bonafide"], args.per_class, seed)
    train_gens = [g for g in args.modern_train.split(",") if g]
    if args.modern and train_gens:
        mod = pd.read_csv(args.modern)
        mod = mod[mod.generator.isin(train_gens)].assign(
            path=lambda d: [str(REPO_ROOT / p) for p in d.path]
        )
        k = min(len(mod), int(args.per_class * args.modern_frac))
        spoof = pd.concat([_take(spoof, args.per_class - k, seed), _take(mod, k, seed)])
        print(f"spoof class: {args.per_class - k} ASVspoof + {k} modern ({train_gens})")
    if args.extra_bona:
        xb = _extra(args.extra_bona, "bonafide")
        k = min(len(xb), int(args.per_class * args.extra_frac))
        bona = pd.concat([_take(bona, args.per_class - k, seed), _take(xb, k, seed)])
        print(f"bona fide class: {args.per_class - k} ASVspoof + {k} from {args.extra_bona}")
    train_df = pd.concat([spoof, bona]).sample(frac=1, random_state=seed).reset_index(drop=True)

    dev = load_manifest(man / "asvspoof19_la_dev.csv")
    dev_df = pd.concat([_take(g, args.dev_n, seed) for _, g in dev.groupby("label")]).reset_index(
        drop=True
    )

    # background voices: short clips from train bona fide speakers, held in RAM
    pool_paths = tr[tr.label == "bonafide"].sample(300, random_state=seed).path.tolist()
    pool = [np.resize(preprocess_wave(load_wave(p, SR), pcfg), 3 * SR) for p in pool_paths]
    print(f"train {len(train_df)} | dev-select {len(dev_df)} | babble pool {len(pool)} | {device}")

    aug = AugmentConfig(enabled=True, codec_prob=0.3, rawboost_prob=0.5)
    situations = V2_SITUATIONS if v2 else {}
    p_babble, p_pink = (0.25, 0.10) if v2 else (0.35, 0.15)

    def mk(df, train, fixed=None):
        ds = NoisyCMDataset(
            df,
            pcfg,
            aug if train else None,
            train=train,
            seed=seed,
            babble_pool=pool,
            babble_prob=p_babble,
            pink_prob=p_pink,
            situations=situations if train else None,
            fixed=fixed,
            serving_order=v2,
        )
        ld = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=train,
            drop_last=train,
            num_workers=args.workers,
            persistent_workers=train and args.workers > 0,
        )
        return ld if train else Cached(ld)

    train_ld = mk(train_df, True)
    dev_sets = {"clean": mk(dev_df, False), "babble5": mk(dev_df, False, ("babble", 5.0))}
    if v2:
        dev_sets["trained_mix"] = mk(dev_df, False, ("mix", list(V2_SITUATIONS)))
        dev_sets["unseen_mix"] = mk(dev_df, False, ("mix", list(HELD_OUT_SITUATIONS)))
        held_gens = []
        if args.modern:
            allgens = set(pd.read_csv(args.modern).generator)
            held_gens = sorted(allgens - set(train_gens))
        if held_gens:
            mod_dev = pd.read_csv(args.modern)
            mod_dev = mod_dev[mod_dev.generator.isin(held_gens)]
            mod_dev = mod_dev.assign(path=lambda d: [str(REPO_ROOT / p) for p in d.path])
            n = min(len(mod_dev), args.dev_n)
            held = pd.concat(
                [_take(mod_dev, n, seed), _take(dev[dev.label == "bonafide"], n, seed)]
            ).reset_index(drop=True)
            dev_sets[f"heldout_gen({'+'.join(held_gens)})"] = mk(held, False)
    select = list(dev_sets)

    model = build_cm("aasist").to(device)
    model.backbone.load_state_dict(
        torch.load(REPO_ROOT / "models" / "pretrained" / args.init, map_location=device)
    )
    n_bona = int((train_df.label == "bonafide").sum())
    w = torch.tensor(
        [(len(train_df) - n_bona) / max(n_bona, 1), 1.0], dtype=torch.float32, device=device
    )
    crit = torch.nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr / 20)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    def evaluate() -> dict[str, float]:
        return {k: _eer(model, ld, device) for k, ld in dev_sets.items()}

    out_path = REPO_ROOT / "models" / "pretrained" / args.out
    log_path = resolve(cfg, "tables") / (args.log or f"finetune_{out_path.stem}_log.csv")
    # resumable like scripts/train_cm.py: a killed run continues after its last finished epoch
    last_path = REPO_ROOT / "models" / "cm" / f"finetune_{out_path.stem}" / "last.pt"
    last_path.parent.mkdir(parents=True, exist_ok=True)
    ident = {
        k: getattr(args, k)
        for k in ("aug", "init", "epochs", "lr", "batch_size", "per_class", "modern_train")
    }

    start_ep, saved = 1, False
    if last_path.exists() and not args.fresh:
        ck = torch.load(last_path, map_location=device)
        if ck["ident"] != ident:
            raise SystemExit(
                f"{last_path} belongs to a different run ({ck['ident']}); "
                "pass --fresh to start over"
            )
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        start_ep, best, saved = ck["epoch"] + 1, ck["best"], ck["saved"]
        log, e0 = ck["log"], ck["e0"]
        print(f"resumed from {last_path}: {ck['epoch']}/{args.epochs} epochs already done")
    else:
        e0 = evaluate()
        print(
            "epoch 0 (start): " + "  ".join(f"{k} {v * 100:.1f}%" for k, v in e0.items()),
            flush=True,
        )
        best = float(np.mean(list(e0.values())))
        log = [{"epoch": 0, **{k: v * 100 for k, v in e0.items()}}]
    if start_ep > args.epochs:
        print("already finished per last.pt -- pass --fresh to retrain")
        return

    for ep in range(start_ep, args.epochs + 1):
        model.train()
        t0, tb, run = time.time(), time.time(), 0.0
        for i, (wav, lab) in enumerate(train_ld):
            wav, lab = wav.to(device, non_blocking=True), lab.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                loss = crit(model(wav), lab)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run += loss.item()
            if i and i % 50 == 0:
                s_per_batch, tb = (time.time() - tb) / 50, time.time()
                left_min = s_per_batch * (len(train_ld) - i) / 60
                print(
                    f"  e{ep} [{i}/{len(train_ld)}] loss {loss.item():.3f} "
                    f"({s_per_batch:.1f}s/batch, ~{left_min:.0f} min left this epoch)",
                    flush=True,
                )
        sched.step()
        e = evaluate()
        print(
            f"epoch {ep}: loss {run / len(train_ld):.3f}  "
            + "  ".join(f"{k} {v * 100:.1f}%" for k, v in e.items())
            + f"  ({time.time() - t0:.0f}s)",
            flush=True,
        )
        log.append(
            {
                "epoch": ep,
                "train_loss": run / len(train_ld),
                **{k: v * 100 for k, v in e.items()},
                "sec": time.time() - t0,
            }
        )
        pd.DataFrame(log).to_csv(log_path, index=False)
        score = float(np.mean(list(e.values())))
        if score < best and e["clean"] <= args.max_clean_eer:
            best, saved = score, True
            torch.save(model.backbone.state_dict(), out_path)
            print(f"  ^ saved -> {out_path.name}", flush=True)
        state = {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": ep,
            "best": best,
            "saved": saved,
            "log": log,
            "e0": e0,
            "ident": ident,
        }
        torch.save(state, last_path.with_suffix(".tmp"))  # a kill mid-write must not corrupt it
        os.replace(last_path.with_suffix(".tmp"), last_path)

    print(
        f"\ndone. best mean dev EER over [{', '.join(select)}] {best * 100:.1f}% "
        f"(start {float(np.mean(list(e0.values()))) * 100:.1f}%)"
    )
    if not saved:
        print("no checkpoint beat the starting model within the clean-EER budget.")


if __name__ == "__main__":
    main()
