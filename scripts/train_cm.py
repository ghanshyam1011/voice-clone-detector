"""Train a raw-waveform countermeasure (RawNet2 / AASIST / AASIST-L) on
ASVspoof19 LA with RawBoost + codec augmentation.

    python scripts/train_cm.py --model aasist --epochs 30
    python scripts/train_cm.py --model rawnet2 --limit-per-class 2000 --epochs 3   # smoke

Checkpoints -> models/cm/<name>/{best,last}.pt   (git-ignored)
Per-epoch dev EER -> results/tables/cm_<name>_trainlog.csv

Resumable: `last.pt` carries model/optimizer/scheduler/scaler state plus the
epoch and running log, and is overwritten every epoch. A restart picks it
straight back up unless --fresh is passed. This matters here because a
40-epoch run spans hours to days on a single laptop and WILL get
interrupted (sleep, app restart) at least once.

Sets *_NUM_THREADS=2 before importing numpy/torch: with N DataLoader
workers each independently defaulting to all-core BLAS, throughput
collapses from thread oversubscription well before 12 cores are actually
short on work -- this was the #1 cause of a measured 13s/batch.
"""

# ruff: noqa: E402  (thread-limit env vars must be set before numpy/torch import)
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

from voiceguard.audio.augment import AugmentConfig
from voiceguard.config import load_config, preprocess_config, resolve
from voiceguard.data import load_manifest
from voiceguard.data.torch_dataset import CMDataset
from voiceguard.eval.metrics import compute_eer
from voiceguard.models.neural import build_cm

torch.set_num_threads(2)


def _subsample(df, k, seed):
    if not k:
        return df
    return (
        pd.concat([g.sample(min(k, len(g)), random_state=seed) for _, g in df.groupby("label")])
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )


@torch.no_grad()
def dev_eer(model, loader, device) -> float:
    model.eval()
    scores, labels = [], []
    for wav, y in loader:
        p = model.score_bonafide(wav.to(device)).float().cpu().numpy()
        scores.append(p)
        labels.append((y == 0).numpy().astype(int))  # 1 = bonafide for metrics
    return compute_eer(np.concatenate(labels), np.concatenate(scores))[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model", default="aasist", choices=["rawnet2", "aasist", "aasist-l", "ssl-aasist"]
    )
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--limit-per-class", type=int, default=None)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="ignore last.pt, start over")
    ap.add_argument(
        "--dev-eval-per-class",
        type=int,
        default=3000,
        help="subsample dev for the per-epoch EER check (full dev is slow); 0 = full",
    )
    ap.add_argument(
        "--codec-prob",
        type=float,
        default=None,
        help="override AugmentConfig.codec_prob (ffmpeg is the slow part)",
    )
    ap.add_argument("--rawboost-prob", type=float, default=None)
    ap.add_argument(
        "--workers",
        type=int,
        default=0,
        help="DataLoader workers. 0 is often FASTER on Windows -- multiprocessing "
        "IPC + per-worker library init can cost more than it saves here.",
    )
    ap.add_argument(
        "--train-per-class",
        type=int,
        default=None,
        help="cap training clips per class (deterministic subset). Speeds up epochs; "
        "use for a first run, then rerun --fresh on the full set.",
    )
    args = ap.parse_args()

    cfg = load_config()
    seed = cfg["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pcfg = preprocess_config(cfg)
    mdir = resolve(cfg, "results").parent / "models" / "cm" / args.model
    mdir.mkdir(parents=True, exist_ok=True)
    tables = resolve(cfg, "tables")
    tables.mkdir(parents=True, exist_ok=True)
    last_path, best_path = mdir / "last.pt", mdir / "best.pt"

    man = resolve(cfg, "manifests")
    train_df = _subsample(
        load_manifest(man / "asvspoof19_la_train.csv"),
        args.train_per_class or args.limit_per_class,
        seed,
    )
    dev_df = _subsample(load_manifest(man / "asvspoof19_la_dev.csv"), args.limit_per_class, seed)
    dev_eval_df = _subsample(dev_df, args.dev_eval_per_class or None, seed)

    aug_kw = {"enabled": not args.no_augment}
    if args.codec_prob is not None:
        aug_kw["codec_prob"] = args.codec_prob
    if args.rawboost_prob is not None:
        aug_kw["rawboost_prob"] = args.rawboost_prob
    aug = AugmentConfig(**aug_kw)
    train_ds = CMDataset(train_df, pcfg, aug, train=True, seed=seed)
    dev_ds = CMDataset(dev_eval_df, pcfg, None, train=False, seed=seed)
    train_ld = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.workers > 0),
    )
    dev_ld = DataLoader(dev_ds, batch_size=args.batch_size, num_workers=args.workers)

    model = build_cm(args.model).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=5e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    start_epoch, best, log = 1, {"eer": 1.0, "epoch": -1}, []
    if last_path.exists() and not args.fresh:
        ckpt = torch.load(last_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best = ckpt["best"]
        log = ckpt["log"]
        print(f"resumed from {last_path}: continuing at epoch {start_epoch}/{args.epochs}")

    print(
        f"{args.model}: {model.param_count() / 1e6:.2f}M params "
        f"({model.param_count(trainable_only=True) / 1e6:.2f}M trainable) | device {device} | "
        f"train {len(train_ds)} dev {len(dev_ds)}/epoch | augment {aug.enabled} "
        f"(codec_prob={aug.codec_prob}) | workers {args.workers}"
    )
    if start_epoch > args.epochs:
        print("already finished per last.pt -- pass --fresh to retrain")
        return

    # bonafide (col 0) is the minority -> up-weight it
    n_bona = int((train_df["label"] == "bonafide").sum())
    n_spoof = int((train_df["label"] == "spoof").sum())
    w = torch.tensor([n_spoof / max(n_bona, 1), 1.0], dtype=torch.float32, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=w)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0, running, tb = time.time(), 0.0, time.time()
        for i, (wav, y) in enumerate(train_ld):
            wav = wav.to(device, non_blocking=True)
            target = y.to(device)  # 0 = bonafide, 1 = spoof -> matches col order
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                loss = criterion(model(wav), target)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item()
            if i % 50 == 0:
                s_per_batch = (time.time() - tb) / 50 if i else 0.0
                tb = time.time()
                eta_min = s_per_batch * (len(train_ld) - i) / 60
                print(
                    f"  e{epoch} [{i}/{len(train_ld)}] loss {loss.item():.4f} "
                    f"({s_per_batch:.1f}s/batch, ~{eta_min:.0f}min left this epoch)",
                    flush=True,
                )
        sched.step()

        eer = dev_eer(model, dev_ld, device)
        dt = time.time() - t0
        avg = running / len(train_ld)
        print(f"epoch {epoch}: loss {avg:.4f}  dev EER {eer * 100:.2f}%  ({dt:.0f}s)")
        log.append({"epoch": epoch, "train_loss": avg, "dev_eer_pct": eer * 100, "sec": dt})
        pd.DataFrame(log).to_csv(tables / f"cm_{args.model}_trainlog.csv", index=False)

        if eer < best["eer"]:
            best = {"eer": eer, "epoch": epoch}
            torch.save(
                {
                    "model": model.state_dict(),
                    "name": args.model,
                    "epoch": epoch,
                    "dev_eer": eer,
                    "front_end": pcfg.fingerprint(),
                },
                best_path,
            )
            print(f"  ^ new best, saved -> {best_path}")

        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "best": best,
                "log": log,
                "args": vars(args),
            },
            last_path,
        )

    print(f"done. best dev EER {best['eer'] * 100:.2f}% @ epoch {best['epoch']}")


if __name__ == "__main__":
    main()
