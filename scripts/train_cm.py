"""Train a raw-waveform countermeasure (RawNet2 / AASIST / AASIST-L) on
ASVspoof19 LA with RawBoost + codec augmentation.

    python scripts/train_cm.py --model aasist --epochs 30
    python scripts/train_cm.py --model rawnet2 --limit-per-class 2000 --epochs 3   # smoke

Checkpoints -> models/cm/<name>/best.pt   (git-ignored)
Per-epoch dev EER -> results/tables/cm_<name>_trainlog.csv
"""

from __future__ import annotations

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
    ap.add_argument("--model", default="aasist", choices=["rawnet2", "aasist", "aasist-l"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--limit-per-class", type=int, default=None)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
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

    man = resolve(cfg, "manifests")
    train_df = _subsample(
        load_manifest(man / "asvspoof19_la_train.csv"), args.limit_per_class, seed
    )
    dev_df = _subsample(load_manifest(man / "asvspoof19_la_dev.csv"), args.limit_per_class, seed)

    aug = AugmentConfig(enabled=not args.no_augment)
    train_ds = CMDataset(train_df, pcfg, aug, train=True, seed=seed)
    dev_ds = CMDataset(dev_df, pcfg, None, train=False, seed=seed)
    train_ld = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
    )
    dev_ld = DataLoader(dev_ds, batch_size=args.batch_size, num_workers=args.workers)

    model = build_cm(args.model).to(device)
    print(
        f"{args.model}: {model.param_count() / 1e6:.2f}M params | device {device} | "
        f"train {len(train_ds)} dev {len(dev_ds)} | augment {aug.enabled}"
    )

    # bonafide (col 0) is the minority -> up-weight it
    n_bona = int((train_df["label"] == "bonafide").sum())
    n_spoof = int((train_df["label"] == "spoof").sum())
    w = torch.tensor([n_spoof / max(n_bona, 1), 1.0], dtype=torch.float32, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=w)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=5e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best = {"eer": 1.0, "epoch": -1}
    log = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running = time.time(), 0.0
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
                print(f"  e{epoch} [{i}/{len(train_ld)}] loss {loss.item():.4f}", flush=True)
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
                mdir / "best.pt",
            )
            print(f"  ^ new best, saved -> {mdir / 'best.pt'}")

    print(f"done. best dev EER {best['eer'] * 100:.2f}% @ epoch {best['epoch']}")


if __name__ == "__main__":
    main()
