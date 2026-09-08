"""Fine-tune pretrained AASIST on modern TTS (from make_modern_spoof.py)
so the acoustic signal fires on current-generation synthesis, not just
2019-era attacks. Keeps a slice of ASVspoof in the mix so it does not
forget the old attacks or start false-flagging real speech.

    python scripts/make_modern_spoof.py --n 400
    python scripts/finetune_modern.py --epochs 12

Saves the best checkpoint to models/pretrained/AASIST_modern.pth. Use it
with `build_scorer("aasist", weights="AASIST_modern.pth")` or
`scripts/serve_demo.py --weights AASIST_modern.pth`.
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

from voiceguard.audio.augment import AugmentConfig
from voiceguard.config import REPO_ROOT, load_config, preprocess_config, resolve
from voiceguard.data import load_manifest
from voiceguard.data.torch_dataset import CMDataset
from voiceguard.eval.metrics import compute_eer
from voiceguard.models.neural import build_cm

torch.set_num_threads(2)
MODERN = REPO_ROOT / "data" / "modern_spoof" / "manifest.csv"
OUT = REPO_ROOT / "models" / "pretrained" / "AASIST_modern.pth"
PRETRAINED = REPO_ROOT / "models" / "pretrained" / "AASIST.pth"


def _abs(p: str) -> str:
    q = REPO_ROOT / p
    return str(q) if q.exists() else p


@torch.no_grad()
def _eer(model, loader, device) -> float:
    model.eval()
    s, y = [], []
    for wav, lab in loader:
        s.append(model.score_bonafide(wav.to(device)).float().cpu().numpy())
        y.append((lab == 0).numpy().astype(int))  # 1 = bonafide for the metric
    return compute_eer(np.concatenate(y), np.concatenate(s))[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--asv-per-class", type=int, default=500, help="ASVspoof clips kept in the mix")
    ap.add_argument("--dev-eer-n", type=int, default=600)
    # don't save a checkpoint whose ASVspoof dev EER regressed past this
    ap.add_argument("--max-dev-eer", type=float, default=0.14)
    args = ap.parse_args()

    if not MODERN.exists():
        raise SystemExit("run scripts/make_modern_spoof.py first")
    if not PRETRAINED.exists():
        raise SystemExit(f"missing {PRETRAINED}")

    cfg = load_config()
    seed = cfg["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pcfg = preprocess_config(cfg)
    man = resolve(cfg, "manifests")

    # modern spoof, split 80/20
    md = pd.read_csv(MODERN)
    md["path"] = md["path"].map(_abs)
    md["label"] = "spoof"
    md = md.sample(frac=1, random_state=seed).reset_index(drop=True)
    cut = int(len(md) * 0.8)
    md_tr, md_ev = md.iloc[:cut], md.iloc[cut:]

    # ASVspoof rehearsal slice
    asv = load_manifest(man / "asvspoof19_la_train.csv")
    k = args.asv_per_class
    asv_tr = pd.concat(
        [g.sample(min(k, len(g)), random_state=seed) for _, g in asv.groupby("label")]
    )
    real_ev = asv[asv.label == "bonafide"].sample(len(md_ev), random_state=seed + 1)

    train_df = (
        pd.concat([md_tr[["path", "label"]], asv_tr[["path", "label"]]])
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )
    modern_eval_df = pd.concat([md_ev[["path", "label"]], real_ev[["path", "label"]]]).reset_index(
        drop=True
    )
    dev = load_manifest(man / "asvspoof19_la_dev.csv")
    dev_df = pd.concat(
        [g.sample(min(args.dev_eer_n, len(g)), random_state=seed) for _, g in dev.groupby("label")]
    ).reset_index(drop=True)

    print(
        f"train {len(train_df)} ({len(md_tr)} modern + {len(asv_tr)} ASV) | "
        f"modern-eval {len(modern_eval_df)} | dev-eval {len(dev_df)} | device {device}"
    )

    aug = AugmentConfig(enabled=True, codec_prob=0.3, rawboost_prob=0.6)
    mk = lambda df, tr, a: DataLoader(  # noqa: E731
        CMDataset(df, pcfg, a, train=tr, seed=seed),
        batch_size=args.batch_size,
        shuffle=tr,
        drop_last=tr,
        num_workers=0,
    )
    train_ld = mk(train_df, True, aug)
    modern_ld = mk(modern_eval_df, False, None)
    dev_ld = mk(dev_df, False, None)

    model = build_cm("aasist").to(device)
    model.backbone.load_state_dict(torch.load(PRETRAINED, map_location=device))
    print("loaded pretrained AASIST")

    n_bona = int((train_df.label == "bonafide").sum())
    n_spoof = len(train_df) - n_bona
    w = torch.tensor([n_spoof / max(n_bona, 1), 1.0], dtype=torch.float32, device=device)
    crit = torch.nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr / 20)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    base_m, base_d = _eer(model, modern_ld, device), _eer(model, dev_ld, device)
    print(f"epoch 0 (pretrained): modern EER {base_m * 100:.1f}%  dev EER {base_d * 100:.1f}%")
    best = {"modern": base_m, "epoch": 0}
    log = [{"epoch": 0, "modern_eer": base_m * 100, "dev_eer": base_d * 100}]

    for ep in range(1, args.epochs + 1):
        model.train()
        t0, run = time.time(), 0.0
        for wav, lab in train_ld:
            wav, lab = wav.to(device, non_blocking=True), lab.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                loss = crit(model(wav), lab)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run += loss.item()
        sched.step()
        me, de = _eer(model, modern_ld, device), _eer(model, dev_ld, device)
        print(
            f"epoch {ep}: loss {run / len(train_ld):.3f}  modern EER {me * 100:.1f}%  "
            f"dev EER {de * 100:.1f}%  ({time.time() - t0:.0f}s)",
            flush=True,
        )
        log.append({"epoch": ep, "modern_eer": me * 100, "dev_eer": de * 100})
        pd.DataFrame(log).to_csv(resolve(cfg, "tables") / "finetune_modern_log.csv", index=False)
        if me < best["modern"] and de <= args.max_dev_eer:
            best = {"modern": me, "dev": de, "epoch": ep}
            torch.save(model.backbone.state_dict(), OUT)
            print(f"  ^ saved -> {OUT.name}  (modern {me * 100:.1f}%, dev {de * 100:.1f}%)")

    print(
        f"\ndone. best modern EER {best['modern'] * 100:.1f}% @ epoch {best['epoch']} "
        f"(pretrained was {base_m * 100:.1f}%)"
    )
    if not OUT.exists():
        print("no checkpoint beat the pretrained modern EER within the dev-EER budget.")


if __name__ == "__main__":
    main()
