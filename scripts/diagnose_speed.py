"""One-shot diagnostic: is AASIST training slow because of the GPU, or
something else? Prints CPU vs GPU forward+backward time for a few
iterations, isolated from data loading entirely (random tensors).

    python scripts/diagnose_speed.py

Expect a few seconds total. If either number is in the tens of seconds or
more per batch, that device path has a real problem worth reporting back.
"""

# ruff: noqa: E402  (thread-limit env vars must be set before torch import)
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import time

import torch

from voiceguard.models.neural import build_cm

BATCH = 8
WARMUP = 2
ITERS = 5


def bench(device_name: str) -> None:
    print(f"\n--- {device_name} ---")
    device = torch.device(device_name)
    model = build_cm("aasist").to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    x = torch.randn(BATCH, 64600, device=device)
    y = torch.randint(0, 2, (BATCH,), device=device)

    t0 = time.time()
    for i in range(WARMUP):
        opt.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        opt.step()
        if device_name == "cuda":
            torch.cuda.synchronize()
        print(f"  warmup {i + 1}/{WARMUP}: {time.time() - t0:.2f}s elapsed so far", flush=True)

    t = time.time()
    for i in range(ITERS):
        opt.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        opt.step()
        if device_name == "cuda":
            torch.cuda.synchronize()
        print(f"  iter {i + 1}/{ITERS}: {time.time() - t:.2f}s cumulative", flush=True)
    dt = time.time() - t
    print(f"  RESULT {device_name}: {dt / ITERS * 1000:.0f} ms/batch (batch={BATCH})")


if __name__ == "__main__":
    print(f"torch {torch.__version__} | cuda available: {torch.cuda.is_available()}")
    bench("cpu")
    if torch.cuda.is_available():
        bench("cuda")
