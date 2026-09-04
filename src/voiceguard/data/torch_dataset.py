"""torch Dataset for training raw-waveform countermeasures.

Pipeline per item:  load -> anti-shortcut front-end (trim + loudness) ->
[augment, train only] -> fix to nb_samp (tile-pad short, random crop long,
matching the AASIST reference) -> (waveform, label).

label: 0 = bonafide, 1 = spoof  (repo convention).
Augmentation, when on, is applied the SAME way to both classes, and varies
every epoch (fresh entropy). Eval is deterministic (no augmentation, no
random crop).
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from voiceguard.audio import load_wave, preprocess_wave
from voiceguard.audio.augment import AugmentConfig, augment_waveform
from voiceguard.audio.preprocess import PreprocessConfig

NB_SAMP = 64600


def _fix_length(
    x: np.ndarray, nb_samp: int, rng: np.random.Generator, random_crop: bool
) -> np.ndarray:
    n = len(x)
    if n >= nb_samp:
        start = int(rng.integers(0, n - nb_samp + 1)) if random_crop else 0
        return x[start : start + nb_samp]
    reps = nb_samp // n + 1
    return np.tile(x, reps)[:nb_samp]


class CMDataset(Dataset):
    def __init__(
        self,
        manifest,
        pcfg: PreprocessConfig,
        augment: AugmentConfig | None = None,
        *,
        nb_samp: int = NB_SAMP,
        train: bool = True,
        seed: int = 42,
    ):
        self.rows = manifest.reset_index(drop=True)
        self.pcfg = pcfg
        self.augment = augment
        self.nb_samp = nb_samp
        self.train = train
        self.seed = seed

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows.iloc[idx]
        do_aug = self.train and self.augment is not None and self.augment.enabled

        if do_aug:
            rng = np.random.default_rng()  # fresh entropy -> new augmentation each epoch
            np.random.seed()  # RawBoost reference uses numpy's global RNG
        else:
            rng = np.random.default_rng(self.seed * 1_000_003 + idx)  # reproducible eval

        y = load_wave(row["path"], self.pcfg.sample_rate)
        y = preprocess_wave(y, self.pcfg)
        if do_aug:
            y = augment_waveform(y, self.pcfg.sample_rate, self.augment, rng)

        y = _fix_length(y, self.nb_samp, rng, random_crop=self.train)
        label = 1 if row["label"] == "spoof" else 0
        return torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32)), label
