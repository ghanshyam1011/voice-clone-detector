"""The fine-tune's data pipeline is where "robust" is made or broken, so its guarantees are
tested directly: held-out situations are never trained on, degradation never depends on the
class (it must not become a shortcut), and v2 runs the front-end last, as serving does."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch

from voiceguard.audio.augment import AugmentConfig
from voiceguard.audio.degrade import STRESS_CONDITIONS
from voiceguard.audio.io import rms_dbfs
from voiceguard.audio.preprocess import PreprocessConfig
from voiceguard.data.torch_dataset import NB_SAMP

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "finetune_robust.py"
_spec = importlib.util.spec_from_file_location("finetune_robust", _PATH)
ft = importlib.util.module_from_spec(_spec)
sys.modules["finetune_robust"] = ft
_spec.loader.exec_module(ft)

SR = 16000
PCFG = PreprocessConfig()
NO_AUG = AugmentConfig(enabled=True, codec_prob=0.0, rawboost_prob=0.0)  # no ffmpeg, no RawBoost


def _voice_like(seconds=3.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    f0 = 120 + 20 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    y = sum(np.sin(h * phase) / h for h in range(1, 12)) * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * t))
    return (0.3 * y / np.abs(y).max() + 0.002 * rng.standard_normal(len(t))).astype(np.float32)


@pytest.fixture
def manifest(tmp_path):
    p = tmp_path / "a.wav"
    sf.write(p, _voice_like(), SR)
    # the SAME audio under both labels: any difference between the two items could only
    # come from the label, which is exactly what must not matter
    return pd.DataFrame({"path": [str(p), str(p)], "label": ["bonafide", "spoof"]})


@pytest.fixture
def pool():
    return [_voice_like(3.0, seed=s) for s in range(1, 5)]


def _ds(manifest, pool, **kw):
    base = dict(train=False, seed=0, babble_pool=pool)
    return ft.NoisyCMDataset(manifest, PCFG, kw.pop("augment", None), **{**base, **kw})


def test_the_held_out_situations_are_never_trained_on():
    assert set(ft.V2_SITUATIONS).isdisjoint(ft.HELD_OUT_SITUATIONS)
    assert set(ft.V2_SITUATIONS) | set(ft.HELD_OUT_SITUATIONS) <= set(STRESS_CONDITIONS)
    assert all(0 < p < 1 for p in ft.V2_SITUATIONS.values())


def test_degradation_does_not_depend_on_the_class(manifest, pool, monkeypatch):
    real = np.random.default_rng
    monkeypatch.setattr(np.random, "default_rng", lambda *a, **k: real(1234))
    ds = _ds(
        manifest,
        pool,
        train=True,
        augment=NO_AUG,
        babble_prob=1.0,
        pink_prob=0.0,
        situations={"reverb": 1.0},
        serving_order=True,
    )
    (x_bona, y_bona), (x_spoof, y_spoof) = ds[0], ds[1]
    assert (y_bona, y_spoof) == (0, 1)
    assert torch.equal(x_bona, x_spoof)
    clean = _ds(manifest, pool, serving_order=True)[0][0]
    assert not torch.allclose(x_bona, clean)  # the draw really did degrade the clip


def test_v2_runs_the_front_end_after_the_degradation(manifest, pool):
    kw = dict(fixed=("situation", "quiet"))
    v1 = _ds(manifest, pool, serving_order=False, **kw)[0][0].numpy()
    v2 = _ds(manifest, pool, serving_order=True, **kw)[0][0].numpy()
    assert rms_dbfs(v1) < -40  # front-end first, then the line made it quiet
    assert rms_dbfs(v2) > rms_dbfs(v1) + 15  # degrade first, then loudness normalisation


@pytest.mark.parametrize("fixed", [("babble", 5.0), ("pink", 10.0), ("mix", ["music", "burst"])])
def test_evaluation_sets_are_deterministic(manifest, pool, fixed):
    ds = _ds(manifest, pool, fixed=fixed, serving_order=True, seed=3)
    assert torch.equal(ds[0][0], ds[0][0])


def test_every_item_is_a_finite_fixed_length_waveform(manifest, pool):
    for situation in ft.V2_SITUATIONS:
        x, _ = _ds(manifest, pool, fixed=("situation", situation), serving_order=True)[0]
        assert x.dtype == torch.float32 and x.shape == (NB_SAMP,) and torch.isfinite(x).all()
