"""The attack generator runs third-party model code, so the parts that guard that are tested:
the checkpoint's eval() string must look like a layer spec and never like code, and a missing
model is an error rather than a download."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "make_modern_clones.py"
_spec = importlib.util.spec_from_file_location("make_modern_clones", _PATH)
mc = importlib.util.module_from_spec(_spec)
sys.modules["make_modern_clones"] = mc
_spec.loader.exec_module(mc)

WAVLM_LARGE_LAYERS = "[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2"


def test_the_real_wavlm_layer_spec_passes_the_guard():
    assert mc._CONV_LAYERS.fullmatch(WAVLM_LARGE_LAYERS)


@pytest.mark.parametrize(
    "evil",
    [
        "__import__('os').system('calc')",
        "[(512,10,5)] + open('x').read()",
        "[(512,10,5)]; import os",
        "[x for x in range(3)]",
        "",
    ],
)
def test_anything_that_looks_like_code_is_refused(evil):
    assert not mc._CONV_LAYERS.fullmatch(evil)


def test_a_missing_model_is_an_error_not_a_download(tmp_path, monkeypatch):
    monkeypatch.setattr(mc, "KNN_CODE", tmp_path / "no-code")
    monkeypatch.setattr(mc, "KNN_WEIGHTS", tmp_path / "no-weights")
    with pytest.raises(SystemExit, match="missing"):
        mc._load_knnvc("cpu")


def test_written_clips_are_16k_mono_and_peak_limited(tmp_path):
    out = tmp_path / "a.wav"
    mc._write(np.sin(np.linspace(0, 400, 24000)).astype(np.float32) * 3.0, 24000, out)
    y, sr = sf.read(out)
    assert sr == 16000 and y.ndim == 1 and np.abs(y).max() <= 0.91
    assert len(y) == pytest.approx(16000, abs=2)  # 1 s at 24 kHz became 1 s at 16 kHz


def test_manifest_rows_use_portable_relative_paths():
    p = mc.REPO_ROOT / "data" / "modern" / "knnvc" / "bona_LA_0001_00.wav"
    r = mc._row("knnvc", "LA_0001", "eval", p, "source=bona")
    assert r["path"] == "data/modern/knnvc/bona_LA_0001_00.wav"
    assert r["label"] == "spoof" and r["family"] == "wavlm-knn-vc"
