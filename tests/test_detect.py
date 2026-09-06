"""Smoke tests for the windowed scorer. Skipped without the pretrained
weights (they're committed to the repo, so normally present)."""

from pathlib import Path

import numpy as np
import pytest

_W = Path(__file__).resolve().parents[1] / "models" / "pretrained" / "AASIST.pth"
needs_weights = pytest.mark.skipif(not _W.exists(), reason="pretrained AASIST weights absent")


@needs_weights
def test_scorer_end_to_end():
    from voiceguard.detect import build_scorer

    sc = build_scorer("aasist", pretrained=True)
    sr = sc.sr
    tone = 0.1 * np.sin(2 * np.pi * 180 * np.arange(int(sr * 6)) / sr).astype(np.float32)
    v = sc.score_waveform(tone, hop_s=1.0)

    assert v.label in {"GENUINE", "UNCERTAIN", "SYNTHETIC"}
    assert 0.0 <= v.risk <= 1.0
    assert len(v.window_risks) >= 3  # 6 s clip, 4 s window, 1 s hop
    assert np.all((v.window_risks >= 0) & (v.window_risks <= 1))
    assert v.smoothed.shape == v.window_risks.shape


@needs_weights
def test_calibration_spreads_the_scores():
    from voiceguard.detect import build_scorer

    sc = build_scorer("aasist", pretrained=True)
    # raw AASIST pins P(synthetic) near 1.0; a value at the operating point
    # must calibrate to roughly 0.5, not stay pinned high
    cal = sc._calibrate_risk(np.array([sc.operating_point]))[0]
    assert 0.3 < cal < 0.7
