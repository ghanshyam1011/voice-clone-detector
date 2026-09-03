"""The anti-shortcut front-end is the point of P0, so it gets real tests."""

import numpy as np
import pytest

from voiceguard.audio.preprocess import (
    PreprocessConfig,
    normalize_loudness,
    preprocess_wave,
    trim_silence,
)

SR = 16000


def _tone(seconds: float, freq: float = 220.0, amp: float = 0.2) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_trim_removes_long_leading_silence():
    cfg = PreprocessConfig(trim_pad_ms=40.0)
    clip = np.concatenate([np.zeros(SR, np.float32), _tone(0.5)])  # 1.0 s silence + 0.5 s tone
    out = trim_silence(clip, cfg)
    pad = 2 * int(SR * cfg.trim_pad_ms / 1000)
    assert out.size < clip.size
    assert abs(out.size - (int(SR * 0.5) + pad)) < SR * 0.1  # ~0.5 s of speech + padding


def test_trim_symmetric_for_both_leading_and_trailing():
    cfg = PreprocessConfig()
    a = np.concatenate([np.zeros(SR, np.float32), _tone(0.4), np.zeros(SR // 2, np.float32)])
    b = np.concatenate([np.zeros(SR // 4, np.float32), _tone(0.4), np.zeros(2 * SR, np.float32)])
    # different silence padding in, near-identical length out
    assert abs(trim_silence(a, cfg).size - trim_silence(b, cfg).size) < SR * 0.1


def test_loudness_normalisation_is_deterministic():
    cfg = PreprocessConfig()
    x = _tone(1.0, amp=0.05)
    assert np.allclose(normalize_loudness(x, cfg), normalize_loudness(x.copy(), cfg))


def test_loudness_brings_quiet_and_loud_clips_together():
    """The core anti-shortcut property: input level must not survive."""
    cfg = PreprocessConfig()
    quiet = normalize_loudness(_tone(1.0, amp=0.01), cfg)
    loud = normalize_loudness(_tone(1.0, amp=0.5), cfg)
    q_rms = np.sqrt(np.mean(quiet**2))
    l_rms = np.sqrt(np.mean(loud**2))
    assert q_rms > 0 and l_rms > 0
    assert abs(20 * np.log10(q_rms / l_rms)) < 2.0  # within 2 dB


def test_peak_ceiling_respected():
    cfg = PreprocessConfig(peak_ceiling_dbfs=-1.0)
    out = preprocess_wave(_tone(1.0, amp=0.9), cfg)
    assert np.max(np.abs(out)) <= 10 ** (-1.0 / 20) + 1e-4


@pytest.mark.parametrize(
    "bad", [np.zeros(SR, np.float32), np.zeros(10, np.float32), np.array([], np.float32)]
)
def test_silence_and_tiny_inputs_do_not_crash(bad):
    out = preprocess_wave(bad, PreprocessConfig())
    assert np.all(np.isfinite(out))


def test_fingerprint_changes_with_settings():
    a = PreprocessConfig(trim_top_db=30.0)
    b = PreprocessConfig(trim_top_db=25.0)
    assert a.fingerprint() != b.fingerprint()
    assert a.fingerprint() == PreprocessConfig(trim_top_db=30.0).fingerprint()
