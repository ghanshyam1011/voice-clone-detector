"""Reproducible acoustic degradations: each one must do what its name says,
deterministically, without producing NaNs or clipping the whole signal."""

import numpy as np
import pytest

from voiceguard.audio import degrade as dg
from voiceguard.audio.codec import _FFMPEG

SR = 16000


def _speechlike(sec=4.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(sec * SR)) / SR
    y = sum(np.sin(2 * np.pi * 150 * k * t) / k for k in range(1, 12))
    y *= 0.5 * (1 + np.sin(2 * np.pi * 3 * t))  # syllable-rate amplitude modulation
    y += 0.01 * rng.standard_normal(len(t))
    return (0.3 * y / np.max(np.abs(y))).astype(np.float32)


def _band_energy_db(y, lo, hi):
    spec = np.abs(np.fft.rfft(y)) ** 2
    f = np.fft.rfftfreq(len(y), 1 / SR)
    return 10 * np.log10(spec[(f >= lo) & (f < hi)].sum() + 1e-12)


def test_mix_at_snr_hits_the_requested_snr():
    y, rng = _speechlike(), np.random.default_rng(1)
    noise = dg.pink_noise(len(y), rng)
    for snr in (0.0, 10.0, 20.0):
        out = dg.mix_at_snr(y, noise, snr)
        resid = out - y  # peak_limit never triggers at these levels
        measured = 20 * np.log10(dg.rms(y) / dg.rms(resid))
        assert measured == pytest.approx(snr, abs=0.5)


def test_reverb_adds_a_tail_after_the_signal_stops():
    burst = np.zeros(2 * SR, np.float32)
    burst[:800] = np.random.default_rng(0).standard_normal(800) * 0.3
    out = dg.reverberate(burst, np.random.default_rng(2), t60=0.6, drr_db=0.0)
    tail = out[int(0.2 * SR) : int(0.8 * SR)]
    assert dg.rms(tail) > 10 * dg.rms(burst[int(0.2 * SR) : int(0.8 * SR)])
    assert dg.rms(out) == pytest.approx(dg.rms(burst), rel=0.5)  # loudness roughly kept


def test_reverb_is_deterministic_for_one_seed_and_differs_across_seeds():
    y = _speechlike()
    a = dg.reverberate(y, np.random.default_rng(5))
    b = dg.reverberate(y, np.random.default_rng(5))
    c = dg.reverberate(y, np.random.default_rng(6))
    assert np.array_equal(a, b)
    assert not np.allclose(a, c)


def test_band_limit_removes_out_of_band_energy_and_keeps_in_band():
    y = np.random.default_rng(3).standard_normal(4 * SR).astype(np.float32) * 0.1
    out = dg.band_limit(y)
    assert _band_energy_db(out, 5000, 8000) < _band_energy_db(y, 5000, 8000) - 30
    assert _band_energy_db(out, 500, 3000) == pytest.approx(_band_energy_db(y, 500, 3000), abs=1.5)


def test_hard_clip_limits_peak_and_actually_clips():
    y = _speechlike()
    out = dg.hard_clip(y, drive_db=18, ceiling=0.4)
    assert np.max(np.abs(out)) <= 0.4 + 1e-6
    assert np.mean(np.abs(out) >= 0.4 - 1e-6) > 0.05


def test_set_level_sets_rms():
    out = dg.set_level(_speechlike(), -45.0)
    assert 20 * np.log10(dg.rms(out)) == pytest.approx(-45.0, abs=0.1)


def test_packet_loss_zeroes_roughly_the_requested_fraction():
    y = np.ones(8 * SR, np.float32)
    out = dg.packet_loss(y, np.random.default_rng(4), loss=0.10)
    frac = float(np.mean(out == 0.0))
    assert 0.09 <= frac <= 0.14  # overlapping gaps can only shrink it, the loop overshoots a little


def test_mains_hum_peaks_at_50_or_60_hz():
    h = dg.mains_hum(4 * SR, np.random.default_rng(7))
    f = np.fft.rfftfreq(len(h), 1 / SR)
    peak = f[np.argmax(np.abs(np.fft.rfft(h)))]
    assert min(abs(peak - 50), abs(peak - 60)) < 1.0


def test_music_and_bursts_are_finite_unit_variance_and_the_right_length():
    for fn in (dg.music_like, dg.bursts):
        x = fn(3 * SR, np.random.default_rng(8))
        assert len(x) == 3 * SR and np.all(np.isfinite(x))
        assert np.std(x) == pytest.approx(1.0, rel=0.05)


NO_FFMPEG = _FFMPEG is None
CODEC_CONDS = ("amr", "opus8k", "g722", "tel_babble", "tel_reverb", *dg.MESSAGING_CONDITIONS)
SYNTHETIC_ONLY = [c for c in dg.STRESS_CONDITIONS if c not in CODEC_CONDS]


def _voices(rng, n):  # stand-in for "three other people talking"
    return dg.pink_noise(n, rng)


def _check_sane(y, out):
    assert out.dtype == np.float32 and np.all(np.isfinite(out))
    assert abs(len(out) - len(y)) <= 640  # codecs may pad/trim a frame
    assert np.max(np.abs(out)) <= 1.0 + 1e-6
    assert dg.rms(out) > 1e-5  # not silenced


@pytest.mark.parametrize("cond", SYNTHETIC_ONLY)
def test_every_non_codec_stress_condition_runs_and_stays_sane(cond):
    y = _speechlike()
    _check_sane(y, dg.apply_condition(cond, y, np.random.default_rng(9), babble=_voices))


@pytest.mark.skipif(NO_FFMPEG, reason="ffmpeg absent")
@pytest.mark.parametrize("cond", CODEC_CONDS)
def test_codec_conditions_run_stay_sane_and_audible(cond):
    y = _speechlike()
    out = dg.apply_condition(cond, y, np.random.default_rng(10), babble=_voices)
    _check_sane(y, out)
    assert dg.rms(out) > 0.01


def test_babble_condition_without_a_source_is_a_clear_error():
    with pytest.raises(ValueError, match="babble"):
        dg.apply_condition("babble5", _speechlike(), np.random.default_rng(0))


def test_unknown_condition_is_a_clear_error():
    with pytest.raises(ValueError, match="unknown condition"):
        dg.apply_condition("underwater", _speechlike(), np.random.default_rng(0))


@pytest.mark.skipif(NO_FFMPEG, reason="ffmpeg absent")
def test_voice_note_conditions_are_wideband_opus_not_the_telephone_band():
    # a messaging-app voice note keeps energy above 4 kHz that the 8 kHz telephone codecs cut
    rng = np.random.default_rng(11)
    y = (0.3 * rng.standard_normal(3 * SR)).astype(np.float32)  # broadband, so any cut shows
    hf = {}
    for cond in ("opus8k", "opus16", "opus24"):
        out = dg.apply_condition(cond, y, np.random.default_rng(12))
        spec = np.abs(np.fft.rfft(out)) ** 2
        f = np.fft.rfftfreq(len(out), 1 / SR)
        hf[cond] = spec[f > 4000].sum() / spec.sum()
    assert hf["opus16"] > 10 * hf["opus8k"] and hf["opus24"] > 10 * hf["opus8k"]
