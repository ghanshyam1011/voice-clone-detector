import numpy as np
import pytest

from voiceguard.audio.channel import FEATURE_NAMES, channel_features

SR = 16000


def _tone_bursts(rng, snr_db=None):
    t = np.arange(SR * 4) / SR
    y = 0.3 * np.sin(2 * np.pi * 220 * t) * (np.sin(2 * np.pi * 2 * t) > 0)  # speech-like on/off
    if snr_db is not None:
        n = rng.standard_normal(len(y))
        n *= np.sqrt(np.mean(y**2) / 10 ** (snr_db / 10)) / n.std()
        y = y + n
    return y.astype(np.float32)


def test_returns_all_named_finite_features():
    f = channel_features(_tone_bursts(np.random.default_rng(0)))
    assert set(f) == set(FEATURE_NAMES)
    assert all(np.isfinite(v) for v in f.values())


def test_background_noise_raises_floor_and_cuts_dynamic_range():
    rng = np.random.default_rng(0)
    clean, noisy = channel_features(_tone_bursts(rng)), channel_features(_tone_bursts(rng, 5))
    assert noisy["floor_rel_db"] > clean["floor_rel_db"] + 10
    assert noisy["dyn_db"] < clean["dyn_db"]
    assert noisy["flatness"] > clean["flatness"]


def test_band_limited_audio_has_low_high_frequency_share():
    rng = np.random.default_rng(1)
    wide = rng.standard_normal(SR * 3).astype(np.float32)
    spec = np.fft.rfft(wide)
    spec[np.fft.rfftfreq(len(wide), 1 / SR) > 3400] = 0
    narrow = np.fft.irfft(spec, len(wide)).astype(np.float32)
    assert channel_features(narrow)["hf_ratio"] < 0.05 < channel_features(wide)["hf_ratio"]


def test_very_short_input_does_not_crash():
    assert np.isfinite(channel_features(np.zeros(100, np.float32))["level_db"])


def test_every_model_input_is_invariant_to_recording_gain():
    y = _tone_bursts(np.random.default_rng(3), snr_db=15)
    base = channel_features(y)
    for gain_db in (-40, -20, 12):
        scaled = channel_features((y * 10 ** (gain_db / 20)).astype(np.float32))
        for k in set(FEATURE_NAMES) - {"level_db"}:  # candidates must be gain-free too
            assert abs(scaled[k] - base[k]) < 0.5, (gain_db, k, scaled[k], base[k])
        # the absolute level, by contrast, moves one-for-one with the gain
        assert abs(scaled["level_db"] - (base["level_db"] + gain_db)) < 0.5


def _syllables(rng, seconds=8.0):
    """Noise carrier under a syllable-rate envelope: sharp onsets, short decays."""
    n = int(seconds * SR)
    env = np.zeros(n)
    attack, decay, t = int(0.02 * SR), int(0.12 * SR), 0.2
    while t < seconds - 0.3:
        i = int(t * SR)
        env[i : i + attack] += np.linspace(0, 1, attack)
        env[i + attack : i + attack + decay] += np.exp(-np.arange(decay) / (0.04 * SR))
        t += rng.uniform(0.15, 0.30)
    y = env * rng.standard_normal(n)
    return (0.3 * y / np.abs(y).max()).astype(np.float32)


def test_clipping_shows_as_a_flat_top_and_a_low_crest_factor():
    y = _syllables(np.random.default_rng(0))
    clipped = np.clip(y * 8, -0.6, 0.6).astype(np.float32)
    a, b = channel_features(y), channel_features(clipped)
    assert b["flat_top_db"] > a["flat_top_db"] + 10
    assert b["crest_db"] < a["crest_db"] - 3


@pytest.mark.parametrize("mains", [50, 60])
def test_mains_hum_is_found_at_either_frequency(mains):
    y = _syllables(np.random.default_rng(1))
    t = np.arange(len(y)) / SR
    hum = 0.04 * sum(np.sin(2 * np.pi * mains * k * t) / k for k in (1, 2, 3, 4))
    clean, humming = channel_features(y), channel_features((y + hum).astype(np.float32))
    assert clean["hum_db"] < 4
    assert humming["hum_db"] > clean["hum_db"] + 8


def test_a_missing_low_band_lowers_the_low_frequency_share():
    wide = np.random.default_rng(2).standard_normal(SR * 3).astype(np.float32)
    spec = np.fft.rfft(wide)
    spec[np.fft.rfftfreq(len(wide), 1 / SR) < 300] = 0
    high_passed = np.fft.irfft(spec, len(wide)).astype(np.float32)
    assert channel_features(high_passed)["lf_ratio_db"] < channel_features(wide)["lf_ratio_db"] - 10


def test_reverberation_raises_and_steady_noise_lowers_the_envelope_correlation():
    from voiceguard.audio.degrade import pink_noise, reverberate

    rng = np.random.default_rng(4)
    y = _syllables(rng)
    dry = channel_features(y)["env_acf"]
    wet = channel_features(reverberate(y, rng, t60=0.8))["env_acf"]
    noisy = channel_features((y + 0.03 * pink_noise(len(y), rng)).astype(np.float32))["env_acf"]
    assert wet > dry + 0.05
    assert noisy < dry - 0.05


def test_a_clean_ordinary_call_gets_no_audio_notes():
    from voiceguard.audio.channel import describe_channel

    clean = dict(flat_top_db=-43.0, hum_db=0.6, env_acf=0.67, floor_rel_db=-17.0)
    assert describe_channel(clean) == []
    # the edge of what clean speech does (its 95th percentile) must still say nothing
    assert (
        describe_channel(dict(flat_top_db=-40.5, hum_db=1.8, env_acf=0.79, floor_rel_db=-6.2)) == []
    )


@pytest.mark.parametrize(
    "affected, word",
    [
        (dict(flat_top_db=-6.0), "distorted"),
        (dict(hum_db=19.0), "hum"),
        (dict(env_acf=0.86), "echo"),
        (dict(floor_rel_db=-4.0), "noise"),
    ],
)
def test_each_measured_degradation_is_named_in_plain_words(affected, word):
    from voiceguard.audio.channel import describe_channel

    clean = dict(flat_top_db=-43.0, hum_db=0.6, env_acf=0.67, floor_rel_db=-17.0)
    notes = describe_channel({**clean, **affected})
    assert len(notes) == 1 and word in notes[0].lower()


def test_several_problems_are_all_reported():
    from voiceguard.audio.channel import describe_channel

    notes = describe_channel(dict(flat_top_db=-6.0, hum_db=19.0, env_acf=0.86, floor_rel_db=-4.0))
    assert len(notes) == 4


def test_the_notes_fire_on_real_signals_not_only_on_hand_made_numbers():
    from voiceguard.audio.channel import describe_channel

    rng = np.random.default_rng(6)
    y = _syllables(rng)
    assert describe_channel(channel_features(y)) == []
    clipped = np.clip(y * 8, -0.6, 0.6).astype(np.float32)
    assert any("distorted" in n for n in describe_channel(channel_features(clipped)))
    t = np.arange(len(y)) / SR
    hum = 0.04 * sum(np.sin(2 * np.pi * 50 * k * t) / k for k in (1, 2, 3, 4))
    assert any("hum" in n for n in describe_channel(channel_features((y + hum).astype(np.float32))))
