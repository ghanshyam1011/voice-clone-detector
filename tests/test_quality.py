"""Input-quality gate: sanitise the audio, find the speech, refuse to judge what
isn't speech, and leave normal audio exactly as it was."""

from pathlib import Path

import numpy as np
import pytest

from voiceguard.audio import quality as q

SR = 16000
DEMO = Path(__file__).resolve().parents[1] / "demo_audio" / "benign.wav"
needs_demo = pytest.mark.skipif(not DEMO.exists(), reason="demo_audio/benign.wav absent")


def _speechlike(sec=6.0, seed=0, amp=0.3):
    """Harmonic source with a 3 Hz syllable-rate envelope: voiced-looking, fluctuating."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(sec * SR)) / SR
    y = sum(np.sin(2 * np.pi * 150 * k * t) / k for k in range(1, 12))
    y *= 0.5 * (1 + np.sin(2 * np.pi * 3 * t))
    y += 0.01 * rng.standard_normal(len(t))
    return (amp * y / np.max(np.abs(y))).astype(np.float32)


@pytest.fixture
def energy_vad(monkeypatch):
    """Force the dependency-free fallback so these tests are deterministic anywhere."""
    monkeypatch.setattr(q, "_vad_model", False)


# ---------- sanitise ----------


def test_normal_audio_is_returned_untouched():
    y = _speechlike()
    out, notes, corrupt = q.sanitise(y)
    assert np.array_equal(out, y) and notes == [] and not corrupt


def test_nan_and_inf_are_replaced_not_propagated():
    y = _speechlike()
    y[100], y[200] = np.nan, np.inf
    out, notes, corrupt = q.sanitise(y)
    assert np.all(np.isfinite(out)) and not corrupt
    assert out[100] == 0.0 and out[200] == 0.0 and "2 non-finite" in notes[0]


def test_mostly_nan_is_a_corrupt_stream():
    y = np.full(SR, np.nan, np.float32)
    _, notes, corrupt = q.sanitise(y)
    assert corrupt and "corrupt" in notes[0]


def test_int16_scaled_samples_are_rescaled():
    y = _speechlike()
    out, notes, _ = q.sanitise(y * 32767)
    assert np.max(np.abs(out)) <= 1.0 and "16-bit" in notes[0]
    assert np.allclose(out, y, atol=1e-3)


def test_dc_offset_is_removed_but_a_tiny_one_is_left_alone():
    y = _speechlike()
    out, notes, _ = q.sanitise(y + 0.3)
    assert abs(float(np.mean(out))) < 1e-3 and "DC offset" in notes[0]
    tiny, notes2, _ = q.sanitise(y + 0.001)
    assert notes2 == [] and np.array_equal(tiny, y + 0.001)


def test_empty_input_does_not_crash():
    out, notes, corrupt = q.sanitise(np.zeros(0, np.float32))
    assert len(out) == 0 and not corrupt


# ---------- analyze: what counts as judgeable speech (fallback detector) ----------


def test_empty_audio_is_no_audio(energy_vad):
    _, qual, segs = q.analyze(np.zeros(0, np.float32))
    assert not qual.usable and qual.status == "no_audio" and segs == []


@pytest.mark.parametrize(
    "name",
    ["silence", "white_noise", "pure_tone", "quiet_noise"],
)
def test_non_speech_is_not_judged(energy_vad, name):
    rng = np.random.default_rng(1)
    t = np.arange(6 * SR) / SR
    y = {
        "silence": np.zeros(6 * SR),
        "white_noise": 0.1 * rng.standard_normal(6 * SR),
        "pure_tone": 0.3 * np.sin(2 * np.pi * 440 * t),
        "quiet_noise": 1e-4 * rng.standard_normal(6 * SR),
    }[name].astype(np.float32)
    _, qual, _ = q.analyze(y)
    assert not qual.usable and qual.status == "no_speech", (name, qual.speech_s)
    assert "no speech" in qual.reason


def test_speech_like_audio_is_usable(energy_vad):
    _, qual, segs = q.analyze(_speechlike(6.0))
    assert qual.usable and qual.status == "ok" and qual.speech_s > 3.0 and segs
    assert not qual.provisional


def test_too_little_speech_is_waiting_then_provisional_then_full(energy_vad):
    y = _speechlike(10.0)
    short = np.concatenate([y[: int(0.6 * SR)], np.zeros(5 * SR, np.float32)])
    assert q.analyze(short)[1].status in ("too_short", "no_speech")
    mid = np.concatenate([y[: int(1.5 * SR)], np.zeros(5 * SR, np.float32)])
    qm = q.analyze(mid)[1]
    assert qm.usable and qm.provisional  # judgeable, but not on its own
    assert not q.analyze(y[: 4 * SR])[1].provisional


def test_very_quiet_speech_is_raised_before_it_reaches_the_front_end(energy_vad):
    y = _speechlike(6.0, amp=0.3) * 1e-3  # about -70 dBFS
    out, qual, _ = q.analyze(y)
    assert qual.usable and any("very quiet" in n for n in qual.notes)
    assert q._rms_dbfs(out) > -45.0


def test_normal_level_speech_passes_through_bit_identical(energy_vad):
    y = _speechlike(6.0)
    out, qual, _ = q.analyze(y)
    assert np.array_equal(out, y) and qual.notes == []


def test_clipping_is_reported(energy_vad):
    y = np.clip(_speechlike(6.0) * 6.0, -1.0, 1.0)
    _, qual, _ = q.analyze(y.astype(np.float32))
    assert qual.clipped_frac > 0.02 and any("clipped" in n for n in qual.notes)


def test_window_limits_the_measured_span(energy_vad):
    y = np.concatenate([np.zeros(20 * SR, np.float32), _speechlike(5.0)])
    _, qual, _ = q.analyze(y, window_s=8.0)
    assert qual.usable and qual.speech_s > 2.0
    _, qual_all, _ = q.analyze(y)
    assert qual_all.duration_s == pytest.approx(25.0, abs=0.01)


def test_quality_serialises_for_the_api(energy_vad):
    d = q.analyze(_speechlike(4.0))[1].as_dict()
    assert {"usable", "status", "reason", "speech_s", "level_dbfs", "notes"} <= set(d)


# ---------- the neural VAD (Silero), when installed + real speech is available ----------


def _silero_available():
    try:
        import silero_vad  # noqa: F401
    except ImportError:
        return False
    return True


needs_silero = pytest.mark.skipif(not _silero_available(), reason="silero-vad not installed")


@needs_silero
def test_silero_rejects_noise_tone_and_silence():
    rng = np.random.default_rng(2)
    t = np.arange(6 * SR) / SR
    for y in (
        np.zeros(6 * SR),
        0.1 * rng.standard_normal(6 * SR),
        0.3 * np.sin(2 * np.pi * 440 * t),
    ):
        _, qual, _ = q.analyze(y.astype(np.float32))
        assert not qual.usable and qual.method == "silero"


def _segs(p):
    return q._segments_from_probs(np.array(p, np.float32))


def test_segmenter_basic_shapes():
    fs = 512 / SR
    ((a, b),) = _segs([0.9] * 50)
    assert a == 0.0 and b == pytest.approx(50 * fs + 0.03)
    assert _segs([0.1] * 50) == []
    assert _segs([0.9] * 5) == []  # 160 ms: shorter than the 250 ms minimum
    assert _segs([]) == []


def test_segmenter_hysteresis_and_short_gaps():
    assert len(_segs([0.9] * 10 + [0.4] * 10 + [0.9] * 10)) == 1  # 0.4 is above the off threshold
    assert len(_segs([0.9] * 10 + [0.1] * 2 + [0.9] * 10)) == 1  # a 64 ms gap is bridged
    assert len(_segs([0.9] * 10 + [0.1] * 10 + [0.9] * 10)) == 2  # a 320 ms gap is not


@needs_silero
@needs_demo
def test_batched_vad_agrees_with_silero_own_segmenter():
    import soundfile as sf
    import torch
    from silero_vad import get_speech_timestamps

    y, _ = sf.read(DEMO, dtype="float32")
    ours = sum(b - a for a, b in q.speech_segments(y)[0])
    theirs = get_speech_timestamps(
        torch.from_numpy(y), q._load_vad(), sampling_rate=SR, return_seconds=True
    )
    assert ours == pytest.approx(sum(t["end"] - t["start"] for t in theirs), abs=0.3)


@needs_silero
@needs_demo
def test_silero_finds_real_speech_and_survives_distortions():
    import soundfile as sf

    y, sr = sf.read(DEMO, dtype="float32")
    assert sr == SR
    assert q.analyze(y)[1].usable
    assert q.analyze((y * 0.001).astype(np.float32))[1].usable  # -60 dB: raised, not lost
    assert q.analyze((y * 32767).astype(np.float32))[1].usable  # int16 range: rescaled
    assert q.analyze((y + 0.3).astype(np.float32))[1].usable  # DC offset: removed


# ---------- streamed PCM ----------


def test_pcm_stream_carries_a_split_sample_into_the_next_chunk():
    y = np.array([0.5, -0.25, 0.125], dtype="<f4")
    raw = y.tobytes()
    dec = q.PcmStream()
    a = dec.feed(raw[:6])  # one whole sample + 2 bytes of the next
    b = dec.feed(raw[6:])
    assert np.array_equal(np.concatenate([a, b]), y)  # nothing lost, nothing misaligned


def test_pcm_stream_turns_nan_and_inf_into_silence():
    raw = np.array([0.5, np.nan, np.inf, -0.5], dtype="<f4").tobytes()
    out = q.PcmStream().feed(raw)
    assert np.array_equal(out, np.array([0.5, 0.0, 0.0, -0.5], np.float32))


def test_pcm_stream_handles_empty_and_tiny_chunks():
    dec = q.PcmStream()
    assert len(dec.feed(b"")) == 0 and len(dec.feed(b"\x00")) == 0
    assert len(dec.feed(b"\x00\x00\x00")) == 1  # 1 + 3 bytes = one sample


def test_stream_rate_keeps_sane_rates_and_ignores_the_rest():
    assert q.stream_rate(48000) == 48000 and q.stream_rate("44100") == 44100
    for bad in (0, -5, 999999999, "abc", None, 1.5e12):
        assert q.stream_rate(bad, 16000) == 16000
