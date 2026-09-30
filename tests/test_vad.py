"""Speech-onset detection -- used to report first-read latency relative to
when the caller actually started talking, not relative to call-connect."""

import numpy as np

from voiceguard.audio.preprocess import PreprocessConfig
from voiceguard.audio.vad import speech_onset_s

SR = 16000


def _tone(seconds: float, freq: float = 220.0, amp: float = 0.2) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_onset_after_leading_silence():
    clip = np.concatenate([np.zeros(SR, np.float32), _tone(1.0)])  # 1.0 s silence + speech
    onset = speech_onset_s(clip, SR)
    assert onset is not None
    assert abs(onset - 1.0) < 0.1  # within ~100 ms of the true onset


def test_onset_near_zero_when_already_speaking():
    clip = _tone(1.0)
    onset = speech_onset_s(clip, SR)
    assert onset is not None
    assert onset < 0.05


def test_no_onset_in_pure_silence():
    assert speech_onset_s(np.zeros(SR, np.float32), SR) is None


def test_no_onset_in_empty_clip():
    assert speech_onset_s(np.zeros(0, np.float32), SR) is None


def test_onset_respects_a_stricter_threshold():
    # a quieter "speech" segment that a stricter (lower top_db) config
    # would not count as speech at all
    quiet = np.concatenate([np.zeros(SR, np.float32), _tone(0.5, amp=0.02)])
    loose = speech_onset_s(quiet, SR, PreprocessConfig(trim_top_db=40.0))
    strict = speech_onset_s(quiet, SR, PreprocessConfig(trim_top_db=5.0))
    assert loose is not None
    assert strict is None or strict > loose - 0.05
