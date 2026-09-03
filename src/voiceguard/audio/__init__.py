"""Audio front-end: one path in, a clean mono 16 kHz waveform out.

`preprocess_file` / `preprocess_wave` are the ONLY entry points feature
extractors and models may use. They enforce identical trim + loudness
handling for both classes so nothing downstream can learn the
silence-duration shortcut (docs/evaluation_protocol.md).
"""

from voiceguard.audio.io import load_wave
from voiceguard.audio.preprocess import (
    PreprocessConfig,
    normalize_loudness,
    preprocess_file,
    preprocess_wave,
    trim_silence,
)

__all__ = [
    "load_wave",
    "PreprocessConfig",
    "preprocess_wave",
    "preprocess_file",
    "trim_silence",
    "normalize_loudness",
]
