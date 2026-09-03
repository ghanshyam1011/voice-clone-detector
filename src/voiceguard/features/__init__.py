"""Feature extractors. All operate on an already-preprocessed waveform
(voiceguard.audio.preprocess_*), never on a raw file, so the anti-shortcut
front-end can never be bypassed.
"""

from voiceguard.features.handcrafted import DIM as HANDCRAFTED_DIM
from voiceguard.features.handcrafted import extract as extract_handcrafted

__all__ = ["extract_handcrafted", "HANDCRAFTED_DIM"]
