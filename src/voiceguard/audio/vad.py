"""Speech-onset detection for latency accounting.

This answers a narrower question than the anti-shortcut front-end's
`trim_silence`: not "where does speech start so we can pad it fairly for
a fairness-preserving model input," but "when, on the wall clock, did the
caller actually start talking" -- so a live session can report its first-read
latency relative to speech, not relative to call-connect (which may include
several seconds of dead air the caller never gets billed for reading a
prompt).

Reuses the exact same `librosa.effects.trim` call as `preprocess.py`, so
"is this speech" is judged identically everywhere in the codebase --
there is deliberately no second, differently-tuned silence detector.
"""

from __future__ import annotations

import librosa
import numpy as np

from voiceguard.audio.preprocess import PreprocessConfig


def speech_onset_s(
    y: np.ndarray, sr: int, cfg: PreprocessConfig | None = None
) -> float | None:
    """Seconds into `y` where sustained non-silence first begins.

    Returns None if the whole clip is at or below the silence threshold
    (no speech detected yet) -- callers should treat that as "keep
    waiting," not as "onset is at t=0."
    """
    cfg = cfg or PreprocessConfig()
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if y.size == 0:
        return None
    if float(np.max(np.abs(y))) < 1e-6:
        # digital silence -- trim() measures dB relative to the clip's own
        # peak, so an all-zero reference makes it degenerate (mirrors the
        # same guard in preprocess.normalize_loudness).
        return None
    _, index = librosa.effects.trim(
        y,
        top_db=cfg.trim_top_db,
        frame_length=cfg.trim_frame_length,
        hop_length=cfg.trim_hop_length,
    )
    if index[1] <= index[0]:  # entire clip below threshold -- mirrors trim_silence
        return None
    return float(index[0]) / sr
