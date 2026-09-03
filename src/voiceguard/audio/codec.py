"""Telephony-codec round-trip via ffmpeg.

Not used by P0 evaluation (that runs clean). It is here for P1 channel
augmentation and for the "encode every clip through the same codec chain
regardless of label" anti-shortcut step. Never apply a codec to one class
only.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

_FFMPEG = shutil.which("ffmpeg")

# name -> (encode args, container extension)
_CODECS: dict[str, tuple[list[str], str]] = {
    "g711_ulaw": (["-ar", "8000", "-c:a", "pcm_mulaw"], "wav"),
    "g722": (["-ar", "16000", "-c:a", "adpcm_g722"], "wav"),
    "amr_nb": (["-ar", "8000", "-c:a", "libopencore_amrnb", "-b:a", "12.2k"], "amr"),
    "opus_8k": (["-ar", "16000", "-c:a", "libopus", "-b:a", "8k"], "ogg"),
}

CODECS = tuple(_CODECS)


def codec_roundtrip(y: np.ndarray, sample_rate: int, codec: str) -> np.ndarray:
    """Encode ``y`` with ``codec`` then decode back to float32 at ``sample_rate``."""
    if _FFMPEG is None:
        raise RuntimeError("ffmpeg not found on PATH")
    if codec not in _CODECS:
        raise ValueError(f"unknown codec {codec!r}; choices: {CODECS}")

    enc_args, ext = _CODECS[codec]
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src, enc, dec = td / "src.wav", td / f"enc.{ext}", td / "dec.wav"
        sf.write(src, y, sample_rate, subtype="PCM_16")
        _run([_FFMPEG, "-y", "-i", str(src), *enc_args, str(enc)])
        _run([_FFMPEG, "-y", "-i", str(enc), "-ar", str(sample_rate), "-ac", "1", str(dec)])
        out, _ = sf.read(dec, dtype="float32", always_2d=False)
    return np.ascontiguousarray(out, dtype=np.float32)


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, capture_output=True)
