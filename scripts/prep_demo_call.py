"""Turn a raw recording (your voice, or a cloned clip) into a demo-ready
call clip: mono 16 kHz, trimmed, loudness-matched, optionally pushed
through a telephony codec so it sounds like an actual phone call.

    python scripts/prep_demo_call.py my_recording.m4a --as enrol
    python scripts/prep_demo_call.py clone_from_elevenlabs.mp3 --as cloned --phone
    python scripts/prep_demo_call.py demo_audio/cloned.wav --check   # score it

`--phone` (default g711_ulaw = landline; `amr_nb` = mobile) also makes the
point that VoiceGuard works on telephony audio, not just clean studio wav.
Output goes to demo_audio/<name>.wav.
"""

# ruff: noqa: E402
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse

import numpy as np
import soundfile as sf

from voiceguard.audio import load_wave
from voiceguard.audio.codec import CODECS, codec_roundtrip
from voiceguard.config import REPO_ROOT

SR = 16000
OUT = REPO_ROOT / "demo_audio"


def _trim(y: np.ndarray, sr: int, top_db: int = 30, pad_ms: int = 60) -> np.ndarray:
    import librosa

    y, _ = librosa.effects.trim(y, top_db=top_db)
    pad = np.zeros(int(pad_ms / 1000 * sr), dtype=np.float32)
    return np.concatenate([pad, y, pad])


def _peaknorm(y: np.ndarray, target: float = 0.95) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    peak = float(np.max(np.abs(y))) or 1.0
    return (target * y / peak).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="any audio file (wav / m4a / mp3 / flac / ogg)")
    ap.add_argument(
        "--as", dest="name", default=None, help="output name, e.g. enrol / benign / cloned"
    )
    ap.add_argument(
        "--phone",
        nargs="?",
        const="g711_ulaw",
        default=None,
        choices=list(CODECS),
        help="apply a telephony codec (default g711_ulaw)",
    )
    ap.add_argument("--max-seconds", type=float, default=20.0)
    ap.add_argument("--check", action="store_true", help="score the result with the detector")
    args = ap.parse_args()

    y = _trim(load_wave(args.input, SR), SR)
    if len(y) > args.max_seconds * SR:
        y = y[: int(args.max_seconds * SR)]
    y = _peaknorm(y)

    if args.phone:
        y = _peaknorm(codec_roundtrip(y, SR, args.phone))

    OUT.mkdir(exist_ok=True)
    stem = args.name or os.path.splitext(os.path.basename(args.input))[0]
    dest = OUT / f"{stem}.wav"
    sf.write(dest, y, SR, subtype="PCM_16")
    tail = f" (phone: {args.phone})" if args.phone else ""
    print(f"wrote demo_audio/{stem}.wav  {len(y) / SR:.1f}s{tail}")

    if args.check:
        from voiceguard.detect import build_scorer

        v = build_scorer("aasist", pretrained=True).score_file(dest)
        print(f"detector: {v.label}  risk {v.risk * 100:.0f}%")


if __name__ == "__main__":
    main()
