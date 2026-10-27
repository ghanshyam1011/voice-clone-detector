"""The sample audio the console offers, in words a non-specialist can follow.

Only files named here are ever served (``/api/demo/clips/<id>``): the id is looked up in this
table and never joined onto a path, so a request cannot reach any other file. The audio itself
lives in ``demo_audio/`` (git-ignored; build it with ``scripts/pick_demo_clips.py``). A clip
that is not on disk is simply not offered.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import soundfile as sf

from voiceguard.config import REPO_ROOT

DEMO_DIR = REPO_ROOT / "demo_audio"


class Clip(NamedTuple):
    role: str  # "register" teaches the system the customer's voice; "call" is played as the call
    title: str
    text: str  # one sentence for a viewer: what the audio IS, never what VoiceGuard will say
    truth: str | None  # "real" or "fake" for a call (so the console can show how it did)


CLIPS: dict[str, Clip] = {
    "enrol": Clip(
        "register",
        "The customer's real voice",
        "About ten seconds of the genuine customer speaking. VoiceGuard learns how they sound.",
        None,
    ),
    "benign": Clip(
        "call",
        "The real customer calls",
        "A genuine call from the same person whose voice was registered.",
        "real",
    ),
    "cloned_knnvc": Clip(
        "call",
        "A voice cloned with a free tool",
        "Someone typed a script and an open-source tool made it sound like the customer. "
        "The hardest example here: it sounds close to the real voice, so it can be missed.",
        "fake",
    ),
    "cloned_1": Clip(
        "call",
        "A cloned voice",
        "Computer-generated speech imitating a real person, made with older cloning technology.",
        "fake",
    ),
    "cloned_2": Clip(
        "call",
        "Another cloned voice",
        "A second computer-generated imitation, from a different cloning system.",
        "fake",
    ),
    "modern_tts": Clip(
        "call",
        "A computer-generated voice",
        "A modern text-to-speech system reading a script. No real person is speaking.",
        "fake",
    ),
}


def clip_path(clip_id: str) -> Path | None:
    """The file for a catalogued id, or None (unknown id, or the audio is not on disk)."""
    if clip_id not in CLIPS:
        return None
    p = DEMO_DIR / f"{clip_id}.wav"
    return p if p.is_file() else None


def available() -> list[dict]:
    """The catalogue entries whose audio exists, in display order."""
    out = []
    for cid, clip in CLIPS.items():
        p = clip_path(cid)
        if p is not None:
            out.append(
                {
                    "id": cid,
                    "role": clip.role,
                    "title": clip.title,
                    "text": clip.text,
                    "truth": clip.truth,
                    "seconds": round(float(sf.info(p).duration), 1),
                }
            )
    return out
