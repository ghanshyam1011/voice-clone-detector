"""Per-session speaker enrolment + live scoring.

Enrolment embeddings live in memory only (privacy posture: no embedding
is written to disk or logs). ``score`` returns cosine similarity to the
enrolled voice and a mismatch risk in [0, 1].

The ``match_sim`` / ``mismatch_sim`` knobs are a documented operating
point, not calibrated — tune them against the actual demo clips.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from voiceguard.config import REPO_ROOT
from voiceguard.speaker.embed import SAMPLE_RATE, SpeakerEmbedder, cosine

_MATCH_SIM = 0.88  # cosine >= this  -> risk ~0 -- fallback when no fitted calibrator exists
_MISMATCH_SIM = 0.55  # cosine <= this  -> risk ~1 -- ditto
_MIN_SCORE_SECONDS = 2.5  # shorter than this -> embedding is unreliable, report unavailable
_CALIB_PATH = REPO_ROOT / "models" / "speaker" / "speaker_calibration.json"


@dataclass
class SpeakerResult:
    enrolled: bool
    similarity: float | None
    risk: float | None
    detail: str

    def as_dict(self) -> dict:
        return {
            "enrolled": self.enrolled,
            "similarity": None if self.similarity is None else round(self.similarity, 4),
            "risk": None if self.risk is None else round(self.risk, 4),
            "detail": self.detail,
        }


class SpeakerVerifier:
    def __init__(
        self,
        embedder: SpeakerEmbedder | None = None,
        *,
        match_sim: float = _MATCH_SIM,
        mismatch_sim: float = _MISMATCH_SIM,
        calib_path: str | Path | None = _CALIB_PATH,
    ):
        self.embedder = embedder or SpeakerEmbedder()
        self.match_sim = match_sim
        self.mismatch_sim = mismatch_sim
        self._enrolled: dict[str, np.ndarray] = {}  # session_id -> mean embedding
        # scripts/calibrate_speaker.py fits this on speaker-disjoint ASVspoof trials;
        # falls back to the match_sim/mismatch_sim linear interpolation if absent.
        self.calibrator = None
        if calib_path and Path(calib_path).exists():
            from voiceguard.eval import Calibrator

            blob = json.loads(Path(calib_path).read_text())
            self.calibrator = Calibrator.from_dict(blob["calibrator"])

    # -- enrolment ---------------------------------------------------
    def enroll(self, session_id: str, wave: np.ndarray) -> None:
        emb = self.embedder.embed(wave)
        prev = self._enrolled.get(session_id)
        merged = emb if prev is None else (prev + emb)
        n = np.linalg.norm(merged)
        self._enrolled[session_id] = merged / n if n > 0 else merged

    def is_enrolled(self, session_id: str) -> bool:
        return session_id in self._enrolled

    def drop(self, session_id: str) -> None:
        self._enrolled.pop(session_id, None)

    # -- scoring ---------------------------------------------------
    def _risk(self, sim: float) -> float:
        if self.calibrator is not None:
            return float(self.calibrator.apply(1.0 - sim)[0])
        span = self.match_sim - self.mismatch_sim
        return float(np.clip((self.match_sim - sim) / span, 0.0, 1.0))

    def score(self, session_id: str, wave: np.ndarray) -> SpeakerResult:
        ref = self._enrolled.get(session_id)
        if ref is None:
            return SpeakerResult(False, None, None, "no enrolled voice for this call")
        wave = np.asarray(wave, dtype=np.float32).reshape(-1)
        if len(wave) < _MIN_SCORE_SECONDS * SAMPLE_RATE:
            return SpeakerResult(True, None, None, "clip too short for a reliable speaker check")
        sim = cosine(self.embedder.embed(wave), ref)
        risk = self._risk(sim)
        if risk >= 0.6:
            detail = f"similarity {sim:.2f} vs enrolled voice (low)"
        elif risk <= 0.25:
            detail = f"similarity {sim:.2f} vs enrolled voice"
        else:
            detail = f"similarity {sim:.2f} vs enrolled voice (borderline)"
        return SpeakerResult(True, sim, risk, detail)
