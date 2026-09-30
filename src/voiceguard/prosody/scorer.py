"""Prosody -> synthetic-likeness risk.

A StandardScaler + LogisticRegression over :data:`FEATURE_NAMES`, fit on
an ASVspoof19-train subset by ``scripts/train_prosody.py`` and saved to
``models/prosody/prosody_lr.joblib`` (tracked — it is a few KB).

`risk` = the model's P(spoof). Weak on its own (that is expected); it is
one signal among several in the fusion.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from voiceguard.config import REPO_ROOT
from voiceguard.prosody.features import FEATURE_NAMES, prosody_features

DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "prosody" / "prosody_lr.joblib"


@dataclass
class ProsodyResult:
    available: bool
    risk: float | None
    detail: str

    def as_dict(self) -> dict:
        return {
            "available": self.available,
            "risk": None if self.risk is None else round(self.risk, 4),
            "detail": self.detail,
        }


class ProsodyScorer:
    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH):
        self.model_path = Path(model_path)
        self._pipe = None
        self._medians = None
        self._calibrator = None

    @property
    def ready(self) -> bool:
        return self._pipe is not None

    def _ensure(self) -> None:
        if self._pipe is not None:
            return
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"{self.model_path} missing — run `python scripts/train_prosody.py`"
            )
        import joblib

        blob = joblib.load(self.model_path)
        self._pipe = blob["pipeline"]
        self._medians = np.asarray(blob["feature_medians"], dtype=np.float32)
        self._calibrator = None
        if blob.get("calibrator"):
            from voiceguard.eval import Calibrator

            self._calibrator = Calibrator.from_dict(blob["calibrator"])

    def warm(self) -> None:
        self._ensure()

    def _impute(self, x: np.ndarray) -> np.ndarray:
        return np.where(np.isfinite(x), x, self._medians)

    def risk_from_features(self, feats: np.ndarray) -> float:
        self._ensure()
        x = self._impute(np.asarray(feats, dtype=np.float32)).reshape(1, -1)
        raw = float(self._pipe.predict_proba(x)[0, 1])  # column 1 = spoof
        if self._calibrator is not None:
            return float(self._calibrator.apply(raw)[0])
        return raw

    def score_file(self, path, sr: int = 16000) -> ProsodyResult:
        from voiceguard.audio import load_wave

        return self.score(load_wave(path, sr), sr)

    def score(self, wave: np.ndarray, sr: int = 16000) -> ProsodyResult:
        feats = prosody_features(wave, sr)
        n_ok = int(np.isfinite(feats).sum())
        if n_ok < len(FEATURE_NAMES) // 2:
            return ProsodyResult(False, None, "not enough voiced speech for a prosody read")
        try:
            r = self.risk_from_features(feats)
        except FileNotFoundError as exc:
            return ProsodyResult(False, None, str(exc))
        if r >= 0.6:
            detail = f"prosody looks synthetic (p={r:.2f})"
        elif r <= 0.35:
            detail = f"prosody looks natural (p={r:.2f})"
        else:
            detail = f"prosody inconclusive (p={r:.2f})"
        return ProsodyResult(True, r, detail)
