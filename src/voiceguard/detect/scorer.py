"""CMScorer: turn a countermeasure model into a risk score over audio.

risk = P(synthetic) = 1 - P(bonafide), from the model softmax. Uncalibrated
(softmax is overconfident) -- fine for a demo, calibration is P3.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from voiceguard.audio import load_wave, preprocess_wave
from voiceguard.audio.preprocess import PreprocessConfig
from voiceguard.config import load_config, preprocess_config, resolve
from voiceguard.data import load_manifest
from voiceguard.eval.metrics import compute_eer
from voiceguard.models.neural import NB_SAMP, build_cm

# verdict bands on the *calibrated* risk (0.5 == the model's EER operating point)
_GENUINE_BELOW = 0.35
_SYNTHETIC_ABOVE = 0.65
_CALIB_TEMPERATURE = 2.0


def _logit(p: np.ndarray | float) -> np.ndarray | float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class Verdict:
    label: str  # "GENUINE" | "UNCERTAIN" | "SYNTHETIC"
    risk: float  # 0-1, EMA-smoothed
    window_risks: np.ndarray  # per-window raw risk
    window_times: np.ndarray  # window centre time (s)
    smoothed: np.ndarray  # EMA timeline, same length as window_risks
    first_high_s: float | None  # time the smoothed risk first crossed _SYNTHETIC_ABOVE

    @property
    def emoji(self) -> str:
        return {"GENUINE": "OK", "UNCERTAIN": "??", "SYNTHETIC": "!!"}[self.label]


def _classify(risk: float) -> str:
    if risk >= _SYNTHETIC_ABOVE:
        return "SYNTHETIC"
    if risk <= _GENUINE_BELOW:
        return "GENUINE"
    return "UNCERTAIN"


class CMScorer:
    def __init__(self, model, pcfg: PreprocessConfig, device, *, name: str = "aasist"):
        self.model = model.eval()
        self.pcfg = pcfg
        self.device = device
        self.name = name
        self.sr = pcfg.sample_rate
        # raw P(synthetic) at the model's EER operating point; 0.999 == "uncalibrated"
        self.operating_point: float = 0.5
        self.dev_eer: float | None = None

    @torch.no_grad()
    def _raw_synth(self, windows: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.ascontiguousarray(windows, np.float32)).to(self.device)
        return 1.0 - self.model.score_bonafide(x).float().cpu().numpy()

    def _calibrate_risk(self, raw_synth: np.ndarray) -> np.ndarray:
        """Map raw P(synthetic) so the operating point sits at 0.5 with spread.
        AASIST's softmax pins everything near 1.0; the ranking is good but the
        magnitude isn't -- this is a logit shift, not real calibration (P3)."""
        shifted = (_logit(raw_synth) - _logit(self.operating_point)) / _CALIB_TEMPERATURE
        return _sigmoid(shifted)

    def score_windows(self, windows: np.ndarray) -> np.ndarray:
        """windows: (N, NB_SAMP) float32 -> (N,) calibrated risk in [0, 1]."""
        return self._calibrate_risk(self._raw_synth(windows))

    def calibrate(self, n_per_class: int = 80) -> None:
        man = resolve(load_config(), "manifests") / "asvspoof19_la_dev.csv"
        df = load_manifest(man)
        parts = [g.sample(min(n_per_class, len(g)), random_state=1) for _, g in df.groupby("label")]
        import pandas as pd

        sub = pd.concat(parts)
        raw, lab = [], []
        for row in sub.itertuples(index=False):
            y = preprocess_wave(load_wave(row.path, self.sr), self.pcfg)
            y = (
                np.tile(y, NB_SAMP // max(len(y), 1) + 1)[:NB_SAMP]
                if len(y) < NB_SAMP
                else y[:NB_SAMP]
            )
            raw.append(float(self._raw_synth(y[None, :])[0]))
            lab.append(1 if row.label == "bonafide" else 0)  # 1 = bonafide for compute_eer
        raw = np.array(raw)
        eer, thr_pbona = compute_eer(np.array(lab), 1.0 - raw)
        self.operating_point = float(np.clip(1.0 - thr_pbona, 1e-4, 1 - 1e-4))
        self.dev_eer = float(eer)

    def score_waveform(self, y: np.ndarray, *, hop_s: float = 1.0, ema: float = 0.6) -> Verdict:
        """Slide a fixed ~4 s window (the AASIST context length) with `hop_s`
        stride; EMA-smooth the per-window risk into a timeline."""
        y = preprocess_wave(np.asarray(y, np.float32), self.pcfg)
        win = NB_SAMP  # 64600 == the length the model's positional param expects
        hop = max(int(hop_s * self.sr), 1)

        if len(y) <= win:
            reps = win // max(len(y), 1) + 1
            starts = [0]
            wins = np.tile(y, reps)[:win][None, :]
        else:
            starts = list(range(0, len(y) - win + 1, hop))
            wins = np.stack([y[s : s + win] for s in starts])

        risks = self.score_windows(wins)
        times = np.array([s / self.sr + win / self.sr / 2 for s in starts])

        smoothed = np.empty_like(risks)
        acc = float(risks[0])
        for i, r in enumerate(risks):
            acc = ema * acc + (1 - ema) * float(r)
            smoothed[i] = acc

        high = np.where(smoothed >= _SYNTHETIC_ABOVE)[0]
        first_high = float(times[high[0]]) if len(high) else None
        final = float(smoothed[-1])
        return Verdict(_classify(final), final, risks, times, smoothed, first_high)

    def score_file(self, path: str | Path, **kw) -> Verdict:
        return self.score_waveform(load_wave(path, self.sr), **kw)


def build_scorer(model_name: str = "aasist", *, pretrained: bool = True, device=None) -> CMScorer:
    cfg = load_config()
    pcfg = preprocess_config(cfg)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_cm(model_name).to(device)

    models_dir = resolve(cfg, "results").parent / "models"
    if pretrained:
        name_map = {"aasist": "AASIST.pth", "aasist-l": "AASIST-L.pth"}
        if model_name not in name_map:
            raise ValueError(f"--pretrained not available for {model_name}")
        wpath = models_dir / "pretrained" / name_map[model_name]
        model.backbone.load_state_dict(torch.load(wpath, map_location=device))
        tag = f"{model_name} (pretrained upstream)"
        calib_path = models_dir / "pretrained" / f"{model_name}_demo_calib.json"
    else:
        ckpt = models_dir / "cm" / model_name / "best.pt"
        model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
        tag = f"{model_name} (ours: {ckpt.name})"
        calib_path = ckpt.with_name("demo_calib.json")

    scorer = CMScorer(model, pcfg, device, name=tag)
    if calib_path.exists():
        c = json.loads(calib_path.read_text())
        scorer.operating_point, scorer.dev_eer = c["operating_point"], c.get("dev_eer")
    else:
        print("calibrating on a dev sample (one-time, ~20s)...", flush=True)
        scorer.calibrate()
        calib_path.write_text(
            json.dumps({"operating_point": scorer.operating_point, "dev_eer": scorer.dev_eer})
        )
    return scorer


__all__ = ["CMScorer", "Verdict", "build_scorer", "NB_SAMP"]
