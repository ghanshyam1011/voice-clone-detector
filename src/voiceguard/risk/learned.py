"""Channel-aware learned fusion: spoof evidence + speaker evidence, conditioned
on how noisy the call is.

Why this exists. The hand-set weighted mean (``fuse``) treats a 0.3 spoof score
the same on a quiet line and in a noisy room, but background voices alone push
a *genuine* caller's score up. One fixed threshold therefore has to choose
between missing clones on clean audio and flagging real callers in noise.
Here each head sees the channel descriptors (``audio.channel``) alongside its
signal, so it learns "0.3 in a noisy room is fine, 0.3 on a clean line is not".

    clone head     genuine vs clone     <- logit(spoof) x channel
    impostor head  genuine vs other human <- logit(speaker risk) x channel
    acoustic risk  = 1 - (1 - p_clone)(1 - p_impostor)     (independent alarms)

Prosody is deliberately not an input: measured at ~46% EER (chance) on the same
trials, it only adds noise. Context risk is mixed in by ``combine_with_context``.

A head is either logistic (linear in the signal and its interactions with the channel) or a
small ensemble of boosted trees, monotone in the signal, that finds its own interactions;
``spec["clone"]["type"] == "gbm"`` selects the latter.

Inference is numpy only (the trees are exported to plain lists); fitting (scikit-learn) lives
in ``scripts/fit_fusion.py``. The fitted parameters are ``models/fusion/fusion.json``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from voiceguard.config import REPO_ROOT

DEFAULT_PATH = REPO_ROOT / "models" / "fusion" / "fusion.json"
# the descriptors the shipped heads were fitted on; a model file names its own (``channel``)
CHANNEL = ("dyn_db", "floor_rel_db", "hf_ratio", "flatness")
# everything a model may condition on: every ``audio.channel`` descriptor except the absolute
# level, which moves with microphone gain and would make the same call "noisy" or "clean" at
# will. Spelled out so this module needs no audio stack; tests/test_learned.py keeps it in sync.
ALLOWED_CHANNEL = (
    *CHANNEL,
    "crest_db",
    "flat_top_db",
    "lf_ratio_db",
    "hum_db",
    "env_acf",
)
STRONG_P = 0.95  # a head this sure is worded as "strong" even if the escalate threshold is higher
CONTEXT_WEIGHT = 0.8  # share of the context risk treated as an independent alarm
_EPS = 1e-3


def _logit(p: float) -> float:
    p = min(max(float(p), _EPS), 1 - _EPS)
    return math.log(p / (1 - p))


def head_features(
    signals: float | list[float],
    channel: dict,
    clamp: tuple[np.ndarray, np.ndarray] | None = None,
    names: tuple[str, ...] | list[str] = CHANNEL,
    main_effects: bool = True,
) -> np.ndarray:
    """[logit(s) per signal, channel x len(names), logit(s) x channel per signal].

    One signal and the default four descriptors -> 9 numbers; two detector scores -> 14.
    Interactions let each score's trustworthiness depend on the channel (e.g. which of
    two detectors to believe in a noisy room).

    ``main_effects=False`` drops the bare channel terms, leaving [logit(s), logit(s) x channel]:
    the channel can then only *reweigh* the signal, never raise an alarm on its own. A
    descriptor describes the recording chain (noise suppression, a messaging-app codec, a
    clean room), not who or what produced the voice, so a head that can alarm on it alone
    learns the chain of its training data (measured: a real WhatsApp voice note, whose
    detector score was unremarkable, was called synthetic at 0.98 on descriptors alone).

    ``clamp`` = (low, high) per channel feature, the range seen when the heads were
    fitted. The heads are linear (with interactions), so a descriptor far outside that
    range would otherwise be extrapolated without limit -- measured: a call with mains
    hum, which no training call had, drove the clone/impostor logits to alarm even
    though both signals were healthy. Clamped, an unfamiliar channel is treated as the
    nearest familiar one."""
    sigs = [signals] if np.isscalar(signals) else list(signals)
    s = np.array([_logit(x) for x in sigs])
    ch = np.array([float(channel[k]) for k in names])
    if clamp is not None:
        ch = np.clip(ch, clamp[0], clamp[1])
    inter = np.outer(s, ch).ravel()
    return np.concatenate([s, ch, inter]) if main_effects else np.concatenate([s, inter])


def tree_features(
    signal: float,
    channel: dict,
    clamp: tuple[np.ndarray, np.ndarray] | None = None,
    names: tuple[str, ...] | list[str] = CHANNEL,
) -> np.ndarray:
    """[logit(s), channel x len(names)]: the inputs of a boosted-tree head, which finds its
    own interactions. Clamped like ``head_features`` so both head types see the same inputs."""
    ch = np.array([float(channel[k]) for k in names])
    if clamp is not None:
        ch = np.clip(ch, clamp[0], clamp[1])
    return np.concatenate([[_logit(signal)], ch])


@dataclass
class _Head:
    mean: np.ndarray
    std: np.ndarray
    coef: np.ndarray
    intercept: float
    main_effects: bool = True  # False: the channel only reweighs the signal (see head_features)

    def prob(self, x: np.ndarray) -> float:
        z = float(((x - self.mean) / self.std) @ self.coef + self.intercept)
        return 1.0 / (1.0 + math.exp(-max(min(z, 30.0), -30.0)))

    @classmethod
    def from_dict(cls, d: dict) -> _Head:
        return cls(
            np.array(d["mean"]),
            np.array(d["std"]),
            np.array(d["coef"]),
            float(d["intercept"]),
            bool(d.get("main_effects", True)),
        )


@dataclass
class _TreeHead:
    """Gradient-boosted trees exported from scikit-learn (``scripts/fit_fusion.py --head gbm``)
    and evaluated in plain Python, so serving still needs no scikit-learn. A tree is parallel
    lists with one entry per node; a leaf has ``feature`` -1 and carries ``value``."""

    baseline: float
    trees: list[dict]

    def prob(self, x: np.ndarray) -> float:
        z = self.baseline
        for t in self.trees:
            feat, thr, left, right = t["feature"], t["threshold"], t["left"], t["right"]
            i = 0
            while feat[i] >= 0:
                i = left[i] if x[feat[i]] <= thr[i] else right[i]
            z += t["value"][i]
        return 1.0 / (1.0 + math.exp(-max(min(z, 30.0), -30.0)))

    @classmethod
    def from_dict(cls, d: dict) -> _TreeHead:
        return cls(float(d["baseline"]), d["trees"])


_LOG_GAIN_LIMIT = 6.0  # exp(+-6): the trust a channel can put in a detector is bounded


@dataclass
class _GainHead:
    """``p = sigmoid(offset + gain(channel) * logit(signal))`` with ``gain = exp(w0 + w . u)``
    and ``u`` the clamped, standardised channel descriptors.

    The channel decides how far to *trust* the detector (a noisy room earns a smaller gain, a
    clean line a larger one) but can never speak for it: the gain is positive by construction,
    so the output rises with the signal for every channel, and a quiet detector stays quiet.
    The linear heads cannot promise that. Their channel terms (bare, or multiplied with the
    signal with a free sign) let a recording style -- noise suppression, a messaging-app codec,
    a very clean room -- raise the alarm with the detector unremarkable: measured, a real
    WhatsApp voice note (detector 0.27) was called synthetic at 0.98."""

    mean: np.ndarray
    std: np.ndarray
    w0: float
    w: np.ndarray
    offset: float

    def gain(self, channel: np.ndarray) -> float:
        z = self.w0 + float(((channel - self.mean) / self.std) @ self.w)
        return math.exp(max(min(z, _LOG_GAIN_LIMIT), -_LOG_GAIN_LIMIT))

    def prob(self, x: np.ndarray) -> float:
        """``x`` = [logit(signal), channel...], the layout of ``tree_features``."""
        z = self.offset + self.gain(x[1:]) * float(x[0])
        return 1.0 / (1.0 + math.exp(-max(min(z, 30.0), -30.0)))

    @classmethod
    def from_dict(cls, d: dict) -> _GainHead:
        return cls(
            np.array(d["mean"]),
            np.array(d["std"]),
            float(d["w0"]),
            np.array(d["w"]),
            float(d["offset"]),
        )


def _head_from_dict(d: dict) -> _Head | _TreeHead | _GainHead:
    kind = d.get("type")
    if kind == "gbm":
        return _TreeHead.from_dict(d)
    if kind == "gain":
        return _GainHead.from_dict(d)
    return _Head.from_dict(d)


class LearnedFusion:
    def __init__(self, spec: dict):
        # a model written before the descriptors were made gain-invariant has no "channel" key
        fitted_on = list(spec.get("channel", ["dyn_db", "floor_db", "hf_ratio", "flatness"]))
        unusable = [k for k in fitted_on if k not in ALLOWED_CHANNEL]
        if unusable:
            raise ValueError(
                f"fusion model was fitted on channel features {fitted_on}; {unusable} are not "
                f"gain-invariant descriptors this code provides ({list(ALLOWED_CHANNEL)}): "
                "refit with scripts/fit_fusion.py"
            )
        self.channel = tuple(fitted_on)
        self.spec = spec
        self.clone = _head_from_dict(spec["clone"])
        self.impostor = _head_from_dict(spec["impostor"])
        self.t_warn = float(spec["t_warn"])
        self.t_esc = float(spec["t_esc"])
        rng = spec.get("channel_range")  # {name: [low, high]} seen at fit time
        self.channel_range = (
            (
                np.array([rng[k][0] for k in self.channel]),
                np.array([rng[k][1] for k in self.channel]),
            )
            if rng
            else None
        )
        # piecewise-linear map of the learned operating points onto the policy's
        # routine thresholds (warn 0.55 / escalate 0.75), so policy.SCENARIOS and
        # the console's score scale keep their meaning.
        self._xs = [0.0, self.t_warn, self.t_esc, 1.0]
        self._ys = [0.0, 0.55, 0.75, 1.0]

    @classmethod
    def load(cls, path: Path | str | None = None) -> LearnedFusion | None:
        p = Path(path) if path else DEFAULT_PATH
        if not p.exists():
            return None
        return cls(json.loads(p.read_text(encoding="utf-8")))

    def detector_mismatch(self, name: str, operating_point: float) -> str | None:
        """Why this fusion should not be served with the given detector, or None if it fits.

        The heads were fitted on one detector's calibrated scores. Another detector -- or the
        same weights calibrated differently -- shifts every score and with it every threshold,
        and nothing downstream would notice."""
        d = self.spec.get("detector")
        if not d:
            return "fusion model does not record which detector it was fitted on (older model)"
        if d.get("name") != name:
            return f"fusion model was fitted on detector {d.get('name')!r}, not {name!r}"
        if abs(float(d["operating_point"]) - operating_point) > 1e-4:
            return (
                f"detector calibration moved: the fusion was fitted at operating point "
                f"{d['operating_point']}, this detector is at {operating_point:.6f}"
            )
        return None

    @property
    def clone_signals(self) -> int:
        """How many detector scores the clone head consumes (1, or 2 with an alt detector)."""
        return int(self.spec.get("clone_signals", 1))

    def probs(
        self,
        spoof: float | None,
        speaker: float | None,
        channel: dict,
        spoof_alt: float | None = None,
    ) -> tuple[float, float, float]:
        """(p_attack, p_clone, p_impostor). A missing signal contributes no alarm."""
        if spoof is not None and self.clone_signals == 2:
            if spoof_alt is None:
                raise ValueError("this fusion model needs the second detector score (spoof_alt)")
            spoof = [spoof, spoof_alt]
        clamp, names = self.channel_range, self.channel

        def head_prob(head, signal) -> float:
            if signal is None:
                return 0.0
            if isinstance(head, (_TreeHead, _GainHead)):
                if not np.isscalar(signal):
                    raise ValueError("this head takes a single detector score")
                return head.prob(tree_features(signal, channel, clamp, names))
            return head.prob(head_features(signal, channel, clamp, names, head.main_effects))

        pc, pi = head_prob(self.clone, spoof), head_prob(self.impostor, speaker)
        return 1.0 - (1.0 - pc) * (1.0 - pi), pc, pi

    def risk(self, p_attack: float) -> float:
        """Map a learned attack probability onto the policy's 0-1 risk scale."""
        return float(np.interp(p_attack, self._xs, self._ys))

    def level(self, p: float) -> str:
        """How loud one head's alarm is, against this model's own operating points:
        ``none`` below 0.3, ``some`` up to the warn threshold (a few signs, not enough to act on),
        ``likely`` from there up to 0.95 or the escalate threshold, whichever is lower (enough to
        ask for a check), ``strong`` beyond (log-odds of about 3 or more). The console words each
        head's finding from this, so its wording cannot drift from the thresholds the decision
        is actually made at."""
        if p >= min(self.t_esc, STRONG_P):
            return "strong"
        if p >= self.t_warn:
            return "likely"
        return "some" if p >= 0.3 else "none"


def combine_with_context(acoustic_risk: float, context_risk: float) -> float:
    """Context is an independent alarm: no context risk leaves the acoustic risk
    untouched (the weighted mean used to dilute it by the context weight)."""
    return 1.0 - (1.0 - acoustic_risk) * (1.0 - CONTEXT_WEIGHT * context_risk)
