"""Per-call session state and the combined assessment step.

A session carries the scenario, the editable call context, and the
rolling smoothing state. `SessionManager.assess()` runs the spoof scorer
and the speaker check on a window, fuses them with the context signal,
applies the policy, and writes a feature-only audit event whenever the
decision changes.

State is in memory only. Enrolment embeddings live in the SpeakerVerifier
(also in memory); nothing here is persisted except the audit JSONL.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import asdict, dataclass, field

import numpy as np

from voiceguard.audit import AuditLog
from voiceguard.risk import CallContext, Signal, decide, fuse
from voiceguard.risk.fusion import DEFAULT_WEIGHTS, FusedRisk
from voiceguard.speaker import SpeakerVerifier

_SPK_WINDOW_S = 6.0  # speaker check wants a longer window than the spoof model
_PROSODY_WINDOW_S = 8.0
_FUSED_EMA = 0.6
# per-signal smoothing on the streaming path (0 = no memory). Prosody is the
# noisiest read frame-to-frame, so it is damped the hardest.
_SIG_EMA = {"spoof": 0.5, "speaker": 0.45, "prosody": 0.8, "context": 0.0}
_SESSION_TTL_S = 3600
_SCENARIOS = ("routine", "transaction", "privileged")


@dataclass
class Session:
    session_id: str
    scenario: str
    context: CallContext
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    fused_ema: float | None = None
    sig_ema: dict = field(default_factory=dict)  # per-signal EMA for the streaming path
    last_action: str | None = None
    last_signals: dict = field(default_factory=dict)  # {"spoof":.., "speaker":.., ...}
    last_fused: float = 0.0
    last_reasons: list = field(default_factory=list)

    def touch(self) -> None:
        self.last_seen = time.time()


class SessionManager:
    def __init__(
        self,
        scorer,
        verifier: SpeakerVerifier,
        audit: AuditLog,
        *,
        sr: int,
        nb_samp: int,
        prosody=None,
    ):
        self.scorer = scorer
        self.verifier = verifier
        self.audit = audit
        self.prosody = prosody  # ProsodyScorer | None
        self.sr = sr
        self.nb_samp = nb_samp
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()  # serialises GPU work across handlers

    # -- lifecycle -------------------------------------------------
    def create(
        self,
        scenario: str = "routine",
        *,
        caller_known: bool = True,
        amount: float = 0.0,
        channel: str = "pstn",
    ) -> Session:
        self._gc()
        sid = uuid.uuid4().hex[:12]
        s = Session(
            sid,
            scenario if scenario in _SCENARIOS else "routine",
            CallContext(caller_known=bool(caller_known), amount=float(amount), channel=channel),
        )
        self._sessions[sid] = s
        return s

    def get(self, sid: str) -> Session | None:
        s = self._sessions.get(sid)
        if s:
            s.touch()
        return s

    def end(self, sid: str) -> None:
        self._sessions.pop(sid, None)
        self.verifier.drop(sid)

    def _gc(self) -> None:
        now = time.time()
        for sid in [k for k, v in self._sessions.items() if now - v.last_seen > _SESSION_TTL_S]:
            self.end(sid)

    # -- inputs -------------------------------------------------
    def enroll(self, sid: str, wave: np.ndarray) -> float:
        with self._lock:
            self.verifier.enroll(sid, wave)
        return len(np.asarray(wave).reshape(-1)) / self.sr

    def acknowledge(self, sid: str, *, by: str = "agent") -> dict:
        """Record that the operator acted on the current alert."""
        s = self._sessions[sid]
        ev = self.audit.write(
            session_id=sid,
            scenario=s.scenario,
            signals=s.last_signals or {},
            fused=s.last_fused,
            action="ACKNOWLEDGED",
            reasons=[f"{by} acknowledged the {s.last_action or 'alert'}"],
        )
        return asdict(ev)

    def set_context(self, sid: str, **kw) -> CallContext:
        s = self._sessions[sid]
        if kw.get("scenario") in _SCENARIOS:
            s.scenario = kw["scenario"]
            s.last_action = None  # re-evaluate against the new thresholds
        if kw.get("caller_known") is not None:
            s.context.caller_known = bool(kw["caller_known"])
        if kw.get("amount") is not None:
            s.context.amount = max(0.0, float(kw["amount"]))
        if kw.get("channel"):
            s.context.channel = str(kw["channel"])
        if kw.get("prior_flags") is not None:
            s.context.prior_flags = max(0, int(kw["prior_flags"]))
        return s.context

    # -- assessment -------------------------------------------------
    def assess(
        self,
        sid: str,
        wave: np.ndarray,
        *,
        spoof_risk: float | None = None,
        level: float | None = None,
        warming: bool = False,
        ema: bool = True,
    ) -> dict:
        s = self._sessions[sid]
        s.touch()
        wave = np.asarray(wave, dtype=np.float32).reshape(-1)

        prosody_r = None
        prosody_detail = ""
        with self._lock:
            if spoof_risk is None:
                w = wave[-self.nb_samp :]
                if len(w) < self.nb_samp:
                    w = np.tile(w, self.nb_samp // max(len(w), 1) + 1)[: self.nb_samp]
                spoof_risk = float(self.scorer.score_windows(w[None, :])[0])
            spk = self.verifier.score(sid, wave[-int(_SPK_WINDOW_S * self.sr) :])
            if self.prosody is not None and not warming:
                pr = self.prosody.score(wave[-int(_PROSODY_WINDOW_S * self.sr) :], self.sr)
                prosody_r, prosody_detail = pr.risk, pr.detail

        ctx_risk, ctx_reasons = s.context.assess()

        # smooth the acoustic reads on the streaming path (single clips: no memory)
        raw = {"spoof": spoof_risk, "speaker": spk.risk, "prosody": prosody_r}
        if ema:
            for k, v in raw.items():
                a = _SIG_EMA[k]
                prev = s.sig_ema.get(k)
                if v is None:
                    s.sig_ema[k] = None
                elif prev is None or a == 0:
                    s.sig_ema[k] = v
                else:
                    s.sig_ema[k] = a * prev + (1 - a) * v
            spoof_risk = s.sig_ema["spoof"]
            spk_risk, prosody_r = s.sig_ema["speaker"], s.sig_ema["prosody"]
        else:
            spk_risk = spk.risk

        signals = [
            Signal(
                "spoof", "Synthetic voice", spoof_risk, DEFAULT_WEIGHTS["spoof"],
                f"score {spoof_risk:.2f}",
            ),
            Signal("speaker", "Speaker mismatch", spk_risk, DEFAULT_WEIGHTS["speaker"], spk.detail),
            Signal(
                "prosody", "Prosody / behaviour", prosody_r, DEFAULT_WEIGHTS["prosody"],
                prosody_detail,
            ),
            Signal(
                "context", "Call context", ctx_risk, DEFAULT_WEIGHTS["context"],
                ctx_reasons[0] if ctx_reasons else "",
            ),
        ]

        instant = fuse(signals)
        if ema:
            s.fused_ema = (
                instant.score
                if s.fused_ema is None
                else _FUSED_EMA * s.fused_ema + (1 - _FUSED_EMA) * instant.score
            )
            smoothed = round(s.fused_ema, 4)
        else:
            # discrete clip analysis — no smoothing across separate clips
            s.fused_ema = instant.score
            smoothed = round(instant.score, 4)
        d = decide(FusedRisk(smoothed, signals, instant.note), s.scenario, ctx_reasons)

        sig_map = {
            "spoof": spoof_risk,
            "speaker": spk_risk,
            "prosody": prosody_r,
            "context": ctx_risk,
        }
        s.last_signals, s.last_fused, s.last_reasons = sig_map, smoothed, d.reasons

        audit_event = None
        if not warming and d.action != s.last_action:
            s.last_action = d.action
            ev = self.audit.write(
                session_id=sid,
                scenario=s.scenario,
                signals=sig_map,
                fused=smoothed,
                action=d.action,
                reasons=d.reasons,
            )
            audit_event = asdict(ev)

        return {
            "spoof": None if spoof_risk is None else round(spoof_risk, 4),
            "speaker": None if spk_risk is None else round(spk_risk, 4),
            "prosody": None if prosody_r is None else round(prosody_r, 4),
            "context": round(ctx_risk, 4),
            "similarity": spk.similarity,
            "enrolled": spk.enrolled,
            "fused": smoothed,
            "instant_fused": instant.score,
            "action": d.action,
            "scenario": s.scenario,
            "reasons": d.reasons,
            "recommendation": d.recommendation,
            "signals": [{"name": x.name, "label": x.label, "value": x.value} for x in signals],
            "context_state": {
                "caller_known": s.context.caller_known,
                "amount": s.context.amount,
                "channel": s.context.channel,
                "prior_flags": s.context.prior_flags,
            },
            "level": level,
            "warming": warming,
            "audit_event": audit_event,
        }
