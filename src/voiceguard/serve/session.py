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
import soxr

from voiceguard.audio import speech_onset_s
from voiceguard.audio.channel import RECENT_SECONDS, channel_features, describe_channel
from voiceguard.audio.quality import MIN_ENROL_SPEECH_S, InputQuality, PcmStream, analyze
from voiceguard.audit import AuditLog
from voiceguard.config import REPO_ROOT
from voiceguard.risk import CallContext, Signal, decide, fuse, waiting
from voiceguard.risk.fusion import DEFAULT_WEIGHTS, FusedRisk
from voiceguard.risk.learned import LearnedFusion, combine_with_context
from voiceguard.speaker import SpeakerVerifier

_SPK_WINDOW_S = 6.0  # speaker check wants a longer window than the spoof model
# serving default, first that exists (a fusion model is fitted for ONE of these; see its
# ``detector`` record and ``LearnedFusion.detector_mismatch``)
DEFAULT_WEIGHTS_PREFERENCE = ("AASIST_robust2.pth", "AASIST_robust.pth")
_PROSODY_WINDOW_S = 8.0
_FUSED_EMA = 0.6
# per-signal smoothing on the streaming path (0 = no memory). Prosody is the
# noisiest read frame-to-frame, so it is damped the hardest.
_SIG_EMA = {"spoof": 0.5, "speaker": 0.45, "prosody": 0.8, "context": 0.0}
_SESSION_TTL_S = 3600
_SPOOF_HOP_S = 1.0  # stride between the ~4 s spoof windows inside one read
_SCENARIOS = ("routine", "transaction", "privileged")
LIVE_BUFFER_S = 12.0  # rolling audio kept per streaming call (REST/WS, gRPC and the evaluators)
LIVE_HOP_S = 0.5  # a streaming call is re-assessed every half second of new audio


class LiveBuffer:
    """The audio of one streaming call, kept so that an assessment always sees the NEWEST audio.

    Chunks go in as they arrive (little-endian float32 PCM at the rate the client declared);
    a separate task takes ``snapshot()`` whenever ``feed`` reports half a second of new audio.
    Feeding never waits for an assessment, so a slow assessment makes the verdict trail the
    call by one assessment and no more. When feeding and assessing shared one loop, each
    assessment consumed only half a second of audio while taking longer than that, so the
    verdict fell further behind real time with every second of the call (measured: 1 s behind
    after 2 s of audio, 3 s behind after 8 s, with no bound)."""

    def __init__(self, sr: int, cap_s: float = LIVE_BUFFER_S, hop_s: float = LIVE_HOP_S):
        self.sr = sr
        self.cap = int(cap_s * sr)
        self.hop = int(hop_s * sr)
        self.in_sr = sr
        self._pcm = PcmStream()
        self.buf = np.zeros(0, dtype=np.float32)
        self.since = 0  # samples fed since the last snapshot
        self.heard = 0  # samples fed in total (the call's audio clock)

    def set_rate(self, rate: int) -> None:
        self.in_sr = rate

    def feed(self, data: bytes) -> bool:
        """Add a chunk; True when enough new audio has arrived to be worth assessing."""
        chunk = self._pcm.feed(data)
        if self.in_sr != self.sr and len(chunk):
            chunk = soxr.resample(chunk, self.in_sr, self.sr).astype(np.float32)
        self.buf = np.concatenate([self.buf, chunk])[-self.cap :]
        self.since += len(chunk)
        self.heard += len(chunk)
        return self.since >= self.hop

    @property
    def unassessed(self) -> bool:
        """Audio has arrived since the last snapshot (possibly less than one hop of it)."""
        return self.since > 0

    def snapshot(self) -> tuple[np.ndarray, float] | None:
        """``(the newest audio, seconds of the call heard so far)``, or None until half a
        second has arrived. Taking it marks the audio as assessed."""
        if len(self.buf) < self.sr // 2:
            return None
        self.since = 0
        return self.buf, self.heard / self.sr


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
    speech_onset_s: float | None = None  # when the caller first started talking,
    # relative to the session's own audio clock -- None until VAD finds it
    first_confirmed_read_s: float | None = None  # latency: onset -> first non-provisional read

    def touch(self) -> None:
        self.last_seen = time.time()


@dataclass
class EnrolResult:
    seconds: float  # length of the clip
    speech_seconds: float  # how much of it was actually speech
    short: bool  # accepted, but under ~4 s of speech: a weak voiceprint


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
        fusion: LearnedFusion | None = None,
    ):
        self.scorer = scorer
        self.verifier = verifier
        self.audit = audit
        self.prosody = prosody  # ProsodyScorer | None
        # channel-aware learned fusion (models/fusion/fusion.json); None -> the
        # hand-set weighted mean in risk.fuse, kept as the documented fallback
        self.fusion = fusion
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
    def enroll(self, sid: str, wave: np.ndarray) -> EnrolResult:
        """Store the caller's voiceprint. Refuses audio that is not clear speech: a
        voiceprint of silence or noise would silently make every later speaker
        check in this call meaningless. Raises ValueError with a reason."""
        wave, quality, _ = analyze(wave, self.sr)
        if not quality.usable or quality.speech_s < MIN_ENROL_SPEECH_S:
            why = quality.reason or f"only {quality.speech_s:.1f} s of speech"
            raise ValueError(
                f"enrolment needs at least {MIN_ENROL_SPEECH_S:.0f} s of clear speech ({why})"
            )
        with self._lock:
            self.verifier.enroll(sid, wave)
        return EnrolResult(len(wave) / self.sr, quality.speech_s, quality.speech_s < 4.0)

    def warm(self) -> float:
        """Push a few seconds of noise through every model once, so the first real call is not
        the one that pays for CUDA start-up, the lazy model loads and the speech detector
        (measured: the first assessment on a cold server stalled for over ten seconds, which
        is the whole of a short demo call). Leaves no session behind. Returns seconds taken."""
        t0 = time.perf_counter()
        noise = (0.05 * np.random.default_rng(0).standard_normal(int(self.sr * 6))).astype(
            np.float32
        )
        analyze(noise, self.sr, window_s=RECENT_SECONDS)
        channel_features(noise, self.sr)
        sid = "warm-" + uuid.uuid4().hex[:8]
        with self._lock:
            self.spoof_read(noise)
            self.verifier.enroll(sid, noise)  # straight to the verifier: noise is not "speech"
            self.verifier.score(sid, noise)
            self.verifier.drop(sid)
        return time.perf_counter() - t0

    def reset_call(self, sid: str) -> None:
        """Start a new call on the same session: forget the previous call's smoothing, alert
        state and speech-onset clock, keep the registered voice and the call details. Without
        this, the end of one clip would bleed into the start of the next."""
        s = self._sessions[sid]
        s.fused_ema = None
        s.sig_ema.clear()
        s.last_action = None
        s.last_signals, s.last_fused, s.last_reasons = {}, 0.0, []
        s.speech_onset_s = None
        s.first_confirmed_read_s = None

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
    def spoof_read(self, wave: np.ndarray) -> tuple[float, np.ndarray]:
        """The one spoof read every transport uses (REST, WebSocket, gRPC):
        ``(smoothed risk, the per-window risks it was smoothed from)``.

        The recent audio goes through the anti-shortcut front-end (trim +
        loudness, ``CMScorer.score_waveform``) and is scored over sliding
        ~4 s windows, then smoothed -- so a single noisy window cannot decide
        the read, and the model sees the same kind of audio it was trained on.
        Callers must hold ``self._lock``."""
        tail = np.asarray(wave, dtype=np.float32).reshape(-1)[-int(RECENT_SECONDS * self.sr) :]
        v = self.scorer.score_waveform(tail, hop_s=_SPOOF_HOP_S)
        return float(v.risk), np.asarray(v.window_risks, dtype=np.float32)

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
        # sanitise + find the speech BEFORE any model sees the audio: silence, noise,
        # NaN, a DC offset or int16-scaled samples must not reach the detectors (they
        # produced the loudest alarm in the product, or crashed), and a read built on
        # little speech is provisional whatever the transport said
        wave, quality, _ = analyze(wave, self.sr, window_s=RECENT_SECONDS)
        if not quality.usable:
            return self._no_read(s, quality, level)
        warming = warming or quality.provisional or len(wave) < self.nb_samp

        if s.speech_onset_s is None:
            s.speech_onset_s = speech_onset_s(wave, self.sr)

        prosody_r = None
        prosody_detail = ""
        with self._lock:
            spoof_windows = None
            if spoof_risk is None:
                spoof_risk, spoof_windows = self.spoof_read(wave)
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
                "spoof",
                "Synthetic voice",
                spoof_risk,
                DEFAULT_WEIGHTS["spoof"],
                f"score {spoof_risk:.2f}",
                provisional=warming,
            ),
            Signal("speaker", "Speaker mismatch", spk_risk, DEFAULT_WEIGHTS["speaker"], spk.detail),
            Signal(
                "prosody",
                "Prosody / behaviour",
                prosody_r,
                DEFAULT_WEIGHTS["prosody"],
                prosody_detail,
            ),
            Signal(
                "context",
                "Call context",
                ctx_risk,
                DEFAULT_WEIGHTS["context"],
                ctx_reasons[0] if ctx_reasons else "",
            ),
        ]

        # what the two learned heads concluded, and what is wrong with the line, for display
        evidence, evidence_level, audio_notes = None, None, _quiet_note(quality)
        channel = None
        if self.fusion is not None:
            ch = channel_features(wave[-int(RECENT_SECONDS * self.sr) :], self.sr)
            channel = {k: round(float(v), 3) for k, v in ch.items() if k != "level_db"}
            p_att, p_clone, p_imp = self.fusion.probs(spoof_risk, spk_risk, ch)
            evidence = {"synthetic": round(p_clone, 4), "voice_mismatch": round(p_imp, 4)}
            evidence_level = {
                "synthetic": self.fusion.level(p_clone),
                "voice_mismatch": self.fusion.level(p_imp),
            }
            audio_notes = audio_notes + describe_channel(ch)
            score = combine_with_context(self.fusion.risk(p_att), ctx_risk)
            notes = []
            if p_clone >= 0.5:
                notes.append("Synthetic-voice evidence (noise-adjusted)")
            if p_imp >= 0.5:
                notes.append("Voice does not match the enrolled speaker (noise-adjusted)")
            # prosody stays on the console for transparency but is not a decision
            # input (measured ~chance), so it must not generate reason codes either
            decision_sigs = [x for x in signals if x.name != "prosody"]
            instant = FusedRisk(
                round(min(1.0, score), 4),
                decision_sigs,
                "; ".join(notes),
                any(x.provisional for x in decision_sigs if x.available),
            )
        else:
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
        d = decide(
            FusedRisk(smoothed, instant.signals, instant.note, instant.has_provisional),
            s.scenario,
            ctx_reasons,
        )
        if not warming and s.first_confirmed_read_s is None and s.speech_onset_s is not None:
            s.first_confirmed_read_s = round(len(wave) / self.sr - s.speech_onset_s, 3)

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
            "provisional": instant.has_provisional,
            "speech_onset_s": s.speech_onset_s,
            "first_confirmed_read_s": s.first_confirmed_read_s,
            "audit_event": audit_event,
            "quality": quality.as_dict(),
            # {"synthetic", "voice_mismatch"} probabilities of the learned heads (None when
            # the hand-set fallback fusion is in use) and plain-English line-quality notes
            "evidence": evidence,
            # the same two findings as none / some / likely / strong, against the learned
            # operating points: what the console words its checks from
            "evidence_level": evidence_level,
            # how the recording looks (noise floor, band, envelope...): numbers for support and
            # for finding out why a real-world recording read the way it did
            "channel": channel,
            "audio_notes": audio_notes,
            # the ~4 s windows the smoothed spoof read was built from (None if supplied)
            "spoof_windows": None
            if spoof_windows is None
            else [round(float(x), 4) for x in spoof_windows],
        }

    def _no_read(self, s: Session, quality: InputQuality, level: float | None) -> dict:
        """Nothing usable to judge. The call context still counts (an unknown caller asking
        for a large transfer is a fact whether or not anyone has spoken), but silence
        is never reported as ALLOW. Session state (smoothing, last action, audit) is left
        exactly as it was, so a pause in the call does not blunt the next real read."""
        ctx_risk, ctx_reasons = s.context.assess()
        signals = [
            Signal("spoof", "Synthetic voice", None, DEFAULT_WEIGHTS["spoof"], quality.reason),
            Signal("speaker", "Speaker mismatch", None, DEFAULT_WEIGHTS["speaker"], quality.reason),
            Signal("prosody", "Prosody / behaviour", None, DEFAULT_WEIGHTS["prosody"], ""),
            Signal(
                "context",
                "Call context",
                ctx_risk,
                DEFAULT_WEIGHTS["context"],
                ctx_reasons[0] if ctx_reasons else "",
            ),
        ]
        score = round(min(1.0, ctx_risk), 4)
        d = decide(FusedRisk(score, signals, "", True), s.scenario, ctx_reasons)
        if d.action == "ALLOW":
            d = waiting(s.scenario, quality.reason)
        else:
            d.reasons.insert(0, f"No usable speech yet ({quality.reason}): judged on context alone")
        return {
            "spoof": None,
            "speaker": None,
            "prosody": None,
            "context": round(ctx_risk, 4),
            "similarity": None,
            "enrolled": self.verifier.is_enrolled(s.session_id),
            "fused": score,
            "instant_fused": score,
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
            "warming": True,
            "provisional": True,
            "speech_onset_s": s.speech_onset_s,
            "first_confirmed_read_s": s.first_confirmed_read_s,
            "audit_event": None,
            "quality": quality.as_dict(),
            "evidence": None,
            "evidence_level": None,
            "channel": None,
            "audio_notes": [],
            "spoof_windows": None,
        }


def _quiet_note(quality: InputQuality) -> list[str]:
    """The gate raised a very quiet signal before the models saw it: say so in plain words."""
    if any(n.startswith("very quiet audio") for n in quality.notes):
        return ["The caller is very quiet or far from the microphone."]
    return []


def build_session_stack(
    model_name: str = "aasist",
    *,
    pretrained: bool = True,
    weights: str | None = None,
    prosody: bool | None = None,
    audit_path: str | None = None,
) -> SessionManager:
    """Build the scorer + speaker verifier + prosody + audit log +
    SessionManager stack. The one place this wiring happens -- every
    transport (REST/WS in :mod:`voiceguard.serve.app`, gRPC in
    :mod:`voiceguard.serve.grpc_service`) calls this and gets the identical
    decision engine, so "same engine, different transport" is actually true
    of the code, not just asserted.

    ``prosody``: compute the prosody branch (~165 ms per assessment). ``None``
    (default) means only when it can affect the decision, i.e. when the learned
    fusion is absent and the hand-set fallback (which weights it) is in use.
    The learned fusion does not use prosody (~chance on the system trials), so
    by default it is neither computed nor shown.

    ``audit_path``: where decisions are logged (default ``logs/session_audit.jsonl``).
    Evaluation runs replay thousands of independent reads and pass a scratch path so
    they never write into the product's real audit trail."""
    from voiceguard.detect import NB_SAMP, build_scorer

    if weights is None and pretrained and model_name == "aasist":
        # the newest noise / room / line robust fine-tune that exists is the default
        for name in DEFAULT_WEIGHTS_PREFERENCE:
            if (REPO_ROOT / "models" / "pretrained" / name).exists():
                weights = name
                break
    src = weights or "upstream"
    print(f"[voiceguard.serve] loading {model_name} ({src}) ...", flush=True)
    scorer = build_scorer(model_name, pretrained=pretrained, weights=weights)
    print(f"[voiceguard.serve] ready -- {scorer.name} on {scorer.device}", flush=True)

    verifier = SpeakerVerifier()  # WavLM model lazy-loads on first enrol / score
    audit = AuditLog(
        audit_path or REPO_ROOT / "logs" / "session_audit.jsonl",
        model=scorer.name,
        front_end=scorer.pcfg.fingerprint(),
    )
    fusion = LearnedFusion.load()
    want_prosody = (fusion is None) if prosody is None else prosody
    prosody_scorer = None
    if want_prosody:
        try:
            from voiceguard.prosody import ProsodyScorer

            prosody_scorer = ProsodyScorer()
            prosody_scorer.warm()
            print("[voiceguard.serve] prosody branch active", flush=True)
        except Exception as exc:  # noqa: BLE001 -- optional: needs [prosody] + the trained model
            print(f"[voiceguard.serve] prosody branch off ({exc})", flush=True)
    else:
        print("[voiceguard.serve] prosody branch off (not a decision input)", flush=True)

    kind = "learned, channel-aware" if fusion else "hand-set weighted mean (no fusion.json)"
    print(f"[voiceguard.serve] fusion: {kind}", flush=True)
    why = fusion and fusion.detector_mismatch(scorer.name, scorer.operating_point)
    if why:  # not fatal (evaluation scripts deliberately swap detectors), but never silent
        print(f"[voiceguard.serve] WARNING: {why}; thresholds may not match", flush=True)
    return SessionManager(
        scorer,
        verifier,
        audit,
        sr=scorer.sr,
        nb_samp=NB_SAMP,
        prosody=prosody_scorer,
        fusion=fusion,
    )
