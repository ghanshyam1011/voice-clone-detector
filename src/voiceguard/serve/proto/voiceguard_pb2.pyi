from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ContextState(_message.Message):
    __slots__ = ("caller_known", "amount", "channel", "prior_flags")
    CALLER_KNOWN_FIELD_NUMBER: _ClassVar[int]
    AMOUNT_FIELD_NUMBER: _ClassVar[int]
    CHANNEL_FIELD_NUMBER: _ClassVar[int]
    PRIOR_FLAGS_FIELD_NUMBER: _ClassVar[int]
    caller_known: bool
    amount: float
    channel: str
    prior_flags: int
    def __init__(self, caller_known: _Optional[bool] = ..., amount: _Optional[float] = ..., channel: _Optional[str] = ..., prior_flags: _Optional[int] = ...) -> None: ...

class CreateSessionRequest(_message.Message):
    __slots__ = ("scenario", "caller_known", "amount", "channel")
    SCENARIO_FIELD_NUMBER: _ClassVar[int]
    CALLER_KNOWN_FIELD_NUMBER: _ClassVar[int]
    AMOUNT_FIELD_NUMBER: _ClassVar[int]
    CHANNEL_FIELD_NUMBER: _ClassVar[int]
    scenario: str
    caller_known: bool
    amount: float
    channel: str
    def __init__(self, scenario: _Optional[str] = ..., caller_known: _Optional[bool] = ..., amount: _Optional[float] = ..., channel: _Optional[str] = ...) -> None: ...

class SessionState(_message.Message):
    __slots__ = ("session_id", "scenario", "context")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    SCENARIO_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    scenario: str
    context: ContextState
    def __init__(self, session_id: _Optional[str] = ..., scenario: _Optional[str] = ..., context: _Optional[_Union[ContextState, _Mapping]] = ...) -> None: ...

class UpdateContextRequest(_message.Message):
    __slots__ = ("session_id", "scenario", "caller_known", "amount", "channel", "prior_flags")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    SCENARIO_FIELD_NUMBER: _ClassVar[int]
    CALLER_KNOWN_FIELD_NUMBER: _ClassVar[int]
    AMOUNT_FIELD_NUMBER: _ClassVar[int]
    CHANNEL_FIELD_NUMBER: _ClassVar[int]
    PRIOR_FLAGS_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    scenario: str
    caller_known: bool
    amount: float
    channel: str
    prior_flags: int
    def __init__(self, session_id: _Optional[str] = ..., scenario: _Optional[str] = ..., caller_known: _Optional[bool] = ..., amount: _Optional[float] = ..., channel: _Optional[str] = ..., prior_flags: _Optional[int] = ...) -> None: ...

class EnrollRequest(_message.Message):
    __slots__ = ("session_id", "audio", "filename")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    AUDIO_FIELD_NUMBER: _ClassVar[int]
    FILENAME_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    audio: bytes
    filename: str
    def __init__(self, session_id: _Optional[str] = ..., audio: _Optional[bytes] = ..., filename: _Optional[str] = ...) -> None: ...

class EnrollResponse(_message.Message):
    __slots__ = ("enrolled", "seconds", "short")
    ENROLLED_FIELD_NUMBER: _ClassVar[int]
    SECONDS_FIELD_NUMBER: _ClassVar[int]
    SHORT_FIELD_NUMBER: _ClassVar[int]
    enrolled: bool
    seconds: float
    short: bool
    def __init__(self, enrolled: _Optional[bool] = ..., seconds: _Optional[float] = ..., short: _Optional[bool] = ...) -> None: ...

class AnalyzeRequest(_message.Message):
    __slots__ = ("session_id", "audio", "filename", "hop")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    AUDIO_FIELD_NUMBER: _ClassVar[int]
    FILENAME_FIELD_NUMBER: _ClassVar[int]
    HOP_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    audio: bytes
    filename: str
    hop: float
    def __init__(self, session_id: _Optional[str] = ..., audio: _Optional[bytes] = ..., filename: _Optional[str] = ..., hop: _Optional[float] = ...) -> None: ...

class AcknowledgeRequest(_message.Message):
    __slots__ = ("session_id", "by")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    BY_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    by: str
    def __init__(self, session_id: _Optional[str] = ..., by: _Optional[str] = ...) -> None: ...

class AuditEvent(_message.Message):
    __slots__ = ("event_id", "utc", "session_id", "action", "fused", "signals", "reasons", "scenario", "model", "front_end", "git_commit")
    class SignalsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    EVENT_ID_FIELD_NUMBER: _ClassVar[int]
    UTC_FIELD_NUMBER: _ClassVar[int]
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    ACTION_FIELD_NUMBER: _ClassVar[int]
    FUSED_FIELD_NUMBER: _ClassVar[int]
    SIGNALS_FIELD_NUMBER: _ClassVar[int]
    REASONS_FIELD_NUMBER: _ClassVar[int]
    SCENARIO_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    FRONT_END_FIELD_NUMBER: _ClassVar[int]
    GIT_COMMIT_FIELD_NUMBER: _ClassVar[int]
    event_id: str
    utc: str
    session_id: str
    action: str
    fused: float
    signals: _containers.ScalarMap[str, float]
    reasons: _containers.RepeatedScalarFieldContainer[str]
    scenario: str
    model: str
    front_end: str
    git_commit: str
    def __init__(self, event_id: _Optional[str] = ..., utc: _Optional[str] = ..., session_id: _Optional[str] = ..., action: _Optional[str] = ..., fused: _Optional[float] = ..., signals: _Optional[_Mapping[str, float]] = ..., reasons: _Optional[_Iterable[str]] = ..., scenario: _Optional[str] = ..., model: _Optional[str] = ..., front_end: _Optional[str] = ..., git_commit: _Optional[str] = ...) -> None: ...

class AcknowledgeResponse(_message.Message):
    __slots__ = ("acknowledged", "audit_event")
    ACKNOWLEDGED_FIELD_NUMBER: _ClassVar[int]
    AUDIT_EVENT_FIELD_NUMBER: _ClassVar[int]
    acknowledged: bool
    audit_event: AuditEvent
    def __init__(self, acknowledged: _Optional[bool] = ..., audit_event: _Optional[_Union[AuditEvent, _Mapping]] = ...) -> None: ...

class GetAuditRequest(_message.Message):
    __slots__ = ("session_id",)
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    def __init__(self, session_id: _Optional[str] = ...) -> None: ...

class AuditTrail(_message.Message):
    __slots__ = ("events",)
    EVENTS_FIELD_NUMBER: _ClassVar[int]
    events: _containers.RepeatedCompositeFieldContainer[AuditEvent]
    def __init__(self, events: _Optional[_Iterable[_Union[AuditEvent, _Mapping]]] = ...) -> None: ...

class AudioChunk(_message.Message):
    __slots__ = ("session_id", "pcm_f32le", "sample_rate")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    PCM_F32LE_FIELD_NUMBER: _ClassVar[int]
    SAMPLE_RATE_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    pcm_f32le: bytes
    sample_rate: int
    def __init__(self, session_id: _Optional[str] = ..., pcm_f32le: _Optional[bytes] = ..., sample_rate: _Optional[int] = ...) -> None: ...

class Assessment(_message.Message):
    __slots__ = ("spoof", "speaker", "prosody", "context", "similarity", "enrolled", "fused", "action", "scenario", "reasons", "recommendation", "context_state", "level", "warming", "provisional", "speech_onset_s", "first_confirmed_read_s", "audit_event")
    SPOOF_FIELD_NUMBER: _ClassVar[int]
    SPEAKER_FIELD_NUMBER: _ClassVar[int]
    PROSODY_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    SIMILARITY_FIELD_NUMBER: _ClassVar[int]
    ENROLLED_FIELD_NUMBER: _ClassVar[int]
    FUSED_FIELD_NUMBER: _ClassVar[int]
    ACTION_FIELD_NUMBER: _ClassVar[int]
    SCENARIO_FIELD_NUMBER: _ClassVar[int]
    REASONS_FIELD_NUMBER: _ClassVar[int]
    RECOMMENDATION_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_STATE_FIELD_NUMBER: _ClassVar[int]
    LEVEL_FIELD_NUMBER: _ClassVar[int]
    WARMING_FIELD_NUMBER: _ClassVar[int]
    PROVISIONAL_FIELD_NUMBER: _ClassVar[int]
    SPEECH_ONSET_S_FIELD_NUMBER: _ClassVar[int]
    FIRST_CONFIRMED_READ_S_FIELD_NUMBER: _ClassVar[int]
    AUDIT_EVENT_FIELD_NUMBER: _ClassVar[int]
    spoof: float
    speaker: float
    prosody: float
    context: float
    similarity: float
    enrolled: bool
    fused: float
    action: str
    scenario: str
    reasons: _containers.RepeatedScalarFieldContainer[str]
    recommendation: str
    context_state: ContextState
    level: float
    warming: bool
    provisional: bool
    speech_onset_s: float
    first_confirmed_read_s: float
    audit_event: AuditEvent
    def __init__(self, spoof: _Optional[float] = ..., speaker: _Optional[float] = ..., prosody: _Optional[float] = ..., context: _Optional[float] = ..., similarity: _Optional[float] = ..., enrolled: _Optional[bool] = ..., fused: _Optional[float] = ..., action: _Optional[str] = ..., scenario: _Optional[str] = ..., reasons: _Optional[_Iterable[str]] = ..., recommendation: _Optional[str] = ..., context_state: _Optional[_Union[ContextState, _Mapping]] = ..., level: _Optional[float] = ..., warming: _Optional[bool] = ..., provisional: _Optional[bool] = ..., speech_onset_s: _Optional[float] = ..., first_confirmed_read_s: _Optional[float] = ..., audit_event: _Optional[_Union[AuditEvent, _Mapping]] = ...) -> None: ...
