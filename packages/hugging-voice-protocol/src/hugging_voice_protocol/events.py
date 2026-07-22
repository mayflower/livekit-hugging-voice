"""Strict version-1 event models for the internal realtime WebSocket."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator

from .audio import MAX_AUDIO_BASE64_CHARS, decode_pcm16_base64
from .errors import ErrorCode

MAX_INSTRUCTIONS_CHARS = 8_000
MAX_VOICE_INSTRUCTIONS_CHARS = 2_000
MAX_CONTEXT_ITEM_CHARS = 16_000
MAX_TEXT_DELTA_CHARS = 4_096
MAX_ERROR_MESSAGE_CHARS = 2_048

EventId = Annotated[str, Field(pattern=r"^evt_[A-Za-z0-9_-]{1,96}$", max_length=100)]
SessionId = Annotated[str, Field(pattern=r"^session_[A-Za-z0-9_-]{1,88}$", max_length=96)]
TurnId = Annotated[str, Field(pattern=r"^turn_[A-Za-z0-9_-]{1,91}$", max_length=96)]
GenerationId = Annotated[str, Field(pattern=r"^gen_[A-Za-z0-9_-]{1,92}$", max_length=96)]
ResponseId = Annotated[str, Field(pattern=r"^resp_[A-Za-z0-9_-]{1,91}$", max_length=96)]
ItemId = Annotated[str, Field(pattern=r"^item_[A-Za-z0-9_-]{1,91}$", max_length=96)]
LanguageCode = Annotated[
    str,
    Field(pattern=r"^[A-Za-z]{2,8}(?:-[A-Za-z0-9]{1,8})*$", max_length=35),
]
VoiceId = Annotated[
    str,
    Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$", max_length=64),
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AudioEncoding(StrEnum):
    PCM_S16LE = "pcm_s16le"


class ConversationRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class ResponseStatus(StrEnum):
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class ResponseReason(StrEnum):
    COMPLETED = "completed"
    CLIENT_CANCELLED = "client_cancelled"
    BARGE_IN = "barge_in"
    EMPTY_TRANSCRIPT = "empty_transcript"
    MODEL_ERROR = "model_error"
    TRANSPORT_ERROR = "transport_error"


class InputAudioFormat(StrictModel):
    encoding: Literal[AudioEncoding.PCM_S16LE] = AudioEncoding.PCM_S16LE
    sample_rate: Literal[16_000] = 16_000
    channels: Literal[1] = 1


class OutputAudioFormat(StrictModel):
    encoding: Literal[AudioEncoding.PCM_S16LE] = AudioEncoding.PCM_S16LE
    sample_rate: Literal[24_000] = 24_000
    channels: Literal[1] = 1


class ServerVADConfig(StrictModel):
    enabled: bool = True
    threshold: float = Field(default=0.6, ge=0.1, le=0.95)
    min_speech_ms: int = Field(default=384, ge=96, le=2_000)
    min_speech_continuation_ms: int = Field(default=192, ge=0, le=1_000)
    min_silence_ms: int = Field(default=500, ge=100, le=3_000)
    speech_pad_ms: int = Field(default=30, ge=0, le=500)
    short_segment_merge_ms: Literal[0] = 0


class SessionConfig(StrictModel):
    instructions: str = Field(default="", max_length=MAX_INSTRUCTIONS_CHARS)
    language: LanguageCode | None = None
    voice: VoiceId | None = None
    voice_instructions: str | None = Field(
        default=None,
        max_length=MAX_VOICE_INSTRUCTIONS_CHARS,
    )
    input_audio_format: InputAudioFormat = Field(default_factory=InputAudioFormat)
    output_audio_format: OutputAudioFormat = Field(default_factory=OutputAudioFormat)
    turn_detection: ServerVADConfig = Field(default_factory=ServerVADConfig)
    interrupt_response: bool = True
    input_audio_transcription: bool = True


class SessionModels(StrictModel):
    vad: Literal["silero-vad"] = "silero-vad"
    stt: Literal["nvidia/parakeet-tdt-0.6b-v3"] = "nvidia/parakeet-tdt-0.6b-v3"
    llm: Literal["google/gemma-4-31B-it"] = "google/gemma-4-31B-it"
    tts: Literal["Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"] = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"


class ModelRevisions(StrictModel):
    vad: str = Field(min_length=1, max_length=96)
    stt: str = Field(pattern=r"^[0-9a-f]{40}$")
    llm: str = Field(pattern=r"^[0-9a-f]{40}$")
    tts: str = Field(pattern=r"^[0-9a-f]{40}$")


class Usage(StrictModel):
    input_text_tokens: int = Field(default=0, ge=0)
    output_text_tokens: int = Field(default=0, ge=0)
    total_text_tokens: int = Field(default=0, ge=0)


class ErrorPayload(StrictModel):
    code: ErrorCode
    message: str = Field(min_length=1, max_length=MAX_ERROR_MESSAGE_CHARS)
    retryable: bool = False
    event_id: EventId | None = None


class ConversationItem(StrictModel):
    id: ItemId
    role: ConversationRole
    content: str = Field(min_length=1, max_length=MAX_CONTEXT_ITEM_CHARS)


class EventBase(StrictModel):
    type: str
    event_id: EventId
    protocol_version: Literal[1] = 1
    session_id: SessionId


class TurnEventBase(EventBase):
    turn_id: TurnId
    turn_revision: int = Field(ge=0)


class ResponseEventBase(TurnEventBase):
    generation_id: GenerationId
    response_id: ResponseId
    item_id: ItemId


class SessionUpdateEvent(EventBase):
    type: Literal["session.update"] = "session.update"
    session: SessionConfig


class InputAudioBufferAppendEvent(EventBase):
    type: Literal["input_audio_buffer.append"] = "input_audio_buffer.append"
    sequence: int = Field(ge=0)
    audio: str = Field(min_length=4, max_length=MAX_AUDIO_BASE64_CHARS)

    @field_validator("audio")
    @classmethod
    def validate_audio(cls, value: str) -> str:
        decode_pcm16_base64(value)
        return value


class InputAudioBufferCommitEvent(EventBase):
    type: Literal["input_audio_buffer.commit"] = "input_audio_buffer.commit"


class InputAudioBufferClearEvent(EventBase):
    type: Literal["input_audio_buffer.clear"] = "input_audio_buffer.clear"


class ConversationItemCreateEvent(EventBase):
    type: Literal["conversation.item.create"] = "conversation.item.create"
    item: ConversationItem


class ResponseCreateEvent(EventBase):
    type: Literal["response.create"] = "response.create"
    instructions: str | None = Field(default=None, max_length=MAX_INSTRUCTIONS_CHARS)


class ResponseCancelEvent(EventBase):
    type: Literal["response.cancel"] = "response.cancel"
    response_id: ResponseId
    generation_id: GenerationId


class SessionCreatedEvent(EventBase):
    type: Literal["session.created"] = "session.created"
    models: SessionModels
    revisions: ModelRevisions
    language: LanguageCode = "de"
    voice: VoiceId = "warm_female"
    supported_languages: tuple[LanguageCode, ...] = Field(
        default=("de",), min_length=1, max_length=32
    )
    supported_voices: tuple[VoiceId, ...] = Field(
        default=("warm_female",), min_length=1, max_length=64
    )
    input_sample_rate: Literal[16_000] = 16_000
    output_sample_rate: Literal[24_000] = 24_000


class ErrorEvent(EventBase):
    type: Literal["error"] = "error"
    error: ErrorPayload


class SpeechStartedEvent(TurnEventBase):
    type: Literal["input_audio_buffer.speech_started"] = "input_audio_buffer.speech_started"
    audio_start_ms: int = Field(ge=0)


class SpeechStoppedEvent(TurnEventBase):
    type: Literal["input_audio_buffer.speech_stopped"] = "input_audio_buffer.speech_stopped"
    audio_end_ms: int = Field(ge=0)


class InputTranscriptionDeltaEvent(TurnEventBase):
    type: Literal["conversation.item.input_audio_transcription.delta"] = (
        "conversation.item.input_audio_transcription.delta"
    )
    item_id: ItemId
    delta: str = Field(min_length=1, max_length=MAX_TEXT_DELTA_CHARS)


class InputTranscriptionCompletedEvent(TurnEventBase):
    type: Literal["conversation.item.input_audio_transcription.completed"] = (
        "conversation.item.input_audio_transcription.completed"
    )
    item_id: ItemId
    transcript: str = Field(max_length=MAX_CONTEXT_ITEM_CHARS)


class ResponseCreatedEvent(ResponseEventBase):
    type: Literal["response.created"] = "response.created"


class ResponseOutputTextDeltaEvent(ResponseEventBase):
    type: Literal["response.output_text.delta"] = "response.output_text.delta"
    delta: str = Field(min_length=1, max_length=MAX_TEXT_DELTA_CHARS)


class ResponseOutputTextDoneEvent(ResponseEventBase):
    type: Literal["response.output_text.done"] = "response.output_text.done"
    text: str = Field(max_length=MAX_CONTEXT_ITEM_CHARS)


class ResponseOutputAudioDeltaEvent(ResponseEventBase):
    type: Literal["response.output_audio.delta"] = "response.output_audio.delta"
    sequence: int = Field(ge=0)
    audio: str = Field(min_length=4, max_length=MAX_AUDIO_BASE64_CHARS)

    @field_validator("audio")
    @classmethod
    def validate_audio(cls, value: str) -> str:
        decode_pcm16_base64(value)
        return value


class ResponseOutputAudioDoneEvent(ResponseEventBase):
    type: Literal["response.output_audio.done"] = "response.output_audio.done"


class ResponseDoneEvent(ResponseEventBase):
    type: Literal["response.done"] = "response.done"
    status: ResponseStatus
    reason: ResponseReason
    usage: Usage = Field(default_factory=Usage)


ClientEvent: TypeAlias = Annotated[
    SessionUpdateEvent
    | InputAudioBufferAppendEvent
    | InputAudioBufferCommitEvent
    | InputAudioBufferClearEvent
    | ConversationItemCreateEvent
    | ResponseCreateEvent
    | ResponseCancelEvent,
    Field(discriminator="type"),
]

ServerEvent: TypeAlias = Annotated[
    SessionCreatedEvent
    | ErrorEvent
    | SpeechStartedEvent
    | SpeechStoppedEvent
    | InputTranscriptionDeltaEvent
    | InputTranscriptionCompletedEvent
    | ResponseCreatedEvent
    | ResponseOutputTextDeltaEvent
    | ResponseOutputTextDoneEvent
    | ResponseOutputAudioDeltaEvent
    | ResponseOutputAudioDoneEvent
    | ResponseDoneEvent,
    Field(discriminator="type"),
]

CLIENT_EVENT_ADAPTER: TypeAdapter[ClientEvent] = TypeAdapter(ClientEvent)
SERVER_EVENT_ADAPTER: TypeAdapter[ServerEvent] = TypeAdapter(ServerEvent)


def parse_client_event_json(data: str | bytes) -> ClientEvent:
    return CLIENT_EVENT_ADAPTER.validate_json(data)


def parse_server_event_json(data: str | bytes) -> ServerEvent:
    return SERVER_EVENT_ADAPTER.validate_json(data)
