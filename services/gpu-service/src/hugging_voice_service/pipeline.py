"""Per-session realtime VAD -> STT -> Gemma -> TTS orchestration."""

from __future__ import annotations

import asyncio
import base64
import logging
import time
import uuid
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from hugging_voice_protocol.audio import decode_pcm16_base64
from hugging_voice_protocol.errors import CloseCode, ErrorCode
from hugging_voice_protocol.events import (
    ClientEvent,
    ConversationItemCreatedEvent,
    ConversationItemCreateEvent,
    ErrorEvent,
    ErrorPayload,
    FunctionCallConversationItem,
    FunctionTool,
    InputAudioBufferAppendEvent,
    InputAudioBufferClearEvent,
    InputAudioBufferCommitEvent,
    InputTranscriptionCompletedEvent,
    InputTranscriptionDeltaEvent,
    ResponseCancelEvent,
    ResponseCreatedEvent,
    ResponseCreateEvent,
    ResponseDoneEvent,
    ResponseOutputAudioDeltaEvent,
    ResponseOutputAudioDoneEvent,
    ResponseOutputFunctionCallDoneEvent,
    ResponseOutputTextDeltaEvent,
    ResponseOutputTextDoneEvent,
    ResponseReason,
    ResponseStatus,
    SessionUpdatedEvent,
    SessionUpdateEvent,
    SpeechStartedEvent,
    SpeechStoppedEvent,
    ToolChoice,
    Usage,
)

from .cancellation import GenerationToken
from .config import SpeechSettings
from .conversation import (
    ConversationRole,
    FunctionCallEntry,
    FunctionCallOutputEntry,
)
from .runtimes.gemma import (
    GemmaMessage,
    TextDelta,
    TextUsage,
    ToolCall,
    ToolCallValidationError,
)
from .schedulers.stt import STTJob
from .schedulers.tts import TTSJob
from .sessions import SessionState
from .telemetry import ServiceTelemetry
from .text_segmenter import SpeechTextSegmenter

logger = logging.getLogger(__name__)


class PipelineEventError(ValueError):
    """A recoverable, structured rejection of one client event."""

    def __init__(self, code: ErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code


class GemmaStreamer(Protocol):
    def stream_response(
        self,
        *,
        messages: Sequence[GemmaMessage],
        instructions: str = "",
        language_instruction: str = "",
        system_prompt: str = "",
        tools: Sequence[Any] = (),
        tool_choice: Any = "auto",
        slot_id: int,
    ) -> AsyncIterator[TextDelta | ToolCall | TextUsage]: ...


class STTScheduling(Protocol):
    async def submit_final(self, job: STTJob) -> str: ...

    async def submit_partial(self, job: STTJob) -> str | None: ...

    async def cancel_session(self, session_id: str) -> None: ...

    async def wait_session_idle(self, session_id: str) -> None: ...


class TTSScheduling(Protocol):
    async def synthesize(self, job: TTSJob) -> None: ...

    async def cancel_generation(self, token: GenerationToken) -> None: ...

    async def wait_session_idle(self, session_id: str) -> None: ...


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


@dataclass(slots=True)
class ResponseContext:
    turn_id: str
    turn_revision: int
    generation_id: str
    response_id: str
    item_id: str
    token: GenerationToken
    instructions: str
    model_language: str
    language_instruction: str
    voice_instructions: str
    system_prompt: str
    tools: tuple[FunctionTool, ...]
    tool_choice: ToolChoice
    started_at: float
    speech_stopped_at: float | None
    text: str = ""
    audio_sequence: int = 0
    usage: TextUsage | None = None
    text_done: bool = False
    audio_done: bool = False
    finalized: bool = False
    cancel_reason: ResponseReason = ResponseReason.CLIENT_CANCELLED


@dataclass(frozen=True, slots=True)
class CompletedTurn:
    turn_id: str
    revision: int
    audio: bytes
    transcription_enabled: bool
    speech_stopped_at: float


class VoicePipeline:
    partial_interval_seconds = 0.5
    completed_turn_queue_size = 4

    def __init__(
        self,
        state: SessionState,
        *,
        stt: STTScheduling,
        tts: TTSScheduling,
        gemma: GemmaStreamer,
        speech: SpeechSettings,
        telemetry: ServiceTelemetry,
    ) -> None:
        self.state = state
        self._stt = stt
        self._tts = tts
        self._gemma = gemma
        self._speech = speech
        self._telemetry = telemetry
        self._response: ResponseContext | None = None
        self._response_task: asyncio.Task[None] | None = None
        self._partial_task: asyncio.Task[None] | None = None
        self._completed_turn: tuple[str, int] | None = None
        self._completed_turns: asyncio.Queue[CompletedTurn] = asyncio.Queue(
            maxsize=self.completed_turn_queue_size
        )
        self._turn_worker_task: asyncio.Task[None] | None = None
        self._turns_idle = asyncio.Event()
        self._turns_idle.set()
        self._draining = False
        self._speech_stopped_at: float | None = None
        self._response_idle = asyncio.Event()
        self._response_idle.set()
        state.pipeline = self

    @property
    def active_response(self) -> ResponseContext | None:
        return self._response

    async def wait_response_idle(self) -> None:
        await self._response_idle.wait()

    async def wait_turns_idle(self) -> None:
        await self._turns_idle.wait()

    async def handle_event(self, event: ClientEvent) -> None:
        if self._draining:
            raise RuntimeError("session is draining")
        if event.session_id != self.state.session_id:
            raise ValueError("event session_id does not match the claimed session")
        self.state.last_activity_at = time.monotonic()
        if not isinstance(event, SessionUpdateEvent | ConversationItemCreateEvent):
            self.state.context_replay_open = False
        if isinstance(event, SessionUpdateEvent):
            await self._update_session(event)
        elif isinstance(event, InputAudioBufferAppendEvent):
            await self._append_audio(event)
        elif isinstance(event, InputAudioBufferCommitEvent):
            await self._commit_audio()
        elif isinstance(event, InputAudioBufferClearEvent):
            await self._clear_audio()
        elif isinstance(event, ConversationItemCreateEvent):
            await self._create_conversation_item(event)
        elif isinstance(event, ResponseCreateEvent):
            await self._start_response(
                response_instructions=event.instructions or "",
                tools=event.tools,
                tool_choice=event.tool_choice,
            )
        elif isinstance(event, ResponseCancelEvent):
            await self._cancel_response(
                reason=ResponseReason.CLIENT_CANCELLED,
                generation_id=event.generation_id,
                response_id=event.response_id,
            )
        else:
            raise ValueError(f"unsupported client event {event.type}")

    async def drain(self) -> None:
        if self._draining:
            return
        self._draining = True
        self.state.partial_epoch += 1
        if self._partial_task is not None:
            self._partial_task.cancel()
            await asyncio.gather(self._partial_task, return_exceptions=True)
            self._partial_task = None
        if self._turn_worker_task is not None:
            self._turn_worker_task.cancel()
            await asyncio.gather(self._turn_worker_task, return_exceptions=True)
            self._turn_worker_task = None
        while not self._completed_turns.empty():
            self._completed_turns.get_nowait()
            self._completed_turns.task_done()
        self._turns_idle.set()
        await self._cancel_response(reason=ResponseReason.TRANSPORT_ERROR)
        await self._stt.cancel_session(self.state.session_id)
        await self._stt.wait_session_idle(self.state.session_id)
        await self._tts.wait_session_idle(self.state.session_id)
        self.state.input_audio_buffer.clear()
        self.state.vad.reset()
        self.state.cancellation.reset()

    async def send_error(
        self,
        code: ErrorCode,
        message: str,
        *,
        retryable: bool = False,
        source_event_id: str | None = None,
    ) -> None:
        await self.state.transport.send(
            ErrorEvent(
                event_id=_id("evt"),
                session_id=self.state.session_id,
                error=ErrorPayload(
                    code=code,
                    message=message,
                    retryable=retryable,
                    event_id=source_event_id,
                ),
            )
        )

    async def _update_session(self, event: SessionUpdateEvent) -> None:
        config = event.session
        language = config.language or self._speech.default_language
        voice = config.voice or self._speech.default_voice
        self._speech.resolve_language(language)
        self._speech.resolve_voice(voice)
        if self.state.tools_frozen and config.tools != self.state.tools:
            raise PipelineEventError(
                ErrorCode.INVALID_TOOL_CONFIGURATION,
                "tools are immutable after the initial session update",
            )
        vad_configuration = (
            config.turn_detection.threshold,
            config.turn_detection.min_speech_ms,
            config.turn_detection.min_speech_continuation_ms,
            config.turn_detection.min_silence_ms,
            config.turn_detection.speech_pad_ms,
        )
        changes_turn_detection = (
            vad_configuration != self.state.vad.configuration
            or config.turn_detection.enabled != self.state.vad_enabled
        )
        if changes_turn_detection and (
            self.state.speech_start_sample is not None
            or self.state.input_audio_buffer.size_bytes > 0
        ):
            raise ValueError("cannot change VAD settings while input audio is buffered")
        self.state.vad.configure(
            threshold=config.turn_detection.threshold,
            min_speech_ms=config.turn_detection.min_speech_ms,
            min_speech_continuation_ms=config.turn_detection.min_speech_continuation_ms,
            min_silence_ms=config.turn_detection.min_silence_ms,
            speech_pad_ms=config.turn_detection.speech_pad_ms,
        )
        self.state.instructions = config.instructions
        self.state.tools = config.tools
        self.state.tool_choice = config.tool_choice
        self._telemetry.tool_schema_bytes.set(
            sum(len(tool.model_dump_json().encode("utf-8")) for tool in config.tools)
        )
        self.state.language = language
        self.state.voice = voice
        self.state.voice_instructions = config.voice_instructions
        self.state.vad_enabled = config.turn_detection.enabled
        self.state.transcription_enabled = config.input_audio_transcription
        self.state.interrupt_response = config.interrupt_response
        await self.state.transport.send(
            SessionUpdatedEvent(
                event_id=_id("evt"),
                session_id=self.state.session_id,
                source_event_id=event.event_id,
            )
        )

    async def _append_audio(self, event: InputAudioBufferAppendEvent) -> None:
        if event.sequence != self.state.next_audio_sequence:
            raise ValueError(
                f"audio sequence conflict: expected={self.state.next_audio_sequence} "
                f"received={event.sequence}"
            )
        self.state.next_audio_sequence += 1
        pcm16 = decode_pcm16_base64(event.audio)
        self.state.input_audio_buffer.append(pcm16)
        if not self.state.vad_enabled:
            return
        signals = await asyncio.to_thread(self.state.vad.process_pcm16, pcm16)
        for signal in signals:
            if signal.kind == "speech_started":
                await self._speech_started(signal.sample_index)
            else:
                await self._speech_stopped(signal.sample_index)
        if self.state.speech_start_sample is not None:
            self._schedule_partial()
        elif self.state.input_audio_buffer.size_bytes > 64_000:
            keep_from = max(
                self.state.input_audio_buffer.first_sample,
                self.state.input_audio_buffer.end_sample - 32_000,
            )
            self.state.input_audio_buffer.discard_before(keep_from)

    async def _speech_started(self, sample_index: int) -> None:
        self.state.current_turn_revision += 1
        self.state.current_turn_id = _id("turn")
        self.state.speech_start_sample = sample_index
        self.state.partial_epoch += 1
        if self.state.pending_call is not None:
            self.state.pending_call = None
            self.state.pending_call_emitted_at = None
            self._telemetry.tool_call_rejections.inc()
        if self.state.interrupt_response and self._response is not None:
            started = time.monotonic()
            await self._cancel_response(reason=ResponseReason.BARGE_IN)
            self._telemetry.barge_in_stop_latency_seconds.observe(time.monotonic() - started)
        await self.state.transport.send(
            SpeechStartedEvent(
                event_id=_id("evt"),
                session_id=self.state.session_id,
                turn_id=self.state.current_turn_id,
                turn_revision=self.state.current_turn_revision,
                audio_start_ms=sample_index * 1_000 // 16_000,
            )
        )

    async def _speech_stopped(self, sample_index: int) -> None:
        turn_id = self.state.current_turn_id
        start = self.state.speech_start_sample
        if turn_id is None or start is None:
            return
        revision = self.state.current_turn_revision
        self._speech_stopped_at = time.monotonic()
        await self.state.transport.send(
            SpeechStoppedEvent(
                event_id=_id("evt"),
                session_id=self.state.session_id,
                turn_id=turn_id,
                turn_revision=revision,
                audio_end_ms=sample_index * 1_000 // 16_000,
            )
        )
        self.state.speech_start_sample = None
        await self._complete_turn(turn_id, revision, start, sample_index)

    def _schedule_partial(self) -> None:
        now = time.monotonic()
        if now - self.state.last_partial_at < self.partial_interval_seconds:
            return
        if self._partial_task is not None and not self._partial_task.done():
            return
        turn_id = self.state.current_turn_id
        start = self.state.speech_start_sample
        if turn_id is None or start is None:
            return
        revision = self.state.current_turn_revision
        epoch = self.state.partial_epoch
        audio = self.state.input_audio_buffer.slice_samples(
            start,
            self.state.input_audio_buffer.end_sample,
        )
        self.state.last_partial_at = now
        self._partial_task = asyncio.create_task(self._run_partial(turn_id, revision, epoch, audio))

    async def _run_partial(self, turn_id: str, revision: int, epoch: int, audio: bytes) -> None:
        def stale() -> bool:
            return (
                self._draining
                or self.state.current_turn_id != turn_id
                or self.state.current_turn_revision != revision
                or self.state.partial_epoch != epoch
            )

        try:
            transcript = await self._stt.submit_partial(
                STTJob(
                    session_id=self.state.session_id,
                    turn_id=turn_id,
                    turn_revision=revision,
                    audio=audio,
                    final=False,
                    is_stale=stale,
                )
            )
            if transcript and not stale() and self.state.transcription_enabled:
                await self.state.transport.send(
                    InputTranscriptionDeltaEvent(
                        event_id=_id("evt"),
                        session_id=self.state.session_id,
                        turn_id=turn_id,
                        turn_revision=revision,
                        item_id=self._user_item_id(turn_id),
                        delta=transcript,
                    )
                )
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.warning(
                "partial_transcription_failed",
                extra={"session_id": self.state.session_id, "turn_id": turn_id, "error": str(exc)},
            )

    async def _complete_turn(
        self,
        turn_id: str,
        revision: int,
        start_sample: int,
        end_sample: int,
    ) -> None:
        turn_key = (turn_id, revision)
        if turn_key == self._completed_turn:
            return
        self._completed_turn = turn_key
        self.state.partial_epoch += 1
        if self._partial_task is not None:
            self._partial_task.cancel()
            await asyncio.gather(self._partial_task, return_exceptions=True)
            self._partial_task = None
        audio = self.state.input_audio_buffer.slice_samples(start_sample, end_sample)
        self.state.input_audio_buffer.discard_before(end_sample)
        completed = CompletedTurn(
            turn_id=turn_id,
            revision=revision,
            audio=audio,
            transcription_enabled=self.state.transcription_enabled,
            speech_stopped_at=self._speech_stopped_at or time.monotonic(),
        )
        self._turns_idle.clear()
        if self._turn_worker_task is None or self._turn_worker_task.done():
            self._turn_worker_task = asyncio.create_task(self._run_completed_turns())
        await self._completed_turns.put(completed)

    async def _run_completed_turns(self) -> None:
        while True:
            completed = await self._completed_turns.get()
            try:
                await self._process_completed_turn(completed)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception(
                    "completed_turn_failed",
                    extra={
                        "session_id": self.state.session_id,
                        "turn_id": completed.turn_id,
                        "turn_revision": completed.revision,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
                await self.send_error(
                    ErrorCode.MODEL_FAILURE,
                    "final transcription failed",
                )
                await self.state.transport.close(
                    code=int(CloseCode.SERVICE_FAILURE),
                    reason=ErrorCode.MODEL_FAILURE,
                )
                return
            finally:
                self._completed_turns.task_done()
                if self._completed_turns.empty():
                    self._turns_idle.set()

    async def _process_completed_turn(self, completed: CompletedTurn) -> None:
        def stale_for_scheduler() -> bool:
            return self._draining

        started = time.monotonic()
        transcript = await self._stt.submit_final(
            STTJob(
                session_id=self.state.session_id,
                turn_id=completed.turn_id,
                turn_revision=completed.revision,
                audio=completed.audio,
                final=True,
                is_stale=stale_for_scheduler,
            )
        )
        self._telemetry.transcription_delay_seconds.observe(time.monotonic() - started)
        self._telemetry.turns.inc()
        item_id = self._user_item_id(completed.turn_id)
        if completed.transcription_enabled:
            await self.state.transport.send(
                InputTranscriptionCompletedEvent(
                    event_id=_id("evt"),
                    session_id=self.state.session_id,
                    turn_id=completed.turn_id,
                    turn_revision=completed.revision,
                    item_id=item_id,
                    transcript=transcript,
                )
            )
        stale_response = self._draining or self.state.current_turn_revision != completed.revision
        if transcript.strip() and not stale_response:
            self.state.conversation.append(item_id=item_id, role="user", content=transcript)
            await self._start_response(
                response_instructions="",
                speech_stopped_at=completed.speech_stopped_at,
            )

    async def _commit_audio(self) -> None:
        if self.state.speech_start_sample is not None:
            await self._speech_stopped(self.state.input_audio_buffer.end_sample)
            self.state.vad.reset()
            self.state.input_audio_buffer.clear()
            return
        if self.state.input_audio_buffer.size_bytes == 0:
            return
        self.state.current_turn_revision += 1
        self.state.current_turn_id = _id("turn")
        start = self.state.input_audio_buffer.first_sample
        end = self.state.input_audio_buffer.end_sample
        self.state.speech_start_sample = start
        await self._speech_started_manual(start)
        await self._speech_stopped(end)
        self.state.vad.reset()
        self.state.input_audio_buffer.clear()

    async def _speech_started_manual(self, sample_index: int) -> None:
        assert self.state.current_turn_id is not None
        await self.state.transport.send(
            SpeechStartedEvent(
                event_id=_id("evt"),
                session_id=self.state.session_id,
                turn_id=self.state.current_turn_id,
                turn_revision=self.state.current_turn_revision,
                audio_start_ms=sample_index * 1_000 // 16_000,
            )
        )

    async def _clear_audio(self) -> None:
        self.state.partial_epoch += 1
        if self._partial_task is not None:
            self._partial_task.cancel()
            await asyncio.gather(self._partial_task, return_exceptions=True)
            self._partial_task = None
        self.state.input_audio_buffer.clear()
        self.state.vad.reset()
        self.state.current_turn_id = None
        self.state.speech_start_sample = None

    async def _create_conversation_item(self, event: ConversationItemCreateEvent) -> None:
        if self._response is not None:
            raise PipelineEventError(
                ErrorCode.TOOL_CALL_STATE_CONFLICT,
                "cannot update conversation while a response is active",
            )
        item = event.item
        if item.type == "message":
            role: ConversationRole = "user" if item.role.value == "user" else "assistant"
            self.state.conversation.append(item_id=item.id, role=role, content=item.content)
        elif item.type == "function_call":
            pending = self.state.pending_call
            if pending is None:
                if not self.state.context_replay_open or self.state.tools_frozen:
                    raise PipelineEventError(
                        ErrorCode.TOOL_CALL_STATE_CONFLICT,
                        "a function call may only be restored during context replay",
                    )
                if item.name not in {tool.function.name for tool in self.state.tools}:
                    raise PipelineEventError(
                        ErrorCode.UNKNOWN_TOOL_NAME,
                        "replayed function call references an unknown tool",
                    )
                self.state.pending_call = item
            elif item != pending:
                raise PipelineEventError(
                    ErrorCode.TOOL_CALL_STATE_CONFLICT,
                    "only the current server-issued function call may be replayed",
                )
        else:
            pending = self.state.pending_call
            if pending is None:
                code = (
                    ErrorCode.DUPLICATE_TOOL_CALL_OUTPUT
                    if self.state.conversation.has_call(item.call_id)
                    else ErrorCode.UNKNOWN_TOOL_CALL_OUTPUT
                )
                raise PipelineEventError(code, "tool output has no matching pending function call")
            if item.call_id != pending.call_id:
                raise PipelineEventError(
                    ErrorCode.UNKNOWN_TOOL_CALL_OUTPUT,
                    "tool output does not match the pending function call",
                )
            if item.name != pending.name:
                raise PipelineEventError(
                    ErrorCode.TOOL_CALL_STATE_CONFLICT,
                    "tool output name does not match the pending function call",
                )
            if (
                item.turn_id,
                item.turn_revision,
                item.generation_id,
                item.response_id,
            ) != (
                pending.turn_id,
                pending.turn_revision,
                pending.generation_id,
                pending.response_id,
            ):
                raise PipelineEventError(
                    ErrorCode.STALE_TOOL_CALL_OUTPUT,
                    "tool output correlations are stale",
                )
            self.state.conversation.commit_tool_exchange(
                call=FunctionCallEntry(
                    item_id=pending.id,
                    call_id=pending.call_id,
                    name=pending.name,
                    arguments=pending.arguments,
                    turn_id=pending.turn_id,
                    turn_revision=pending.turn_revision,
                    generation_id=pending.generation_id,
                    response_id=pending.response_id,
                ),
                output=FunctionCallOutputEntry(
                    item_id=item.id,
                    call_id=item.call_id,
                    name=item.name,
                    output=item.output,
                    is_error=item.is_error,
                    turn_id=item.turn_id,
                    turn_revision=item.turn_revision,
                    generation_id=item.generation_id,
                    response_id=item.response_id,
                ),
            )
            self.state.pending_call = None
            self.state.tool_result_ack_at = time.monotonic()
            if self.state.pending_call_emitted_at is not None:
                self._telemetry.tool_result_wait_seconds.observe(
                    self.state.tool_result_ack_at - self.state.pending_call_emitted_at
                )
            logger.info(
                "tool_result_acknowledged",
                extra={
                    "session_id": self.state.session_id,
                    "turn_id": item.turn_id,
                    "generation_id": item.generation_id,
                    "response_id": item.response_id,
                    "call_id": item.call_id,
                    "tool_name": item.name,
                    "result_size": len(item.output),
                    "is_error": item.is_error,
                },
            )
        await self.state.transport.send(
            ConversationItemCreatedEvent(
                event_id=_id("evt"),
                session_id=self.state.session_id,
                source_event_id=event.event_id,
                item_id=item.id,
            )
        )

    async def _start_response(
        self,
        *,
        response_instructions: str,
        tools: Sequence[Any] | None = None,
        tool_choice: Any | None = None,
        speech_stopped_at: float | None = None,
    ) -> None:
        if self._response is not None:
            raise ValueError("a response is already active for this session")
        if self.state.pending_call is not None:
            raise PipelineEventError(
                ErrorCode.TOOL_CALL_STATE_CONFLICT,
                "pending tool call requires a confirmed output before another response",
            )
        self.state.tools_frozen = True
        if self.state.current_turn_id is None:
            self.state.current_turn_revision += 1
            self.state.current_turn_id = _id("turn")
        turn_id = self.state.current_turn_id
        generation_id = _id("gen")
        response_id = _id("resp")
        item_id = _id("item")
        token = self.state.cancellation.start(
            turn_id=turn_id,
            turn_revision=self.state.current_turn_revision,
            generation_id=generation_id,
        )
        instructions = self.state.instructions
        if response_instructions.strip():
            instructions = f"{instructions}\n{response_instructions}".strip()
        language = self._speech.resolve_language(self.state.language)
        voice = self._speech.resolve_voice(self.state.voice)
        context = ResponseContext(
            turn_id=turn_id,
            turn_revision=self.state.current_turn_revision,
            generation_id=generation_id,
            response_id=response_id,
            item_id=item_id,
            token=token,
            instructions=instructions,
            model_language=language.model_language,
            language_instruction=language.response_instruction,
            voice_instructions=voice.render(
                language.model_language,
                self.state.voice_instructions,
            ),
            system_prompt=self._speech.system_prompt,
            tools=tuple(self.state.tools if tools is None else tools),
            tool_choice=self.state.tool_choice if tool_choice is None else tool_choice,
            started_at=time.monotonic(),
            speech_stopped_at=(
                self._speech_stopped_at if speech_stopped_at is None else speech_stopped_at
            ),
        )
        self._response = context
        self._response_idle.clear()
        self.state.current_generation_id = generation_id
        self.state.current_response_id = response_id
        await self.state.transport.send(
            ResponseCreatedEvent(
                **self._response_fields(context),
            )
        )
        self._response_task = asyncio.create_task(self._run_response(context))

    async def _run_response(self, context: ResponseContext) -> None:
        segments: asyncio.Queue[str | None] = asyncio.Queue(maxsize=8)
        tts_task: asyncio.Task[None] | None = None
        segmenter = SpeechTextSegmenter()
        first_text_at: float | None = None
        function_call: ToolCall | None = None
        self._telemetry.llm_jobs_active.inc()
        try:
            async for event in self._gemma.stream_response(
                messages=self.state.conversation.messages(),
                instructions=context.instructions,
                language_instruction=context.language_instruction,
                system_prompt=context.system_prompt,
                tools=context.tools,
                tool_choice=context.tool_choice,
                slot_id=self.state.slot.index,
            ):
                if not self.state.cancellation.is_current(context.token):
                    self._telemetry.stale_chunks_dropped.inc()
                    raise asyncio.CancelledError
                if isinstance(event, TextUsage):
                    context.usage = event
                    continue
                if isinstance(event, ToolCall):
                    if context.text or function_call is not None:
                        raise ValueError("Gemma emitted mixed or multiple tool output")
                    function_call = event
                    self._telemetry.tool_decision_seconds.observe(
                        time.monotonic() - context.started_at
                    )
                    continue
                if tts_task is None:
                    tts_task = asyncio.create_task(self._run_tts(context, segments))
                if first_text_at is None:
                    first_text_at = time.monotonic()
                    self._telemetry.llm_ttft_seconds.observe(first_text_at - context.started_at)
                    if self.state.tool_result_ack_at is not None:
                        self._telemetry.tool_result_to_first_text_seconds.observe(
                            first_text_at - self.state.tool_result_ack_at
                        )
                context.text += event.text
                await self.state.transport.send(
                    ResponseOutputTextDeltaEvent(
                        **self._response_fields(context),
                        delta=event.text,
                    )
                )
                for segment in segmenter.feed(event.text):
                    await segments.put(segment)
            if function_call is not None:
                pending = FunctionCallConversationItem(
                    id=context.item_id,
                    call_id=function_call.call_id,
                    name=function_call.name,
                    arguments=function_call.arguments,
                    turn_id=context.turn_id,
                    turn_revision=context.turn_revision,
                    generation_id=context.generation_id,
                    response_id=context.response_id,
                )
                self.state.pending_call = pending
                self.state.pending_call_emitted_at = time.monotonic()
                self._telemetry.tool_call_generations.inc()
                logger.info(
                    "tool_call_emitted",
                    extra={
                        "session_id": self.state.session_id,
                        "turn_id": context.turn_id,
                        "generation_id": context.generation_id,
                        "response_id": context.response_id,
                        "call_id": pending.call_id,
                        "tool_name": pending.name,
                        "argument_size": len(pending.arguments),
                        "duration_seconds": round(
                            self.state.pending_call_emitted_at - context.started_at, 3
                        ),
                    },
                )
                await self.state.transport.send(
                    ResponseOutputFunctionCallDoneEvent(
                        **self._response_fields(context),
                        call_id=pending.call_id,
                        name=pending.name,
                        arguments=pending.arguments,
                    )
                )
                await self._finalize_response(
                    context,
                    status=ResponseStatus.COMPLETED,
                    reason=ResponseReason.TOOL_CALL,
                )
                return
            for segment in segmenter.flush():
                await segments.put(segment)
            if tts_task is None:
                raise ValueError("Gemma returned neither text nor a tool call")
            await segments.put(None)
            await self._send_text_done(context)
            await tts_task
            await self._send_audio_done(context)
            await self._finalize_response(
                context,
                status=ResponseStatus.COMPLETED,
                reason=ResponseReason.COMPLETED,
            )
            if context.text.strip():
                self.state.conversation.append(
                    item_id=context.item_id,
                    role="assistant",
                    content=context.text,
                )
        except asyncio.CancelledError:
            if tts_task is not None:
                tts_task.cancel()
                await asyncio.gather(tts_task, return_exceptions=True)
            if function_call is None:
                await self._send_text_done(context)
                await self._send_audio_done(context)
            await self._finalize_response(
                context,
                status=ResponseStatus.CANCELLED,
                reason=context.cancel_reason,
            )
        except Exception as exc:
            if tts_task is not None:
                tts_task.cancel()
                await asyncio.gather(tts_task, return_exceptions=True)
            logger.exception(
                "response_generation_failed",
                extra={
                    "session_id": self.state.session_id,
                    "turn_id": context.turn_id,
                    "generation_id": context.generation_id,
                    "response_id": context.response_id,
                    "duration_seconds": round(time.monotonic() - context.started_at, 3),
                },
            )
            if isinstance(exc, ToolCallValidationError):
                self._telemetry.tool_call_parse_failures.inc()
            await self.send_error(
                (exc.code if isinstance(exc, ToolCallValidationError) else ErrorCode.MODEL_FAILURE),
                str(exc) or type(exc).__name__,
            )
            if function_call is None:
                await self._send_text_done(context)
                await self._send_audio_done(context)
            await self._finalize_response(
                context,
                status=ResponseStatus.FAILED,
                reason=ResponseReason.MODEL_ERROR,
            )
        finally:
            self._telemetry.llm_jobs_active.dec()
            self._telemetry.llm_duration_seconds.observe(time.monotonic() - context.started_at)
            if self._response is context:
                self._response = None
                self._response_task = None
                self.state.current_generation_id = None
                self.state.current_response_id = None
                self._response_idle.set()

    async def _run_tts(
        self,
        context: ResponseContext,
        segments: asyncio.Queue[str | None],
    ) -> None:
        async def send_frame(frame: bytes) -> None:
            if not self.state.cancellation.is_current(context.token):
                self._telemetry.stale_chunks_dropped.inc()
                return
            if context.audio_sequence == 0 and context.speech_stopped_at is not None:
                self._telemetry.first_audio_latency_seconds.observe(
                    time.monotonic() - context.speech_stopped_at
                )
            if context.audio_sequence == 0 and self.state.tool_result_ack_at is not None:
                self._telemetry.tool_result_to_first_audio_seconds.observe(
                    time.monotonic() - self.state.tool_result_ack_at
                )
            await self.state.transport.send(
                ResponseOutputAudioDeltaEvent(
                    **self._response_fields(context),
                    sequence=context.audio_sequence,
                    audio=base64.b64encode(frame).decode("ascii"),
                )
            )
            context.audio_sequence += 1

        while True:
            segment = await segments.get()
            if segment is None:
                return
            await self._tts.synthesize(
                TTSJob(
                    token=context.token,
                    text=segment,
                    language=context.model_language,
                    instructions=context.voice_instructions,
                    is_current=lambda: self.state.cancellation.is_current(context.token),
                    on_frame=send_frame,
                )
            )

    async def _cancel_response(
        self,
        *,
        reason: ResponseReason,
        generation_id: str | None = None,
        response_id: str | None = None,
    ) -> None:
        context = self._response
        if context is None:
            return
        if response_id is not None and response_id != context.response_id:
            raise ValueError("response.cancel response_id does not match the active response")
        cancelled = self.state.cancellation.cancel(generation_id)
        if cancelled is None:
            return
        context.cancel_reason = reason
        logger.info(
            "response_generation_cancelled",
            extra={
                "session_id": self.state.session_id,
                "turn_id": context.turn_id,
                "generation_id": context.generation_id,
                "response_id": context.response_id,
                "reason": reason,
                "duration_seconds": round(time.monotonic() - context.started_at, 3),
            },
        )
        self._telemetry.turns_cancelled.inc()
        dropped = await self.state.transport.cancel_generation(context.generation_id)
        self._telemetry.stale_chunks_dropped.inc(dropped)
        await self._tts.cancel_generation(context.token)
        task = self._response_task
        if task is not None and task is not asyncio.current_task():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if not context.finalized:
            await self._send_text_done(context)
            await self._send_audio_done(context)
            await self._finalize_response(
                context,
                status=ResponseStatus.CANCELLED,
                reason=reason,
            )

    async def _send_text_done(self, context: ResponseContext) -> None:
        if context.text_done:
            return
        context.text_done = True
        await self.state.transport.send(
            ResponseOutputTextDoneEvent(
                **self._response_fields(context),
                text=context.text,
            )
        )

    async def _send_audio_done(self, context: ResponseContext) -> None:
        if context.audio_done:
            return
        context.audio_done = True
        await self.state.transport.send(
            ResponseOutputAudioDoneEvent(
                **self._response_fields(context),
            )
        )

    async def _finalize_response(
        self,
        context: ResponseContext,
        *,
        status: ResponseStatus,
        reason: ResponseReason,
    ) -> None:
        if context.finalized:
            return
        context.finalized = True
        usage = context.usage or TextUsage(0, 0, 0)
        await self.state.transport.send(
            ResponseDoneEvent(
                **self._response_fields(context),
                status=status,
                reason=reason,
                usage=Usage(
                    input_text_tokens=usage.prompt_tokens,
                    output_text_tokens=usage.completion_tokens,
                    total_text_tokens=usage.total_tokens,
                ),
            )
        )
        self.state.cancellation.finish(context.token)
        if reason is not ResponseReason.TOOL_CALL and self.state.tool_result_ack_at is not None:
            self.state.tool_result_ack_at = None
            self.state.pending_call_emitted_at = None
        if usage.completion_tokens > 0:
            duration = max(time.monotonic() - context.started_at, 1e-9)
            self._telemetry.llm_tokens_per_second.observe(usage.completion_tokens / duration)

    def _response_fields(self, context: ResponseContext) -> dict[str, Any]:
        return {
            "event_id": _id("evt"),
            "session_id": self.state.session_id,
            "turn_id": context.turn_id,
            "turn_revision": context.turn_revision,
            "generation_id": context.generation_id,
            "response_id": context.response_id,
            "item_id": context.item_id,
        }

    @staticmethod
    def _user_item_id(turn_id: str) -> str:
        return f"item_{turn_id.removeprefix('turn_')}"
