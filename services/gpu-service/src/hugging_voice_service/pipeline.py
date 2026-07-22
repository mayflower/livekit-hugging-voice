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
from hugging_voice_protocol.errors import ErrorCode
from hugging_voice_protocol.events import (
    ClientEvent,
    ConversationItemCreateEvent,
    ErrorEvent,
    ErrorPayload,
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
    ResponseOutputTextDeltaEvent,
    ResponseOutputTextDoneEvent,
    ResponseReason,
    ResponseStatus,
    SessionUpdateEvent,
    SpeechStartedEvent,
    SpeechStoppedEvent,
    Usage,
)

from .cancellation import GenerationToken
from .config import SpeechSettings
from .conversation import ConversationRole
from .runtimes.gemma import GemmaMessage, TextDelta, TextUsage
from .schedulers.stt import STTJob
from .schedulers.tts import TTSJob
from .sessions import SessionState
from .telemetry import ServiceTelemetry
from .text_segmenter import SpeechTextSegmenter

logger = logging.getLogger(__name__)


class GemmaStreamer(Protocol):
    def stream_response(
        self,
        *,
        messages: Sequence[GemmaMessage],
        instructions: str = "",
        language_instruction: str = "",
        system_prompt: str = "",
        max_tokens: int = 256,
    ) -> AsyncIterator[TextDelta | TextUsage]: ...


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
    started_at: float
    speech_stopped_at: float | None
    text: str = ""
    audio_sequence: int = 0
    usage: TextUsage | None = None
    text_done: bool = False
    audio_done: bool = False
    finalized: bool = False
    cancel_reason: ResponseReason = ResponseReason.CLIENT_CANCELLED


class VoicePipeline:
    partial_interval_seconds = 0.5

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
        self._completed_turns: set[tuple[str, int]] = set()
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

    async def handle_event(self, event: ClientEvent) -> None:
        if self._draining:
            raise RuntimeError("session is draining")
        if event.session_id != self.state.session_id:
            raise ValueError("event session_id does not match the claimed session")
        self.state.last_activity_at = time.monotonic()
        if isinstance(event, SessionUpdateEvent):
            await self._update_session(event)
        elif isinstance(event, InputAudioBufferAppendEvent):
            await self._append_audio(event)
        elif isinstance(event, InputAudioBufferCommitEvent):
            await self._commit_audio()
        elif isinstance(event, InputAudioBufferClearEvent):
            await self._clear_audio()
        elif isinstance(event, ConversationItemCreateEvent):
            self._create_conversation_item(event)
        elif isinstance(event, ResponseCreateEvent):
            await self._start_response(response_instructions=event.instructions or "")
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
        await self._cancel_response(reason=ResponseReason.TRANSPORT_ERROR)
        await self._stt.cancel_session(self.state.session_id)
        await self._stt.wait_session_idle(self.state.session_id)
        await self._tts.wait_session_idle(self.state.session_id)
        self.state.input_audio_buffer.clear()
        self.state.vad.reset()
        self.state.cancellation.reset()

    async def send_error(self, code: ErrorCode, message: str, *, retryable: bool = False) -> None:
        await self.state.transport.send(
            ErrorEvent(
                event_id=_id("evt"),
                session_id=self.state.session_id,
                error=ErrorPayload(code=code, message=message, retryable=retryable),
            )
        )

    async def _update_session(self, event: SessionUpdateEvent) -> None:
        if self.state.current_turn_id is not None and self.state.speech_start_sample is not None:
            raise ValueError("cannot change VAD settings during an active speech turn")
        config = event.session
        language = config.language or self._speech.default_language
        voice = config.voice or self._speech.default_voice
        self._speech.resolve_language(language)
        self._speech.resolve_voice(voice)
        self.state.instructions = config.instructions
        self.state.language = language
        self.state.voice = voice
        self.state.voice_instructions = config.voice_instructions
        self.state.vad_enabled = config.turn_detection.enabled
        self.state.transcription_enabled = config.input_audio_transcription
        self.state.interrupt_response = config.interrupt_response
        self.state.vad.configure(
            threshold=config.turn_detection.threshold,
            min_speech_ms=config.turn_detection.min_speech_ms,
            min_speech_continuation_ms=config.turn_detection.min_speech_continuation_ms,
            min_silence_ms=config.turn_detection.min_silence_ms,
            speech_pad_ms=config.turn_detection.speech_pad_ms,
        )
        self.state.input_audio_buffer.clear()

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
        if turn_key in self._completed_turns:
            return
        self._completed_turns.add(turn_key)
        self.state.partial_epoch += 1
        if self._partial_task is not None:
            self._partial_task.cancel()
            await asyncio.gather(self._partial_task, return_exceptions=True)
            self._partial_task = None
        audio = self.state.input_audio_buffer.slice_samples(start_sample, end_sample)

        def stale() -> bool:
            return self._draining or self.state.current_turn_revision != revision

        started = time.monotonic()
        try:
            transcript = await self._stt.submit_final(
                STTJob(
                    session_id=self.state.session_id,
                    turn_id=turn_id,
                    turn_revision=revision,
                    audio=audio,
                    final=True,
                    is_stale=stale,
                )
            )
        finally:
            self.state.input_audio_buffer.discard_before(end_sample)
        self._telemetry.transcription_delay_seconds.observe(time.monotonic() - started)
        self._telemetry.turns.inc()
        item_id = self._user_item_id(turn_id)
        if self.state.transcription_enabled:
            await self.state.transport.send(
                InputTranscriptionCompletedEvent(
                    event_id=_id("evt"),
                    session_id=self.state.session_id,
                    turn_id=turn_id,
                    turn_revision=revision,
                    item_id=item_id,
                    transcript=transcript,
                )
            )
        if transcript.strip() and not stale():
            self.state.conversation.append(item_id=item_id, role="user", content=transcript)
            await self._start_response(response_instructions="")

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

    def _create_conversation_item(self, event: ConversationItemCreateEvent) -> None:
        if self._response is not None:
            raise ValueError("cannot replay conversation while a response is active")
        role: ConversationRole = "user" if event.item.role.value == "user" else "assistant"
        self.state.conversation.append(
            item_id=event.item.id,
            role=role,
            content=event.item.content,
        )

    async def _start_response(self, *, response_instructions: str) -> None:
        if self._response is not None:
            raise ValueError("a response is already active for this session")
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
            started_at=time.monotonic(),
            speech_stopped_at=self._speech_stopped_at,
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
        tts_task = asyncio.create_task(self._run_tts(context, segments))
        segmenter = SpeechTextSegmenter()
        first_text_at: float | None = None
        self._telemetry.llm_jobs_active.inc()
        try:
            async for event in self._gemma.stream_response(
                messages=self.state.conversation.messages(),
                instructions=context.instructions,
                language_instruction=context.language_instruction,
                system_prompt=context.system_prompt,
            ):
                if not self.state.cancellation.is_current(context.token):
                    self._telemetry.stale_chunks_dropped.inc()
                    raise asyncio.CancelledError
                if isinstance(event, TextUsage):
                    context.usage = event
                    continue
                if first_text_at is None:
                    first_text_at = time.monotonic()
                    self._telemetry.llm_ttft_seconds.observe(first_text_at - context.started_at)
                context.text += event.text
                await self.state.transport.send(
                    ResponseOutputTextDeltaEvent(
                        **self._response_fields(context),
                        delta=event.text,
                    )
                )
                for segment in segmenter.feed(event.text):
                    await segments.put(segment)
            for segment in segmenter.flush():
                await segments.put(segment)
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
            tts_task.cancel()
            await asyncio.gather(tts_task, return_exceptions=True)
            await self._send_text_done(context)
            await self._send_audio_done(context)
            await self._finalize_response(
                context,
                status=ResponseStatus.CANCELLED,
                reason=context.cancel_reason,
            )
        except Exception as exc:
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
            await self.send_error(
                ErrorCode.MODEL_FAILURE,
                str(exc) or type(exc).__name__,
            )
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
