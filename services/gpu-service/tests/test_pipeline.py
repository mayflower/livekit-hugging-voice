from __future__ import annotations

import asyncio
import base64
from collections.abc import AsyncIterator, Sequence
from typing import Any

import pytest
from hugging_voice_protocol.errors import ErrorCode
from hugging_voice_protocol.events import (
    ConversationItemCreateEvent,
    FunctionCallConversationItem,
    FunctionCallOutputConversationItem,
    FunctionDefinition,
    FunctionTool,
    InputAudioBufferAppendEvent,
    InputAudioBufferCommitEvent,
    InputTranscriptionCompletedEvent,
    ResponseCancelEvent,
    ResponseCreateEvent,
    ResponseDoneEvent,
    ResponseOutputAudioDeltaEvent,
    ResponseOutputAudioDoneEvent,
    ResponseOutputFunctionCallDoneEvent,
    ResponseOutputTextDeltaEvent,
    ResponseOutputTextDoneEvent,
    ResponseSpeakEvent,
    ServerEvent,
    ServerVADConfig,
    SessionConfig,
    SessionUpdateEvent,
)
from hugging_voice_service.cancellation import GenerationToken
from hugging_voice_service.capacity import SessionSlot
from hugging_voice_service.config import SpeechSettings, TranscriptionSettings
from hugging_voice_service.pipeline import PipelineEventError, VoicePipeline
from hugging_voice_service.runtimes.llama_cpp_chat import (
    GemmaMessage,
    TextDelta,
    TextUsage,
    ToolCall,
)
from hugging_voice_service.runtimes.silero import SessionVAD
from hugging_voice_service.schedulers.stt import STTJob
from hugging_voice_service.schedulers.tts import TTSJob
from hugging_voice_service.sessions import SessionState
from hugging_voice_service.telemetry import ServiceTelemetry


class Probability:
    def __init__(self, value: float) -> None:
        self._value = value

    def item(self) -> float:
        return self._value


class VADModel:
    def __init__(self, probability: float = 0.0) -> None:
        self._probability = probability

    def __call__(self, samples: Any, sample_rate: int) -> Probability:
        del samples, sample_rate
        return Probability(self._probability)

    def reset_states(self) -> None:
        return None


class RecordingTransport:
    def __init__(self) -> None:
        self.events: list[ServerEvent] = []

    async def send(self, event: ServerEvent) -> None:
        self.events.append(event)

    async def cancel_generation(self, generation_id: str) -> int:
        del generation_id
        return 0

    async def close(self, *, code: int, reason: str) -> None:
        del code, reason


class UnusedSTT:
    async def submit_final(self, job: STTJob) -> str:
        del job
        return ""

    async def submit_partial(self, job: STTJob) -> str | None:
        del job
        return None

    async def cancel_session(self, session_id: str) -> None:
        del session_id

    async def wait_session_idle(self, session_id: str) -> None:
        del session_id


class SequencedBlockingSTT(UnusedSTT):
    def __init__(self) -> None:
        self.jobs: list[STTJob] = []
        self.first_started = asyncio.Event()
        self.release_first = asyncio.Event()

    async def submit_final(self, job: STTJob) -> str:
        self.jobs.append(job)
        if len(self.jobs) == 1:
            self.first_started.set()
            await self.release_first.wait()
            return "Erster Turn"
        return "Zweiter Turn"


class RecordingPartialSTT(UnusedSTT):
    def __init__(self) -> None:
        self.jobs: list[STTJob] = []

    async def submit_partial(self, job: STTJob) -> str | None:
        self.jobs.append(job)
        return None


class ImmediateTTS:
    async def synthesize(self, job: TTSJob) -> None:
        if job.is_current():
            await job.on_frame(bytes(960))

    async def cancel_generation(self, token: GenerationToken) -> None:
        del token

    async def wait_session_idle(self, session_id: str) -> None:
        del session_id


class RecordingTTS(ImmediateTTS):
    def __init__(self) -> None:
        self.jobs: list[TTSJob] = []

    async def synthesize(self, job: TTSJob) -> None:
        self.jobs.append(job)
        await super().synthesize(job)


class GateTTS(RecordingTTS):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def synthesize(self, job: TTSJob) -> None:
        self.jobs.append(job)
        self.started.set()
        await self.release.wait()
        await ImmediateTTS.synthesize(self, job)


class BlockingTTS(ImmediateTTS):
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def synthesize(self, job: TTSJob) -> None:
        del job
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


class TwoRoundGemma:
    def __init__(self) -> None:
        self.first_started = asyncio.Event()
        self.calls = 0

    async def stream_response(
        self,
        *,
        messages: Sequence[GemmaMessage],
        instructions: str = "",
        language_instruction: str = "",
        system_prompt: str = "",
        tools: object = (),
        tool_choice: object = "auto",
        slot_id: int = 0,
        max_tokens: int | None = None,
    ) -> AsyncIterator[TextDelta | TextUsage]:
        del (
            messages,
            instructions,
            language_instruction,
            system_prompt,
            tools,
            tool_choice,
            slot_id,
            max_tokens,
        )
        self.calls += 1
        if self.calls == 1:
            self.first_started.set()
            await asyncio.Event().wait()
        yield TextDelta("Neue Antwort. ")
        yield TextUsage(prompt_tokens=4, completion_tokens=3, total_tokens=7)


class ImmediateGemma:
    async def stream_response(
        self,
        *,
        messages: Sequence[GemmaMessage],
        instructions: str = "",
        language_instruction: str = "",
        system_prompt: str = "",
        tools: object = (),
        tool_choice: object = "auto",
        slot_id: int = 0,
        max_tokens: int | None = None,
    ) -> AsyncIterator[TextDelta | TextUsage]:
        del (
            messages,
            instructions,
            language_instruction,
            system_prompt,
            tools,
            tool_choice,
            slot_id,
            max_tokens,
        )
        yield TextDelta("Hallo. ")
        yield TextUsage(prompt_tokens=2, completion_tokens=1, total_tokens=3)


class ForbiddenGemma:
    def stream_response(self, **kwargs: object) -> AsyncIterator[TextDelta]:
        del kwargs
        raise AssertionError("direct response.speak must not call the LLM")


class ToolGemma:
    async def stream_response(
        self,
        *,
        messages: Sequence[GemmaMessage],
        instructions: str = "",
        language_instruction: str = "",
        system_prompt: str = "",
        tools: object = (),
        tool_choice: object = "auto",
        slot_id: int = 0,
        max_tokens: int | None = None,
    ) -> AsyncIterator[ToolCall | TextUsage]:
        del (
            messages,
            instructions,
            language_instruction,
            system_prompt,
            tools,
            tool_choice,
            max_tokens,
        )
        assert slot_id == 0
        yield ToolCall(call_id="call_add", name="add_numbers", arguments='{"a":19,"b":23}')
        yield TextUsage(prompt_tokens=4, completion_tokens=8, total_tokens=12)


class ToolThenTextGemma:
    def __init__(self) -> None:
        self.calls = 0

    async def stream_response(
        self,
        *,
        messages: Sequence[GemmaMessage],
        instructions: str = "",
        language_instruction: str = "",
        system_prompt: str = "",
        tools: object = (),
        tool_choice: object = "auto",
        slot_id: int = 0,
        max_tokens: int | None = None,
    ) -> AsyncIterator[ToolCall | TextDelta | TextUsage]:
        del messages, instructions, language_instruction, system_prompt, tools, slot_id, max_tokens
        self.calls += 1
        if self.calls == 1:
            yield ToolCall(
                call_id="call_add",
                name="add_numbers",
                arguments='{"a":19,"b":23}',
            )
            yield TextUsage(prompt_tokens=4, completion_tokens=8, total_tokens=12)
            return
        assert tool_choice == "none"
        yield TextDelta("Das Ergebnis ist 42.")
        yield TextUsage(prompt_tokens=12, completion_tokens=5, total_tokens=17)


class BlockingPrefillGemma(ImmediateGemma):
    def __init__(self) -> None:
        self.prefill_calls: list[dict[str, Any]] = []
        self.prefill_started = asyncio.Event()
        self.release_prefill = asyncio.Event()
        self.stream_started = asyncio.Event()

    async def prefill(self, **kwargs: Any) -> int:
        self.prefill_calls.append(kwargs)
        self.prefill_started.set()
        await self.release_prefill.wait()
        return 17

    async def stream_response(
        self,
        *,
        messages: Sequence[GemmaMessage],
        instructions: str = "",
        language_instruction: str = "",
        system_prompt: str = "",
        tools: object = (),
        tool_choice: object = "auto",
        slot_id: int = 0,
        max_tokens: int | None = None,
    ) -> AsyncIterator[TextDelta | TextUsage]:
        self.stream_started.set()
        async for event in super().stream_response(
            messages=messages,
            instructions=instructions,
            language_instruction=language_instruction,
            system_prompt=system_prompt,
            tools=tools,
            tool_choice=tool_choice,
            slot_id=slot_id,
            max_tokens=max_tokens,
        ):
            yield event


class TokenRecordingGemma(ImmediateGemma):
    def __init__(self) -> None:
        self.max_tokens: list[int | None] = []

    async def stream_response(
        self,
        *,
        messages: Sequence[GemmaMessage],
        instructions: str = "",
        language_instruction: str = "",
        system_prompt: str = "",
        tools: object = (),
        tool_choice: object = "auto",
        slot_id: int = 0,
        max_tokens: int | None = None,
    ) -> AsyncIterator[TextDelta | TextUsage]:
        self.max_tokens.append(max_tokens)
        async for event in super().stream_response(
            messages=messages,
            instructions=instructions,
            language_instruction=language_instruction,
            system_prompt=system_prompt,
            tools=tools,
            tool_choice=tool_choice,
            slot_id=slot_id,
            max_tokens=max_tokens,
        ):
            yield event


def make_state(*, probability: float = 0.0) -> tuple[SessionState, RecordingTransport]:
    transport = RecordingTransport()
    state = SessionState(
        session_id="session_pipeline",
        slot=SessionSlot(index=0),
        transport=transport,
        vad=SessionVAD(
            min_speech_ms=96,
            model_factory=lambda: VADModel(probability),
            sample_tensor_factory=lambda samples: samples,
        ),
    )
    state.conversation.append(item_id="item_user", role="user", content="Hallo")
    return state, transport


async def wait_for_response_end(pipeline: VoicePipeline) -> None:
    await asyncio.wait_for(pipeline.wait_response_idle(), timeout=1.0)


@pytest.mark.asyncio
async def test_instruction_update_preserves_buffered_audio_and_vad_state() -> None:
    state, transport = make_state()
    pipeline = VoicePipeline(
        state,
        stt=UnusedSTT(),
        tts=ImmediateTTS(),
        gemma=ImmediateGemma(),
        speech=SpeechSettings(),
        telemetry=ServiceTelemetry(),
    )
    buffered_audio = bytes(1_280)
    state.input_audio_buffer.append(buffered_audio)
    state.vad.process_pcm16(bytes(256 * 2))
    assert state.vad.buffered_bytes == 256 * 2

    await pipeline.handle_event(
        SessionUpdateEvent(
            event_id="evt_update",
            session_id=state.session_id,
            session=SessionConfig(
                instructions="Neue Anweisung",
                turn_detection=ServerVADConfig(min_speech_ms=96),
            ),
        )
    )

    assert state.instructions == "Neue Anweisung"
    assert state.input_audio_buffer.size_bytes == len(buffered_audio)
    assert state.vad.buffered_bytes == 256 * 2
    assert transport.events[-1].type == "session.updated"


@pytest.mark.asyncio
async def test_partial_stt_is_disabled_by_default_and_uses_a_bounded_sliding_window() -> None:
    state, _ = make_state()
    stt = RecordingPartialSTT()
    pipeline = VoicePipeline(
        state,
        stt=stt,
        tts=ImmediateTTS(),
        gemma=ImmediateGemma(),
        speech=SpeechSettings(),
        telemetry=ServiceTelemetry(),
    )
    state.current_turn_id = "turn_partial"
    state.current_turn_revision = 0
    state.speech_start_sample = 0
    state.input_audio_buffer.append(bytes(10 * 16_000 * 2))
    pipeline._schedule_partial()
    assert pipeline._partial_task is None

    enabled = VoicePipeline(
        state,
        stt=stt,
        tts=ImmediateTTS(),
        gemma=ImmediateGemma(),
        speech=SpeechSettings(),
        transcription=TranscriptionSettings(
            partial_enabled=True,
            partial_interval_ms=1_000,
            partial_max_audio_ms=4_000,
        ),
        telemetry=ServiceTelemetry(),
    )
    enabled._schedule_partial()
    assert enabled._partial_task is not None
    await enabled._partial_task
    assert len(stt.jobs) == 1
    assert len(stt.jobs[0].audio) == 4_000 * 16 * 2
    for _ in range(5):
        state.input_audio_buffer.append(bytes(16_000 * 2))
        state.last_partial_at = 0
        enabled._schedule_partial()
        assert enabled._partial_task is not None
        await enabled._partial_task
    assert all(len(job.audio) <= 4_000 * 16 * 2 for job in stt.jobs)
    assert sum(len(job.audio) for job in stt.jobs) <= len(stt.jobs) * 4_000 * 16 * 2


@pytest.mark.asyncio
async def test_prefix_prefill_is_deduplicated_and_first_response_waits_for_it() -> None:
    state, _ = make_state()
    gemma = BlockingPrefillGemma()
    pipeline = VoicePipeline(
        state,
        stt=UnusedSTT(),
        tts=ImmediateTTS(),
        gemma=gemma,
        speech=SpeechSettings(),
        telemetry=ServiceTelemetry(),
    )
    update = SessionUpdateEvent(
        event_id="evt_update",
        session_id=state.session_id,
        session=SessionConfig(
            instructions="Be concise.",
            turn_detection=ServerVADConfig(min_speech_ms=96),
        ),
    )
    await pipeline.handle_event(update)
    await asyncio.wait_for(gemma.prefill_started.wait(), timeout=1.0)
    await pipeline.handle_event(update.model_copy(update={"event_id": "evt_update_again"}))
    assert len(gemma.prefill_calls) == 1
    assert gemma.prefill_calls[0]["slot_id"] == state.slot.index

    await pipeline.handle_event(
        ResponseCreateEvent(event_id="evt_response", session_id=state.session_id)
    )
    await asyncio.sleep(0)
    assert not gemma.stream_started.is_set()
    gemma.release_prefill.set()
    await wait_for_response_end(pipeline)
    assert gemma.stream_started.is_set()

    await pipeline.handle_event(
        update.model_copy(
            update={
                "event_id": "evt_changed",
                "session": update.session.model_copy(
                    update={"instructions": "A different instruction."}
                ),
            }
        )
    )

    with pytest.raises(PipelineEventError, match="tools are immutable"):
        await pipeline.handle_event(
            update.model_copy(
                update={
                    "event_id": "evt_tools_changed",
                    "session": update.session.model_copy(
                        update={
                            "tools": (
                                FunctionTool(
                                    function=FunctionDefinition(
                                        name="lookup",
                                        description="Look up a value.",
                                        parameters={
                                            "type": "object",
                                            "properties": {},
                                            "additionalProperties": False,
                                        },
                                    )
                                ),
                            )
                        }
                    ),
                }
            )
        )


@pytest.mark.asyncio
async def test_pipeline_applies_separate_voice_and_tool_token_limits() -> None:
    normal_state, _ = make_state()
    normal_gemma = TokenRecordingGemma()
    normal = VoicePipeline(
        normal_state,
        stt=UnusedSTT(),
        tts=ImmediateTTS(),
        gemma=normal_gemma,
        speech=SpeechSettings(),
        telemetry=ServiceTelemetry(),
    )
    await normal.handle_event(
        ResponseCreateEvent(event_id="evt_normal", session_id=normal_state.session_id)
    )
    await wait_for_response_end(normal)

    tool_state, _ = make_state()
    tool_state.tools = (
        FunctionTool(
            function=FunctionDefinition(
                name="get_time",
                description="Get the current time.",
                parameters={"type": "object", "properties": {}},
            )
        ),
    )
    tool_gemma = TokenRecordingGemma()
    tool = VoicePipeline(
        tool_state,
        stt=UnusedSTT(),
        tts=ImmediateTTS(),
        gemma=tool_gemma,
        speech=SpeechSettings(),
        telemetry=ServiceTelemetry(),
    )
    await tool.handle_event(
        ResponseCreateEvent(event_id="evt_tool", session_id=tool_state.session_id)
    )
    await wait_for_response_end(tool)

    assert normal_gemma.max_tokens == [128]
    assert tool_gemma.max_tokens == [96]


@pytest.mark.asyncio
async def test_vad_update_with_buffered_audio_is_rejected_atomically() -> None:
    state, _ = make_state()
    pipeline = VoicePipeline(
        state,
        stt=UnusedSTT(),
        tts=ImmediateTTS(),
        gemma=ImmediateGemma(),
        speech=SpeechSettings(),
        telemetry=ServiceTelemetry(),
    )
    state.instructions = "Alt"
    state.input_audio_buffer.append(bytes(1_280))

    with pytest.raises(ValueError, match="input audio is buffered"):
        await pipeline.handle_event(
            SessionUpdateEvent(
                event_id="evt_update",
                session_id=state.session_id,
                session=SessionConfig(
                    instructions="Darf nicht übernommen werden",
                    turn_detection=ServerVADConfig(
                        threshold=0.7,
                        min_speech_ms=96,
                    ),
                ),
            )
        )

    assert state.instructions == "Alt"
    assert state.vad.configuration[0] == 0.6
    assert state.input_audio_buffer.size_bytes == 1_280


@pytest.mark.asyncio
async def test_final_stt_does_not_block_audio_ingestion_for_the_next_turn() -> None:
    state, transport = make_state()
    state.vad_enabled = False
    stt = SequencedBlockingSTT()
    pipeline = VoicePipeline(
        state,
        stt=stt,
        tts=ImmediateTTS(),
        gemma=ImmediateGemma(),
        speech=SpeechSettings(),
        telemetry=ServiceTelemetry(),
    )

    for sequence in range(2):
        await pipeline.handle_event(
            InputAudioBufferAppendEvent(
                event_id=f"evt_audio_{sequence}",
                session_id=state.session_id,
                sequence=sequence,
                audio=base64.b64encode(bytes(1_280)).decode("ascii"),
            )
        )
        await asyncio.wait_for(
            pipeline.handle_event(
                InputAudioBufferCommitEvent(
                    event_id=f"evt_commit_{sequence}",
                    session_id=state.session_id,
                )
            ),
            timeout=0.1,
        )
        if sequence == 0:
            await asyncio.wait_for(stt.first_started.wait(), timeout=1.0)

    assert len(stt.jobs) == 1
    assert pipeline._completed_turns.qsize() == 1
    stt.release_first.set()
    await asyncio.wait_for(pipeline.wait_turns_idle(), timeout=1.0)
    await wait_for_response_end(pipeline)

    completed = [
        event for event in transport.events if isinstance(event, InputTranscriptionCompletedEvent)
    ]
    assert [event.transcript for event in completed] == ["Erster Turn", "Zweiter Turn"]
    assert [entry.content for entry in state.conversation.entries if entry.role == "user"][-1] == (
        "Zweiter Turn"
    )


@pytest.mark.asyncio
async def test_drain_cancels_blocked_final_stt_worker_and_clears_pending_turns() -> None:
    state, _ = make_state()
    state.vad_enabled = False
    stt = SequencedBlockingSTT()
    pipeline = VoicePipeline(
        state,
        stt=stt,
        tts=ImmediateTTS(),
        gemma=ImmediateGemma(),
        speech=SpeechSettings(),
        telemetry=ServiceTelemetry(),
    )

    for sequence in range(2):
        await pipeline.handle_event(
            InputAudioBufferAppendEvent(
                event_id=f"evt_audio_{sequence}",
                session_id=state.session_id,
                sequence=sequence,
                audio=base64.b64encode(bytes(1_280)).decode("ascii"),
            )
        )
        await pipeline.handle_event(
            InputAudioBufferCommitEvent(
                event_id=f"evt_commit_{sequence}",
                session_id=state.session_id,
            )
        )
        if sequence == 0:
            await asyncio.wait_for(stt.first_started.wait(), timeout=1.0)

    await asyncio.wait_for(pipeline.drain(), timeout=1.0)

    assert pipeline._turn_worker_task is None
    assert pipeline._completed_turns.empty()
    await asyncio.wait_for(pipeline.wait_turns_idle(), timeout=0.1)


@pytest.mark.asyncio
async def test_cancel_before_first_token_then_new_generation_completes_cleanly() -> None:
    state, transport = make_state()
    gemma = TwoRoundGemma()
    pipeline = VoicePipeline(
        state,
        stt=UnusedSTT(),
        tts=ImmediateTTS(),
        gemma=gemma,
        speech=SpeechSettings(),
        telemetry=ServiceTelemetry(),
    )
    await pipeline.handle_event(
        ResponseCreateEvent(
            event_id="evt_create_1",
            session_id=state.session_id,
        )
    )
    await asyncio.wait_for(gemma.first_started.wait(), timeout=1.0)
    first = pipeline.active_response
    assert first is not None
    await pipeline.handle_event(
        ResponseCancelEvent(
            event_id="evt_cancel_1",
            session_id=state.session_id,
            response_id=first.response_id,
            generation_id=first.generation_id,
        )
    )

    await pipeline.handle_event(
        ResponseCreateEvent(
            event_id="evt_create_2",
            session_id=state.session_id,
        )
    )
    await wait_for_response_end(pipeline)

    done = [event for event in transport.events if isinstance(event, ResponseDoneEvent)]
    assert [event.reason.value for event in done] == ["client_cancelled", "completed"]
    assert done[0].generation_id != done[1].generation_id
    text = [
        event.delta for event in transport.events if isinstance(event, ResponseOutputTextDeltaEvent)
    ]
    assert text == ["Neue Antwort. "]
    assert sum(isinstance(event, ResponseOutputAudioDeltaEvent) for event in transport.events) == 1


@pytest.mark.asyncio
async def test_cancel_while_tts_is_blocked_emits_one_terminal_lifecycle() -> None:
    state, transport = make_state()
    tts = BlockingTTS()
    pipeline = VoicePipeline(
        state,
        stt=UnusedSTT(),
        tts=tts,
        gemma=ImmediateGemma(),
        speech=SpeechSettings(),
        telemetry=ServiceTelemetry(),
    )
    await pipeline.handle_event(
        ResponseCreateEvent(event_id="evt_create", session_id=state.session_id)
    )
    await asyncio.wait_for(tts.started.wait(), timeout=1.0)
    active = pipeline.active_response
    assert active is not None
    await pipeline.handle_event(
        ResponseCancelEvent(
            event_id="evt_cancel",
            session_id=state.session_id,
            response_id=active.response_id,
            generation_id=active.generation_id,
        )
    )

    assert tts.cancelled.is_set()
    done = [event for event in transport.events if isinstance(event, ResponseDoneEvent)]
    assert len(done) == 1
    assert done[0].reason.value == "client_cancelled"
    assert not any(isinstance(event, ResponseOutputAudioDeltaEvent) for event in transport.events)


@pytest.mark.asyncio
async def test_vad_barge_in_preserves_the_barge_in_terminal_reason() -> None:
    state, transport = make_state(probability=1.0)
    gemma = TwoRoundGemma()
    pipeline = VoicePipeline(
        state,
        stt=UnusedSTT(),
        tts=ImmediateTTS(),
        gemma=gemma,
        speech=SpeechSettings(),
        telemetry=ServiceTelemetry(),
    )
    await pipeline.handle_event(
        ResponseCreateEvent(event_id="evt_create", session_id=state.session_id)
    )
    await asyncio.wait_for(gemma.first_started.wait(), timeout=1.0)

    await pipeline.handle_event(
        InputAudioBufferAppendEvent(
            event_id="evt_audio",
            session_id=state.session_id,
            sequence=0,
            audio=base64.b64encode(bytes(512 * 3 * 2)).decode("ascii"),
        )
    )

    done = [event for event in transport.events if isinstance(event, ResponseDoneEvent)]
    assert len(done) == 1
    assert done[0].reason.value == "barge_in"


@pytest.mark.asyncio
async def test_tool_generation_is_silent_and_never_starts_tts() -> None:
    state, transport = make_state()
    tts = BlockingTTS()
    pipeline = VoicePipeline(
        state,
        stt=UnusedSTT(),
        tts=tts,
        gemma=ToolGemma(),
        speech=SpeechSettings(),
        telemetry=ServiceTelemetry(),
    )
    await pipeline.handle_event(
        ResponseCreateEvent(event_id="evt_tool", session_id=state.session_id)
    )
    await wait_for_response_end(pipeline)
    calls = [
        event
        for event in transport.events
        if isinstance(event, ResponseOutputFunctionCallDoneEvent)
    ]
    assert len(calls) == 1 and calls[0].arguments == '{"a":19,"b":23}'
    assert not tts.started.is_set()
    assert not any(
        isinstance(
            event,
            ResponseOutputTextDeltaEvent
            | ResponseOutputTextDoneEvent
            | ResponseOutputAudioDeltaEvent
            | ResponseOutputAudioDoneEvent,
        )
        for event in transport.events
    )
    done = [event for event in transport.events if isinstance(event, ResponseDoneEvent)]
    assert len(done) == 1 and done[0].reason.value == "tool_call"


@pytest.mark.asyncio
async def test_tool_result_ack_then_final_response_runs_through_service_pipeline() -> None:
    state, transport = make_state()
    gemma = ToolThenTextGemma()
    pipeline = VoicePipeline(
        state,
        stt=UnusedSTT(),
        tts=ImmediateTTS(),
        gemma=gemma,
        speech=SpeechSettings(),
        telemetry=ServiceTelemetry(),
    )
    await pipeline.handle_event(
        ResponseCreateEvent(event_id="evt_tool", session_id=state.session_id)
    )
    await wait_for_response_end(pipeline)
    pending = state.pending_call
    assert pending is not None

    await pipeline.handle_event(
        ConversationItemCreateEvent(
            event_id="evt_tool_output",
            session_id=state.session_id,
            item=FunctionCallOutputConversationItem(
                id="item_tool_output",
                call_id=pending.call_id,
                name=pending.name,
                output="42",
                is_error=False,
                turn_id=pending.turn_id,
                turn_revision=pending.turn_revision,
                generation_id=pending.generation_id,
                response_id=pending.response_id,
            ),
        )
    )
    assert state.pending_call is None
    assert transport.events[-1].type == "conversation.item.created"

    await pipeline.handle_event(
        ResponseCreateEvent(
            event_id="evt_final",
            session_id=state.session_id,
            tool_choice="none",
        )
    )
    await wait_for_response_end(pipeline)

    done = [event for event in transport.events if isinstance(event, ResponseDoneEvent)]
    assert [event.reason.value for event in done] == ["tool_call", "completed"]
    assert any(
        isinstance(event, ResponseOutputTextDeltaEvent) and "42" in event.delta
        for event in transport.events
    )
    assert any(isinstance(event, ResponseOutputAudioDeltaEvent) for event in transport.events)


@pytest.mark.asyncio
async def test_function_call_replay_is_rejected_after_bootstrap() -> None:
    state, _ = make_state()
    state.context_replay_open = False
    pipeline = VoicePipeline(
        state,
        stt=UnusedSTT(),
        tts=ImmediateTTS(),
        gemma=ImmediateGemma(),
        speech=SpeechSettings(),
        telemetry=ServiceTelemetry(),
    )
    event = ConversationItemCreateEvent(
        event_id="evt_injected_call",
        session_id=state.session_id,
        item=FunctionCallConversationItem(
            id="item_injected_call",
            call_id="call_injected",
            name="add_numbers",
            arguments='{"a":19,"b":23}',
            turn_id="turn_injected",
            turn_revision=0,
            generation_id="gen_injected",
            response_id="resp_injected",
        ),
    )
    with pytest.raises(PipelineEventError) as caught:
        await pipeline.handle_event(event)
    assert caught.value.code is ErrorCode.TOOL_CALL_STATE_CONFLICT


@pytest.mark.asyncio
async def test_response_speak_emits_text_audio_without_calling_llm() -> None:
    state, transport = make_state()
    tts = RecordingTTS()
    pipeline = VoicePipeline(
        state,
        stt=UnusedSTT(),
        tts=tts,
        gemma=ForbiddenGemma(),
        speech=SpeechSettings(),
        telemetry=ServiceTelemetry(),
    )

    await pipeline.handle_event(
        ResponseSpeakEvent(
            event_id="evt_speak",
            session_id=state.session_id,
            text="Ich prüfe das kurz.",
        )
    )
    await wait_for_response_end(pipeline)

    assert [job.priority for job in tts.jobs] == ["filler_or_explicit_say"]
    assert any(
        isinstance(event, ResponseOutputTextDoneEvent) and event.text == "Ich prüfe das kurz."
        for event in transport.events
    )
    assert any(isinstance(event, ResponseOutputAudioDeltaEvent) for event in transport.events)
    assert state.conversation.entries[-1].content == "Ich prüfe das kurz."


@pytest.mark.asyncio
async def test_response_speak_is_cancellable_during_tts() -> None:
    state, transport = make_state()
    tts = BlockingTTS()
    pipeline = VoicePipeline(
        state,
        stt=UnusedSTT(),
        tts=tts,
        gemma=ForbiddenGemma(),
        speech=SpeechSettings(),
        telemetry=ServiceTelemetry(),
    )
    await pipeline.handle_event(
        ResponseSpeakEvent(
            event_id="evt_speak",
            session_id=state.session_id,
            text="Ich prüfe das kurz.",
        )
    )
    await asyncio.wait_for(tts.started.wait(), timeout=1.0)
    active = pipeline.active_response
    assert active is not None
    await pipeline.handle_event(
        ResponseCancelEvent(
            event_id="evt_cancel_speak",
            session_id=state.session_id,
            response_id=active.response_id,
            generation_id=active.generation_id,
        )
    )
    assert tts.cancelled.is_set()
    done = [event for event in transport.events if isinstance(event, ResponseDoneEvent)]
    assert len(done) == 1
    assert done[0].reason.value == "client_cancelled"


@pytest.mark.asyncio
async def test_response_speak_during_pending_tool_call_preserves_atomic_tool_state() -> None:
    state, _ = make_state()
    tts = RecordingTTS()
    pipeline = VoicePipeline(
        state,
        stt=UnusedSTT(),
        tts=tts,
        gemma=ToolGemma(),
        speech=SpeechSettings(),
        telemetry=ServiceTelemetry(),
    )
    await pipeline.handle_event(
        ResponseCreateEvent(event_id="evt_tool", session_id=state.session_id)
    )
    await wait_for_response_end(pipeline)
    pending = state.pending_call
    assert pending is not None

    await pipeline.handle_event(
        ResponseSpeakEvent(
            event_id="evt_speak",
            session_id=state.session_id,
            text="Ich prüfe das kurz.",
        )
    )
    await wait_for_response_end(pipeline)

    assert state.pending_call == pending
    assert all(entry.content != "Ich prüfe das kurz." for entry in state.conversation.entries)
    assert [job.priority for job in tts.jobs] == ["filler_or_explicit_say"]


@pytest.mark.asyncio
async def test_tool_result_is_acked_during_speak_and_final_reply_is_deferred() -> None:
    state, transport = make_state()
    tts = GateTTS()
    gemma = ToolThenTextGemma()
    pipeline = VoicePipeline(
        state,
        stt=UnusedSTT(),
        tts=tts,
        gemma=gemma,
        speech=SpeechSettings(),
        telemetry=ServiceTelemetry(),
    )
    await pipeline.handle_event(
        ResponseCreateEvent(event_id="evt_tool", session_id=state.session_id)
    )
    await wait_for_response_end(pipeline)
    pending = state.pending_call
    assert pending is not None

    await pipeline.handle_event(
        ResponseSpeakEvent(
            event_id="evt_speak",
            session_id=state.session_id,
            text="Ich prüfe das kurz.",
        )
    )
    await asyncio.wait_for(tts.started.wait(), timeout=1.0)
    await pipeline.handle_event(
        ConversationItemCreateEvent(
            event_id="evt_tool_output",
            session_id=state.session_id,
            item=FunctionCallOutputConversationItem(
                id="item_tool_output",
                call_id=pending.call_id,
                name=pending.name,
                output="42",
                is_error=False,
                turn_id=pending.turn_id,
                turn_revision=pending.turn_revision,
                generation_id=pending.generation_id,
                response_id=pending.response_id,
            ),
        )
    )
    assert state.pending_call is None
    assert transport.events[-1].type == "conversation.item.created"

    await pipeline.handle_event(
        ResponseCreateEvent(
            event_id="evt_final",
            session_id=state.session_id,
            tool_choice="none",
        )
    )
    assert gemma.calls == 1
    tts.release.set()
    await wait_for_response_end(pipeline)
    assert gemma.calls == 2

    done = [event for event in transport.events if isinstance(event, ResponseDoneEvent)]
    assert [event.reason.value for event in done] == [
        "tool_call",
        "completed",
        "completed",
    ]
    assert [job.priority for job in tts.jobs] == [
        "filler_or_explicit_say",
        "final_response",
    ]
