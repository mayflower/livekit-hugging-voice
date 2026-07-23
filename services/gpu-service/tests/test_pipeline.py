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
    InputAudioBufferAppendEvent,
    ResponseCancelEvent,
    ResponseCreateEvent,
    ResponseDoneEvent,
    ResponseOutputAudioDeltaEvent,
    ResponseOutputAudioDoneEvent,
    ResponseOutputFunctionCallDoneEvent,
    ResponseOutputTextDeltaEvent,
    ResponseOutputTextDoneEvent,
    ServerEvent,
)
from hugging_voice_service.cancellation import GenerationToken
from hugging_voice_service.capacity import SessionSlot
from hugging_voice_service.config import SpeechSettings
from hugging_voice_service.pipeline import PipelineEventError, VoicePipeline
from hugging_voice_service.runtimes.gemma import GemmaMessage, TextDelta, TextUsage, ToolCall
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


class ImmediateTTS:
    async def synthesize(self, job: TTSJob) -> None:
        if job.is_current():
            await job.on_frame(bytes(960))

    async def cancel_generation(self, token: GenerationToken) -> None:
        del token

    async def wait_session_idle(self, session_id: str) -> None:
        del session_id


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
    ) -> AsyncIterator[TextDelta | TextUsage]:
        del messages, instructions, language_instruction, system_prompt, tools, tool_choice, slot_id
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
    ) -> AsyncIterator[TextDelta | TextUsage]:
        del messages, instructions, language_instruction, system_prompt, tools, tool_choice, slot_id
        yield TextDelta("Hallo. ")
        yield TextUsage(prompt_tokens=2, completion_tokens=1, total_tokens=3)


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
    ) -> AsyncIterator[ToolCall | TextUsage]:
        del messages, instructions, language_instruction, system_prompt, tools, tool_choice
        assert slot_id == 0
        yield ToolCall(call_id="call_add", name="add_numbers", arguments='{"a":19,"b":23}')
        yield TextUsage(prompt_tokens=4, completion_tokens=8, total_tokens=12)


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
