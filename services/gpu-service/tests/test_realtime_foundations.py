from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from typing import cast

import pytest
from fastapi import WebSocket
from hugging_voice_protocol.events import ResponseCreatedEvent, ResponseOutputAudioDeltaEvent
from hugging_voice_service.auth import AuthenticationError, TokenAuthenticator
from hugging_voice_service.cancellation import GenerationCancellation
from hugging_voice_service.capacity import CapacityManager, SlotState
from hugging_voice_service.conversation import (
    Conversation,
    FunctionCallEntry,
    FunctionCallOutputEntry,
    ToolExchangeGroup,
)
from hugging_voice_service.realtime import WebSocketTransport
from hugging_voice_service.runtimes.silero import SessionVAD
from hugging_voice_service.sessions import BoundedAudioBuffer, SessionRegistry, SessionTransport
from hugging_voice_service.telemetry import ServiceTelemetry
from hugging_voice_service.text_segmenter import SpeechTextSegmenter
from starlette.websockets import WebSocketState


def test_token_authenticator_requires_a_valid_secret_and_exact_bearer_header(
    tmp_path: Path,
) -> None:
    secret = tmp_path / "token"
    secret.write_text("correct-horse\n", encoding="utf-8")
    authenticator = TokenAuthenticator.from_file(secret)

    assert authenticator.authenticate_header("Bearer correct-horse")
    assert not authenticator.authenticate_header("Bearer wrong")
    assert not authenticator.authenticate_header("bearer correct-horse")
    assert not authenticator.authenticate_header(None)
    assert not authenticator.authenticate_header(f"Bearer {'x' * 4_097}")

    with pytest.raises(AuthenticationError):
        TokenAuthenticator("")
    with pytest.raises(AuthenticationError):
        TokenAuthenticator("contains whitespace")
    with pytest.raises(AuthenticationError):
        TokenAuthenticator.from_file(tmp_path / "missing")


def test_conversation_is_completed_text_only_and_bounded() -> None:
    conversation = Conversation(max_messages=2, max_characters=12)
    conversation.append(item_id="item_1", role="user", content=" eins ")
    conversation.append(item_id="item_2", role="assistant", content="zwei")
    conversation.append(item_id="item_3", role="user", content="drei")

    assert [entry.item_id for entry in conversation.entries] == ["item_2", "item_3"]
    assert [message.content for message in conversation.messages()] == ["zwei", "drei"]
    with pytest.raises(ValueError):
        conversation.append(item_id="item_4", role="user", content="  ")


def test_generation_tokens_never_revalidate_after_cancel_or_replacement() -> None:
    cancellation = GenerationCancellation("session_a")
    first = cancellation.start(turn_id="turn_a", turn_revision=0, generation_id="gen_a")
    assert cancellation.is_current(first)
    assert cancellation.cancel("other") is None
    assert cancellation.is_current(first)
    assert cancellation.cancel("gen_a") == first
    assert not cancellation.is_current(first)

    second = cancellation.start(turn_id="turn_b", turn_revision=1, generation_id="gen_b")
    assert cancellation.is_current(second)
    assert not cancellation.finish(first)
    assert cancellation.is_current(second)
    assert cancellation.finish(second)


def test_audio_buffer_tracks_absolute_samples_and_rejects_overflow() -> None:
    audio = BoundedAudioBuffer(max_seconds=1, sample_rate=4)
    audio.append(b"\x01\x00\x02\x00")
    assert (audio.first_sample, audio.end_sample) == (0, 2)
    assert audio.slice_samples(1, 2) == b"\x02\x00"
    audio.discard_before(1)
    audio.append(b"\x03\x00\x04\x00")
    assert audio.slice_samples(1, 4) == b"\x02\x00\x03\x00\x04\x00"
    with pytest.raises(BufferError):
        audio.append(b"\x00" * 4)


def test_german_text_segmenter_preserves_abbreviations_and_bounds_segments() -> None:
    segmenter = SpeechTextSegmenter(max_characters=80)
    assert segmenter.feed("Ein vollständiger Satz.") == ["Ein vollständiger Satz."]
    assert segmenter.feed("Dr. Müller misst 3.14 Meter. Danach geht es weiter. ") == [
        "Dr. Müller misst 3.14 Meter.",
        "Danach geht es weiter.",
    ]
    long = "Wort " * 30
    segments = segmenter.feed(long) + segmenter.flush()
    assert " ".join(segments) == long.strip()
    assert all(len(segment) <= 80 for segment in segments)


def test_tool_exchange_is_committed_and_trimmed_atomically() -> None:
    conversation = Conversation(max_messages=2, max_characters=100)
    call = FunctionCallEntry(
        item_id="item_call",
        call_id="call_1",
        name="add_numbers",
        arguments='{"a":19,"b":23}',
        turn_id="turn_1",
        turn_revision=0,
        generation_id="gen_1",
        response_id="resp_1",
    )
    output = FunctionCallOutputEntry(
        item_id="item_output",
        call_id="call_1",
        name="add_numbers",
        output="42",
        is_error=False,
        turn_id="turn_1",
        turn_revision=0,
        generation_id="gen_1",
        response_id="resp_1",
    )
    conversation.commit_tool_exchange(call=call, output=output)
    assert isinstance(conversation.groups[0], ToolExchangeGroup)
    messages = conversation.messages()
    assert messages[0].tool_calls[0].call_id == messages[1].tool_call_id == "call_1"
    conversation.append(item_id="item_next", role="user", content="Weiter")
    assert len(conversation.groups) == 1
    assert conversation.entries[0].item_id == "item_next"

    conflicting = FunctionCallOutputEntry(
        item_id="item_bad",
        call_id="call_wrong",
        name="add_numbers",
        output="42",
        is_error=False,
        turn_id="turn_1",
        turn_revision=0,
        generation_id="gen_1",
        response_id="resp_1",
    )
    with pytest.raises(ValueError, match="do not match"):
        Conversation().commit_tool_exchange(call=call, output=conflicting)


@pytest.mark.asyncio
async def test_capacity_has_no_wait_queue_and_draining_or_stuck_slots_stay_occupied() -> None:
    telemetry = ServiceTelemetry()
    capacity = CapacityManager(2, telemetry=telemetry)
    first = await capacity.claim("session_1")
    second = await capacity.claim("session_2")
    assert first is not None and second is not None
    assert await capacity.claim("session_3") is None

    await capacity.begin_release(first)
    assert first.state is SlotState.DRAINING
    assert await capacity.claim("session_3") is None
    await capacity.complete_release(first, drained=False)
    assert first.state.value == SlotState.STUCK.value
    assert await capacity.claim("session_3") is None

    await capacity.begin_release(second)
    await capacity.complete_release(second, drained=True)
    replacement = await capacity.claim("session_3")
    assert replacement is second
    assert (await capacity.report()) == {
        "total": 2,
        "active": 1,
        "draining": 0,
        "stuck": 1,
        "available": 0,
    }


@pytest.mark.asyncio
async def test_service_drain_atomically_revokes_all_future_admission() -> None:
    capacity = CapacityManager(1, telemetry=ServiceTelemetry())
    await capacity.begin_service_drain()
    assert await capacity.claim("session_late") is None
    assert (await capacity.report())["available"] == 0
    await asyncio.sleep(0)


class NoopTransport:
    async def send(self, event: object) -> None:
        del event

    async def cancel_generation(self, generation_id: str) -> int:
        del generation_id
        return 0

    async def close(self, *, code: int, reason: str) -> None:
        del code, reason


class StuckPipeline:
    async def drain(self) -> None:
        await asyncio.Event().wait()


class FailingPipeline:
    async def drain(self) -> None:
        raise RuntimeError("drain failed")


class DelayedPipeline:
    async def drain(self) -> None:
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_vad_construction_failure_returns_claimed_capacity() -> None:
    capacity = CapacityManager(1, telemetry=ServiceTelemetry())
    registry = SessionRegistry(capacity, drain_timeout=1.0)

    def fail_vad_construction() -> SessionVAD:
        raise RuntimeError("VAD construction failed")

    with pytest.raises(RuntimeError, match="VAD construction failed"):
        await registry.create(
            session_id="session_vad_failure",
            transport=cast(SessionTransport, NoopTransport()),
            vad_factory=fail_vad_construction,
        )

    assert not await registry.states()
    assert (await capacity.report())["available"] == 1


@pytest.mark.asyncio
async def test_release_timeout_quarantines_slot_instead_of_reusing_it() -> None:
    capacity = CapacityManager(1, telemetry=ServiceTelemetry())
    registry = SessionRegistry(capacity, drain_timeout=0.01)
    state = await registry.create(
        session_id="session_stuck",
        transport=cast(SessionTransport, NoopTransport()),
        vad_factory=lambda: cast(SessionVAD, object()),
    )
    assert state is not None
    state.pipeline = StuckPipeline()

    assert not await registry.release(state)
    report = await capacity.report()
    assert report["stuck"] == 1
    assert report["available"] == 0
    assert await capacity.claim("session_replacement") is None


@pytest.mark.asyncio
async def test_successful_release_does_not_retain_tasks_and_remains_idempotent() -> None:
    capacity = CapacityManager(1, telemetry=ServiceTelemetry())
    registry = SessionRegistry(capacity, drain_timeout=1.0)
    state = await registry.create(
        session_id="session_released",
        transport=cast(SessionTransport, NoopTransport()),
        vad_factory=lambda: cast(SessionVAD, object()),
    )
    assert state is not None

    assert await registry.release(state)
    assert await registry.release(state)
    assert not registry._releases
    assert (await capacity.report())["available"] == 1


@pytest.mark.asyncio
async def test_cancelled_release_waiter_does_not_retain_completed_task() -> None:
    capacity = CapacityManager(1, telemetry=ServiceTelemetry())
    registry = SessionRegistry(capacity, drain_timeout=1.0)
    state = await registry.create(
        session_id="session_cancelled_waiter",
        transport=cast(SessionTransport, NoopTransport()),
        vad_factory=lambda: cast(SessionVAD, object()),
    )
    assert state is not None
    state.pipeline = DelayedPipeline()

    waiter = asyncio.create_task(registry.release(state))
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    await asyncio.sleep(0.02)

    assert not registry._releases
    assert (await capacity.report())["available"] == 1


@pytest.mark.asyncio
async def test_release_failure_quarantines_slot_and_drops_completed_task(
    caplog: pytest.LogCaptureFixture,
) -> None:
    capacity = CapacityManager(1, telemetry=ServiceTelemetry())
    registry = SessionRegistry(capacity, drain_timeout=1.0)
    state = await registry.create(
        session_id="session_failed",
        transport=cast(SessionTransport, NoopTransport()),
        vad_factory=lambda: cast(SessionVAD, object()),
    )
    assert state is not None
    state.pipeline = FailingPipeline()

    assert not await registry.release(state)
    assert not registry._releases
    report = await capacity.report()
    assert report["stuck"] == 1
    assert report["available"] == 0
    assert "session_drain_failed" in caplog.text


def test_telemetry_exports_every_required_wave_three_metric() -> None:
    rendered = ServiceTelemetry().render().decode()
    names = {
        "hugging_voice_sessions_active",
        "hugging_voice_sessions_available",
        "hugging_voice_sessions_rejected_total",
        "hugging_voice_sessions_draining",
        "hugging_voice_sessions_stuck",
        "hugging_voice_turns_total",
        "hugging_voice_turns_cancelled_total",
        "hugging_voice_stt_queue_seconds",
        "hugging_voice_stt_inference_seconds",
        "hugging_voice_transcription_delay_seconds",
        "hugging_voice_llm_ttft_seconds",
        "hugging_voice_llm_duration_seconds",
        "hugging_voice_llm_tokens_per_second",
        "hugging_voice_tts_queue_seconds",
        "hugging_voice_tts_ttfa_seconds",
        "hugging_voice_tts_duration_seconds",
        "hugging_voice_tts_audio_seconds",
        "hugging_voice_first_audio_latency_seconds",
        "hugging_voice_barge_in_stop_latency_seconds",
        "hugging_voice_stale_chunks_dropped_total",
        "hugging_voice_websocket_errors_total",
        "hugging_voice_gpu_memory_bytes",
    }
    assert all(name in rendered for name in names)


class BlockingWebSocket:
    def __init__(self) -> None:
        self.application_state = WebSocketState.CONNECTED
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.sent: list[str] = []

    async def accept(self, subprotocol: str | None = None) -> None:
        assert subprotocol == "hugging-voice-livekit.v2"

    async def send_text(self, data: str) -> None:
        if not self.sent:
            self.started.set()
            await self.release.wait()
        self.sent.append(data)

    async def close(self, code: int = 1000, reason: str | None = None) -> None:
        del code, reason
        self.application_state = WebSocketState.DISCONNECTED


def response_created(generation_id: str) -> ResponseCreatedEvent:
    return ResponseCreatedEvent(
        event_id=f"evt_{generation_id}",
        session_id="session_transport",
        turn_id="turn_transport",
        turn_revision=0,
        generation_id=generation_id,
        response_id=f"resp_{generation_id}",
        item_id=f"item_{generation_id}",
    )


def audio_delta(generation_id: str) -> ResponseOutputAudioDeltaEvent:
    return ResponseOutputAudioDeltaEvent(
        event_id=f"evt_audio_{generation_id}",
        session_id="session_transport",
        turn_id="turn_transport",
        turn_revision=0,
        generation_id=generation_id,
        response_id=f"resp_{generation_id}",
        item_id=f"item_{generation_id}",
        sequence=0,
        audio=base64.b64encode(bytes(960)).decode(),
    )


@pytest.mark.asyncio
async def test_transport_purges_queued_audio_for_only_the_cancelled_generation() -> None:
    websocket = BlockingWebSocket()
    transport = WebSocketTransport(cast(WebSocket, websocket), max_messages=8)
    await transport.start()
    await transport.send(response_created("gen_old"))
    await asyncio.wait_for(websocket.started.wait(), timeout=1.0)
    await transport.send(audio_delta("gen_old"))
    await transport.send(audio_delta("gen_new"))

    assert await transport.cancel_generation("gen_old") == 1
    websocket.release.set()
    await transport.close(code=1000, reason="done")
    sent = [json.loads(payload) for payload in websocket.sent]
    assert [event["type"] for event in sent] == [
        "response.created",
        "response.output_audio.delta",
    ]
    assert sent[-1]["generation_id"] == "gen_new"
