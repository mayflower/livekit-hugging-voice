from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass
from typing import Any

import pytest
from aiohttp import WSMsgType, web
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
    ModelRevisions,
    ResponseCreatedEvent,
    ResponseDoneEvent,
    ResponseOutputAudioDeltaEvent,
    ResponseOutputAudioDoneEvent,
    ResponseOutputTextDeltaEvent,
    ResponseOutputTextDoneEvent,
    ResponseReason,
    ResponseStatus,
    SessionCreatedEvent,
    SessionModels,
    SessionUpdateEvent,
    SpeechStartedEvent,
    SpeechStoppedEvent,
    Usage,
    parse_client_event_json,
)
from livekit import rtc
from livekit.agents import Agent, AgentSession, APIConnectOptions
from livekit.agents.llm import ChatContext, RealtimeError
from livekit.plugins.hugging_voice.realtime import PartialTranscription, RealtimeModel

REVISION = "c" * 40


def common_response_fields(generation: int) -> dict[str, Any]:
    return {
        "event_id": f"evt_response_{generation}",
        "session_id": "session_contract",
        "turn_id": f"turn_{generation}",
        "turn_revision": generation,
        "generation_id": f"gen_{generation}",
        "response_id": f"resp_{generation}",
        "item_id": f"item_assistant_{generation}",
    }


@dataclass
class ContractServer:
    port: int
    cancel_first: bool = False
    omit_audio: bool = False
    ignore_response: bool = False
    send_transcription: bool = True
    open_audio_frames: int = 2
    pause_response: bool = False

    def __post_init__(self) -> None:
        self.events: asyncio.Queue[ClientEvent] = asyncio.Queue()
        self._runner: web.AppRunner | None = None
        self._response_count = 0
        self.cancel_count = 0
        self.continue_response = asyncio.Event()

    @property
    def url(self) -> str:
        return f"ws://127.0.0.1:{self.port}/v1/realtime"

    async def start(self) -> None:
        app = web.Application()
        app.router.add_get("/v1/realtime", self._websocket)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        await web.TCPSite(self._runner, "127.0.0.1", self.port).start()

    async def close(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()

    async def next_event(self) -> ClientEvent:
        return await asyncio.wait_for(self.events.get(), timeout=1.0)

    async def _websocket(self, request: web.Request) -> web.WebSocketResponse:
        assert request.headers["Authorization"] == "Bearer contract-secret"
        websocket = web.WebSocketResponse(protocols=("hugging-voice-livekit.v1",))
        await websocket.prepare(request)
        await websocket.send_str(
            SessionCreatedEvent(
                event_id="evt_session_created",
                session_id="session_contract",
                models=SessionModels(),
                revisions=ModelRevisions(
                    vad="6.2.1",
                    stt=REVISION,
                    llm=REVISION,
                    tts=REVISION,
                ),
            ).model_dump_json()
        )
        sent_transcription = False
        async for message in websocket:
            if message.type is not WSMsgType.TEXT:
                continue
            event = parse_client_event_json(message.data)
            await self.events.put(event)
            if (
                event.type == "session.update"
                and not sent_transcription
                and self.send_transcription
            ):
                sent_transcription = True
                await self._send_transcription(websocket)
            elif event.type == "response.create":
                self._response_count += 1
                if self.ignore_response:
                    continue
                if self.cancel_first and self._response_count == 1:
                    await self._send_open_response(websocket, self._response_count)
                else:
                    await self._send_response(websocket, self._response_count)
            elif event.type == "response.cancel":
                self.cancel_count += 1
                await self._send_cancelled_response(websocket, self._response_count)
        return websocket

    async def _send_transcription(self, websocket: web.WebSocketResponse) -> None:
        events = [
            SpeechStartedEvent(
                event_id="evt_speech_start",
                session_id="session_contract",
                turn_id="turn_input",
                turn_revision=0,
                audio_start_ms=0,
            ),
            SpeechStoppedEvent(
                event_id="evt_speech_stop",
                session_id="session_contract",
                turn_id="turn_input",
                turn_revision=0,
                audio_end_ms=600,
            ),
            InputTranscriptionDeltaEvent(
                event_id="evt_partial",
                session_id="session_contract",
                turn_id="turn_input",
                turn_revision=0,
                item_id="item_input",
                delta="Hal",
            ),
            InputTranscriptionCompletedEvent(
                event_id="evt_final",
                session_id="session_contract",
                turn_id="turn_input",
                turn_revision=0,
                item_id="item_input",
                transcript="Hallo",
            ),
        ]
        for event in events:
            await websocket.send_str(event.model_dump_json())

    async def _send_response(self, websocket: web.WebSocketResponse, generation: int) -> None:
        fields = common_response_fields(generation)
        await websocket.send_str(ResponseCreatedEvent(**fields).model_dump_json())
        if self.pause_response:
            await self.continue_response.wait()
        events = [
            ResponseOutputTextDeltaEvent(**fields, delta="Guten Tag. "),
            ResponseOutputTextDoneEvent(**fields, text="Guten Tag. "),
        ]
        if not self.omit_audio:
            events.append(
                ResponseOutputAudioDeltaEvent(
                    **fields,
                    sequence=0,
                    audio=base64.b64encode(bytes(960)).decode(),
                )
            )
        events.extend(
            [
                ResponseOutputAudioDoneEvent(**fields),
                ResponseDoneEvent(
                    **fields,
                    status=ResponseStatus.COMPLETED,
                    reason=ResponseReason.COMPLETED,
                    usage=Usage(
                        input_text_tokens=4,
                        output_text_tokens=3,
                        total_text_tokens=7,
                    ),
                ),
            ]
        )
        for event in events:
            await websocket.send_str(event.model_dump_json())

    async def _send_open_response(
        self,
        websocket: web.WebSocketResponse,
        generation: int,
    ) -> None:
        fields = common_response_fields(generation)
        await websocket.send_str(ResponseCreatedEvent(**fields).model_dump_json())
        for sequence in range(self.open_audio_frames):
            await websocket.send_str(
                ResponseOutputAudioDeltaEvent(
                    **fields,
                    sequence=sequence,
                    audio=base64.b64encode(bytes(960)).decode(),
                ).model_dump_json()
            )

    async def _send_cancelled_response(
        self,
        websocket: web.WebSocketResponse,
        generation: int,
    ) -> None:
        fields = common_response_fields(generation)
        events = [
            ResponseOutputAudioDeltaEvent(
                **fields,
                sequence=99,
                audio=base64.b64encode(bytes(960)).decode(),
            ),
            ResponseOutputAudioDoneEvent(**fields),
            ResponseDoneEvent(
                **fields,
                status=ResponseStatus.CANCELLED,
                reason=ResponseReason.CLIENT_CANCELLED,
                usage=Usage(),
            ),
        ]
        for event in events:
            await websocket.send_str(event.model_dump_json())


@pytest.mark.asyncio
async def test_native_session_maps_transcription_text_audio_and_metrics(
    unused_tcp_port: int,
) -> None:
    server = ContractServer(unused_tcp_port)
    await server.start()
    model = RealtimeModel(
        base_url=server.url,
        token="contract-secret",
        conn_options=APIConnectOptions(max_retry=0, timeout=1.0),
    )
    session = model.session()
    partials: list[PartialTranscription] = []
    finals: list[object] = []
    metrics: list[object] = []
    speech: list[str] = []
    session.on("hugging_voice_partial_transcription", partials.append)
    session.on("input_audio_transcription_completed", finals.append)
    session.on("metrics_collected", metrics.append)
    session.on("input_speech_started", lambda event: speech.append("started"))
    session.on("input_speech_stopped", lambda event: speech.append("stopped"))
    try:
        generation = await asyncio.wait_for(session.generate_reply(), timeout=1.0)
        messages = [message async for message in generation.message_stream]
        assert len(messages) == 1
        message = messages[0]
        text = "".join([delta async for delta in message.text_stream])
        audio = [frame async for frame in message.audio_stream]
        modalities = await message.modalities

        assert text == "Guten Tag. "
        assert len(audio) == 1
        assert (audio[0].sample_rate, audio[0].num_channels, len(audio[0].data)) == (
            24_000,
            1,
            480,
        )
        assert modalities == ["text", "audio"]
        assert generation.response_id == "resp_1"
        assert speech == ["started", "stopped"]
        assert partials == [
            PartialTranscription(
                item_id="item_input",
                transcript="Hal",
                turn_id="turn_input",
                turn_revision=0,
            )
        ]
        assert len(finals) == 1
        assert session.chat_ctx.get_by_id("item_input") is not None
        assert session.chat_ctx.get_by_id("item_assistant_1") is not None
        assert any(getattr(metric, "request_id", None) == "resp_1" for metric in metrics)
    finally:
        await model.aclose()
        await server.close()


@pytest.mark.asyncio
async def test_modalities_are_available_before_response_streams_finish(
    unused_tcp_port: int,
) -> None:
    server = ContractServer(unused_tcp_port, send_transcription=False, pause_response=True)
    await server.start()
    model = RealtimeModel(base_url=server.url, token="contract-secret")
    session = model.session()
    try:
        generation = await asyncio.wait_for(session.generate_reply(), timeout=1.0)
        message = await anext(generation.message_stream.__aiter__())
        assert await asyncio.wait_for(message.modalities, timeout=0.1) == ["text", "audio"]
        server.continue_response.set()
        assert len([frame async for frame in message.audio_stream]) == 1
    finally:
        server.continue_response.set()
        await model.aclose()
        await server.close()


@pytest.mark.asyncio
async def test_audio_commit_instruction_and_append_only_context_map_to_client_events(
    unused_tcp_port: int,
) -> None:
    server = ContractServer(unused_tcp_port, send_transcription=False)
    await server.start()
    model = RealtimeModel(base_url=server.url, token="contract-secret")
    session = model.session()
    try:
        await wait_client_event(server, "session.update")
        frame = rtc.AudioFrame(
            data=bytes(48_000 * 40 // 1_000 * 2 * 2),
            sample_rate=48_000,
            num_channels=2,
            samples_per_channel=48_000 * 40 // 1_000,
        )
        session.push_audio(frame)
        session.commit_audio()
        append = await wait_client_event(server, "input_audio_buffer.append")
        commit = await wait_client_event(server, "input_audio_buffer.commit")
        assert isinstance(append, InputAudioBufferAppendEvent)
        assert isinstance(commit, InputAudioBufferCommitEvent)
        assert append.sequence == 0
        assert len(decode_pcm16_base64(append.audio)) == 1_280

        session.clear_audio()
        clear = await wait_client_event(server, "input_audio_buffer.clear")
        assert isinstance(clear, InputAudioBufferClearEvent)

        await session.update_instructions("Neue Anweisung")
        update = await wait_client_event(server, "session.update")
        assert isinstance(update, SessionUpdateEvent)
        assert update.session.instructions == "Neue Anweisung"

        context = ChatContext.empty()
        context.add_message(id="livekit-user", role="user", content="Kontext")
        await session.update_chat_ctx(context)
        replay = await wait_client_event(server, "conversation.item.create")
        assert isinstance(replay, ConversationItemCreateEvent)
        assert replay.item.content == "Kontext"

        changed = ChatContext.empty()
        changed.add_message(id="livekit-user", role="user", content="Mutiert")
        with pytest.raises(RealtimeError, match="append-only"):
            await session.update_chat_ctx(changed)
    finally:
        await model.aclose()
        await server.close()


@pytest.mark.asyncio
async def test_input_audio_queue_full_is_explicit_and_closes_session() -> None:
    model = RealtimeModel(
        base_url="ws://127.0.0.1:1/v1/realtime",
        token="contract-secret",
        conn_options=APIConnectOptions(max_retry=0, timeout=0.1),
    )
    session = model.session()
    frame = rtc.AudioFrame(
        data=bytes(640 * 2),
        sample_rate=16_000,
        num_channels=1,
        samples_per_channel=640,
    )
    try:
        for _ in range(64):
            session.push_audio(frame)
        with pytest.raises(RealtimeError, match="queue is full"):
            session.push_audio(frame)
        await asyncio.sleep(0)
    finally:
        await model.aclose()


@pytest.mark.asyncio
async def test_agent_session_starts_with_only_native_realtime_model(
    unused_tcp_port: int,
) -> None:
    server = ContractServer(unused_tcp_port, send_transcription=False)
    await server.start()
    model = RealtimeModel(base_url=server.url, token="contract-secret")
    agent_session: AgentSession[dict[str, Any]] = AgentSession(llm=model)
    try:
        await asyncio.wait_for(
            agent_session.start(
                agent=Agent(instructions="Antworte knapp auf Deutsch."),
                record=False,
            ),
            timeout=2.0,
        )
        update = await wait_client_event(server, "session.update")
        assert isinstance(update, SessionUpdateEvent)
        assert update.session.instructions == "Antworte knapp auf Deutsch."
        assert agent_session.llm is model
    finally:
        await agent_session.aclose()
        await model.aclose()
        await server.close()


@dataclass
class CapacityServer:
    port: int

    def __post_init__(self) -> None:
        self.hits = 0
        self._runner: web.AppRunner | None = None

    @property
    def url(self) -> str:
        return f"ws://127.0.0.1:{self.port}/v1/realtime"

    async def start(self) -> None:
        app = web.Application()
        app.router.add_get("/v1/realtime", self._websocket)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        await web.TCPSite(self._runner, "127.0.0.1", self.port).start()

    async def close(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()

    async def _websocket(self, request: web.Request) -> web.WebSocketResponse:
        self.hits += 1
        websocket = web.WebSocketResponse(protocols=("hugging-voice-livekit.v1",))
        await websocket.prepare(request)
        await websocket.send_str(
            ErrorEvent(
                event_id="evt_capacity",
                session_id="session_capacity",
                error=ErrorPayload(
                    code=ErrorCode.SESSION_LIMIT_REACHED,
                    message="full",
                    retryable=True,
                ),
            ).model_dump_json()
        )
        await websocket.close(code=4429)
        return websocket


@pytest.mark.asyncio
async def test_base_urls_retry_next_endpoint_after_authoritative_4429(
    unused_tcp_port_factory: Any,
) -> None:
    capacity = CapacityServer(unused_tcp_port_factory())
    contract = ContractServer(unused_tcp_port_factory())
    await capacity.start()
    await contract.start()
    model = RealtimeModel(
        base_urls=[capacity.url, contract.url],
        token="contract-secret",
        conn_options=APIConnectOptions(max_retry=0, timeout=1.0),
    )
    session = model.session()
    try:
        generation = await asyncio.wait_for(session.generate_reply(), timeout=1.0)
        assert generation.response_id == "resp_1"
        assert capacity.hits == 1
    finally:
        await model.aclose()
        await capacity.close()
        await contract.close()


@dataclass
class ReconnectServer:
    port: int

    def __post_init__(self) -> None:
        self.allow_first = asyncio.Event()
        self.allow_second = asyncio.Event()
        self.first_replay = asyncio.Event()
        self.second_replay = asyncio.Event()
        self.connections = 0
        self.replays: dict[int, list[ConversationItemCreateEvent]] = {1: [], 2: []}
        self.updates: dict[int, SessionUpdateEvent] = {}
        self.types: dict[int, list[str]] = {1: [], 2: []}
        self._runner: web.AppRunner | None = None

    @property
    def url(self) -> str:
        return f"ws://127.0.0.1:{self.port}/v1/realtime"

    async def start(self) -> None:
        app = web.Application()
        app.router.add_get("/v1/realtime", self._websocket)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        await web.TCPSite(self._runner, "127.0.0.1", self.port).start()

    async def close(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()

    async def _websocket(self, request: web.Request) -> web.WebSocketResponse:
        self.connections += 1
        connection = self.connections
        websocket = web.WebSocketResponse(protocols=("hugging-voice-livekit.v1",))
        await websocket.prepare(request)
        await (self.allow_first if connection == 1 else self.allow_second).wait()
        await websocket.send_str(
            SessionCreatedEvent(
                event_id=f"evt_session_{connection}",
                session_id=f"session_reconnect_{connection}",
                models=SessionModels(),
                revisions=ModelRevisions(
                    vad="6.2.1",
                    stt=REVISION,
                    llm=REVISION,
                    tts=REVISION,
                ),
            ).model_dump_json()
        )
        async for message in websocket:
            if message.type is not WSMsgType.TEXT:
                continue
            event = parse_client_event_json(message.data)
            self.types.setdefault(connection, []).append(event.type)
            if isinstance(event, SessionUpdateEvent):
                self.updates[connection] = event
            elif isinstance(event, ConversationItemCreateEvent):
                self.replays.setdefault(connection, []).append(event)
                if connection == 1:
                    self.first_replay.set()
                else:
                    self.second_replay.set()
            elif event.type == "response.create" and connection == 1:
                await websocket.close(code=1012)
        return websocket


@dataclass
class FlappingServer:
    port: int

    def __post_init__(self) -> None:
        self.connections = 0
        self._runner: web.AppRunner | None = None

    @property
    def url(self) -> str:
        return f"ws://127.0.0.1:{self.port}/v1/realtime"

    async def start(self) -> None:
        app = web.Application()
        app.router.add_get("/v1/realtime", self._websocket)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        await web.TCPSite(self._runner, "127.0.0.1", self.port).start()

    async def close(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()

    async def _websocket(self, request: web.Request) -> web.WebSocketResponse:
        self.connections += 1
        websocket = web.WebSocketResponse(protocols=("hugging-voice-livekit.v1",))
        await websocket.prepare(request)
        await websocket.send_str(
            SessionCreatedEvent(
                event_id=f"evt_flap_{self.connections}",
                session_id=f"session_flap_{self.connections}",
                models=SessionModels(),
                revisions=ModelRevisions(
                    vad="6.2.1",
                    stt=REVISION,
                    llm=REVISION,
                    tts=REVISION,
                ),
            ).model_dump_json()
        )
        await websocket.receive()
        await websocket.close(code=1012)
        return websocket


@pytest.mark.asyncio
async def test_post_handshake_disconnects_exhaust_the_bounded_retry_budget(
    unused_tcp_port: int,
) -> None:
    server = FlappingServer(unused_tcp_port)
    await server.start()
    model = RealtimeModel(
        base_url=server.url,
        token="contract-secret",
        conn_options=APIConnectOptions(max_retry=1, retry_interval=0.01, timeout=1.0),
    )
    session = model.session()
    try:
        await asyncio.wait_for(session._closed.wait(), timeout=1.0)
        assert server.connections == 2
    finally:
        await model.aclose()
        await server.close()


@pytest.mark.asyncio
async def test_reconnect_fails_active_request_replays_context_and_buffers_no_audio(
    unused_tcp_port: int,
) -> None:
    server = ReconnectServer(unused_tcp_port)
    await server.start()
    model = RealtimeModel(
        base_url=server.url,
        token="contract-secret",
        instructions="initial",
        conn_options=APIConnectOptions(max_retry=2, retry_interval=0.01, timeout=1.0),
    )
    session = model.session()
    context = ChatContext.empty()
    context.add_message(id="custom-user-id", role="user", content="Bestätigter Kontext")
    await session.update_chat_ctx(context)
    reconnected = asyncio.Event()
    session.on("session_reconnected", lambda event: reconnected.set())
    server.allow_first.set()
    try:
        await asyncio.wait_for(server.first_replay.wait(), timeout=1.0)
        pending = session.generate_reply()
        with pytest.raises(RealtimeError, match="disconnect"):
            await asyncio.wait_for(pending, timeout=1.0)

        frame = rtc.AudioFrame(
            data=bytes(640 * 2),
            sample_rate=16_000,
            num_channels=1,
            samples_per_channel=640,
        )
        with pytest.raises(RealtimeError, match="not buffered"):
            session.push_audio(frame)
        await session.update_instructions("changed")
        server.allow_second.set()
        await asyncio.wait_for(reconnected.wait(), timeout=1.0)
        await asyncio.wait_for(server.second_replay.wait(), timeout=1.0)

        assert server.replays[1][0].item.content == "Bestätigter Kontext"
        assert server.replays[2][0].item.id == server.replays[1][0].item.id
        assert server.updates[1].session.instructions == "initial"
        assert server.updates[2].session.instructions == "changed"
        assert "response.create" not in server.types[2]
    finally:
        await model.aclose()
        await server.close()


async def wait_client_event(server: ContractServer, event_type: str) -> ClientEvent:
    while True:
        event = await server.next_event()
        if event.type == event_type:
            return event


@pytest.mark.asyncio
async def test_interrupt_purges_local_audio_and_late_old_generation_is_ignored(
    unused_tcp_port: int,
) -> None:
    server = ContractServer(unused_tcp_port, cancel_first=True)
    await server.start()
    model = RealtimeModel(
        base_url=server.url,
        token="contract-secret",
        conn_options=APIConnectOptions(max_retry=0, timeout=1.0),
    )
    session = model.session()
    audio_events = 0
    audio_ready = asyncio.Event()

    def server_event(event: object) -> None:
        nonlocal audio_events
        if getattr(event, "type", None) == "response.output_audio.delta":
            audio_events += 1
            if audio_events == 2:
                audio_ready.set()

    session.on("hugging_voice_server_event_received", server_event)
    try:
        first = await asyncio.wait_for(session.generate_reply(), timeout=1.0)
        messages = [message async for message in first.message_stream]
        await asyncio.wait_for(audio_ready.wait(), timeout=1.0)
        session.interrupt()
        assert [frame async for frame in messages[0].audio_stream] == []
        await wait_client_event(server, "response.cancel")

        second = await asyncio.wait_for(session.generate_reply(), timeout=1.0)
        second_messages = [message async for message in second.message_stream]
        assert "".join([delta async for delta in second_messages[0].text_stream]) == "Guten Tag. "
        assert len([frame async for frame in second_messages[0].audio_stream]) == 1
        assert second.response_id == "resp_2"
    finally:
        await model.aclose()
        await server.close()


@pytest.mark.asyncio
async def test_interrupt_before_first_output_closes_generation_exactly_once(
    unused_tcp_port: int,
) -> None:
    server = ContractServer(unused_tcp_port, cancel_first=True, open_audio_frames=0)
    await server.start()
    model = RealtimeModel(base_url=server.url, token="contract-secret")
    session = model.session()
    try:
        generation = await asyncio.wait_for(session.generate_reply(), timeout=1.0)
        message = await anext(generation.message_stream.__aiter__())
        session.interrupt()
        assert [delta async for delta in message.text_stream] == []
        assert [frame async for frame in message.audio_stream] == []
        assert await message.modalities == ["text", "audio"]
        await wait_client_event(server, "response.cancel")
        session.interrupt()
        await asyncio.sleep(0)
        assert server.cancel_count == 1
    finally:
        await model.aclose()
        await server.close()


@pytest.mark.asyncio
async def test_response_without_audio_emits_error_instead_of_silent_success(
    unused_tcp_port: int,
) -> None:
    server = ContractServer(unused_tcp_port, omit_audio=True)
    await server.start()
    model = RealtimeModel(base_url=server.url, token="contract-secret")
    session = model.session()
    errors: list[object] = []
    session.on("error", errors.append)
    try:
        generation = await asyncio.wait_for(session.generate_reply(), timeout=1.0)
        message = await anext(generation.message_stream.__aiter__())
        assert "".join([delta async for delta in message.text_stream]) == "Guten Tag. "
        assert [frame async for frame in message.audio_stream] == []
        assert await message.modalities == ["text", "audio"]
        assert any("without audio" in str(getattr(error, "error", "")) for error in errors)
    finally:
        await model.aclose()
        await server.close()


@pytest.mark.asyncio
async def test_generate_reply_future_times_out_and_close_never_leaves_it_hanging(
    unused_tcp_port: int,
) -> None:
    server = ContractServer(unused_tcp_port, ignore_response=True)
    await server.start()
    model = RealtimeModel(
        base_url=server.url,
        token="contract-secret",
        conn_options=APIConnectOptions(max_retry=0, timeout=0.2),
    )
    session = model.session()
    try:
        timed_out = session.generate_reply()
        with pytest.raises(Exception, match="response.created timed out"):
            await asyncio.wait_for(timed_out, timeout=1.0)

        pending = session.generate_reply()
        await model.aclose()
        with pytest.raises(Exception, match="closed"):
            await asyncio.wait_for(pending, timeout=1.0)
    finally:
        await model.aclose()
        await server.close()
