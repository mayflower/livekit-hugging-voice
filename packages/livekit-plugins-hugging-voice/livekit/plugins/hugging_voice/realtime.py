"""Native LiveKit Agents realtime adapter for the Hugging Voice protocol."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeVar, cast

import aiohttp
from hugging_voice_protocol.audio import decode_pcm16_base64, encode_pcm16_base64
from hugging_voice_protocol.errors import CloseCode, ErrorCode
from hugging_voice_protocol.events import (
    ClientEvent,
    ConversationItem,
    ConversationItemCreateEvent,
    ConversationRole,
    ErrorEvent,
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
    ServerEvent,
    SessionConfig,
    SessionCreatedEvent,
    SessionUpdateEvent,
    SpeechStartedEvent,
    SpeechStoppedEvent,
    parse_server_event_json,
)
from livekit import rtc
from livekit.agents import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions, utils
from livekit.agents.llm import (
    ChatContext,
    ChatMessage,
    FunctionCall,
    GenerationCreatedEvent,
    InputSpeechStartedEvent,
    InputSpeechStoppedEvent,
    InputTranscriptionCompleted,
    MessageGeneration,
    RealtimeCapabilities,
    RealtimeError,
    RealtimeModelError,
    RealtimeSessionReconnectedEvent,
    Tool,
    ToolChoice,
    ToolContext,
)
from livekit.agents.llm import (
    RealtimeModel as LiveKitRealtimeModel,
)
from livekit.agents.llm import (
    RealtimeSession as LiveKitRealtimeSession,
)
from livekit.agents.metrics import RealtimeModelMetrics
from livekit.agents.metrics.base import Metadata
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import aio
from livekit.agents.utils.aio.channel import ChanEmpty, ChanFull

from .audio import InputAudioProcessor
from .endpoint_resolver import EndpointResolver
from .options import resolve_base_urls, resolve_token

logger = logging.getLogger(__name__)

WEBSOCKET_SUBPROTOCOL = "hugging-voice-livekit.v1"
MODEL_NAME = "hugging-voice-gemma4-parakeet-qwen3-tts"
PROVIDER_NAME = "hugging-voice"
OUTBOUND_QUEUE_SIZE = 128
INPUT_AUDIO_QUEUE_SIZE = 64
TEXT_CHANNEL_SIZE = 64
AUDIO_CHANNEL_SIZE = 128
RETRY_RESET_AFTER_SECONDS = 30.0

PluginEvent = Literal[
    "hugging_voice_server_event_received",
    "hugging_voice_client_event_queued",
    "hugging_voice_partial_transcription",
]
ChannelValue = TypeVar("ChannelValue")


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


@dataclass(frozen=True, slots=True)
class ServerEventReceived:
    type: str
    event_id: str


@dataclass(frozen=True, slots=True)
class ClientEventQueued:
    type: str
    event_id: str


@dataclass(frozen=True, slots=True)
class PartialTranscription:
    item_id: str
    transcript: str
    turn_id: str
    turn_revision: int


@dataclass(frozen=True, slots=True)
class _Command:
    kind: Literal[
        "session_update",
        "conversation_item",
        "audio_append",
        "audio_commit",
        "audio_clear",
        "response_create",
        "response_cancel",
    ]
    event_id: str
    payload: object = None


@dataclass(frozen=True, slots=True)
class _AudioControl:
    kind: Literal["commit", "clear"]


_AUDIO_STOP = object()
_SEND_STOP = object()


@dataclass(slots=True)
class _ResponseState:
    response_id: str
    generation_id: str
    message_id: str
    text_channel: aio.Chan[str]
    audio_channel: aio.Chan[rtc.AudioFrame]
    modalities: asyncio.Future[list[Literal["text", "audio"]]]
    created_at: float
    user_initiated: bool
    text: str = ""
    first_audio_at: float | None = None
    saw_text: bool = False
    saw_audio: bool = False
    cancelled: bool = False
    finalized: bool = False


class _CapacityError(RealtimeError):
    pass


class _FatalProtocolError(RealtimeError):
    pass


class _FatalSessionError(RealtimeError):
    pass


class RealtimeModel(LiveKitRealtimeModel):
    """One model factory backed only by the local Hugging Voice service."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        base_urls: Sequence[str] | None = None,
        headless_dns: str | None = None,
        headless_port: int = 8765,
        headless_tls: bool = False,
        token: str | None = None,
        token_file: str | Path | None = None,
        language: Literal["de"] = "de",
        voice: Literal["de_standard_01"] = "de_standard_01",
        instructions: str = "",
        http_session: aiohttp.ClientSession | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        if language != "de":
            raise ValueError("Hugging Voice supports only language='de'")
        if voice != "de_standard_01":
            raise ValueError("Hugging Voice supports only voice='de_standard_01'")
        if len(instructions) > 8_000:
            raise ValueError("instructions exceed the 8,000 character protocol limit")
        super().__init__(
            capabilities=RealtimeCapabilities(
                message_truncation=False,
                turn_detection=True,
                user_transcription=True,
                auto_tool_reply_generation=False,
                audio_output=True,
                manual_function_calls=False,
                mutable_chat_context=False,
                mutable_instructions=True,
                mutable_tools=False,
                per_response_tool_choice=False,
                supports_say=False,
            )
        )
        if headless_dns is not None:
            if base_url is not None or base_urls is not None:
                raise ValueError("headless_dns cannot be combined with base_url/base_urls")
            self._resolver = EndpointResolver(
                headless_dns=headless_dns,
                headless_port=headless_port,
                headless_tls=headless_tls,
            )
        else:
            urls = resolve_base_urls(base_url=base_url, base_urls=base_urls)
            self._resolver = EndpointResolver(static_urls=urls)
        self._token = resolve_token(token=token, token_file=token_file)
        self._instructions = instructions
        self._conn_options = conn_options
        self._http_session = http_session
        self._owns_http_session = http_session is None
        self._sessions: set[RealtimeSession] = set()
        self._closed = False

    @property
    def model(self) -> str:
        return MODEL_NAME

    @property
    def provider(self) -> str:
        return PROVIDER_NAME

    @property
    def base_urls(self) -> tuple[str, ...]:
        return self._resolver.configured_urls

    def session(self) -> RealtimeSession:
        if self._closed:
            raise RuntimeError("RealtimeModel is closed")
        session = RealtimeSession(self)
        self._sessions.add(session)
        return session

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        await asyncio.gather(
            *(session.aclose() for session in tuple(self._sessions)),
            return_exceptions=True,
        )
        self._sessions.clear()
        if self._owns_http_session and self._http_session is not None:
            await self._http_session.close()
        self._http_session = None

    def _client(self) -> aiohttp.ClientSession:
        if self._http_session is None:
            self._http_session = aiohttp.ClientSession()
        return self._http_session

    def _discard_session(self, session: RealtimeSession) -> None:
        self._sessions.discard(session)


class RealtimeSession(LiveKitRealtimeSession[PluginEvent]):
    def __init__(self, model: RealtimeModel) -> None:
        super().__init__(model)
        self._model = model
        self._chat_ctx = ChatContext.empty()
        self._tools = ToolContext([])
        self._instructions = model._instructions
        self._outbound: asyncio.Queue[_Command | object] = asyncio.Queue(OUTBOUND_QUEUE_SIZE)
        self._audio_input: asyncio.Queue[rtc.AudioFrame | _AudioControl | object] = asyncio.Queue(
            INPUT_AUDIO_QUEUE_SIZE
        )
        self._audio_processor = InputAudioProcessor()
        self._session_id: str | None = None
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._connected = asyncio.Event()
        self._ever_connected = False
        self._closing = False
        self._closed = asyncio.Event()
        self._sequence = 0
        self._response: _ResponseState | None = None
        self._pending_generation: asyncio.Future[GenerationCreatedEvent] | None = None
        self._pending_event_id: str | None = None
        self._pending_timeout_task: asyncio.Task[None] | None = None
        self._wire_item_ids: dict[str, str] = {}
        self._connected_at = 0.0
        self._main_task = asyncio.create_task(self._run())
        self._audio_task = asyncio.create_task(self._audio_loop())
        self._fatal_task: asyncio.Task[None] | None = None
        self._cycle_connected = False

    @property
    def chat_ctx(self) -> ChatContext:
        return self._chat_ctx

    @property
    def tools(self) -> ToolContext:
        return self._tools

    async def update_instructions(self, instructions: str) -> None:
        if len(instructions) > 8_000:
            raise RealtimeError("instructions exceed the 8,000 character limit")
        self._instructions = instructions
        if self._connected.is_set():
            await self._queue_async(_Command("session_update", _id("evt")))

    async def update_chat_ctx(self, chat_ctx: ChatContext) -> None:
        current = self._validated_messages(self._chat_ctx)
        replacement = self._validated_messages(chat_ctx)
        if len(replacement) > 30:
            raise RealtimeError("Hugging Voice chat context is limited to 30 messages")
        if replacement[: len(current)] != current:
            raise RealtimeError("Hugging Voice supports only append-only chat context updates")
        additions = replacement[len(current) :]
        self._chat_ctx = chat_ctx.copy()
        if self._connected.is_set():
            for message in additions:
                await self._queue_async(self._conversation_command(message))

    async def update_tools(self, tools: list[Tool]) -> None:
        if tools:
            raise RealtimeError("Hugging Voice does not support tools")
        self._tools = ToolContext([])

    def update_options(
        self,
        *,
        tool_choice: NotGivenOr[ToolChoice | None] = NOT_GIVEN,
    ) -> None:
        if utils.is_given(tool_choice) and tool_choice not in {None, "none"}:
            raise RealtimeError("Hugging Voice does not support tool choice")

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        self._ensure_audio_may_queue()
        self._put_audio_nowait(frame)

    def push_video(self, frame: rtc.VideoFrame) -> None:
        del frame
        raise RealtimeError("Hugging Voice does not support video input")

    def generate_reply(
        self,
        *,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        tools: NotGivenOr[list[Tool]] = NOT_GIVEN,
    ) -> asyncio.Future[GenerationCreatedEvent]:
        if utils.is_given(tools) and tools:
            raise RealtimeError("Hugging Voice does not support per-response tools")
        if utils.is_given(tool_choice) and tool_choice != "none":
            raise RealtimeError("Hugging Voice does not support per-response tool choice")
        if self._pending_generation is not None or self._response is not None:
            raise RealtimeError("a response is already pending or active")
        if self._ever_connected and not self._connected.is_set():
            raise RealtimeError("cannot generate a response while disconnected")
        response_instructions = instructions if utils.is_given(instructions) else None
        if response_instructions is not None and len(response_instructions) > 8_000:
            raise RealtimeError("response instructions exceed the protocol limit")
        future = asyncio.get_running_loop().create_future()
        event_id = _id("evt")
        self._pending_generation = future
        self._pending_event_id = event_id
        self._pending_timeout_task = asyncio.create_task(
            self._expire_generation_request(future, event_id)
        )
        self._queue_sync(_Command("response_create", event_id, response_instructions))
        return future

    def commit_audio(self) -> None:
        self._ensure_audio_may_queue()
        self._put_audio_nowait(_AudioControl("commit"))

    def clear_audio(self) -> None:
        self._ensure_audio_may_queue()
        self._put_audio_nowait(_AudioControl("clear"))

    def interrupt(self) -> None:
        response = self._response
        if response is None:
            return
        self._queue_sync(
            _Command(
                "response_cancel",
                _id("evt"),
                (response.response_id, response.generation_id),
            )
        )
        response.cancelled = True
        self._emit_cancelled_metrics(response)
        self._finish_response(response, cancelled=True, add_to_chat=False)

    def truncate(
        self,
        *,
        message_id: str,
        modalities: list[Literal["text", "audio"]],
        audio_end_ms: int,
        audio_transcript: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        del message_id, modalities, audio_end_ms, audio_transcript
        raise RealtimeError("Hugging Voice does not support message truncation")

    async def aclose(self) -> None:
        if self._closing:
            await self._closed.wait()
            return
        self._closing = True
        self._connected.clear()
        self._fail_pending(RealtimeError("realtime session closed"))
        if self._response is not None:
            self._response.cancelled = True
            self._finish_response(self._response, cancelled=True, add_to_chat=False)
        self._put_stop(self._audio_input, _AUDIO_STOP)
        self._put_stop(self._outbound, _SEND_STOP)
        ws = self._ws
        if ws is not None:
            await ws.close()
        for task in (self._audio_task, self._main_task):
            task.cancel()
        await asyncio.gather(self._audio_task, self._main_task, return_exceptions=True)
        self._model._discard_session(self)
        self._closed.set()

    async def _run(self) -> None:
        retries = 0
        try:
            while not self._closing:
                try:
                    await self._connection_cycle(reconnected=self._ever_connected)
                    raise ConnectionError("Hugging Voice WebSocket closed")
                except asyncio.CancelledError:
                    raise
                except _FatalProtocolError as exc:
                    self._emit_error(exc, recoverable=False)
                    return
                except _FatalSessionError as exc:
                    self._emit_error(exc, recoverable=False)
                    return
                except Exception as exc:
                    if (
                        self._cycle_connected
                        and time.monotonic() - self._connected_at >= RETRY_RESET_AFTER_SECONDS
                    ):
                        retries = 0
                    self._disconnect_cleanup(exc)
                    recoverable = retries < self._model._conn_options.max_retry
                    self._emit_error(exc, recoverable=recoverable)
                    if not recoverable:
                        return
                    delay = self._model._conn_options._interval_for_retry(retries)
                    retries += 1
                    await asyncio.sleep(delay)
        finally:
            self._connected.clear()
            if not self._closing:
                self._fail_pending(RealtimeError("realtime connection ended"))
                if self._response is not None:
                    self._finish_response(self._response, cancelled=True, add_to_chat=False)
                self._closing = True
                if self._ws is not None:
                    await self._ws.close()
                    self._ws = None
                self._audio_task.cancel()
                await asyncio.gather(self._audio_task, return_exceptions=True)
                self._model._discard_session(self)
                self._closed.set()

    async def _connection_cycle(self, *, reconnected: bool) -> None:
        self._cycle_connected = False
        started = time.monotonic()
        capacity_errors = 0
        last_error: Exception | None = None
        endpoints = await self._model._resolver.resolve(
            self._model._client(),
            token=self._model._token,
        )
        for endpoint in endpoints:
            try:
                ws, created = await self._open_endpoint(endpoint)
            except _CapacityError as exc:
                capacity_errors += 1
                self._model._resolver.invalidate()
                last_error = exc
                continue
            except Exception as exc:
                last_error = exc
                continue
            self._ws = ws
            self._session_id = created.session_id
            self._sequence = 0
            await self._send_direct(self._session_update_event())
            for message in self._validated_messages(self._chat_ctx):
                await self._send_direct(self._command_event(self._conversation_command(message)))
            self._connected_at = time.monotonic()
            self._connected.set()
            self._cycle_connected = True
            self._report_connection_acquired(time.monotonic() - started)
            if reconnected:
                self.emit("session_reconnected", RealtimeSessionReconnectedEvent())
            self._ever_connected = True
            sender = asyncio.create_task(self._send_loop(ws))
            receiver = asyncio.create_task(self._receive_loop(ws))
            try:
                done, pending = await asyncio.wait(
                    {sender, receiver},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                for task in done:
                    task.result()
            finally:
                for task in (sender, receiver):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(sender, receiver, return_exceptions=True)
                await ws.close()
            return
        if capacity_errors == len(endpoints):
            raise _CapacityError("all Hugging Voice endpoints are at capacity")
        raise last_error or ConnectionError("no Hugging Voice endpoint was reachable")

    async def _open_endpoint(
        self, endpoint: str
    ) -> tuple[aiohttp.ClientWebSocketResponse, SessionCreatedEvent]:
        try:
            ws = await asyncio.wait_for(
                self._model._client().ws_connect(
                    endpoint,
                    headers={"Authorization": f"Bearer {self._model._token}"},
                    protocols=(WEBSOCKET_SUBPROTOCOL,),
                    max_msg_size=128 * 1024,
                    autoclose=True,
                    autoping=True,
                ),
                timeout=self._model._conn_options.timeout,
            )
        except TimeoutError as exc:
            raise ConnectionError("Hugging Voice connection timed out") from exc
        try:
            if ws.protocol != WEBSOCKET_SUBPROTOCOL:
                raise _FatalProtocolError("server did not negotiate the v1 subprotocol")
            event = await asyncio.wait_for(
                self._receive_event(ws),
                timeout=self._model._conn_options.timeout,
            )
            if isinstance(event, SessionCreatedEvent):
                return ws, event
            if isinstance(event, ErrorEvent):
                if event.error.code is ErrorCode.SESSION_LIMIT_REACHED:
                    raise _CapacityError(event.error.message)
                if event.error.code in {
                    ErrorCode.AUTHENTICATION_FAILED,
                    ErrorCode.INVALID_CONFIGURATION,
                }:
                    raise _FatalProtocolError(event.error.message)
                raise RealtimeError(event.error.message)
            raise _FatalProtocolError("first server event was not session.created")
        except Exception:
            await ws.close()
            raise

    async def _send_loop(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        while True:
            command = await self._outbound.get()
            if command is _SEND_STOP:
                return
            assert isinstance(command, _Command)
            await self._send_direct(self._command_event(command), ws=ws)

    async def _receive_loop(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        while True:
            event = await self._receive_event(ws)
            self.emit(
                "hugging_voice_server_event_received",
                ServerEventReceived(type=event.type, event_id=event.event_id),
            )
            self._handle_server_event(event)

    async def _receive_event(self, ws: aiohttp.ClientWebSocketResponse) -> ServerEvent:
        message = await ws.receive()
        if message.type is aiohttp.WSMsgType.TEXT:
            try:
                return parse_server_event_json(message.data)
            except ValueError as exc:
                raise _FatalProtocolError("server emitted an invalid v1 event") from exc
        if message.type in {
            aiohttp.WSMsgType.CLOSE,
            aiohttp.WSMsgType.CLOSED,
            aiohttp.WSMsgType.CLOSING,
        }:
            if ws.close_code == int(CloseCode.SERVICE_RESTART):
                raise ConnectionError("Hugging Voice service is restarting")
            raise ConnectionError(f"Hugging Voice WebSocket closed with code {ws.close_code}")
        if message.type is aiohttp.WSMsgType.ERROR:
            raise ConnectionError("Hugging Voice WebSocket failed") from ws.exception()
        raise _FatalProtocolError("server emitted a non-text WebSocket message")

    async def _send_direct(
        self,
        event: ClientEvent,
        *,
        ws: aiohttp.ClientWebSocketResponse | None = None,
    ) -> None:
        socket = ws or self._ws
        if socket is None:
            raise ConnectionError("Hugging Voice WebSocket is unavailable")
        await socket.send_str(event.model_dump_json())

    def _handle_server_event(self, event: ServerEvent) -> None:
        if isinstance(event, SpeechStartedEvent):
            self.emit("input_speech_started", InputSpeechStartedEvent())
        elif isinstance(event, SpeechStoppedEvent):
            self.emit(
                "input_speech_stopped",
                InputSpeechStoppedEvent(user_transcription_enabled=True),
            )
        elif isinstance(event, InputTranscriptionDeltaEvent):
            self.emit(
                "hugging_voice_partial_transcription",
                PartialTranscription(
                    item_id=event.item_id,
                    transcript=event.delta,
                    turn_id=event.turn_id,
                    turn_revision=event.turn_revision,
                ),
            )
        elif isinstance(event, InputTranscriptionCompletedEvent):
            self.emit(
                "input_audio_transcription_completed",
                InputTranscriptionCompleted(
                    item_id=event.item_id,
                    transcript=event.transcript,
                    is_final=True,
                ),
            )
            if event.transcript.strip() and self._chat_ctx.get_by_id(event.item_id) is None:
                self._chat_ctx.add_message(
                    id=event.item_id,
                    role="user",
                    content=event.transcript,
                )
        elif isinstance(event, ResponseCreatedEvent):
            self._response_created(event)
        elif isinstance(event, ResponseOutputTextDeltaEvent):
            response = self._matching_response(event.generation_id)
            if response is not None:
                response.saw_text = True
                response.text += event.delta
                self._channel_send(response.text_channel, event.delta)
        elif isinstance(event, ResponseOutputTextDoneEvent):
            response = self._matching_response(event.generation_id)
            if response is not None:
                response.text_channel.close()
        elif isinstance(event, ResponseOutputAudioDeltaEvent):
            response = self._matching_response(event.generation_id)
            if response is not None:
                payload = decode_pcm16_base64(event.audio)
                if len(payload) != 960:
                    raise _FatalProtocolError("server audio frame is not exactly 20 ms")
                response.saw_audio = True
                if response.first_audio_at is None:
                    response.first_audio_at = time.monotonic()
                self._channel_send(
                    response.audio_channel,
                    rtc.AudioFrame(
                        data=payload,
                        sample_rate=24_000,
                        num_channels=1,
                        samples_per_channel=480,
                    ),
                )
        elif isinstance(event, ResponseOutputAudioDoneEvent):
            response = self._matching_response(event.generation_id)
            if response is not None:
                response.audio_channel.close()
        elif isinstance(event, ResponseDoneEvent):
            response = self._matching_response(event.generation_id)
            if response is not None:
                missing_audio = event.status.value == "completed" and not response.saw_audio
                cancelled = event.status.value != "completed" or missing_audio
                if missing_audio:
                    self._emit_error(
                        RealtimeError("Hugging Voice response completed without audio"),
                        recoverable=False,
                    )
                self._emit_metrics(response, event)
                self._finish_response(
                    response,
                    cancelled=cancelled,
                    add_to_chat=not cancelled,
                )
        elif isinstance(event, ErrorEvent):
            self._emit_error(RealtimeError(event.error.message), recoverable=event.error.retryable)

    def _response_created(self, event: ResponseCreatedEvent) -> None:
        if self._response is not None:
            raise _FatalProtocolError("server created overlapping responses")
        loop = asyncio.get_running_loop()
        text_channel = aio.Chan[str](TEXT_CHANNEL_SIZE)
        audio_channel = aio.Chan[rtc.AudioFrame](AUDIO_CHANNEL_SIZE)
        modalities = loop.create_future()
        modalities.set_result(["text", "audio"])
        user_initiated = self._pending_generation is not None
        response = _ResponseState(
            response_id=event.response_id,
            generation_id=event.generation_id,
            message_id=event.item_id,
            text_channel=text_channel,
            audio_channel=audio_channel,
            modalities=modalities,
            created_at=time.monotonic(),
            user_initiated=user_initiated,
        )
        self._response = response
        message_channel = aio.Chan[MessageGeneration](1)
        function_channel = aio.Chan[FunctionCall](1)
        message_channel.send_nowait(
            MessageGeneration(
                message_id=event.item_id,
                text_stream=text_channel,
                audio_stream=audio_channel,
                modalities=modalities,
            )
        )
        message_channel.close()
        function_channel.close()
        generation = GenerationCreatedEvent(
            message_stream=message_channel,
            function_stream=function_channel,
            user_initiated=user_initiated,
            response_id=event.response_id,
        )
        pending = self._pending_generation
        self._pending_generation = None
        self._pending_event_id = None
        if self._pending_timeout_task is not None:
            self._pending_timeout_task.cancel()
            self._pending_timeout_task = None
        if pending is not None and not pending.done():
            pending.set_result(generation)
        self.emit("generation_created", generation)

    def _finish_response(
        self,
        response: _ResponseState,
        *,
        cancelled: bool,
        add_to_chat: bool,
    ) -> None:
        if response.finalized:
            return
        response.finalized = True
        response.cancelled = response.cancelled or cancelled
        if cancelled:
            self._drain_channel(response.audio_channel)
        response.text_channel.close()
        response.audio_channel.close()
        if add_to_chat and response.text.strip():
            existing = self._chat_ctx.get_by_id(response.message_id)
            if existing is None:
                self._chat_ctx.add_message(
                    id=response.message_id,
                    role="assistant",
                    content=response.text,
                    interrupted=False,
                )
        if self._response is response:
            self._response = None

    def _emit_metrics(self, response: _ResponseState, event: ResponseDoneEvent) -> None:
        duration = max(time.monotonic() - response.created_at, 0.0)
        output_tokens = event.usage.output_text_tokens
        self.emit(
            "metrics_collected",
            RealtimeModelMetrics(
                label=self._model.label,
                request_id=response.response_id,
                timestamp=time.time(),
                duration=duration,
                session_duration=max(time.monotonic() - self._connected_at, 0.0),
                ttft=(
                    -1
                    if response.first_audio_at is None
                    else response.first_audio_at - response.created_at
                ),
                cancelled=event.status.value != "completed",
                input_tokens=event.usage.input_text_tokens,
                output_tokens=output_tokens,
                total_tokens=event.usage.total_text_tokens,
                tokens_per_second=output_tokens / duration if duration > 0 else 0.0,
                input_token_details=RealtimeModelMetrics.InputTokenDetails(
                    text_tokens=event.usage.input_text_tokens
                ),
                output_token_details=RealtimeModelMetrics.OutputTokenDetails(
                    text_tokens=output_tokens
                ),
                metadata=Metadata(
                    model_name=self._model.model, model_provider=self._model.provider
                ),
            ),
        )

    def _emit_cancelled_metrics(self, response: _ResponseState) -> None:
        duration = max(time.monotonic() - response.created_at, 0.0)
        self.emit(
            "metrics_collected",
            RealtimeModelMetrics(
                label=self._model.label,
                request_id=response.response_id,
                timestamp=time.time(),
                duration=duration,
                session_duration=max(time.monotonic() - self._connected_at, 0.0),
                ttft=(
                    -1
                    if response.first_audio_at is None
                    else response.first_audio_at - response.created_at
                ),
                cancelled=True,
                input_token_details=RealtimeModelMetrics.InputTokenDetails(),
                output_token_details=RealtimeModelMetrics.OutputTokenDetails(),
                metadata=Metadata(
                    model_name=self._model.model, model_provider=self._model.provider
                ),
            ),
        )

    async def _audio_loop(self) -> None:
        try:
            while True:
                item = await self._audio_input.get()
                if item is _AUDIO_STOP:
                    return
                if isinstance(item, rtc.AudioFrame):
                    frames = await asyncio.to_thread(self._audio_processor.push, item)
                    for frame in frames:
                        await self._queue_async(
                            _Command("audio_append", _id("evt"), bytes(frame.data))
                        )
                else:
                    assert isinstance(item, _AudioControl)
                    if item.kind == "commit":
                        frames = await asyncio.to_thread(self._audio_processor.flush)
                        for frame in frames:
                            await self._queue_async(
                                _Command("audio_append", _id("evt"), bytes(frame.data))
                            )
                        await self._queue_async(_Command("audio_commit", _id("evt")))
                    else:
                        await asyncio.to_thread(self._audio_processor.clear)
                        await self._queue_async(_Command("audio_clear", _id("evt")))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._schedule_fatal(exc)

    def _command_event(self, command: _Command) -> ClientEvent:
        session_id = self._session_id
        if session_id is None:
            raise ConnectionError("Hugging Voice session ID is unavailable")
        if command.kind == "session_update":
            return self._session_update_event(event_id=command.event_id)
        if command.kind == "conversation_item":
            assert isinstance(command.payload, ChatMessage)
            text = command.payload.text_content
            assert text is not None
            return ConversationItemCreateEvent(
                event_id=command.event_id,
                session_id=session_id,
                item=ConversationItem(
                    id=self._wire_item_id(command.payload.id),
                    role=ConversationRole(command.payload.role),
                    content=text,
                ),
            )
        if command.kind == "audio_append":
            assert isinstance(command.payload, bytes)
            event = InputAudioBufferAppendEvent(
                event_id=command.event_id,
                session_id=session_id,
                sequence=self._sequence,
                audio=encode_pcm16_base64(command.payload),
            )
            self._sequence += 1
            return event
        if command.kind == "audio_commit":
            return InputAudioBufferCommitEvent(
                event_id=command.event_id,
                session_id=session_id,
            )
        if command.kind == "audio_clear":
            return InputAudioBufferClearEvent(
                event_id=command.event_id,
                session_id=session_id,
            )
        if command.kind == "response_create":
            return ResponseCreateEvent(
                event_id=command.event_id,
                session_id=session_id,
                instructions=cast(str | None, command.payload),
            )
        assert command.kind == "response_cancel"
        response_id, generation_id = cast(tuple[str, str], command.payload)
        return ResponseCancelEvent(
            event_id=command.event_id,
            session_id=session_id,
            response_id=response_id,
            generation_id=generation_id,
        )

    def _session_update_event(self, *, event_id: str | None = None) -> SessionUpdateEvent:
        session_id = self._session_id
        if session_id is None:
            raise ConnectionError("Hugging Voice session ID is unavailable")
        return SessionUpdateEvent(
            event_id=event_id or _id("evt"),
            session_id=session_id,
            session=SessionConfig(instructions=self._instructions),
        )

    def _conversation_command(self, message: ChatMessage) -> _Command:
        return _Command("conversation_item", _id("evt"), message)

    def _validated_messages(self, context: ChatContext) -> list[ChatMessage]:
        messages: list[ChatMessage] = []
        for item in context.items:
            if not isinstance(item, ChatMessage):
                raise RealtimeError("Hugging Voice chat context accepts only text messages")
            if item.role not in {"user", "assistant"}:
                raise RealtimeError("Hugging Voice chat roles are user and assistant only")
            if item.text_content is None or len(item.content) != 1:
                raise RealtimeError(
                    "Hugging Voice chat messages must contain exactly one text part"
                )
            messages.append(item)
        return messages

    def _wire_item_id(self, livekit_id: str) -> str:
        mapped = self._wire_item_ids.get(livekit_id)
        if mapped is not None:
            return mapped
        if re.fullmatch(r"item_[A-Za-z0-9_-]{1,91}", livekit_id):
            mapped = livekit_id
        else:
            mapped = f"item_{hashlib.sha256(livekit_id.encode()).hexdigest()[:24]}"
        self._wire_item_ids[livekit_id] = mapped
        return mapped

    async def _queue_async(self, command: _Command) -> None:
        try:
            self._outbound.put_nowait(command)
        except asyncio.QueueFull as exc:
            error = _FatalSessionError("Hugging Voice outbound queue is full")
            self._schedule_fatal(error)
            raise error from exc
        self.emit(
            "hugging_voice_client_event_queued",
            ClientEventQueued(type=command.kind, event_id=command.event_id),
        )

    def _queue_sync(self, command: _Command) -> None:
        if self._closing:
            raise RealtimeError("realtime session is closed")
        try:
            self._outbound.put_nowait(command)
        except asyncio.QueueFull as exc:
            self._schedule_fatal(RealtimeError("Hugging Voice outbound queue is full"))
            raise RealtimeError("Hugging Voice outbound queue is full") from exc
        self.emit(
            "hugging_voice_client_event_queued",
            ClientEventQueued(type=command.kind, event_id=command.event_id),
        )

    def _put_audio_nowait(self, item: rtc.AudioFrame | _AudioControl) -> None:
        try:
            self._audio_input.put_nowait(item)
        except asyncio.QueueFull as exc:
            self._schedule_fatal(RealtimeError("Hugging Voice input audio queue is full"))
            raise RealtimeError("Hugging Voice input audio queue is full") from exc

    def _ensure_audio_may_queue(self) -> None:
        if self._closing:
            raise RealtimeError("realtime session is closed")
        if self._ever_connected and not self._connected.is_set():
            raise RealtimeError("audio is not buffered while Hugging Voice reconnects")

    def _disconnect_cleanup(self, error: Exception) -> None:
        self._connected.clear()
        self._ws = None
        self._session_id = None
        self._sequence = 0
        self._audio_processor.reset()
        self._clear_queue(self._audio_input)
        self._clear_queue(self._outbound)
        self._fail_pending(RealtimeError(f"response failed during disconnect: {error}"))
        if self._response is not None:
            self._finish_response(self._response, cancelled=True, add_to_chat=False)

    def _fail_pending(self, error: Exception) -> None:
        pending = self._pending_generation
        self._pending_generation = None
        self._pending_event_id = None
        if self._pending_timeout_task is not None:
            self._pending_timeout_task.cancel()
            self._pending_timeout_task = None
        if pending is not None and not pending.done():
            pending.set_exception(error)

    def _emit_error(self, error: Exception, *, recoverable: bool) -> None:
        self.emit(
            "error",
            RealtimeModelError(
                timestamp=time.time(),
                label=self._model.label,
                error=error,
                recoverable=recoverable,
            ),
        )

    def _schedule_fatal(self, error: Exception) -> None:
        self._emit_error(error, recoverable=False)
        if self._fatal_task is None:
            self._fatal_task = asyncio.create_task(self.aclose())

    def _matching_response(self, generation_id: str) -> _ResponseState | None:
        response = self._response
        if response is None or response.generation_id != generation_id:
            return None
        return response

    async def _expire_generation_request(
        self,
        future: asyncio.Future[GenerationCreatedEvent],
        event_id: str,
    ) -> None:
        try:
            await asyncio.sleep(self._model._conn_options.timeout)
            if self._pending_generation is future and self._pending_event_id == event_id:
                self._pending_generation = None
                self._pending_event_id = None
                self._pending_timeout_task = None
                if not future.done():
                    future.set_exception(RealtimeError("response.created timed out"))
        except asyncio.CancelledError:
            return

    @staticmethod
    def _channel_send(channel: aio.Chan[ChannelValue], value: ChannelValue) -> None:
        try:
            channel.send_nowait(value)
        except ChanFull as exc:
            raise _FatalSessionError("LiveKit generation channel is full") from exc

    @staticmethod
    def _drain_channel(channel: aio.Chan[ChannelValue]) -> None:
        while True:
            try:
                channel.recv_nowait()
            except (ChanEmpty, aio.ChanClosed):
                return

    @staticmethod
    def _clear_queue(queue: asyncio.Queue[object]) -> None:
        while True:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                return

    @staticmethod
    def _put_stop(queue: asyncio.Queue[object], sentinel: object) -> None:
        while True:
            try:
                queue.put_nowait(sentinel)
                return
            except asyncio.QueueFull:
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    return


__all__ = [
    "ClientEventQueued",
    "PartialTranscription",
    "RealtimeModel",
    "RealtimeSession",
    "ServerEventReceived",
]
