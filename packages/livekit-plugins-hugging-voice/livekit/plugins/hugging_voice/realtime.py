"""Native LiveKit Agents realtime adapter for the Hugging Voice protocol."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
import uuid
from collections.abc import AsyncIterable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeVar, cast

import aiohttp
from hugging_voice_protocol.audio import decode_pcm16_base64, encode_pcm16_base64
from hugging_voice_protocol.errors import CloseCode, ErrorCode
from hugging_voice_protocol.events import (
    ClientEvent,
    ConversationItemCreatedEvent,
    ConversationItemCreateEvent,
    ConversationRole,
    ErrorEvent,
    FunctionCallConversationItem,
    FunctionCallOutputConversationItem,
    InputAudioBufferAppendEvent,
    InputAudioBufferClearEvent,
    InputAudioBufferCommitEvent,
    InputTranscriptionCompletedEvent,
    InputTranscriptionDeltaEvent,
    MessageConversationItem,
    ResponseCancelEvent,
    ResponseCreatedEvent,
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
    SessionCreatedEvent,
    SessionUpdatedEvent,
    SessionUpdateEvent,
    SpeechStartedEvent,
    SpeechStoppedEvent,
    parse_server_event_json,
)
from hugging_voice_protocol.events import (
    FunctionTool as ProtocolFunctionTool,
)
from livekit import rtc
from livekit.agents import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions, utils
from livekit.agents.llm import (
    ChatContext,
    ChatMessage,
    FunctionCall,
    FunctionCallOutput,
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
from livekit.agents.llm.tool_context import ProviderTool
from livekit.agents.metrics import RealtimeModelMetrics
from livekit.agents.metrics.base import Metadata
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import aio
from livekit.agents.utils.aio.channel import ChanEmpty

from .audio import InputAudioProcessor
from .endpoint_resolver import EndpointResolver
from .options import resolve_base_urls, resolve_token

logger = logging.getLogger(__name__)

WEBSOCKET_SUBPROTOCOL = "hugging-voice-livekit.v2"
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


def _ack_timeout_error(command_kind: str) -> RealtimeError:
    code = (
        ErrorCode.SESSION_UPDATE_ACK_TIMEOUT
        if command_kind == "session_update"
        else ErrorCode.CONVERSATION_ACK_TIMEOUT
    )
    return RealtimeError(f"{code}: acknowledgement timed out for {command_kind}")


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
        "response_speak",
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
    message_channel: aio.Chan[MessageGeneration]
    function_channel: aio.Chan[FunctionCall]
    direct_say: bool = False
    text: str = ""
    first_audio_at: float | None = None
    next_audio_sequence: int = 0
    saw_text: bool = False
    saw_audio: bool = False
    cancelled: bool = False
    finalized: bool = False
    message_started: bool = False
    pending_call: ResponseOutputFunctionCallDoneEvent | None = None


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
        language: str | None = None,
        voice: str | None = None,
        voice_instructions: str | None = None,
        instructions: str = "",
        http_session: aiohttp.ClientSession | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        if len(instructions) > 8_000:
            raise ValueError("instructions exceed the 8,000 character protocol limit")
        try:
            validated = SessionConfig(
                language=language,
                voice=voice,
                voice_instructions=voice_instructions,
            )
        except ValueError as exc:
            raise ValueError(f"invalid speech configuration: {exc}") from exc
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
                per_response_tool_choice=True,
                supports_say=True,
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
        self._language = validated.language
        self._voice = validated.voice
        self._voice_instructions = validated.voice_instructions
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
        self._language = model._language
        self._voice = model._voice
        self._voice_instructions = model._voice_instructions
        self._turn_detection = ServerVADConfig()
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
        self._pending_direct_say = False
        self._pending_timeout_task: asyncio.Task[None] | None = None
        self._ack_waiters: dict[str, asyncio.Future[None]] = {}
        self._ack_tasks: set[asyncio.Task[None]] = set()
        self._say_tasks: set[asyncio.Task[None]] = set()
        self._configuration_lock = asyncio.Lock()
        self._configuration_idle = asyncio.Event()
        self._configuration_idle.set()
        self._default_tool_choice: ToolChoice = "auto"
        self._tools_frozen = False
        self._wire_item_ids: dict[str, str] = {}
        self._invalidated_call_ids: set[str] = set()
        self._non_model_replay_item_ids: set[str] = set()
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
        async with self._configuration_lock:
            config = self._session_config(instructions=instructions)
            if self._connected.is_set():
                await self._queue_configuration_update(config)
            self._instructions = instructions

    async def update_speech_options(
        self,
        *,
        language: str | None = None,
        voice: str | None = None,
        voice_instructions: NotGivenOr[str | None] = NOT_GIVEN,
    ) -> None:
        """Update language, voice, or Qwen speaking-style instructions."""
        async with self._configuration_lock:
            try:
                validated = self._session_config(
                    language=self._language if language is None else language,
                    voice=self._voice if voice is None else voice,
                    voice_instructions=voice_instructions,
                )
            except ValueError as exc:
                raise RealtimeError(f"invalid speech configuration: {exc}") from exc
            if self._connected.is_set():
                await self._queue_configuration_update(validated)
            self._language = validated.language
            self._voice = validated.voice
            self._voice_instructions = validated.voice_instructions

    async def update_chat_ctx(self, chat_ctx: ChatContext) -> None:
        current = list(self._chat_ctx.items)
        replacement = list(chat_ctx.items)
        if len(replacement) > 30:
            raise RealtimeError("Hugging Voice chat context is limited to 30 items")
        if replacement[: len(current)] != current:
            raise RealtimeError("Hugging Voice supports only append-only chat context updates")
        additions = replacement[len(current) :]
        commands = [(item, self._conversation_command(item)) for item in additions]
        if not self._connected.is_set():
            self._chat_ctx = chat_ctx.copy()
            return
        for _, command in commands:
            self._command_event(command)
        for item, command in commands:
            await self._queue_with_ack(command)
            self._chat_ctx = ChatContext([*self._chat_ctx.items, item])

    async def update_tools(self, tools: list[Tool]) -> None:
        context = ToolContext(tools)
        protocol_tools = self._protocol_tools(context)
        async with self._configuration_lock:
            if self._tools_frozen and protocol_tools != self._protocol_tools(self._tools):
                raise RealtimeError("Hugging Voice tools are immutable after connection")
            config = self._session_config(tools=protocol_tools)
            if self._connected.is_set():
                await self._queue_configuration_update(config)
            self._tools = context

    def update_options(
        self,
        *,
        tool_choice: NotGivenOr[ToolChoice | None] = NOT_GIVEN,
    ) -> None:
        if utils.is_given(tool_choice):
            previous = self._default_tool_choice
            candidate: ToolChoice = "auto" if tool_choice is None else tool_choice
            config = self._session_config(tool_choice=candidate)
            if self._connected.is_set():
                if not self._configuration_idle.is_set():
                    raise RealtimeError("a configuration update is already pending")
                self._configuration_idle.clear()
                try:
                    self._queue_sync_with_ack(
                        _Command("session_update", _id("evt"), config),
                        on_rejected=lambda: self._restore_tool_choice(candidate, previous),
                        on_complete=self._configuration_idle.set,
                    )
                except Exception:
                    self._configuration_idle.set()
                    raise
            self._default_tool_choice = candidate

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
        if self._pending_generation is not None or self._response is not None:
            raise RealtimeError("a response is already pending or active")
        if self._ever_connected and not self._connected.is_set():
            raise RealtimeError("cannot generate a response while disconnected")
        response_instructions = instructions if utils.is_given(instructions) else None
        if response_instructions is not None and len(response_instructions) > 8_000:
            raise RealtimeError("response instructions exceed the protocol limit")
        event_id = _id("evt")
        response_tools = self._tools if not utils.is_given(tools) else ToolContext(tools)
        response_choice = (
            self._default_tool_choice if not utils.is_given(tool_choice) else tool_choice
        )
        payload = (response_instructions, self._protocol_tools(response_tools), response_choice)
        future = asyncio.get_running_loop().create_future()
        self._pending_generation = future
        self._pending_event_id = event_id
        self._pending_direct_say = False
        self._pending_timeout_task = asyncio.create_task(
            self._expire_generation_request(future, event_id)
        )
        try:
            self._queue_sync(_Command("response_create", event_id, payload))
        except Exception:
            self._pending_generation = None
            self._pending_event_id = None
            self._pending_timeout_task.cancel()
            self._pending_timeout_task = None
            future.cancel()
            raise
        return future

    def say(
        self,
        text: str | AsyncIterable[str],
    ) -> asyncio.Future[GenerationCreatedEvent]:
        if self._pending_generation is not None or self._response is not None:
            raise RealtimeError("a response is already pending or active")
        if self._ever_connected and not self._connected.is_set():
            raise RealtimeError("cannot speak while disconnected")
        event_id = _id("evt")
        future = asyncio.get_running_loop().create_future()
        self._pending_generation = future
        self._pending_event_id = event_id
        self._pending_direct_say = True
        self._pending_timeout_task = asyncio.create_task(
            self._expire_generation_request(future, event_id)
        )
        if isinstance(text, str):
            try:
                self._validate_say_text(text)
                self._queue_sync(_Command("response_speak", event_id, text))
            except Exception:
                self._clear_failed_say(future)
                raise
            return future
        task = asyncio.create_task(self._collect_and_queue_say(text, event_id, future))
        self._say_tasks.add(task)
        task.add_done_callback(self._say_tasks.discard)

        def cancel_unfinished_collector(
            completed: asyncio.Future[GenerationCreatedEvent],
        ) -> None:
            if (completed.cancelled() or completed.exception() is not None) and not task.done():
                task.cancel()

        future.add_done_callback(cancel_unfinished_collector)
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
        self._fail_ack_waiters(RealtimeError("realtime session closed"))
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
        for task in tuple(self._ack_tasks):
            task.cancel()
        for task in tuple(self._say_tasks):
            task.cancel()
        await asyncio.gather(*self._ack_tasks, *self._say_tasks, return_exceptions=True)
        self._ack_tasks.clear()
        self._say_tasks.clear()
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
            if self._language is None:
                self._language = created.language
            if self._voice is None:
                self._voice = created.voice
            self._turn_detection = created.turn_detection
            self._sequence = 0
            await self._send_direct_with_ack(self._session_update_event(), ws)
            for item in self._chat_ctx.items:
                if self._skip_invalidated_replay(item):
                    continue
                event = self._command_event(self._conversation_command(item))
                assert isinstance(event, ConversationItemCreateEvent)
                await self._send_direct_with_ack(event, ws)
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
                raise _FatalProtocolError("server did not negotiate the v2 subprotocol")
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
            await self._handle_server_event(event)

    async def _receive_event(self, ws: aiohttp.ClientWebSocketResponse) -> ServerEvent:
        message = await ws.receive()
        if message.type is aiohttp.WSMsgType.TEXT:
            try:
                return parse_server_event_json(message.data)
            except ValueError as exc:
                raise _FatalProtocolError("server emitted an invalid v2 event") from exc
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

    async def _send_direct_with_ack(
        self,
        event: SessionUpdateEvent | ConversationItemCreateEvent,
        ws: aiohttp.ClientWebSocketResponse,
    ) -> None:
        await self._send_direct(event, ws=ws)
        while True:
            received = await asyncio.wait_for(
                self._receive_event(ws), timeout=self._model._conn_options.timeout
            )
            if isinstance(received, SessionUpdatedEvent | ConversationItemCreatedEvent):
                if received.source_event_id == event.event_id:
                    return
                raise _FatalProtocolError("server acknowledged the wrong bootstrap event")
            if isinstance(received, ErrorEvent) and received.error.event_id == event.event_id:
                raise RealtimeError(received.error.message)
            await self._handle_server_event(received)

    async def _handle_server_event(self, event: ServerEvent) -> None:
        if isinstance(event, SessionUpdatedEvent | ConversationItemCreatedEvent):
            waiter = self._ack_waiters.pop(event.source_event_id, None)
            if waiter is not None and not waiter.done():
                waiter.set_result(None)
        elif isinstance(event, SpeechStartedEvent):
            self._invalidate_dangling_calls()
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
        elif isinstance(
            event,
            ResponseCreatedEvent
            | ResponseOutputTextDeltaEvent
            | ResponseOutputTextDoneEvent
            | ResponseOutputAudioDeltaEvent
            | ResponseOutputAudioDoneEvent
            | ResponseOutputFunctionCallDoneEvent
            | ResponseDoneEvent,
        ):
            await self._handle_response_event(event)
        elif isinstance(event, ErrorEvent):
            error = RealtimeError(event.error.message)
            if event.error.event_id == self._pending_event_id:
                self._fail_pending(error)
            if event.error.event_id is not None:
                waiter = self._ack_waiters.pop(event.error.event_id, None)
                if waiter is not None and not waiter.done():
                    waiter.set_exception(error)
            self._emit_error(error, recoverable=event.error.retryable)

    async def _handle_response_event(
        self,
        event: (
            ResponseCreatedEvent
            | ResponseOutputTextDeltaEvent
            | ResponseOutputTextDoneEvent
            | ResponseOutputAudioDeltaEvent
            | ResponseOutputAudioDoneEvent
            | ResponseOutputFunctionCallDoneEvent
            | ResponseDoneEvent
        ),
    ) -> None:
        if isinstance(event, ResponseCreatedEvent):
            self._response_created(event)
        elif isinstance(event, ResponseOutputTextDeltaEvent):
            response = self._matching_response(event.generation_id)
            if response is not None:
                await self._start_message(response)
                response.saw_text = True
                response.text += event.delta
                await response.text_channel.send(event.delta)
        elif isinstance(event, ResponseOutputTextDoneEvent):
            response = self._matching_response(event.generation_id)
            if response is not None:
                response.text_channel.close()
        elif isinstance(event, ResponseOutputAudioDeltaEvent):
            response = self._matching_response(event.generation_id)
            if response is not None:
                if event.sequence != response.next_audio_sequence:
                    raise _FatalProtocolError(
                        "server audio sequence conflict: "
                        f"expected={response.next_audio_sequence} received={event.sequence}"
                    )
                response.next_audio_sequence += 1
                await self._start_message(response)
                payload = decode_pcm16_base64(event.audio)
                if len(payload) != 960:
                    raise _FatalProtocolError("server audio frame is not exactly 20 ms")
                response.saw_audio = True
                if response.first_audio_at is None:
                    response.first_audio_at = time.monotonic()
                await response.audio_channel.send(
                    rtc.AudioFrame(
                        data=payload,
                        sample_rate=24_000,
                        num_channels=1,
                        samples_per_channel=480,
                    )
                )
        elif isinstance(event, ResponseOutputAudioDoneEvent):
            response = self._matching_response(event.generation_id)
            if response is not None:
                response.audio_channel.close()
        elif isinstance(event, ResponseOutputFunctionCallDoneEvent):
            response = self._matching_response(event.generation_id)
            if response is not None:
                if response.pending_call is not None or response.saw_text or response.saw_audio:
                    raise _FatalProtocolError("invalid mixed or duplicate function call response")
                response.pending_call = event
        elif isinstance(event, ResponseDoneEvent):
            response = self._matching_response(event.generation_id)
            if response is not None:
                if event.reason.value == "tool_call":
                    call_event = response.pending_call
                    if call_event is None or response.message_started:
                        raise _FatalProtocolError("tool response is missing its function call")
                    call = FunctionCall(
                        id=call_event.item_id,
                        call_id=call_event.call_id,
                        name=call_event.name,
                        arguments=call_event.arguments,
                        extra={
                            "hugging_voice": {
                                "turn_id": call_event.turn_id,
                                "turn_revision": call_event.turn_revision,
                                "generation_id": call_event.generation_id,
                                "response_id": call_event.response_id,
                            }
                        },
                    )
                    self._chat_ctx.insert(call)
                    await response.function_channel.send(call)
                    self._emit_metrics(response, event)
                    self._finish_response(response, cancelled=False, add_to_chat=False)
                    return
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

    async def _queue_with_ack(self, command: _Command) -> None:
        if len(self._ack_waiters) >= 32:
            raise RealtimeError("too many pending Hugging Voice acknowledgements")
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._ack_waiters[command.event_id] = future
        try:
            await self._queue_async(command)
            await asyncio.wait_for(future, timeout=self._model._conn_options.timeout)
        except TimeoutError as exc:
            error = _ack_timeout_error(command.kind)
            self._schedule_fatal(error)
            raise error from exc
        finally:
            self._ack_waiters.pop(command.event_id, None)

    async def _queue_configuration_update(self, config: SessionConfig) -> None:
        await self._configuration_idle.wait()
        self._configuration_idle.clear()
        try:
            await self._queue_with_ack(_Command("session_update", _id("evt"), config))
        finally:
            self._configuration_idle.set()

    def _queue_sync_with_ack(
        self,
        command: _Command,
        *,
        on_rejected: Callable[[], None] | None = None,
        on_complete: Callable[[], None] | None = None,
    ) -> None:
        if len(self._ack_waiters) >= 32:
            raise RealtimeError("too many pending Hugging Voice acknowledgements")
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._ack_waiters[command.event_id] = future
        try:
            self._queue_sync(command)
        except Exception:
            self._ack_waiters.pop(command.event_id, None)
            raise
        task = asyncio.create_task(
            self._observe_background_ack(
                command,
                future,
                on_rejected=on_rejected,
                on_complete=on_complete,
            )
        )
        self._ack_tasks.add(task)
        task.add_done_callback(self._ack_tasks.discard)

    async def _observe_background_ack(
        self,
        command: _Command,
        future: asyncio.Future[None],
        *,
        on_rejected: Callable[[], None] | None,
        on_complete: Callable[[], None] | None,
    ) -> None:
        try:
            await asyncio.wait_for(future, timeout=self._model._conn_options.timeout)
        except asyncio.CancelledError:
            return
        except TimeoutError:
            if not self._closing:
                self._schedule_fatal(_ack_timeout_error(command.kind))
        except RealtimeError:
            # The server event and reconnect path already report this rejection.
            if on_rejected is not None:
                on_rejected()
            return
        finally:
            self._ack_waiters.pop(command.event_id, None)
            if on_complete is not None:
                on_complete()

    def _response_created(self, event: ResponseCreatedEvent) -> None:
        if self._response is not None:
            raise _FatalProtocolError("server created overlapping responses")
        self._tools_frozen = True
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
            message_channel=aio.Chan[MessageGeneration](1),
            function_channel=aio.Chan[FunctionCall](1),
            direct_say=self._pending_direct_say,
        )
        self._response = response
        generation = GenerationCreatedEvent(
            message_stream=response.message_channel,
            function_stream=response.function_channel,
            user_initiated=user_initiated,
            response_id=event.response_id,
        )
        pending = self._pending_generation
        self._pending_generation = None
        self._pending_event_id = None
        self._pending_direct_say = False
        if self._pending_timeout_task is not None:
            self._pending_timeout_task.cancel()
            self._pending_timeout_task = None
        if pending is not None and not pending.done():
            pending.set_result(generation)
        self.emit("generation_created", generation)

    async def _start_message(self, response: _ResponseState) -> None:
        if response.message_started:
            return
        response.message_started = True
        await response.message_channel.send(
            MessageGeneration(
                message_id=response.message_id,
                text_stream=response.text_channel,
                audio_stream=response.audio_channel,
                modalities=response.modalities,
            ),
        )
        response.message_channel.close()

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
        response.message_channel.close()
        response.function_channel.close()
        if add_to_chat and response.text.strip():
            existing = self._chat_ctx.get_by_id(response.message_id)
            if existing is None:
                self._chat_ctx.add_message(
                    id=response.message_id,
                    role="assistant",
                    content=response.text,
                    interrupted=False,
                )
                if response.direct_say and self._has_dangling_tool_call():
                    # A filler that completed before its FunctionCallOutput is
                    # visible to LiveKit but deliberately absent from the
                    # service's atomic model context. Do not replay it later.
                    self._non_model_replay_item_ids.add(response.message_id)
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
            config = command.payload
            if config is not None and not isinstance(config, SessionConfig):
                raise RealtimeError("invalid queued Hugging Voice session configuration")
            return self._session_update_event(event_id=command.event_id, config=config)
        if command.kind == "conversation_item":
            item = command.payload
            if isinstance(item, ChatMessage):
                text = item.text_content
                if text is None:
                    raise RealtimeError("chat message has no text content")
                wire_item: object = MessageConversationItem(
                    id=self._wire_item_id(item.id),
                    role=ConversationRole(item.role),
                    content=text,
                )
            elif isinstance(item, FunctionCall):
                meta = item.extra.get("hugging_voice", {})
                required_meta = {
                    "turn_id",
                    "turn_revision",
                    "generation_id",
                    "response_id",
                }
                if not required_meta.issubset(meta):
                    raise RealtimeError(
                        "dangling external function calls cannot be added to Hugging Voice"
                    )
                wire_item = FunctionCallConversationItem(
                    id=self._wire_item_id(item.id),
                    call_id=item.call_id,
                    name=item.name,
                    arguments=item.arguments,
                    turn_id=meta["turn_id"],
                    turn_revision=meta["turn_revision"],
                    generation_id=meta["generation_id"],
                    response_id=meta["response_id"],
                )
            elif isinstance(item, FunctionCallOutput):
                if item.call_id in self._invalidated_call_ids:
                    raise RealtimeError("function output belongs to an invalidated tool call")
                call = next(
                    (
                        entry
                        for entry in self._chat_ctx.items
                        if isinstance(entry, FunctionCall) and entry.call_id == item.call_id
                    ),
                    None,
                )
                if call is None:
                    raise RealtimeError("function output references an unknown call")
                meta = call.extra.get("hugging_voice", {})
                wire_item = FunctionCallOutputConversationItem(
                    id=self._wire_item_id(item.id),
                    call_id=item.call_id,
                    name=item.name or call.name,
                    output=item.output,
                    is_error=item.is_error,
                    turn_id=meta["turn_id"],
                    turn_revision=meta["turn_revision"],
                    generation_id=meta["generation_id"],
                    response_id=meta["response_id"],
                )
            else:
                raise RealtimeError("unsupported Hugging Voice context item")
            return ConversationItemCreateEvent(
                event_id=command.event_id,
                session_id=session_id,
                item=cast(Any, wire_item),
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
            instructions, tools, tool_choice = cast(
                tuple[str | None, tuple[ProtocolFunctionTool, ...], ToolChoice], command.payload
            )
            return ResponseCreateEvent(
                event_id=command.event_id,
                session_id=session_id,
                instructions=instructions,
                tools=tools,
                tool_choice=cast(Any, tool_choice),
            )
        if command.kind == "response_speak":
            assert isinstance(command.payload, str)
            return ResponseSpeakEvent(
                event_id=command.event_id,
                session_id=session_id,
                text=command.payload,
            )
        assert command.kind == "response_cancel"
        response_id, generation_id = cast(tuple[str, str], command.payload)
        return ResponseCancelEvent(
            event_id=command.event_id,
            session_id=session_id,
            response_id=response_id,
            generation_id=generation_id,
        )

    async def _collect_and_queue_say(
        self,
        chunks: AsyncIterable[str],
        event_id: str,
        future: asyncio.Future[GenerationCreatedEvent],
    ) -> None:
        parts: list[str] = []
        size = 0
        try:
            async for chunk in chunks:
                if not isinstance(chunk, str):
                    raise RealtimeError("say() stream yielded a non-string value")
                size += len(chunk)
                if size > 500:
                    raise RealtimeError("say() text exceeds the 500 character limit")
                parts.append(chunk)
            text = "".join(parts)
            self._validate_say_text(text)
            if (
                self._pending_generation is not future
                or self._pending_event_id != event_id
                or future.done()
            ):
                return
            self._queue_sync(_Command("response_speak", event_id, text))
        except asyncio.CancelledError:
            self._clear_failed_say(future)
            raise
        except Exception as exc:
            self._clear_failed_say(future, exc)

    @staticmethod
    def _validate_say_text(text: str) -> None:
        try:
            ResponseSpeakEvent(
                event_id="evt_validation",
                session_id="session_validation",
                text=text,
            )
        except ValueError as exc:
            raise RealtimeError(f"invalid say() text: {exc}") from exc

    def _clear_failed_say(
        self,
        future: asyncio.Future[GenerationCreatedEvent],
        error: Exception | None = None,
    ) -> None:
        owns_pending_request = self._pending_generation is future
        if owns_pending_request:
            self._pending_generation = None
            self._pending_event_id = None
            self._pending_direct_say = False
            if self._pending_timeout_task is not None:
                self._pending_timeout_task.cancel()
                self._pending_timeout_task = None
        if future.done():
            return
        if error is None:
            future.cancel()
        else:
            future.set_exception(error)

    def _session_update_event(
        self,
        *,
        event_id: str | None = None,
        config: SessionConfig | None = None,
    ) -> SessionUpdateEvent:
        session_id = self._session_id
        if session_id is None:
            raise ConnectionError("Hugging Voice session ID is unavailable")
        return SessionUpdateEvent(
            event_id=event_id or _id("evt"),
            session_id=session_id,
            session=config or self._session_config(),
        )

    def _session_config(
        self,
        *,
        instructions: str | None = None,
        language: str | None = None,
        voice: str | None = None,
        voice_instructions: NotGivenOr[str | None] = NOT_GIVEN,
        tools: NotGivenOr[tuple[ProtocolFunctionTool, ...]] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
    ) -> SessionConfig:
        return SessionConfig(
            instructions=self._instructions if instructions is None else instructions,
            tools=self._protocol_tools(self._tools) if not utils.is_given(tools) else tools,
            tool_choice=cast(
                Any,
                self._default_tool_choice if not utils.is_given(tool_choice) else tool_choice,
            ),
            language=self._language if language is None else language,
            voice=self._voice if voice is None else voice,
            voice_instructions=(
                self._voice_instructions
                if not utils.is_given(voice_instructions)
                else voice_instructions
            ),
            turn_detection=self._turn_detection,
        )

    def _restore_tool_choice(self, candidate: ToolChoice, previous: ToolChoice) -> None:
        if self._default_tool_choice == candidate:
            self._default_tool_choice = previous

    def _conversation_command(self, item: object) -> _Command:
        return _Command("conversation_item", _id("evt"), item)

    def _invalidate_dangling_calls(self) -> None:
        completed = {
            item.call_id for item in self._chat_ctx.items if isinstance(item, FunctionCallOutput)
        }
        self._invalidated_call_ids.update(
            item.call_id
            for item in self._chat_ctx.items
            if isinstance(item, FunctionCall) and item.call_id not in completed
        )

    def _skip_invalidated_replay(self, item: object) -> bool:
        return (isinstance(item, ChatMessage) and item.id in self._non_model_replay_item_ids) or (
            isinstance(item, FunctionCall | FunctionCallOutput)
            and item.call_id in self._invalidated_call_ids
        )

    def _has_dangling_tool_call(self) -> bool:
        completed = {
            item.call_id for item in self._chat_ctx.items if isinstance(item, FunctionCallOutput)
        }
        return any(
            item.call_id not in completed
            for item in self._chat_ctx.items
            if isinstance(item, FunctionCall)
        )

    @staticmethod
    def _protocol_tools(context: ToolContext) -> tuple[ProtocolFunctionTool, ...]:
        if any(isinstance(tool, ProviderTool) for tool in context.flatten()):
            raise RealtimeError("provider tools are not supported by Hugging Voice")
        try:
            parsed = tuple(
                ProtocolFunctionTool.model_validate(tool)
                for tool in context.parse_function_tools("openai", strict=True)
            )
            return tuple(sorted(parsed, key=lambda tool: tool.function.name))
        except ValueError as exc:
            raise RealtimeError(f"invalid Hugging Voice tool schema: {exc}") from exc

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
        self._fail_ack_waiters(
            RealtimeError(f"context acknowledgement failed during disconnect: {error}")
        )
        if self._response is not None:
            self._finish_response(self._response, cancelled=True, add_to_chat=False)

    def _fail_pending(self, error: Exception) -> None:
        pending = self._pending_generation
        self._pending_generation = None
        self._pending_event_id = None
        self._pending_direct_say = False
        if self._pending_timeout_task is not None:
            self._pending_timeout_task.cancel()
            self._pending_timeout_task = None
        if pending is not None and not pending.done():
            pending.set_exception(error)

    def _fail_ack_waiters(self, error: Exception) -> None:
        for waiter in self._ack_waiters.values():
            if not waiter.done():
                waiter.set_exception(error)
        self._ack_waiters.clear()

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
                self._pending_direct_say = False
                self._pending_timeout_task = None
                if not future.done():
                    future.set_exception(RealtimeError("response.created timed out"))
        except asyncio.CancelledError:
            return

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
