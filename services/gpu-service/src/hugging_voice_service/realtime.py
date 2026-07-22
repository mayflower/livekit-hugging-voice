"""Authenticated realtime WebSocket admission and service-wide schedulers."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Callable
from typing import cast

from fastapi import WebSocket
from hugging_voice_protocol.audio import MAX_AUDIO_BASE64_CHARS
from hugging_voice_protocol.errors import CloseCode, ErrorCode
from hugging_voice_protocol.events import (
    ErrorEvent,
    ErrorPayload,
    ModelRevisions,
    ResponseDoneEvent,
    ResponseOutputAudioDeltaEvent,
    ServerEvent,
    SessionCreatedEvent,
    SessionModels,
    parse_client_event_json,
)
from pydantic import ValidationError
from starlette.websockets import WebSocketDisconnect, WebSocketState

from .capacity import CapacityManager
from .lifecycle import LifecyclePhase, ServiceLifecycle
from .pipeline import GemmaStreamer, VoicePipeline
from .runtimes.silero import SessionVAD
from .schedulers.stt import STTRuntime, STTScheduler
from .schedulers.tts import TTSRuntime, TTSScheduler
from .sessions import SessionRegistry, SessionState, SessionTransport

logger = logging.getLogger(__name__)

WEBSOCKET_SUBPROTOCOL = "hugging-voice-livekit.v1"
MAX_INBOUND_MESSAGE_CHARS = MAX_AUDIO_BASE64_CHARS + 8_192


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


class OutboundQueueFull(RuntimeError):
    pass


class InvalidProtocolEvent(ValueError):
    pass


class InboundQueueFull(RuntimeError):
    pass


class WebSocketTransport:
    """One bounded, serialized sender for a single accepted WebSocket."""

    def __init__(self, websocket: WebSocket, *, max_messages: int) -> None:
        self._websocket = websocket
        self._queue: asyncio.Queue[ServerEvent | None] = asyncio.Queue(max_messages)
        self._sender: asyncio.Task[None] | None = None
        self._closed = False
        self._cancelled_generations: set[str] = set()

    async def start(self) -> None:
        if self._sender is not None:
            raise RuntimeError("WebSocket transport already started")
        await self._websocket.accept(subprotocol=WEBSOCKET_SUBPROTOCOL)
        self._sender = asyncio.create_task(self._send_loop())

    async def send(self, event: ServerEvent) -> None:
        if self._closed:
            raise ConnectionError("WebSocket transport is closed")
        if self._sender is None or self._sender.done():
            raise ConnectionError("WebSocket sender is unavailable")
        if (
            isinstance(event, ResponseOutputAudioDeltaEvent)
            and event.generation_id in self._cancelled_generations
        ):
            return
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull as exc:
            raise OutboundQueueFull("outbound WebSocket queue is full") from exc

    async def cancel_generation(self, generation_id: str) -> int:
        self._cancelled_generations.add(generation_id)
        retained: list[ServerEvent | None] = []
        dropped = 0
        while True:
            try:
                event = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if (
                isinstance(event, ResponseOutputAudioDeltaEvent)
                and event.generation_id == generation_id
            ):
                dropped += 1
            else:
                retained.append(event)
        for event in retained:
            self._queue.put_nowait(event)
        return dropped

    async def close(self, *, code: int, reason: str) -> None:
        if self._closed:
            return
        self._closed = True
        sender = self._sender
        if sender is not None:
            try:
                await asyncio.wait_for(self._queue.put(None), timeout=1.0)
                await asyncio.wait_for(
                    asyncio.gather(sender, return_exceptions=True),
                    timeout=2.0,
                )
            except TimeoutError:
                sender.cancel()
                await asyncio.gather(sender, return_exceptions=True)
            self._sender = None
        if self._websocket.application_state is not WebSocketState.DISCONNECTED:
            try:
                await self._websocket.close(code=code, reason=reason[:123])
            except RuntimeError:
                pass

    async def _send_loop(self) -> None:
        while True:
            event = await self._queue.get()
            if event is None:
                return
            if (
                isinstance(event, ResponseOutputAudioDeltaEvent)
                and event.generation_id in self._cancelled_generations
            ):
                continue
            await self._websocket.send_text(event.model_dump_json())
            if isinstance(event, ResponseDoneEvent):
                self._cancelled_generations.discard(event.generation_id)


class RealtimeService:
    """Owns admission, two shared schedulers, and isolated session state."""

    def __init__(
        self,
        lifecycle: ServiceLifecycle,
        *,
        vad_factory: Callable[[], SessionVAD] | None = None,
    ) -> None:
        self.lifecycle = lifecycle
        self.settings = lifecycle.settings
        self.capacity = CapacityManager(
            self.settings.server.max_sessions,
            telemetry=lifecycle.telemetry,
        )
        self.registry = SessionRegistry(
            self.capacity,
            drain_timeout=self.settings.server.drain_timeout_seconds,
        )
        self._vad_factory = vad_factory or self._new_vad
        self._stt: STTScheduler | None = None
        self._tts: TTSScheduler | None = None
        self._started = False
        self._draining = False

    @property
    def ready(self) -> bool:
        return self._started and not self._draining and self.lifecycle.ready

    async def start(self) -> None:
        if self._started:
            raise RuntimeError("realtime service already started")
        if not self.lifecycle.ready:
            return
        if self.lifecycle.parakeet is None or self.lifecycle.qwen is None:
            raise RuntimeError("speech runtimes are unavailable after lifecycle startup")
        self._stt = STTScheduler(
            cast(STTRuntime, self.lifecycle.parakeet),
            telemetry=self.lifecycle.telemetry,
        )
        self._tts = TTSScheduler(
            cast(TTSRuntime, self.lifecycle.qwen),
            telemetry=self.lifecycle.telemetry,
        )
        await self._stt.start()
        await self._tts.start()
        self._started = True

    async def handle_websocket(self, websocket: WebSocket) -> None:
        session_id = _id("session")
        if not self._authenticated(websocket):
            await self._reject(
                websocket,
                session_id=session_id,
                code=ErrorCode.AUTHENTICATION_FAILED,
                message="missing or invalid bearer token",
                close_code=CloseCode.AUTHENTICATION_ERROR,
                with_subprotocol=self._offers_subprotocol(websocket),
            )
            return
        if not self._offers_subprotocol(websocket):
            await self._reject(
                websocket,
                session_id=session_id,
                code=ErrorCode.INVALID_CONFIGURATION,
                message=f"required WebSocket subprotocol is {WEBSOCKET_SUBPROTOCOL}",
                close_code=CloseCode.PROTOCOL_ERROR,
                with_subprotocol=False,
            )
            return
        if not self.ready:
            draining = self._draining or self.lifecycle.phase is LifecyclePhase.DRAINING
            await self._reject(
                websocket,
                session_id=session_id,
                code=ErrorCode.SERVICE_DRAINING if draining else ErrorCode.MODEL_FAILURE,
                message="service is draining" if draining else "service is not ready",
                close_code=(CloseCode.SERVICE_RESTART if draining else CloseCode.SERVICE_FAILURE),
                with_subprotocol=True,
            )
            return

        transport = WebSocketTransport(
            websocket,
            max_messages=self.settings.server.outbound_queue_size,
        )
        await transport.start()
        state = await self.registry.create(
            session_id=session_id,
            transport=transport,
            vad=self._vad_factory(),
        )
        if state is None:
            await self._send_transport_error(
                transport,
                session_id=session_id,
                code=ErrorCode.SESSION_LIMIT_REACHED,
                message="all session slots are occupied",
            )
            await transport.close(
                code=int(CloseCode.SESSION_LIMIT_REACHED),
                reason=ErrorCode.SESSION_LIMIT_REACHED,
            )
            return

        logger.info(
            "realtime_session_admitted",
            extra={"session_id": session_id, "slot": state.slot.index},
        )
        try:
            pipeline = self._new_pipeline(state)
            await transport.send(self._session_created(session_id))
            await self._serve_session(websocket, state, pipeline)
        except OutboundQueueFull:
            self.lifecycle.telemetry.websocket_errors.inc()
            await transport.close(
                code=int(CloseCode.SESSION_CONFLICT),
                reason=ErrorCode.QUEUE_OVERFLOW,
            )
        except InboundQueueFull:
            self.lifecycle.telemetry.websocket_errors.inc()
            await self._send_transport_error(
                transport,
                session_id=session_id,
                code=ErrorCode.QUEUE_OVERFLOW,
                message="inbound WebSocket queue is full",
            )
            await transport.close(
                code=int(CloseCode.SESSION_CONFLICT),
                reason=ErrorCode.QUEUE_OVERFLOW,
            )
        except InvalidProtocolEvent:
            self.lifecycle.telemetry.websocket_errors.inc()
            await self._send_transport_error(
                transport,
                session_id=session_id,
                code=ErrorCode.INVALID_EVENT,
                message="invalid protocol event",
            )
            await transport.close(
                code=int(CloseCode.PROTOCOL_ERROR),
                reason=ErrorCode.INVALID_EVENT,
            )
        except ValueError as exc:
            self.lifecycle.telemetry.websocket_errors.inc()
            await self._send_transport_error(
                transport,
                session_id=session_id,
                code=ErrorCode.SESSION_STATE_CONFLICT,
                message=str(exc) or "session state conflict",
            )
            await transport.close(
                code=int(CloseCode.SESSION_CONFLICT),
                reason=ErrorCode.SESSION_STATE_CONFLICT,
            )
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            self.lifecycle.telemetry.websocket_errors.inc()
            logger.warning(
                "realtime_session_failed",
                extra={"session_id": session_id, "error": f"{type(exc).__name__}: {exc}"},
            )
            await self._send_transport_error(
                transport,
                session_id=session_id,
                code=ErrorCode.MODEL_FAILURE,
                message="realtime session failed",
            )
            await transport.close(
                code=int(CloseCode.SERVICE_FAILURE),
                reason=ErrorCode.MODEL_FAILURE,
            )
        finally:
            drained = await self.registry.release(state)
            logger.info(
                "realtime_session_released",
                extra={
                    "session_id": session_id,
                    "slot": state.slot.index,
                    "drained": drained,
                    "connected_seconds": round(time.monotonic() - state.connected_at, 3),
                },
            )
            await transport.close(code=1000, reason="session closed")

    async def usage_report(self) -> dict[str, object]:
        states = await self.registry.states()
        now = time.monotonic()
        return {
            "active_sessions": sum(state.lifecycle.value == "active" for state in states),
            "active_responses": sum(state.current_response_id is not None for state in states),
            "sessions": [
                {
                    "session_id": state.session_id,
                    "lifecycle": state.lifecycle,
                    "connected_seconds": round(now - state.connected_at, 3),
                    "turn_revision": state.current_turn_revision,
                    "response_active": state.current_response_id is not None,
                }
                for state in states
            ],
        }

    async def aclose(self) -> None:
        if self._draining:
            return
        self._draining = True
        await self.capacity.begin_service_drain()
        self.lifecycle.begin_drain()
        if self._started:
            try:
                await asyncio.wait_for(
                    self.registry.wait_connections_drained(),
                    timeout=self.settings.server.drain_timeout_seconds,
                )
            except TimeoutError:
                states = await self.registry.states()
                for state in states:
                    if state.lifecycle.value == "active":
                        await self._send_transport_error(
                            state.transport,
                            session_id=state.session_id,
                            code=ErrorCode.SERVICE_DRAINING,
                            message="service drain timeout reached",
                        )
                        await state.transport.close(
                            code=int(CloseCode.SERVICE_RESTART),
                            reason=ErrorCode.SERVICE_DRAINING,
                        )
                await asyncio.gather(
                    *(self.registry.release(state) for state in states),
                    return_exceptions=True,
                )
            if self._stt is not None:
                await self._stt.aclose()
            if self._tts is not None:
                await self._tts.aclose()
        self._stt = None
        self._tts = None
        self._started = False

    async def _serve_session(
        self,
        websocket: WebSocket,
        state: SessionState,
        pipeline: VoicePipeline,
    ) -> None:
        inbound: asyncio.Queue[str | None] = asyncio.Queue(self.settings.server.inbound_queue_size)

        async def receive() -> None:
            try:
                while True:
                    message = await websocket.receive()
                    if message["type"] == "websocket.disconnect":
                        return
                    text = message.get("text")
                    if text is None or len(text) > MAX_INBOUND_MESSAGE_CHARS:
                        raise ValueError("WebSocket message must be bounded UTF-8 JSON text")
                    try:
                        inbound.put_nowait(text)
                    except asyncio.QueueFull as exc:
                        raise InboundQueueFull("inbound WebSocket queue is full") from exc
            finally:
                try:
                    inbound.put_nowait(None)
                except asyncio.QueueFull:
                    pass

        async def consume() -> None:
            while True:
                raw = await inbound.get()
                if raw is None:
                    return
                try:
                    event = parse_client_event_json(raw)
                    await pipeline.handle_event(event)
                except ValidationError as exc:
                    raise InvalidProtocolEvent("invalid protocol event") from exc

        receiver = asyncio.create_task(receive())
        consumer = asyncio.create_task(consume())
        done, pending = await asyncio.wait(
            {receiver, consumer},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        for task in done:
            task.result()

    def _new_vad(self) -> SessionVAD:
        config = self.settings.vad
        return SessionVAD(
            threshold=config.threshold,
            min_speech_ms=config.min_speech_ms,
            min_speech_continuation_ms=config.min_speech_continuation_ms,
            min_silence_ms=config.min_silence_ms,
            speech_pad_ms=config.speech_pad_ms,
        )

    def _new_pipeline(self, state: SessionState) -> VoicePipeline:
        if self._stt is None or self._tts is None or self.lifecycle.gemma is None:
            raise RuntimeError("realtime schedulers are not ready")
        return VoicePipeline(
            state,
            stt=self._stt,
            tts=self._tts,
            gemma=cast(GemmaStreamer, self.lifecycle.gemma),
            telemetry=self.lifecycle.telemetry,
        )

    def _session_created(self, session_id: str) -> SessionCreatedEvent:
        lock = self.lifecycle.lock
        if lock is None:
            raise RuntimeError("verified model lock is unavailable")
        revisions = {model.id: model.revision for model in lock.models}
        return SessionCreatedEvent(
            event_id=_id("evt"),
            session_id=session_id,
            models=SessionModels(),
            revisions=ModelRevisions(
                vad=revisions["silero-vad"],
                stt=revisions["nvidia/parakeet-tdt-0.6b-v3"],
                llm=revisions["google/gemma-4-31B-it"],
                tts=revisions["Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"],
            ),
        )

    def _authenticated(self, websocket: WebSocket) -> bool:
        authenticator = self.lifecycle.authenticator
        return authenticator is not None and authenticator.authenticate_header(
            websocket.headers.get("authorization")
        )

    @staticmethod
    def _offers_subprotocol(websocket: WebSocket) -> bool:
        return WEBSOCKET_SUBPROTOCOL in websocket.scope.get("subprotocols", [])

    async def _reject(
        self,
        websocket: WebSocket,
        *,
        session_id: str,
        code: ErrorCode,
        message: str,
        close_code: CloseCode,
        with_subprotocol: bool,
    ) -> None:
        await websocket.accept(subprotocol=WEBSOCKET_SUBPROTOCOL if with_subprotocol else None)
        await websocket.send_text(
            ErrorEvent(
                event_id=_id("evt"),
                session_id=session_id,
                error=ErrorPayload(code=code, message=message, retryable=False),
            ).model_dump_json()
        )
        await websocket.close(code=int(close_code), reason=code)
        self.lifecycle.telemetry.websocket_errors.inc()

    @staticmethod
    async def _send_transport_error(
        transport: SessionTransport,
        *,
        session_id: str,
        code: ErrorCode,
        message: str,
    ) -> None:
        try:
            await transport.send(
                ErrorEvent(
                    event_id=_id("evt"),
                    session_id=session_id,
                    error=ErrorPayload(code=code, message=message, retryable=False),
                )
            )
        except (ConnectionError, OutboundQueueFull):
            pass
