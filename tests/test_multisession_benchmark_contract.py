from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import aiohttp
import pytest
from aiohttp import WSMsgType, web
from hugging_voice_protocol.errors import ErrorCode
from hugging_voice_protocol.events import (
    ErrorEvent,
    ErrorPayload,
    ModelRevisions,
    SessionCreatedEvent,
    SessionModels,
    SessionUpdatedEvent,
    SessionUpdateEvent,
    parse_client_event_json,
)

REPO_ROOT = Path(__file__).parents[1]
REVISION = "c" * 40


def load_runner() -> ModuleType:
    path = REPO_ROOT / "benchmarks" / "multisession_soak.py"
    spec = importlib.util.spec_from_file_location("benchmark_multisession_contract", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class BenchmarkContractServer:
    def __init__(self, port: int, capacity: int) -> None:
        self.port = port
        self.capacity = capacity
        self.active = 0
        self._next_session = 0
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
        assert request.headers["Authorization"] == "Bearer contract-secret"
        websocket = web.WebSocketResponse(protocols=("hugging-voice-livekit.v2",))
        await websocket.prepare(request)
        if self.active >= self.capacity:
            await websocket.send_str(
                ErrorEvent(
                    event_id="evt_capacity_rejected",
                    session_id="session_capacity_rejected",
                    error=ErrorPayload(
                        code=ErrorCode.SESSION_LIMIT_REACHED,
                        message="configured benchmark capacity exhausted",
                        retryable=True,
                    ),
                ).model_dump_json()
            )
            await websocket.close()
            return websocket

        index = self._next_session
        self._next_session += 1
        self.active += 1
        session_id = f"session_benchmark_{index:02d}"
        await websocket.send_str(
            SessionCreatedEvent(
                event_id=f"evt_session_created_{index:02d}",
                session_id=session_id,
                models=SessionModels(),
                revisions=ModelRevisions(
                    vad="6.2.1",
                    stt=REVISION,
                    llm=REVISION,
                    tts=REVISION,
                ),
                llama_slot_id=index,
                supported_languages=("de", "en", "fr", "it"),
                supported_voices=(
                    "warm_female",
                    "clear_female",
                    "warm_male",
                    "clear_male",
                    "friendly_neutral",
                ),
            ).model_dump_json()
        )
        try:
            async for message in websocket:
                if message.type is not WSMsgType.TEXT:
                    continue
                event = parse_client_event_json(message.data)
                if isinstance(event, SessionUpdateEvent):
                    await websocket.send_str(
                        SessionUpdatedEvent(
                            event_id=f"evt_ack_{event.event_id}",
                            session_id=session_id,
                            source_event_id=event.event_id,
                        ).model_dump_json()
                    )
        finally:
            self.active -= 1
        return websocket


def make_client(
    module: ModuleType,
    http: aiohttp.ClientSession,
    url: str,
    index: int,
    concurrency: int,
    canaries: list[str],
) -> Any:
    canary = canaries[index]
    return module.SoakSession(
        profile_id="test-profile",
        session_concurrency=concurrency,
        arrival_mode="barrier",
        workload="mixed",
        session_index=index,
        label=f"session-{index:02d}",
        session=http,
        url=url,
        token="contract-secret",
        audio=b"\x00\x00" * 640,
        realtime_audio=False,
        canary=canary,
        forbidden_canaries=frozenset(canaries) - {canary},
        tool_delay_seconds=0.0,
    )


@pytest.mark.parametrize("session_count", [1, 2, 4, 8])
@pytest.mark.asyncio
async def test_benchmark_connects_n_isolated_sessions(
    unused_tcp_port: int, session_count: int
) -> None:
    module = load_runner()
    server = BenchmarkContractServer(unused_tcp_port, capacity=session_count)
    await server.start()
    canaries = module.build_canaries(session_count, 1234)
    async with aiohttp.ClientSession() as http:
        clients = [
            make_client(module, http, server.url, index, session_count, canaries)
            for index in range(session_count)
        ]
        try:
            await asyncio.gather(*(client.connect() for client in clients))
            assert len({client.session_id for client in clients}) == session_count
            assert {client.slot_id for client in clients} == set(range(session_count))
            assert len(set(canaries)) == session_count
        finally:
            await asyncio.gather(*(client.close() for client in clients))
            await server.close()


@pytest.mark.asyncio
async def test_benchmark_capacity_rejection_is_explicit(unused_tcp_port: int) -> None:
    module = load_runner()
    server = BenchmarkContractServer(unused_tcp_port, capacity=4)
    await server.start()
    canaries = module.build_canaries(5, 1234)
    async with aiohttp.ClientSession() as http:
        clients = [make_client(module, http, server.url, index, 5, canaries) for index in range(5)]
        try:
            results = await asyncio.gather(
                *(client.connect() for client in clients), return_exceptions=True
            )
            failures = [result for result in results if isinstance(result, Exception)]
            assert len(failures) == 1
            assert "expected session.created, got error" in str(failures[0])
        finally:
            await asyncio.gather(*(client.close() for client in clients))
            await server.close()
