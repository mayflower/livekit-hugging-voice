from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

import aiohttp
import pytest
from aiohttp import web
from livekit.plugins.hugging_voice.endpoint_resolver import EndpointResolver


@dataclass
class CapacityHTTPServer:
    port: int
    payload: object

    def __post_init__(self) -> None:
        self.authorization: str | None = None
        self._runner: web.AppRunner | None = None

    @property
    def websocket_url(self) -> str:
        return f"ws://127.0.0.1:{self.port}/v1/realtime"

    async def start(self) -> None:
        app = web.Application()
        app.router.add_get("/v1/capacity", self._capacity)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        await web.TCPSite(self._runner, "127.0.0.1", self.port).start()

    async def close(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()

    async def _capacity(self, request: web.Request) -> web.Response:
        self.authorization = request.headers.get("Authorization")
        return web.json_response(self.payload)


@pytest.mark.asyncio
async def test_capacity_orders_available_by_load_and_leaves_full_last(
    unused_tcp_port_factory: Callable[[], int],
) -> None:
    idle = CapacityHTTPServer(unused_tcp_port_factory(), {"total": 2, "active": 0, "available": 2})
    busy = CapacityHTTPServer(unused_tcp_port_factory(), {"total": 2, "active": 1, "available": 1})
    full = CapacityHTTPServer(unused_tcp_port_factory(), {"total": 2, "active": 2, "available": 0})
    for server in (idle, busy, full):
        await server.start()
    resolver = EndpointResolver(
        static_urls=[full.websocket_url, busy.websocket_url, idle.websocket_url],
        randomizer=random.Random(7),
    )
    try:
        async with aiohttp.ClientSession() as session:
            ordered = await resolver.resolve(session, token="resolver-secret")
        assert ordered == (idle.websocket_url, busy.websocket_url, full.websocket_url)
        assert all(
            server.authorization == "Bearer resolver-secret" for server in (idle, busy, full)
        )
    finally:
        for server in (idle, busy, full):
            await server.close()


@pytest.mark.asyncio
async def test_capacity_accepts_operator_configured_pool_size(
    unused_tcp_port: int,
) -> None:
    server = CapacityHTTPServer(
        unused_tcp_port,
        {"total": 20, "active": 7, "available": 13},
    )
    await server.start()
    resolver = EndpointResolver(static_urls=[server.websocket_url])
    try:
        async with aiohttp.ClientSession() as session:
            assert await resolver.resolve(session, token="secret") == (server.websocket_url,)
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_malformed_capacity_is_bounded_unknown_not_a_false_available_claim(
    unused_tcp_port_factory: Callable[[], int],
) -> None:
    good = CapacityHTTPServer(unused_tcp_port_factory(), {"total": 2, "active": 0, "available": 2})
    malformed = CapacityHTTPServer(unused_tcp_port_factory(), {"total": 99, "active": -1})
    await good.start()
    await malformed.start()
    resolver = EndpointResolver(static_urls=[malformed.websocket_url, good.websocket_url])
    try:
        async with aiohttp.ClientSession() as session:
            ordered = await resolver.resolve(session, token="secret")
        assert ordered[0] == good.websocket_url
        assert ordered[1] == malformed.websocket_url
    finally:
        await good.close()
        await malformed.close()


@pytest.mark.asyncio
async def test_headless_dns_formats_ipv4_and_ipv6_and_uses_short_cache() -> None:
    calls = 0

    async def dns(host: str, port: int) -> tuple[str, ...]:
        nonlocal calls
        calls += 1
        assert (host, port) == ("hugging-voice-headless.voice.svc", 8765)
        return ("127.0.0.9", "2001:db8::9", "not-an-address")

    resolver = EndpointResolver(
        headless_dns="hugging-voice-headless.voice.svc",
        dns_resolver=dns,
        capacity_timeout=0.05,
        cache_ttl=2.0,
    )
    async with aiohttp.ClientSession() as session:
        first = await resolver.resolve(session, token="secret")
        second = await resolver.resolve(session, token="secret")
    assert first == (
        "ws://127.0.0.9:8765/v1/realtime",
        "ws://[2001:db8::9]:8765/v1/realtime",
    )
    assert second == first
    assert calls == 1


def test_resolver_modes_are_mutually_exclusive_and_bounded() -> None:
    with pytest.raises(ValueError):
        EndpointResolver()
    with pytest.raises(ValueError):
        EndpointResolver(static_urls=["ws://one"], headless_dns="service")
    with pytest.raises(ValueError):
        EndpointResolver(headless_dns="service", headless_port=0)
