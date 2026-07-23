"""Small capacity-aware resolver for static or headless-service endpoints."""

from __future__ import annotations

import asyncio
import ipaddress
import random
import socket
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from urllib.parse import urlparse, urlunparse

import aiohttp

DNSResolver = Callable[[str, int], Awaitable[Sequence[str]]]


@dataclass(frozen=True, slots=True)
class CapacitySnapshot:
    url: str
    total: int
    active: int
    available: int


async def _system_dns(host: str, port: int) -> Sequence[str]:
    loop = asyncio.get_running_loop()
    records = await loop.getaddrinfo(
        host,
        port,
        family=socket.AF_UNSPEC,
        type=socket.SOCK_STREAM,
    )
    return tuple(dict.fromkeys(str(record[4][0]) for record in records))


class EndpointResolver:
    def __init__(
        self,
        *,
        static_urls: Sequence[str] = (),
        headless_dns: str | None = None,
        headless_port: int = 8765,
        headless_tls: bool = False,
        cache_ttl: float = 2.0,
        capacity_timeout: float = 0.5,
        dns_resolver: DNSResolver = _system_dns,
        randomizer: random.Random | None = None,
    ) -> None:
        if bool(static_urls) == bool(headless_dns):
            raise ValueError("configure static_urls or headless_dns, exactly one")
        if not 1 <= headless_port <= 65_535:
            raise ValueError("headless_port is outside the TCP port range")
        if not 0.1 <= cache_ttl <= 30.0 or not 0.05 <= capacity_timeout <= 5.0:
            raise ValueError("resolver timeout/cache settings are outside safe bounds")
        self._static_urls = tuple(static_urls)
        self._headless_dns = headless_dns
        self._headless_port = headless_port
        self._headless_tls = headless_tls
        self._cache_ttl = cache_ttl
        self._capacity_timeout = capacity_timeout
        self._dns_resolver = dns_resolver
        self._random = randomizer or random.SystemRandom()
        self._cached: tuple[str, ...] = ()
        self._cached_at = 0.0
        self._lock = asyncio.Lock()

    @property
    def configured_urls(self) -> tuple[str, ...]:
        return self._static_urls

    async def resolve(self, session: aiohttp.ClientSession, *, token: str) -> tuple[str, ...]:
        now = time.monotonic()
        if self._cached and now - self._cached_at < self._cache_ttl:
            return self._cached
        async with self._lock:
            now = time.monotonic()
            if self._cached and now - self._cached_at < self._cache_ttl:
                return self._cached
            endpoints = await self._discover()
            ordered: tuple[str, ...]
            if len(endpoints) == 1:
                ordered = endpoints
            else:
                ordered = await self._order_by_capacity(session, endpoints, token=token)
            self._cached = ordered[:32]
            self._cached_at = time.monotonic()
            return self._cached

    def invalidate(self) -> None:
        self._cached = ()
        self._cached_at = 0.0

    async def _discover(self) -> tuple[str, ...]:
        if self._static_urls:
            return self._static_urls
        assert self._headless_dns is not None
        addresses = await self._dns_resolver(self._headless_dns, self._headless_port)
        scheme = "wss" if self._headless_tls else "ws"
        endpoints: list[str] = []
        for address in addresses[:64]:
            try:
                parsed = ipaddress.ip_address(address.split("%", 1)[0])
            except ValueError:
                continue
            host = f"[{address}]" if parsed.version == 6 else address
            endpoint = f"{scheme}://{host}:{self._headless_port}/v1/realtime"
            if endpoint not in endpoints:
                endpoints.append(endpoint)
        if not endpoints:
            raise ConnectionError(
                f"headless DNS returned no usable addresses: {self._headless_dns}"
            )
        return tuple(endpoints)

    async def _order_by_capacity(
        self,
        session: aiohttp.ClientSession,
        endpoints: tuple[str, ...],
        *,
        token: str,
    ) -> tuple[str, ...]:
        results = await asyncio.gather(
            *(self._capacity(session, endpoint, token=token) for endpoint in endpoints),
            return_exceptions=True,
        )
        available: list[CapacitySnapshot] = []
        full: list[CapacitySnapshot] = []
        unknown: list[str] = []
        for endpoint, result in zip(endpoints, results, strict=True):
            if isinstance(result, BaseException):
                unknown.append(endpoint)
            elif result.available > 0:
                available.append(result)
            else:
                full.append(result)
        self._random.shuffle(available)
        available.sort(key=lambda item: (item.active, -item.available))
        self._random.shuffle(full)
        return tuple([item.url for item in available] + unknown + [item.url for item in full])

    async def _capacity(
        self,
        session: aiohttp.ClientSession,
        endpoint: str,
        *,
        token: str,
    ) -> CapacitySnapshot:
        parsed = urlparse(endpoint)
        http_scheme = "https" if parsed.scheme == "wss" else "http"
        capacity_url = urlunparse((http_scheme, parsed.netloc, "/v1/capacity", "", "", ""))
        timeout = aiohttp.ClientTimeout(total=self._capacity_timeout)
        async with session.get(
            capacity_url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=timeout,
        ) as response:
            if response.status != 200:
                raise ConnectionError(f"capacity endpoint returned HTTP {response.status}")
            payload = await response.json()
        try:
            total = int(payload["total"])
            active = int(payload["active"])
            available = int(payload["available"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ConnectionError("capacity endpoint returned an invalid payload") from exc
        if not 1 <= total <= 64 or min(active, available) < 0 or active + available > total:
            raise ConnectionError("capacity endpoint returned impossible values")
        return CapacitySnapshot(
            url=endpoint,
            total=total,
            active=active,
            available=available,
        )


__all__ = ["CapacitySnapshot", "EndpointResolver"]
