"""Small same-origin LiveKit voice demo and token issuer."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import ssl
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from secrets import token_urlsafe
from typing import Final
from urllib.parse import urlsplit

import aiohttp
from aiohttp import WSMsgType, web
from livekit import api

STATIC_ROOT: Final = Path(__file__).with_name("static")
LANGUAGE_PATTERN: Final = re.compile(r"^[A-Za-z]{2,8}(?:-[A-Za-z0-9]{1,8})*$")
VOICE_PATTERN: Final = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
MAX_BODY_BYTES: Final = 4_096
LOGGER: Final = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class WebSettings:
    api_key: str
    api_secret: str
    agent_name: str = "hugging-voice"
    livekit_internal_url: str = "ws://livekit:7880"
    livekit_public_url: str | None = None
    host: str = "0.0.0.0"
    port: int = 3_000
    tls_cert_file: Path | None = None
    tls_key_file: Path | None = None

    @classmethod
    def from_environment(cls) -> WebSettings:
        api_key = os.environ.get("LIVEKIT_API_KEY", "").strip()
        api_secret = os.environ.get("LIVEKIT_API_SECRET", "").strip()
        if not api_key or not api_secret:
            raise ValueError("LIVEKIT_API_KEY and LIVEKIT_API_SECRET are required")
        cert = os.environ.get("HUGGING_VOICE_WEB_TLS_CERT_FILE", "").strip()
        key = os.environ.get("HUGGING_VOICE_WEB_TLS_KEY_FILE", "").strip()
        if bool(cert) != bool(key):
            raise ValueError("both web TLS certificate and key files must be configured")
        port = int(os.environ.get("HUGGING_VOICE_WEB_INTERNAL_PORT", "3000"))
        if not 1 <= port <= 65_535:
            raise ValueError("web port must be between 1 and 65535")
        return cls(
            api_key=api_key,
            api_secret=api_secret,
            agent_name=os.environ.get("HUGGING_VOICE_AGENT_NAME", "hugging-voice").strip(),
            livekit_internal_url=os.environ.get("LIVEKIT_INTERNAL_URL", "ws://livekit:7880").rstrip(
                "/"
            ),
            livekit_public_url=(os.environ.get("LIVEKIT_PUBLIC_URL", "").strip() or None),
            host=os.environ.get("HUGGING_VOICE_WEB_HOST", "0.0.0.0"),
            port=port,
            tls_cert_file=Path(cert) if cert else None,
            tls_key_file=Path(key) if key else None,
        )


@dataclass(frozen=True, slots=True)
class SpeechSelection:
    language: str | None
    voice: str | None
    voice_instructions: str | None


SETTINGS_KEY: web.AppKey[WebSettings] = web.AppKey("settings", WebSettings)
HTTP_CLIENT_KEY: web.AppKey[aiohttp.ClientSession] = web.AppKey(
    "http_client", aiohttp.ClientSession
)


def parse_speech_selection(payload: object) -> SpeechSelection:
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")
    unknown = set(payload) - {"language", "voice", "voice_instructions"}
    if unknown:
        raise ValueError(f"unknown option: {sorted(unknown)[0]}")

    def optional_text(name: str, maximum: int) -> str | None:
        value = payload.get(name)
        if value is None or value == "":
            return None
        if not isinstance(value, str):
            raise ValueError(f"{name} must be a string")
        value = value.strip()
        if not value or len(value) > maximum:
            raise ValueError(f"{name} must contain between 1 and {maximum} characters")
        return value

    language = optional_text("language", 35)
    voice = optional_text("voice", 64)
    instructions = optional_text("voice_instructions", 2_000)
    if language is not None and LANGUAGE_PATTERN.fullmatch(language) is None:
        raise ValueError("language is not a valid configured language ID")
    if voice is not None and VOICE_PATTERN.fullmatch(voice) is None:
        raise ValueError("voice is not a valid configured voice ID")
    return SpeechSelection(language, voice, instructions)


def create_join_response(
    settings: WebSettings,
    selection: SpeechSelection,
    *,
    request_host: str,
    request_secure: bool,
) -> dict[str, str]:
    room_name = f"voice-{token_urlsafe(12)}"
    identity = f"web-{token_urlsafe(10)}"
    metadata = json.dumps(
        {
            key: value
            for key, value in {
                "language": selection.language,
                "voice": selection.voice,
                "voice_instructions": selection.voice_instructions,
            }.items()
            if value is not None
        },
        separators=(",", ":"),
    )
    token = (
        api.AccessToken(settings.api_key, settings.api_secret)
        .with_identity(identity)
        .with_name("Web voice user")
        .with_ttl(timedelta(minutes=10))
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=True,
                can_subscribe=True,
                can_publish_data=True,
            )
        )
        .with_room_config(
            api.RoomConfiguration(
                agents=[
                    api.RoomAgentDispatch(
                        agent_name=settings.agent_name,
                        metadata=metadata,
                    )
                ]
            )
        )
        .to_jwt()
    )
    if settings.livekit_public_url is not None:
        server_url = settings.livekit_public_url
    else:
        scheme = "wss" if request_secure else "ws"
        server_url = f"{scheme}://{request_host}"
    return {
        "server_url": server_url,
        "participant_token": token,
        "room_name": room_name,
        "identity": identity,
    }


def create_app(settings: WebSettings) -> web.Application:
    app = web.Application(client_max_size=MAX_BODY_BYTES)
    app[SETTINGS_KEY] = settings
    app.router.add_get("/", _index)
    app.router.add_get("/health", _health)
    app.router.add_post("/api/join", _join)
    app.router.add_route("*", "/rtc", _livekit_proxy)
    app.router.add_route("*", "/rtc/{tail:.*}", _livekit_proxy)
    app.on_startup.append(_startup)
    app.on_cleanup.append(_cleanup)
    return app


async def _index(request: web.Request) -> web.FileResponse:
    del request
    return web.FileResponse(
        STATIC_ROOT / "index.html",
        headers={
            "Cache-Control": "no-store",
            "Content-Security-Policy": (
                "default-src 'self'; script-src 'self' https://cdn.jsdelivr.net "
                "'sha256-D0z/DmnKPg1Q+CVk9cDu2kDBAX4D8bUQbF2RGarKLNw='; "
                "style-src 'self' 'unsafe-inline'; connect-src 'self' ws: wss:; "
                "media-src 'self' blob:; img-src 'self' data:"
            ),
            "X-Content-Type-Options": "nosniff",
            "Referrer-Policy": "no-referrer",
        },
    )


async def _health(request: web.Request) -> web.Response:
    del request
    return web.json_response({"status": "ready"})


async def _join(request: web.Request) -> web.Response:
    origin = request.headers.get("Origin")
    if origin is not None and urlsplit(origin).netloc != request.host:
        raise web.HTTPForbidden(text="cross-origin token requests are forbidden")
    try:
        payload = await request.json(loads=json.loads)
        selection = parse_speech_selection(payload)
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        raise web.HTTPBadRequest(text=str(exc)) from exc
    settings = request.app[SETTINGS_KEY]
    return web.json_response(
        create_join_response(
            settings,
            selection,
            request_host=request.host,
            request_secure=request.secure,
        )
    )


async def _livekit_proxy(request: web.Request) -> web.StreamResponse:
    if request.headers.get("Upgrade", "").lower() != "websocket":
        raise web.HTTPBadRequest(text="LiveKit signaling requires a WebSocket upgrade")
    settings = request.app[SETTINGS_KEY]
    client = request.app.get(HTTP_CLIENT_KEY)
    if not isinstance(client, aiohttp.ClientSession):
        raise web.HTTPServiceUnavailable(text="LiveKit proxy is not ready")
    protocols = tuple(
        protocol.strip()
        for protocol in request.headers.get("Sec-WebSocket-Protocol", "").split(",")
        if protocol.strip()
    )
    downstream = web.WebSocketResponse(protocols=protocols, max_msg_size=1 * 1024 * 1024)
    await downstream.prepare(request)
    upstream_url = f"{settings.livekit_internal_url}{request.rel_url}"
    upstream_headers = {
        name: value
        for name in ("Authorization", "Origin", "User-Agent")
        if (value := request.headers.get(name)) is not None
    }
    try:
        async with client.ws_connect(
            upstream_url,
            headers=upstream_headers,
            protocols=protocols,
            max_msg_size=1 * 1024 * 1024,
            autoclose=True,
            autoping=True,
        ) as upstream:
            await _bridge_websockets(downstream, upstream)
    except (aiohttp.ClientError, TimeoutError) as exc:
        await downstream.close(code=1011, message=b"LiveKit signaling unavailable")
        LOGGER.warning("LiveKit signaling proxy failed: %s", exc)
    return downstream


async def _bridge_websockets(
    downstream: web.WebSocketResponse,
    upstream: aiohttp.ClientWebSocketResponse,
) -> None:
    async def to_upstream() -> None:
        async for message in downstream:
            if message.type is WSMsgType.TEXT:
                await upstream.send_str(message.data)
            elif message.type is WSMsgType.BINARY:
                await upstream.send_bytes(message.data)
            elif message.type is WSMsgType.ERROR:
                break

    async def to_downstream() -> None:
        async for message in upstream:
            if message.type is WSMsgType.TEXT:
                await downstream.send_str(message.data)
            elif message.type is WSMsgType.BINARY:
                await downstream.send_bytes(message.data)
            elif message.type is WSMsgType.ERROR:
                break

    tasks = {asyncio.create_task(to_upstream()), asyncio.create_task(to_downstream())}
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    for task in done:
        task.result()


async def _startup(app: web.Application) -> None:
    app[HTTP_CLIENT_KEY] = aiohttp.ClientSession()


async def _cleanup(app: web.Application) -> None:
    client = app.get(HTTP_CLIENT_KEY)
    if isinstance(client, aiohttp.ClientSession):
        await client.close()


def _ssl_context(settings: WebSettings) -> ssl.SSLContext | None:
    if settings.tls_cert_file is None or settings.tls_key_file is None:
        return None
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(settings.tls_cert_file, settings.tls_key_file)
    return context


def main() -> None:
    settings = WebSettings.from_environment()
    web.run_app(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        ssl_context=_ssl_context(settings),
        access_log_format='%a "%r" %s %Tf',
    )


if __name__ == "__main__":
    main()
