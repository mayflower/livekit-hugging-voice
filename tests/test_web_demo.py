from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest
from aiohttp import WSMsgType, web
from aiohttp.test_utils import TestClient, TestServer
from livekit import api

REPO_ROOT = Path(__file__).parents[1]


def load_example_module(name: str, relative_path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


web_demo = load_example_module("hugging_voice_web_demo", "examples/minimal-livekit-agent/web.py")
agent_demo = load_example_module(
    "hugging_voice_agent_demo", "examples/minimal-livekit-agent/agent.py"
)


def web_settings(**overrides: object) -> object:
    values = {
        "api_key": "testkey",
        "api_secret": "test-api-secret-that-is-more-than-32-bytes",
        "agent_name": "hugging-voice",
        "livekit_internal_url": "ws://127.0.0.1:1",
    }
    values.update(overrides)
    return web_demo.WebSettings(**values)


def test_join_response_contains_scoped_token_and_explicit_dispatch() -> None:
    selection = web_demo.parse_speech_selection(
        {"language": "en", "voice": "warm_female", "voice_instructions": "Speak warmly."}
    )

    response = web_demo.create_join_response(
        web_settings(), selection, request_host="voice.example:3000", request_secure=True
    )
    claims = api.TokenVerifier("testkey", "test-api-secret-that-is-more-than-32-bytes").verify(
        response["participant_token"]
    )

    assert response["server_url"] == "wss://voice.example:3000"
    assert response["room_name"].startswith("voice-")
    assert claims.video.room_join is True
    assert claims.video.room == response["room_name"]
    assert claims.room_config is not None
    dispatch = claims.room_config.agents[0]
    assert dispatch.agent_name == "hugging-voice"
    assert json.loads(dispatch.metadata) == {
        "language": "en",
        "voice": "warm_female",
        "voice_instructions": "Speak warmly.",
    }


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"unknown": "value"},
        {"language": "../en"},
        {"voice": "not a voice"},
        {"voice_instructions": "x" * 2_001},
    ],
)
def test_join_options_reject_invalid_input(payload: object) -> None:
    with pytest.raises(ValueError):
        web_demo.parse_speech_selection(payload)


@pytest.mark.asyncio
async def test_web_app_serves_ui_and_rejects_cross_origin_token_requests() -> None:
    async with TestClient(TestServer(web_demo.create_app(web_settings()))) as client:
        health = await client.get("/health")
        assert health.status == 200
        assert await health.json() == {"status": "ready"}

        index = await client.get("/")
        html = await index.text()
        assert index.status == 200
        assert "setMicrophoneEnabled" in html
        assert "lk.transcription" in html
        assert "RoomEvent.DataReceived" in html
        assert "hugging_voice.tool_call" in html
        assert "add_numbers · count_characters" in html
        assert "Content-Security-Policy" in index.headers
        inline_script = re.search(r"<script>(.*)</script>", html, flags=re.DOTALL)
        assert inline_script is not None
        script_hash = base64.b64encode(
            hashlib.sha256(inline_script.group(1).encode()).digest()
        ).decode()
        assert f"'sha256-{script_hash}'" in index.headers["Content-Security-Policy"]

        forbidden = await client.post(
            "/api/join", json={}, headers={"Origin": "https://untrusted.example"}
        )
        assert forbidden.status == 403


@pytest.mark.asyncio
async def test_livekit_websocket_proxy_bridges_text_and_binary() -> None:
    async def echo(request: web.Request) -> web.WebSocketResponse:
        assert request.headers["Authorization"] == "Bearer room-token"
        socket = web.WebSocketResponse()
        await socket.prepare(request)
        async for message in socket:
            if message.type is WSMsgType.TEXT:
                await socket.send_str(message.data)
            elif message.type is WSMsgType.BINARY:
                await socket.send_bytes(message.data)
        return socket

    upstream_app = web.Application()
    upstream_app.router.add_get("/rtc", echo)
    async with TestServer(upstream_app) as upstream:
        internal_url = str(upstream.make_url("/")).rstrip("/").replace("http://", "ws://")
        settings = web_settings(livekit_internal_url=internal_url)
        async with TestClient(TestServer(web_demo.create_app(settings))) as client:
            socket = await client.ws_connect(
                "/rtc?access_token=redacted", headers={"Authorization": "Bearer room-token"}
            )
            await socket.send_str("hello")
            assert (await socket.receive()).data == "hello"
            await socket.send_bytes(b"\x01\x02")
            assert (await socket.receive()).data == b"\x01\x02"
            await socket.close()


def test_agent_dispatch_metadata_overrides_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HUGGING_VOICE_LANGUAGE", "de")
    monkeypatch.setenv("HUGGING_VOICE_VOICE", "warm_male")

    options = agent_demo.speech_options(
        json.dumps({"language": "en", "voice": "clear_female", "voice_instructions": "Calm"})
    )

    assert options.language == "en"
    assert options.voice == "clear_female"
    assert options.voice_instructions == "Calm"


def test_agent_dispatch_metadata_rejects_unknown_options() -> None:
    with pytest.raises(ValueError, match="unknown speech option"):
        agent_demo.speech_options('{"locale":"en"}')


def test_demo_agent_exposes_two_native_function_tools() -> None:
    assert agent_demo.add_numbers.info.name == "add_numbers"
    assert agent_demo.count_characters.info.name == "count_characters"


@pytest.mark.asyncio
async def test_demo_tools_publish_visible_lifecycle_events() -> None:
    published: list[tuple[str, bool, str]] = []

    class LocalParticipant:
        async def publish_data(self, payload: str, *, reliable: bool, topic: str) -> None:
            published.append((payload, reliable, topic))

    class Room:
        local_participant = LocalParticipant()

    class Userdata:
        room = Room()

    class FunctionCall:
        def __init__(self, call_id: str, name: str) -> None:
            self.call_id = call_id
            self.name = name

    class Context:
        def __init__(self, call_id: str, name: str) -> None:
            self.userdata = Userdata()
            self.function_call = FunctionCall(call_id, name)

    assert await agent_demo.add_numbers(Context("call-1", "add_numbers"), a=2, b=3) == "5"
    assert (
        await agent_demo.count_characters(Context("call-2", "count_characters"), text="Grüße")
        == "5"
    )

    events = [json.loads(payload) for payload, _, _ in published]
    assert all(reliable for _, reliable, _ in published)
    assert all(topic == "hugging_voice.tool_call" for _, _, topic in published)
    assert [(event["call_id"], event["status"]) for event in events] == [
        ("call-1", "running"),
        ("call-1", "completed"),
        ("call-2", "running"),
        ("call-2", "completed"),
    ]
    assert events[1] == {
        "version": 1,
        "call_id": "call-1",
        "name": "add_numbers",
        "status": "completed",
        "arguments": {"a": 2, "b": 3},
        "result": "5",
    }
    assert events[3] == {
        "version": 1,
        "call_id": "call-2",
        "name": "count_characters",
        "status": "completed",
        "arguments": {"text": "Grüße"},
        "result": "5",
    }


@pytest.mark.asyncio
async def test_tool_display_event_rejects_oversized_payload() -> None:
    class LocalParticipant:
        async def publish_data(self, payload: str, *, reliable: bool, topic: str) -> None:
            raise AssertionError("oversized payload must not be published")

    class Room:
        local_participant = LocalParticipant()

    class Userdata:
        room = Room()

    class FunctionCall:
        call_id = "call-large"
        name = "test"

    class Context:
        userdata = Userdata()
        function_call = FunctionCall()

    with pytest.raises(ValueError, match="size limit"):
        await agent_demo._publish_tool_event(
            Context(), status="running", arguments={"text": "x" * 4_096}
        )
