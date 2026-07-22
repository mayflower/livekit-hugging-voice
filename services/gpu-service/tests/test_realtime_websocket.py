from __future__ import annotations

import asyncio
import hashlib
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI, WebSocket
from hugging_voice_service.app import create_app
from hugging_voice_service.auth import TokenAuthenticator
from hugging_voice_service.config import ServerSettings, ServiceSettings
from hugging_voice_service.lifecycle import LifecyclePhase, ServiceLifecycle
from hugging_voice_service.llama_process import LlamaProcessState
from hugging_voice_service.model_manifest import LockedFile, LockedModel, ModelLock
from hugging_voice_service.realtime import RealtimeService
from hugging_voice_service.runtimes.gemma import GemmaMessage, TextDelta, TextUsage
from hugging_voice_service.runtimes.silero import SessionVAD
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

REVISION = "b" * 40
TOKEN = "websocket-test-secret"


class ReadyLlama:
    state = LlamaProcessState.READY
    failure: str | None = None

    def __init__(self) -> None:
        self.failure_event = asyncio.Event()

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None


class TestParakeet:
    load_count = 1

    async def transcribe_partial(self, pcm16: bytes) -> str:
        del pcm16
        return "Hallo"

    async def transcribe_final(self, pcm16: bytes) -> str:
        del pcm16
        return "Hallo Welt"

    def load(self) -> None:
        return None

    def warmup(self) -> None:
        return None

    def close(self) -> None:
        return None


class TestQwen:
    load_count = 1

    async def stream_pcm16_frames(
        self,
        text: str,
        *,
        voice: str,
        cancelled: Callable[[], bool],
    ) -> AsyncIterator[bytes]:
        assert text.strip()
        assert voice == "de_standard_01"
        if not cancelled():
            yield bytes(960)

    def load(self) -> None:
        return None

    def warmup(self) -> None:
        return None

    def close(self) -> None:
        return None


class TestGemma:
    async def stream_response(
        self,
        *,
        messages: list[GemmaMessage],
        instructions: str = "",
        max_tokens: int = 256,
    ) -> AsyncIterator[TextDelta | TextUsage]:
        del instructions, max_tokens
        assert messages[-1].content == "Hallo"
        yield TextDelta("Guten Tag. ")
        yield TextUsage(prompt_tokens=3, completion_tokens=2, total_tokens=5)

    async def warmup(self) -> None:
        return None

    async def aclose(self) -> None:
        return None


class CanaryGemma(TestGemma):
    async def stream_response(
        self,
        *,
        messages: list[GemmaMessage],
        instructions: str = "",
        max_tokens: int = 256,
    ) -> AsyncIterator[TextDelta | TextUsage]:
        del instructions, max_tokens
        canary = messages[-1].content
        if "ALPHA" in canary:
            await asyncio.sleep(0.1)
            text = "ALPHA. "
        elif "BETA" in canary:
            text = "BETA. "
        else:
            raise AssertionError(f"missing canary in {canary!r}")
        yield TextDelta(text)
        yield TextUsage(prompt_tokens=2, completion_tokens=1, total_tokens=3)


class Probability:
    def item(self) -> float:
        return 0.0


class SilentVADModel:
    def __call__(self, samples: Any, sample_rate: int) -> Probability:
        del samples, sample_rate
        return Probability()

    def reset_states(self) -> None:
        return None


def locked_model(model_id: str, *, package: bool = False) -> LockedModel:
    if package:
        return LockedModel(
            delivery="python-package",
            id=model_id,
            source_repo="pypi:silero-vad",
            revision="6.2.1",
            files=(),
            license="MIT",
        )
    return LockedModel(
        delivery="huggingface",
        id=model_id,
        source_repo="test/model",
        revision=REVISION,
        files=(
            LockedFile(
                path="model.bin",
                size=1,
                sha256=hashlib.sha256(b"x").hexdigest(),
            ),
        ),
        license="Apache-2.0",
    )


def make_ready_service(tmp_path: Path, *, gemma: TestGemma | None = None) -> RealtimeService:
    token_file = tmp_path / "token"
    token_file.write_text(TOKEN, encoding="utf-8")
    settings = ServiceSettings(
        server=ServerSettings(
            token_file=token_file,
            max_sessions=2,
            drain_timeout_seconds=1.0,
        )
    )
    lifecycle = ServiceLifecycle(settings)
    lifecycle.authenticator = TokenAuthenticator.from_file(token_file)
    lifecycle.lock = ModelLock(
        models=(
            locked_model("google/gemma-4-31B-it"),
            locked_model("nvidia/parakeet-tdt-0.6b-v3"),
            locked_model("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"),
            locked_model("silero-vad", package=True),
        )
    )
    lifecycle.llama = ReadyLlama()
    lifecycle.parakeet = TestParakeet()
    lifecycle.qwen = TestQwen()
    lifecycle.gemma = gemma or TestGemma()
    lifecycle.phase = LifecyclePhase.READY
    lifecycle.telemetry.ready.set(1)
    return RealtimeService(
        lifecycle,
        vad_factory=lambda: SessionVAD(
            model_factory=SilentVADModel,
            sample_tensor_factory=lambda samples: samples,
        ),
    )


def make_test_app(service: RealtimeService) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        del app
        await service.start()
        try:
            yield
        finally:
            await service.aclose()

    app = FastAPI(lifespan=lifespan)

    @app.websocket("/v1/realtime")
    async def realtime(websocket: WebSocket) -> None:
        await service.handle_websocket(websocket)

    return app


def headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN}"}


def send_context_and_response(websocket: Any, session_id: str, canary: str) -> None:
    websocket.send_json(
        {
            "type": "conversation.item.create",
            "event_id": f"evt_context_{canary.lower()}",
            "protocol_version": 1,
            "session_id": session_id,
            "item": {
                "id": f"item_context_{canary.lower()}",
                "role": "user",
                "content": canary,
            },
        }
    )
    websocket.send_json(
        {
            "type": "response.create",
            "event_id": f"evt_response_{canary.lower()}",
            "protocol_version": 1,
            "session_id": session_id,
        }
    )


def receive_through_done(websocket: Any, initial: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events = initial
    while events[-1]["type"] != "response.done":
        events.append(websocket.receive_json())
    return events


@pytest.mark.asyncio
async def test_v1_operations_are_authenticated_and_report_realtime_state(tmp_path: Path) -> None:
    service = make_ready_service(tmp_path)
    await service.start()
    app = create_app(
        service.settings,
        lifecycle=service.lifecycle,
        realtime=service,
    )
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            assert (await client.get("/health/ready")).status_code == 200
            assert (await client.get("/v1/models")).status_code == 401
            capacity = (await client.get("/v1/capacity", headers=headers())).json()
            assert capacity == {
                "total": 2,
                "active": 0,
                "draining": 0,
                "stuck": 0,
                "available": 2,
            }
            assert (await client.get("/v1/pool", headers=headers())).status_code == 200
            usage = (await client.get("/v1/usage", headers=headers())).json()
            assert usage == {"active_sessions": 0, "active_responses": 0, "sessions": []}
            models = (await client.get("/v1/models", headers=headers())).json()
            assert len(models["models"]) == 4
    finally:
        await service.aclose()
        await service.lifecycle.aclose()


def test_authenticated_handshake_and_complete_text_audio_response(tmp_path: Path) -> None:
    service = make_ready_service(tmp_path)
    with TestClient(make_test_app(service)) as client:
        with client.websocket_connect(
            "/v1/realtime",
            headers=headers(),
            subprotocols=["hugging-voice-livekit.v1"],
        ) as websocket:
            created = websocket.receive_json()
            assert created["type"] == "session.created"
            assert created["revisions"] == {
                "vad": "6.2.1",
                "stt": REVISION,
                "llm": REVISION,
                "tts": REVISION,
            }
            session_id = created["session_id"]
            websocket.send_json(
                {
                    "type": "conversation.item.create",
                    "event_id": "evt_context",
                    "protocol_version": 1,
                    "session_id": session_id,
                    "item": {"id": "item_context", "role": "user", "content": "Hallo"},
                }
            )
            websocket.send_json(
                {
                    "type": "response.create",
                    "event_id": "evt_response",
                    "protocol_version": 1,
                    "session_id": session_id,
                }
            )

            events: list[dict[str, Any]] = []
            while not events or events[-1]["type"] != "response.done":
                events.append(websocket.receive_json())

            assert [event["type"] for event in events] == [
                "response.created",
                "response.output_text.delta",
                "response.output_text.done",
                "response.output_audio.delta",
                "response.output_audio.done",
                "response.done",
            ]
            assert events[-1]["status"] == "completed"
            assert events[-1]["usage"]["total_text_tokens"] == 5
            assert events[3]["sequence"] == 0


def test_auth_and_subprotocol_fail_before_any_capacity_claim(tmp_path: Path) -> None:
    service = make_ready_service(tmp_path)
    with TestClient(make_test_app(service)) as client:
        with client.websocket_connect(
            "/v1/realtime",
            subprotocols=["hugging-voice-livekit.v1"],
        ) as websocket:
            assert websocket.receive_json()["error"]["code"] == "authentication_failed"
            try:
                websocket.receive_json()
            except WebSocketDisconnect as exc:
                assert exc.code == 4401

        with client.websocket_connect(
            "/v1/realtime",
            headers=headers(),
            subprotocols=["wrong.v1"],
        ) as websocket:
            assert websocket.receive_json()["error"]["code"] == "invalid_configuration"
            try:
                websocket.receive_json()
            except WebSocketDisconnect as exc:
                assert exc.code == 4400


def test_invalid_event_is_structured_and_closed_as_protocol_error(tmp_path: Path) -> None:
    service = make_ready_service(tmp_path)
    with TestClient(make_test_app(service)) as client:
        with client.websocket_connect(
            "/v1/realtime",
            headers=headers(),
            subprotocols=["hugging-voice-livekit.v1"],
        ) as websocket:
            created = websocket.receive_json()
            websocket.send_json(
                {
                    "type": "not.supported",
                    "event_id": "evt_bad",
                    "protocol_version": 1,
                    "session_id": created["session_id"],
                }
            )
            assert websocket.receive_json()["error"]["code"] == "invalid_event"
            try:
                websocket.receive_json()
            except WebSocketDisconnect as exc:
                assert exc.code == 4400


def test_draining_service_rejects_new_sessions_with_1012(tmp_path: Path) -> None:
    service = make_ready_service(tmp_path)
    with TestClient(make_test_app(service)) as client:
        service.lifecycle.begin_drain()
        with client.websocket_connect(
            "/v1/realtime",
            headers=headers(),
            subprotocols=["hugging-voice-livekit.v1"],
        ) as websocket:
            assert websocket.receive_json()["error"]["code"] == "service_draining"
            try:
                websocket.receive_json()
            except WebSocketDisconnect as exc:
                assert exc.code == 1012


def test_third_connection_is_structurally_rejected_without_a_wait_queue(tmp_path: Path) -> None:
    service = make_ready_service(tmp_path)
    with TestClient(make_test_app(service)) as client:
        with client.websocket_connect(
            "/v1/realtime",
            headers=headers(),
            subprotocols=["hugging-voice-livekit.v1"],
        ) as first:
            assert first.receive_json()["type"] == "session.created"
            with client.websocket_connect(
                "/v1/realtime",
                headers=headers(),
                subprotocols=["hugging-voice-livekit.v1"],
            ) as second:
                assert second.receive_json()["type"] == "session.created"
                with client.websocket_connect(
                    "/v1/realtime",
                    headers=headers(),
                    subprotocols=["hugging-voice-livekit.v1"],
                ) as third:
                    error = third.receive_json()
                    assert error["error"]["code"] == "session_limit_reached"
                    try:
                        third.receive_json()
                    except WebSocketDisconnect as exc:
                        assert exc.code == 4429
            with client.websocket_connect(
                "/v1/realtime",
                headers=headers(),
                subprotocols=["hugging-voice-livekit.v1"],
            ) as replacement:
                assert replacement.receive_json()["type"] == "session.created"


def test_two_canary_sessions_are_isolated_and_one_cancel_cannot_cross(tmp_path: Path) -> None:
    service = make_ready_service(tmp_path, gemma=CanaryGemma())
    with TestClient(make_test_app(service)) as client:
        with client.websocket_connect(
            "/v1/realtime",
            headers=headers(),
            subprotocols=["hugging-voice-livekit.v1"],
        ) as alpha:
            alpha_session = alpha.receive_json()["session_id"]
            with client.websocket_connect(
                "/v1/realtime",
                headers=headers(),
                subprotocols=["hugging-voice-livekit.v1"],
            ) as beta:
                beta_session = beta.receive_json()["session_id"]
                send_context_and_response(alpha, alpha_session, "ALPHA")
                send_context_and_response(beta, beta_session, "BETA")
                alpha_created = alpha.receive_json()
                beta_created = beta.receive_json()
                assert alpha_created["type"] == beta_created["type"] == "response.created"

                alpha.send_json(
                    {
                        "type": "response.cancel",
                        "event_id": "evt_cancel_alpha",
                        "protocol_version": 1,
                        "session_id": alpha_session,
                        "response_id": alpha_created["response_id"],
                        "generation_id": alpha_created["generation_id"],
                    }
                )
                alpha_events = receive_through_done(alpha, [alpha_created])
                beta_events = receive_through_done(beta, [beta_created])

                assert alpha_events[-1]["reason"] == "client_cancelled"
                assert not any(
                    event["type"] == "response.output_text.delta" for event in alpha_events
                )
                assert beta_events[-1]["status"] == "completed"
                assert [
                    event["delta"]
                    for event in beta_events
                    if event["type"] == "response.output_text.delta"
                ] == ["BETA. "]
                for field in (
                    "session_id",
                    "turn_id",
                    "generation_id",
                    "response_id",
                    "item_id",
                ):
                    assert alpha_created[field] != beta_created[field]
