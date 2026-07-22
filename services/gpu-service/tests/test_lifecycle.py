from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
from pathlib import Path

import httpx
import pytest
from hugging_voice_service.app import create_app
from hugging_voice_service.config import ModelSettings, ServerSettings, ServiceSettings
from hugging_voice_service.lifecycle import LifecyclePhase, ServiceLifecycle
from hugging_voice_service.llama_process import LlamaProcessState
from hugging_voice_service.model_manifest import LockedFile, LockedModel, ModelLock, render_lock
from hugging_voice_service.realtime import RealtimeService

REVISION = "a" * 40


class FakeLlama:
    def __init__(self) -> None:
        self.state = LlamaProcessState.STOPPED
        self.failure: str | None = None
        self.failure_event = asyncio.Event()
        self.starts = 0
        self.stops = 0

    async def start(self) -> None:
        self.starts += 1
        self.state = LlamaProcessState.READY

    async def stop(self) -> None:
        self.stops += 1
        self.state = LlamaProcessState.STOPPED


class FakeBlockingRuntime:
    def __init__(self, *, fail_warmup: bool = False) -> None:
        self.load_count = 0
        self.warmups = 0
        self.closed = 0
        self.fail_warmup = fail_warmup

    def load(self) -> None:
        self.load_count += 1

    def warmup(self) -> None:
        self.warmups += 1
        if self.fail_warmup:
            raise RuntimeError("deliberate warmup failure")

    def close(self) -> None:
        self.closed += 1


class FakeGemma:
    def __init__(self) -> None:
        self.warmups = 0
        self.closed = 0

    async def warmup(self) -> None:
        self.warmups += 1

    async def aclose(self) -> None:
        self.closed += 1


def locked_file(root: Path, model_id: str, name: str, content: bytes) -> LockedFile:
    path = root / model_id / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return LockedFile(path=name, size=len(content), sha256=hashlib.sha256(content).hexdigest())


def make_settings_and_lock(tmp_path: Path) -> ServiceSettings:
    root = tmp_path / "models"
    models = (
        LockedModel(
            delivery="huggingface",
            id="google/gemma-4-31B-it",
            source_repo="ggml-org/gemma-4-31B-it-GGUF",
            revision=REVISION,
            files=(locked_file(root, "google/gemma-4-31B-it", "gemma.gguf", b"gemma"),),
            license="Apache-2.0",
        ),
        LockedModel(
            delivery="huggingface",
            id="nvidia/parakeet-tdt-0.6b-v3",
            source_repo="nvidia/parakeet-tdt-0.6b-v3",
            revision=REVISION,
            files=(
                locked_file(
                    root,
                    "nvidia/parakeet-tdt-0.6b-v3",
                    "parakeet-tdt-0.6b-v3.nemo",
                    b"parakeet",
                ),
            ),
            license="CC-BY-4.0",
        ),
        LockedModel(
            delivery="huggingface",
            id="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            source_repo="Serveurperso/Qwen3-TTS-GGUF",
            revision=REVISION,
            files=(
                locked_file(
                    root,
                    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                    "qwen-talker-1.7b-customvoice-BF16.gguf",
                    b"talker",
                ),
                locked_file(
                    root,
                    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                    "qwen-tokenizer-12hz-BF16.gguf",
                    b"codec",
                ),
            ),
            license="Apache-2.0",
        ),
        LockedModel(
            delivery="python-package",
            id="silero-vad",
            source_repo="pypi:silero-vad",
            revision="6.2.1",
            files=(),
            license="MIT",
        ),
    )
    lock_path = tmp_path / "manifest.lock.json"
    lock_path.write_text(render_lock(ModelLock(models=models)), encoding="utf-8")
    binary = tmp_path / "llama-server"
    binary.write_bytes(b"binary")
    token_file = tmp_path / "token"
    token_file.write_text("test-secret", encoding="utf-8")
    return ServiceSettings(
        server=ServerSettings(token_file=token_file),
        models=ModelSettings(
            root=root,
            lock_file=lock_path,
            llama_server_binary=binary,
        ),
    )


@pytest.mark.asyncio
async def test_lifecycle_loads_each_runtime_once_and_reports_real_readiness(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(importlib.metadata, "version", lambda package: "6.2.1")
    settings = make_settings_and_lock(tmp_path)
    llama = FakeLlama()
    parakeet = FakeBlockingRuntime()
    qwen = FakeBlockingRuntime()
    gemma = FakeGemma()
    lifecycle = ServiceLifecycle(
        settings,
        cuda_probe=lambda: None,
        llama_factory=lambda binary, model, config: llama,
        parakeet_factory=lambda checkpoint: parakeet,
        qwen_factory=lambda talker, codec: qwen,
        gemma_factory=lambda port, violation: gemma,
        gpu_memory_probe=lambda: 123_456,
    )
    await lifecycle.start()
    assert lifecycle.ready
    assert lifecycle.phase.value == LifecyclePhase.READY.value
    assert (llama.starts, parakeet.load_count, qwen.load_count, gemma.warmups) == (1, 1, 1, 1)
    metrics = lifecycle.telemetry.render().decode()
    assert 'hugging_voice_model_loads_total{model="gemma"} 1.0' in metrics
    assert 'hugging_voice_model_loads_total{model="parakeet"} 1.0' in metrics
    assert 'hugging_voice_model_loads_total{model="qwen_tts"} 1.0' in metrics

    realtime = RealtimeService(lifecycle)
    await realtime.start()
    app = create_app(settings, lifecycle=lifecycle, realtime=realtime)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        assert (await client.get("/health/live")).status_code == 200
        assert (await client.get("/health/ready")).status_code == 200
        models = (
            await client.get(
                "/v1/models",
                headers={"Authorization": "Bearer test-secret"},
            )
        ).json()
        assert models["llama_cpp_commit"] == settings.models.llama_cpp_commit
        assert models["quantization"] == "Q4_0"
        assert (await client.get("/metrics")).status_code == 200
        metrics = (await client.get("/metrics")).text
        assert "hugging_voice_gpu_memory_bytes 123456.0" in metrics

    await realtime.aclose()
    assert lifecycle.phase.value == LifecyclePhase.DRAINING.value
    await lifecycle.aclose()
    assert (llama.stops, parakeet.closed, qwen.closed, gemma.closed) == (1, 1, 1, 1)


@pytest.mark.asyncio
async def test_missing_cuda_fails_before_any_model_constructor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(importlib.metadata, "version", lambda package: "6.2.1")
    settings = make_settings_and_lock(tmp_path)
    constructed = 0

    def forbidden_factory(*args: object) -> FakeLlama:
        nonlocal constructed
        constructed += 1
        return FakeLlama()

    lifecycle = ServiceLifecycle(
        settings,
        cuda_probe=lambda: (_ for _ in ()).throw(RuntimeError("CUDA unavailable")),
        llama_factory=forbidden_factory,
    )
    await lifecycle.start()
    assert lifecycle.phase is LifecyclePhase.FAILED
    assert "CUDA unavailable" in (lifecycle.error or "")
    assert constructed == 0
    assert not lifecycle.ready


@pytest.mark.asyncio
async def test_missing_auth_secret_fails_before_model_verification(tmp_path: Path) -> None:
    settings = ServiceSettings(
        server=ServerSettings(token_file=tmp_path / "missing-token"),
        models=ModelSettings(
            root=tmp_path / "models",
            lock_file=tmp_path / "missing-lock.json",
            llama_server_binary=tmp_path / "llama-server",
        ),
    )
    lifecycle = ServiceLifecycle(settings)
    await lifecycle.start()

    assert lifecycle.phase is LifecyclePhase.FAILED
    assert "unable to read bearer token file" in (lifecycle.error or "")
    assert lifecycle.lock is None
    assert lifecycle.llama is None


@pytest.mark.asyncio
async def test_warmup_failure_cleans_started_resources_and_stays_unready(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(importlib.metadata, "version", lambda package: "6.2.1")
    settings = make_settings_and_lock(tmp_path)
    llama = FakeLlama()
    parakeet = FakeBlockingRuntime()
    qwen = FakeBlockingRuntime(fail_warmup=True)
    lifecycle = ServiceLifecycle(
        settings,
        cuda_probe=lambda: None,
        llama_factory=lambda binary, model, config: llama,
        parakeet_factory=lambda checkpoint: parakeet,
        qwen_factory=lambda talker, codec: qwen,
        gemma_factory=lambda port, violation: FakeGemma(),
    )
    await lifecycle.start()
    assert lifecycle.phase is LifecyclePhase.FAILED
    assert "deliberate warmup failure" in (lifecycle.error or "")
    assert (llama.stops, parakeet.closed, qwen.closed) == (1, 1, 1)
    assert not lifecycle.ready


@pytest.mark.asyncio
async def test_hash_failure_prevents_all_runtime_construction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(importlib.metadata, "version", lambda package: "6.2.1")
    settings = make_settings_and_lock(tmp_path)
    gemma_file = settings.models.root / "google/gemma-4-31B-it" / "gemma.gguf"
    gemma_file.write_bytes(b"tampered")
    constructors = 0

    def forbidden_factory(*args: object) -> FakeLlama:
        nonlocal constructors
        constructors += 1
        return FakeLlama()

    lifecycle = ServiceLifecycle(
        settings,
        cuda_probe=lambda: None,
        llama_factory=forbidden_factory,
    )
    await lifecycle.start()
    assert lifecycle.phase is LifecyclePhase.FAILED
    assert "model verification failed" in (lifecycle.error or "")
    assert constructors == 0


@pytest.mark.asyncio
async def test_unexpected_llama_exit_immediately_revokes_readiness(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(importlib.metadata, "version", lambda package: "6.2.1")
    settings = make_settings_and_lock(tmp_path)
    llama = FakeLlama()
    lifecycle = ServiceLifecycle(
        settings,
        cuda_probe=lambda: None,
        llama_factory=lambda binary, model, config: llama,
        parakeet_factory=lambda checkpoint: FakeBlockingRuntime(),
        qwen_factory=lambda talker, codec: FakeBlockingRuntime(),
        gemma_factory=lambda port, violation: FakeGemma(),
    )
    await lifecycle.start()
    assert lifecycle.phase.value == LifecyclePhase.READY.value
    llama.failure = "llama-server exited unexpectedly with code 9"
    llama.state = LlamaProcessState.FAILED
    llama.failure_event.set()
    await asyncio.sleep(0)
    assert lifecycle.phase is LifecyclePhase.FAILED
    assert not lifecycle.ready
    assert lifecycle.error == llama.failure
    await lifecycle.aclose()
