"""Offline verification and concrete single-load model lifecycle."""

from __future__ import annotations

import asyncio
import logging
import wave
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from .auth import TokenAuthenticator
from .config import ServiceSettings, SpeechSettings
from .llama_process import LlamaProcess, LlamaProcessState
from .model_manifest import LockedModel, ModelLock, load_lock, verify_lock
from .runtimes.gemma import GemmaRuntime
from .runtimes.parakeet import ParakeetRuntime
from .runtimes.qwen_tts import QwenTTSRuntime
from .telemetry import ServiceTelemetry

logger = logging.getLogger(__name__)

# Both talker variants come from the same locked model repository and share the
# codec. Only the base talker (default voice_clone mode) is in the shipped
# manifest; voice_design requires adding the VoiceDesign talker to
# models/manifest.yaml and re-running the prefetch.
QWEN_TALKER_FILES: dict[str, str] = {
    "voice_clone": "qwen-talker-1.7b-base-BF16.gguf",
    "voice_design": "qwen-talker-1.7b-voicedesign-BF16.gguf",
}


def _verify_voice_references(speech: SpeechSettings) -> None:
    """Fail startup early when a frozen voice reference is missing or corrupt."""

    if speech.tts_mode != "voice_clone":
        return
    for voice_id in speech.voices:
        for language in speech.languages:
            reference = speech.resolve_voice_reference(voice_id, language)
            path = speech.voice_reference_path(reference)
            try:
                with wave.open(str(path), "rb") as recording:
                    if recording.getnframes() <= 0:
                        raise RuntimeError(
                            f"voice reference {path} for {voice_id}/{language} is empty"
                        )
            except (OSError, wave.Error) as exc:
                raise RuntimeError(
                    f"voice reference {path} for {voice_id}/{language} is missing or invalid"
                ) from exc


class LifecyclePhase(StrEnum):
    CREATED = "created"
    LOADING_AUTH = "loading_auth"
    VERIFYING_MODELS = "verifying_models"
    CHECKING_CUDA = "checking_cuda"
    STARTING_LLAMA = "starting_llama"
    LOADING_PARAKEET = "loading_parakeet"
    LOADING_QWEN = "loading_qwen"
    WARMING_GEMMA = "warming_gemma"
    READY = "ready"
    DRAINING = "draining"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


class ManagedLlama(Protocol):
    state: LlamaProcessState
    failure: str | None
    failure_event: asyncio.Event

    async def start(self) -> None: ...

    async def stop(self) -> None: ...


class ManagedParakeet(Protocol):
    load_count: int

    def load(self) -> None: ...

    def warmup(self) -> None: ...

    def close(self) -> None: ...


class ManagedQwen(Protocol):
    load_count: int

    def load(self) -> None: ...

    def warmup(self) -> None: ...

    def close(self) -> None: ...


class ManagedGemma(Protocol):
    async def warmup(self) -> None: ...

    async def aclose(self) -> None: ...


LlamaFactory = Callable[[Path, Path, ServiceSettings], ManagedLlama]
ParakeetFactory = Callable[[Path], ManagedParakeet]
QwenFactory = Callable[[Path, Path, SpeechSettings], ManagedQwen]
GemmaFactory = Callable[[int, Callable[[], None]], ManagedGemma]
CudaProbe = Callable[[], None]
GpuMemoryProbe = Callable[[], int]


def _require_cuda() -> None:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("GPU runtime dependencies are not installed") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; CPU fallback is disabled")


def _gpu_memory_bytes() -> int:
    import torch

    free, total = torch.cuda.mem_get_info()
    return int(total - free)


def _llama_factory(binary: Path, model: Path, settings: ServiceSettings) -> LlamaProcess:
    return LlamaProcess(
        binary=binary,
        model=model,
        port=settings.models.llama_port,
        parallel_slots=settings.models.llama_parallel_slots,
        context_size=settings.models.llama_context_size,
        startup_timeout=settings.models.llama_startup_timeout_seconds,
        shutdown_timeout=settings.models.llama_shutdown_timeout_seconds,
    )


def _gemma_factory(port: int, violation: Callable[[], None]) -> GemmaRuntime:
    return GemmaRuntime(port=port, reasoning_violation=violation)


def _qwen_factory(talker: Path, codec: Path, speech: SpeechSettings) -> QwenTTSRuntime:
    language = speech.resolve_language(speech.default_language)
    voice = speech.resolve_voice(speech.default_voice)
    voice_references: list[tuple[str, Path, str]] = []
    if speech.tts_mode == "voice_clone":
        for voice_id in speech.voices:
            for language_id, language_settings in speech.languages.items():
                reference = speech.resolve_voice_reference(voice_id, language_id)
                voice_references.append(
                    (
                        language_settings.model_language,
                        speech.voice_reference_path(reference),
                        reference.text,
                    )
                )
    return QwenTTSRuntime(
        talker,
        codec,
        mode=speech.tts_mode,
        warmup_language=language.model_language,
        warmup_instructions=voice.render(language.model_language),
        voice_references=tuple(voice_references),
        do_sample=speech.generation.do_sample,
        temperature=speech.generation.temperature,
        top_k=speech.generation.top_k,
        top_p=speech.generation.top_p,
        repetition_penalty=speech.generation.repetition_penalty,
    )


class ServiceLifecycle:
    """Create each expensive runtime once and expose readiness from real state."""

    def __init__(
        self,
        settings: ServiceSettings,
        *,
        telemetry: ServiceTelemetry | None = None,
        cuda_probe: CudaProbe = _require_cuda,
        llama_factory: LlamaFactory = _llama_factory,
        parakeet_factory: ParakeetFactory = ParakeetRuntime,
        qwen_factory: QwenFactory = _qwen_factory,
        gemma_factory: GemmaFactory = _gemma_factory,
        gpu_memory_probe: GpuMemoryProbe = _gpu_memory_bytes,
    ) -> None:
        self.settings = settings
        self.telemetry = telemetry or ServiceTelemetry()
        self._cuda_probe = cuda_probe
        self._llama_factory = llama_factory
        self._parakeet_factory = parakeet_factory
        self._qwen_factory = qwen_factory
        self._gemma_factory = gemma_factory
        self._gpu_memory_probe = gpu_memory_probe
        self._gpu_memory_observation_failed = False
        self.phase = LifecyclePhase.CREATED
        self.error: str | None = None
        self.lock: ModelLock | None = None
        self.authenticator: TokenAuthenticator | None = None
        self.llama: ManagedLlama | None = None
        self.parakeet: ManagedParakeet | None = None
        self.qwen: ManagedQwen | None = None
        self.gemma: ManagedGemma | None = None
        self._llama_monitor: asyncio.Task[None] | None = None
        self._start_lock = asyncio.Lock()

    @property
    def ready(self) -> bool:
        return (
            self.phase is LifecyclePhase.READY
            and self.llama is not None
            and self.llama.state is LlamaProcessState.READY
        )

    @property
    def live(self) -> bool:
        return self.phase is not LifecyclePhase.FAILED

    async def start(self) -> None:
        async with self._start_lock:
            if self.phase is not LifecyclePhase.CREATED:
                raise RuntimeError(f"cannot start service lifecycle from {self.phase}")
            try:
                self.phase = LifecyclePhase.LOADING_AUTH
                self.authenticator = await asyncio.to_thread(
                    TokenAuthenticator.from_file,
                    self.settings.server.token_file,
                )
                self.phase = LifecyclePhase.VERIFYING_MODELS
                self.lock = await asyncio.to_thread(load_lock, self.settings.models.lock_file)
                await asyncio.to_thread(
                    verify_lock,
                    self.lock,
                    self.settings.models.root,
                )
                await asyncio.to_thread(_verify_voice_references, self.settings.speech)
                artifacts = self._required_artifacts(self.lock)

                self.phase = LifecyclePhase.CHECKING_CUDA
                await asyncio.to_thread(self._cuda_probe)

                self.phase = LifecyclePhase.STARTING_LLAMA
                self.llama = self._llama_factory(
                    self.settings.models.llama_server_binary,
                    artifacts["gemma"],
                    self.settings,
                )
                await self.llama.start()
                self.telemetry.model_loads.labels(model="gemma").inc()
                self._llama_monitor = asyncio.create_task(self._monitor_llama())

                self.phase = LifecyclePhase.LOADING_PARAKEET
                self.parakeet = self._parakeet_factory(artifacts["parakeet"])
                await asyncio.to_thread(self.parakeet.load)
                await asyncio.to_thread(self.parakeet.warmup)
                self.telemetry.model_loads.labels(model="parakeet").inc()

                self.phase = LifecyclePhase.LOADING_QWEN
                self.qwen = self._qwen_factory(
                    artifacts["qwen_talker"],
                    artifacts["qwen_codec"],
                    self.settings.speech,
                )
                await asyncio.to_thread(self.qwen.load)
                await asyncio.to_thread(self.qwen.warmup)
                self.telemetry.model_loads.labels(model="qwen_tts").inc()

                self.phase = LifecyclePhase.WARMING_GEMMA
                self.gemma = self._gemma_factory(
                    self.settings.models.llama_port,
                    self.telemetry.reasoning_violations.inc,
                )
                await self.gemma.warmup()

                self.phase = LifecyclePhase.READY
                self.telemetry.ready.set(1)
            except asyncio.CancelledError:
                self.error = "startup cancelled"
                failed_stage = self.phase.value
                self.phase = LifecyclePhase.FAILED
                self.telemetry.ready.set(0)
                await self._close_resources()
                raise
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"
                failed_stage = self.phase.value
                self.telemetry.lifecycle_failures.labels(stage=failed_stage).inc()
                self.phase = LifecyclePhase.FAILED
                self.telemetry.ready.set(0)
                await self._close_resources()
                logger.exception("service_startup_failed", extra={"stage": failed_stage})

    def begin_drain(self) -> None:
        if self.phase is LifecyclePhase.READY:
            self.phase = LifecyclePhase.DRAINING
            self.telemetry.ready.set(0)

    async def aclose(self) -> None:
        if self.phase is LifecyclePhase.STOPPED:
            return
        self.telemetry.ready.set(0)
        self.phase = LifecyclePhase.STOPPING
        await self._close_resources()
        self.phase = LifecyclePhase.STOPPED

    def model_report(self) -> dict[str, object]:
        return {
            "llama_cpp_commit": self.settings.models.llama_cpp_commit,
            "quantization": "Q4_0",
            "models": []
            if self.lock is None
            else [model.model_dump(mode="json") for model in self.lock.models],
            "phase": self.phase,
        }

    async def observe_gpu_memory(self) -> None:
        """Refresh device-wide allocated memory without affecting readiness."""
        if self.phase not in {LifecyclePhase.READY, LifecyclePhase.DRAINING}:
            return
        try:
            observed = await asyncio.to_thread(self._gpu_memory_probe)
            if observed < 0:
                raise ValueError("GPU memory probe returned a negative value")
        except (ImportError, RuntimeError, ValueError) as exc:
            if not self._gpu_memory_observation_failed:
                logger.warning("gpu_memory_observation_unavailable", extra={"error": str(exc)})
                self._gpu_memory_observation_failed = True
            return
        self.telemetry.gpu_memory_bytes.set(observed)

    async def _monitor_llama(self) -> None:
        llama = self.llama
        if llama is None:
            return
        await llama.failure_event.wait()
        if self.phase not in {LifecyclePhase.STOPPING, LifecyclePhase.STOPPED}:
            self.error = llama.failure or "llama-server failed"
            self.phase = LifecyclePhase.FAILED
            self.telemetry.ready.set(0)
            self.telemetry.lifecycle_failures.labels(stage="llama_runtime").inc()

    async def _close_resources(self) -> None:
        if self.gemma is not None:
            await self.gemma.aclose()
            self.gemma = None
        if self.qwen is not None:
            await asyncio.to_thread(self.qwen.close)
            self.qwen = None
        if self.parakeet is not None:
            await asyncio.to_thread(self.parakeet.close)
            self.parakeet = None
        if self.llama is not None:
            await self.llama.stop()
            self.llama = None
        if self._llama_monitor is not None:
            self._llama_monitor.cancel()
            await asyncio.gather(self._llama_monitor, return_exceptions=True)
            self._llama_monitor = None

    def _required_artifacts(self, lock: ModelLock) -> dict[str, Path]:
        models = {model.id: model for model in lock.models}
        required = {
            "google/gemma-4-31B-it",
            "nvidia/parakeet-tdt-0.6b-v3",
            "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
            "silero-vad",
        }
        missing = required - models.keys()
        if missing:
            raise RuntimeError(f"model lock is missing required entries: {sorted(missing)}")
        talker_file = QWEN_TALKER_FILES[self.settings.speech.tts_mode]
        return {
            "gemma": self._only_file(models["google/gemma-4-31B-it"]),
            "parakeet": self._only_file(models["nvidia/parakeet-tdt-0.6b-v3"]),
            "qwen_talker": self._named_file(
                models["Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"],
                talker_file,
            ),
            "qwen_codec": self._named_file(
                models["Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"],
                "qwen-tokenizer-12hz-BF16.gguf",
            ),
        }

    def _only_file(self, model: LockedModel) -> Path:
        if len(model.files) != 1:
            raise RuntimeError(f"expected one locked file for {model.id}, got {len(model.files)}")
        return self.settings.models.root / model.id / model.files[0].path

    def _named_file(self, model: LockedModel, name: str) -> Path:
        match = next((file for file in model.files if file.path == name), None)
        if match is None:
            raise RuntimeError(f"model lock for {model.id} is missing {name}")
        return self.settings.models.root / model.id / match.path
