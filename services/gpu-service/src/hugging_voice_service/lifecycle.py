"""Offline verification and concrete single-load model lifecycle."""

from __future__ import annotations

import asyncio
import logging
import wave
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Protocol, cast

import aiohttp

from .auth import TokenAuthenticator
from .config import ServiceSettings, SpeechSettings, TTSSettings
from .llama_process import LlamaProcess, LlamaProcessError, LlamaProcessState
from .llm_profiles import LLMProfile, resolve_llm_profile
from .model_manifest import LockedModel, ModelLock, load_lock, verify_lock
from .runtimes.llama_cpp_chat import LlamaCppChatRuntime
from .runtimes.parakeet import ParakeetRuntime
from .runtimes.qwen_tts import (
    QwenCudaGraphTTSRuntime,
    QwenTTSRuntime,
    QwenTTSRuntimePool,
)
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
    @property
    def load_count(self) -> int: ...

    def load(self) -> None: ...

    def warmup(self) -> None: ...

    def close(self) -> None: ...


class ManagedGemma(Protocol):
    async def warmup(self) -> None: ...

    async def aclose(self) -> None: ...


LlamaFactory = Callable[[Path, Path, ServiceSettings], ManagedLlama]
ParakeetFactory = Callable[[Path], ManagedParakeet]
QwenFactory = Callable[[Path, Path | None, SpeechSettings, TTSSettings], ManagedQwen]
GemmaFactory = Callable[[int, int, LLMProfile, Callable[[], None]], ManagedGemma]
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
    profile = resolve_llm_profile(settings.models.llm_profile)
    return LlamaProcess(
        binary=binary,
        model=model,
        profile=profile,
        port=settings.models.llama_port,
        parallel_slots=settings.models.llama_parallel_slots,
        context_size=settings.models.llama_context_size,
        flash_attention=settings.models.llama_flash_attention,
        continuous_batching=settings.models.llama_continuous_batching,
        batch_size=settings.models.llama_batch_size,
        ubatch_size=settings.models.llama_ubatch_size,
        cache_type_k=settings.models.llama_cache_type_k,
        cache_type_v=settings.models.llama_cache_type_v,
        cache_reuse=settings.models.llama_cache_reuse,
        metrics=settings.models.llama_metrics,
        startup_timeout=settings.models.llama_startup_timeout_seconds,
        shutdown_timeout=settings.models.llama_shutdown_timeout_seconds,
    )


def _gemma_factory(
    port: int,
    parallel_slots: int,
    profile: LLMProfile,
    violation: Callable[[], None],
) -> LlamaCppChatRuntime:
    return LlamaCppChatRuntime(
        port=port,
        profile=profile,
        parallel_slots=parallel_slots,
        reasoning_violation=violation,
    )


def _voice_references(speech: SpeechSettings) -> tuple[tuple[str, Path, str], ...]:
    references: list[tuple[str, Path, str]] = []
    for voice_id in speech.voices:
        for language_id, language_settings in speech.languages.items():
            reference = speech.resolve_voice_reference(voice_id, language_id)
            references.append(
                (
                    language_settings.model_language,
                    speech.voice_reference_path(reference),
                    reference.text,
                )
            )
    return tuple(references)


def _qwen_factory(
    primary: Path,
    secondary: Path | None,
    speech: SpeechSettings,
    tts: TTSSettings,
) -> QwenTTSRuntimePool:
    language = speech.resolve_language(speech.default_language)
    voice = speech.resolve_voice(speech.default_voice)
    references = _voice_references(speech) if speech.tts_mode == "voice_clone" else ()
    if tts.profile == "qwen3_tts_0_6b_cuda":
        return QwenTTSRuntimePool(
            [
                QwenCudaGraphTTSRuntime(
                    primary,
                    voice_references=references,
                    chunk_size=tts.chunk_size,
                    do_sample=speech.generation.do_sample,
                    temperature=speech.generation.temperature,
                    top_k=speech.generation.top_k,
                    top_p=speech.generation.top_p,
                    repetition_penalty=speech.generation.repetition_penalty,
                )
                for _ in range(tts.worker_count)
            ]
        )
    if secondary is None:
        raise RuntimeError("compatibility Qwen profile requires the shared codec")
    return QwenTTSRuntimePool(
        [
            QwenTTSRuntime(
                primary,
                secondary,
                mode=speech.tts_mode,
                warmup_language=language.model_language,
                warmup_instructions=voice.render(language.model_language),
                voice_references=references,
                do_sample=speech.generation.do_sample,
                temperature=speech.generation.temperature,
                top_k=speech.generation.top_k,
                top_p=speech.generation.top_p,
                repetition_penalty=speech.generation.repetition_penalty,
                chunk_size=tts.chunk_size,
            )
        ]
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
                if self.lock.profile_id != self.settings.profile_id:
                    raise RuntimeError(
                        "model lock profile mismatch: "
                        f"config={self.settings.profile_id!r} "
                        f"lock={self.lock.profile_id!r}"
                    )
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
                    artifacts["qwen_primary"],
                    artifacts.get("qwen_secondary"),
                    self.settings.speech,
                    self.settings.tts,
                )
                await asyncio.to_thread(self.qwen.load)
                await asyncio.to_thread(self.qwen.warmup)
                self.telemetry.model_loads.labels(model="qwen_tts").inc(
                    self.settings.tts.worker_count
                )

                self.phase = LifecyclePhase.WARMING_GEMMA
                self.gemma = self._gemma_factory(
                    self.settings.models.llama_port,
                    self.settings.models.llama_parallel_slots,
                    resolve_llm_profile(self.settings.models.llm_profile),
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
            "profile_id": self.settings.profile_id,
            "llama_cpp_commit": self.settings.models.llama_cpp_commit,
            "llm_profile": self.settings.models.llm_profile,
            "quantization": resolve_llm_profile(self.settings.models.llm_profile).quantization,
            "tts_profile": self.settings.tts.profile,
            "tts_worker_count": self.settings.tts.worker_count,
            "max_sessions": self.settings.server.max_sessions,
            "llama_parallel_slots": self.settings.models.llama_parallel_slots,
            "llama_context_size": self.settings.models.llama_context_size,
            "vad_min_silence_ms": self.settings.vad.min_silence_ms,
            "tts_chunk_size": self.settings.tts.chunk_size,
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

    async def llama_metrics(self) -> bytes:
        """Best-effort raw metrics from the loopback-only child process."""

        llama = self.llama
        scrape = None if llama is None else getattr(llama, "metrics", None)
        if scrape is None:
            return b""
        try:
            return cast(bytes, await scrape())
        except (aiohttp.ClientError, LlamaProcessError, TimeoutError) as exc:
            self.telemetry.llama_metrics_scrape_failures.inc()
            logger.warning("llama_metrics_scrape_failed", extra={"error": str(exc)})
            return b""

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
        tts_model_id = (
            "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
            if self.settings.tts.profile == "qwen3_tts_0_6b_cuda"
            else "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
        )
        llm_profile = resolve_llm_profile(self.settings.models.llm_profile)
        required = {
            llm_profile.model_id,
            "nvidia/parakeet-tdt-0.6b-v3",
            tts_model_id,
            "silero-vad",
        }
        missing = required - models.keys()
        if missing:
            raise RuntimeError(f"model lock is missing required entries: {sorted(missing)}")
        artifacts = {
            "gemma": self._named_file(
                models[llm_profile.model_id],
                llm_profile.local_artifact_key,
            ),
            "parakeet": self._only_file(models["nvidia/parakeet-tdt-0.6b-v3"]),
        }
        if self.settings.tts.profile == "qwen3_tts_0_6b_cuda":
            artifacts["qwen_primary"] = self.settings.models.root / tts_model_id
            return artifacts
        talker_file = QWEN_TALKER_FILES[self.settings.speech.tts_mode]
        artifacts.update(
            {
                "qwen_primary": self._named_file(
                    models[tts_model_id],
                    talker_file,
                ),
                "qwen_secondary": self._named_file(
                    models[tts_model_id],
                    "qwen-tokenizer-12hz-BF16.gguf",
                ),
            }
        )
        return artifacts

    def _only_file(self, model: LockedModel) -> Path:
        if len(model.files) != 1:
            raise RuntimeError(f"expected one locked file for {model.id}, got {len(model.files)}")
        return self.settings.models.root / model.id / model.files[0].path

    def _named_file(self, model: LockedModel, name: str) -> Path:
        match = next((file for file in model.files if file.path == name), None)
        if match is None:
            raise RuntimeError(f"model lock for {model.id} is missing {name}")
        return self.settings.models.root / model.id / match.path
