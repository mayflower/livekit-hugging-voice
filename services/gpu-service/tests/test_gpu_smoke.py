from __future__ import annotations

import os
import wave
from pathlib import Path
from typing import cast

import pytest
from hugging_voice_service.config import load_settings
from hugging_voice_service.lifecycle import ServiceLifecycle
from hugging_voice_service.pipeline import GemmaStreamer
from hugging_voice_service.runtimes.llama_cpp_chat import GemmaMessage, TextDelta
from hugging_voice_service.schedulers.stt import STTRuntime
from hugging_voice_service.schedulers.tts import TTSRuntime
from hugging_voice_service.text_segmenter import SpeechTextSegmenter

REPO_ROOT = Path(__file__).parents[3]


def require_real_gpu_assets() -> None:
    if os.environ.get("HV_RUN_GPU_TESTS") != "1":
        pytest.skip("set HV_RUN_GPU_TESTS=1 to run real GPU model tests")
    lock = REPO_ROOT / "models" / "manifest.lock.json"
    model_root = REPO_ROOT / ".models"
    if not lock.is_file() or not model_root.is_dir():
        pytest.skip("real model lock/.models directory is unavailable")
    try:
        import torch
    except ImportError:
        pytest.skip("GPU optional dependencies are not installed")
    if not torch.cuda.is_available():
        pytest.skip("PyTorch cannot access an NVIDIA GPU")


def load_german_smoke_audio() -> bytes:
    configured = os.environ.get("HV_GPU_SMOKE_WAV")
    if not configured:
        pytest.skip("set HV_GPU_SMOKE_WAV to a short German 16 kHz mono PCM16 WAV")
    path = Path(configured)
    if not path.is_file():
        pytest.skip(f"HV_GPU_SMOKE_WAV does not exist: {path}")
    with wave.open(str(path), "rb") as source:
        if source.getparams()[:3] != (1, 2, 16_000) or source.getcomptype() != "NONE":
            pytest.fail("HV_GPU_SMOKE_WAV must be mono, PCM16, and 16 kHz")
        pcm16 = source.readframes(source.getnframes())
    if not pcm16:
        pytest.fail("HV_GPU_SMOKE_WAV contains no audio frames")
    return pcm16


@pytest.mark.gpu
@pytest.mark.asyncio
async def test_real_service_lifecycle_loads_and_warms_all_models() -> None:
    require_real_gpu_assets()
    settings = load_settings(REPO_ROOT / "services" / "gpu-service" / "config" / "default.yaml")
    if not settings.server.token_file.is_file():
        pytest.skip(f"mounted bearer secret is unavailable: {settings.server.token_file}")
    lifecycle = ServiceLifecycle(settings)
    try:
        await lifecycle.start()
        assert lifecycle.ready, lifecycle.error
        assert lifecycle.parakeet is not None and lifecycle.parakeet.load_count == 1
        assert lifecycle.qwen is not None and lifecycle.qwen.load_count == 1
    finally:
        await lifecycle.aclose()


@pytest.mark.gpu
@pytest.mark.asyncio
async def test_real_german_stt_gemma_and_tts_smoke() -> None:
    require_real_gpu_assets()
    pcm16 = load_german_smoke_audio()
    settings = load_settings(REPO_ROOT / "services" / "gpu-service" / "config" / "default.yaml")
    if not settings.server.token_file.is_file():
        pytest.skip(f"mounted bearer secret is unavailable: {settings.server.token_file}")
    lifecycle = ServiceLifecycle(settings)
    try:
        await lifecycle.start()
        assert lifecycle.ready, lifecycle.error
        assert lifecycle.parakeet is not None
        transcript = await cast(STTRuntime, lifecycle.parakeet).transcribe_final(pcm16)
        assert transcript.strip(), "real Parakeet transcription was empty"

        assert lifecycle.gemma is not None
        visible = ""
        async for event in cast(GemmaStreamer, lifecycle.gemma).stream_response(
            messages=[GemmaMessage(role="user", content=transcript)],
            slot_id=0,
        ):
            if isinstance(event, TextDelta):
                visible += event.text
        assert visible.strip(), "real Gemma stream had no visible response"
        assert "<think" not in visible.lower()

        assert lifecycle.qwen is not None
        tts = cast(TTSRuntime, lifecycle.qwen)
        language = settings.speech.resolve_language(settings.speech.default_language)
        voice = settings.speech.resolve_voice(settings.speech.default_voice)
        ref_audio = None
        ref_text = None
        if settings.speech.tts_mode == "voice_clone":
            reference = settings.speech.resolve_voice_reference(
                settings.speech.default_voice, settings.speech.default_language
            )
            ref_audio = settings.speech.voice_reference_path(reference)
            ref_text = reference.text
        frames: list[bytes] = []
        segmenter = SpeechTextSegmenter()
        for segment in segmenter.feed(visible) + segmenter.flush():
            async for frame in tts.stream_pcm16_frames(
                segment,
                language=language.model_language,
                instructions=voice.render(language.model_language),
                ref_audio=ref_audio,
                ref_text=ref_text,
                cancelled=lambda: False,
            ):
                frames.append(frame)
        assert frames and all(len(frame) == 960 for frame in frames)
    finally:
        await lifecycle.aclose()
