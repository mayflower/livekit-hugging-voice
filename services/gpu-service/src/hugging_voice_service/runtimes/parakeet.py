"""One shared, CUDA-only nano-parakeet runtime loaded from a local NeMo file."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, cast


class ParakeetModel(Protocol):
    def transcribe(self, audio: Any, timestamps: bool = False) -> str: ...


ModelFactory = Callable[[Path], ParakeetModel]


def _require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Parakeet requires CUDA; CPU fallback is disabled")


def _load_local_model(checkpoint: Path) -> ParakeetModel:
    import sentencepiece as spm
    import torch
    from nano_parakeet import ParakeetTDT
    from nano_parakeet._loader import (
        get_bundled_tokenizer_proto,
        load_nemo_state_dict,
        remap_state_dict,
    )

    model = ParakeetTDT()
    state = remap_state_dict(load_nemo_state_dict(str(checkpoint), map_location="cpu"))
    missing, _unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f"Parakeet checkpoint is missing keys: {missing[:10]}")
    model = model.to("cuda").eval()
    model.encoder.to(torch.float16)
    model.decoder.to(torch.float16)
    model.joint.to(torch.float16)
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.LoadFromSerializedProto(get_bundled_tokenizer_proto())
    model.sp = tokenizer
    model.warmup()
    return cast(ParakeetModel, model)


class ParakeetRuntime:
    model_id = "nvidia/parakeet-tdt-0.6b-v3"
    language = "de"
    sample_rate = 16_000
    compute_type = "float16"

    def __init__(
        self,
        checkpoint: Path,
        *,
        model_factory: ModelFactory = _load_local_model,
        cuda_probe: Callable[[], None] = _require_cuda,
    ) -> None:
        self._checkpoint = checkpoint
        self._model_factory = model_factory
        self._cuda_probe = cuda_probe
        self._model: ParakeetModel | None = None
        self.load_count = 0

    def load(self) -> None:
        if self._model is not None:
            raise RuntimeError("Parakeet runtime is already loaded")
        self._cuda_probe()
        if not self._checkpoint.is_file():
            raise FileNotFoundError(f"missing local Parakeet checkpoint: {self._checkpoint}")
        self._model = self._model_factory(self._checkpoint)
        self.load_count += 1

    def warmup(self) -> None:
        transcript = self._transcribe_pcm16(bytes(self.sample_rate * 2))
        if not isinstance(transcript, str):
            raise RuntimeError("Parakeet warmup did not return text")

    async def transcribe_partial(self, pcm16: bytes) -> str:
        return await asyncio.to_thread(self._transcribe_pcm16, pcm16)

    async def transcribe_final(self, pcm16: bytes) -> str:
        return await asyncio.to_thread(self._transcribe_pcm16, pcm16)

    def close(self) -> None:
        self._model = None

    def _transcribe_pcm16(self, pcm16: bytes) -> str:
        import numpy as np

        if self._model is None:
            raise RuntimeError("Parakeet runtime is not loaded")
        if len(pcm16) % 2:
            raise ValueError("Parakeet input must contain complete PCM16 samples")
        audio = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0
        result = self._model.transcribe(audio, timestamps=False)
        if not isinstance(result, str):
            raise RuntimeError("Parakeet returned an unexpected transcription result")
        return result.strip()
