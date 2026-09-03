"""One shared, CUDA-only nano-parakeet runtime loaded from a local NeMo file."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Protocol, cast


class ParakeetModel(Protocol):
    def transcribe(self, audio: Any, timestamps: bool = False) -> str: ...


ModelFactory = Callable[[Path], ParakeetModel]
ObserveSeconds = Callable[[float], None]


def _ignore_seconds(seconds: float) -> None:
    del seconds


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
    """Parakeet against second-aligned audio, because the shape sets the cost.

    Each distinct input length makes the encoder pick kernels and allocate for
    a tensor shape it has not seen, and that dominates the call: measured
    locally on 2-5 s of speech, a fresh length costs ~740 ms against ~20 ms for
    a repeated one. Turn audio is never twice the same length, so every turn
    paid it -- which is the 1.6-1.9 s the deployment measured while the same
    model transcribes a repeated buffer in 60 ms.

    Padding to the next whole second bounds the shapes to one per second, and
    ``warmup`` pays for all of them once. That cache is per thread, not per
    process -- on the default executor a shape stayed expensive until the
    particular worker had seen it -- so inference is pinned to one owned
    thread and warmed inside it. Silence at the end is transcribed as
    silence: the padded and unpadded transcripts matched on every reference
    recording tried. Rounding to a single fixed length would flatten the shapes
    further but did change transcripts, so the second is the smaller bucket
    that stays neutral.
    """

    model_id = "nvidia/parakeet-tdt-0.6b-v3"
    language = "de"
    sample_rate = 16_000
    compute_type = "float16"
    # Covers a conversational turn; anything longer pays for its bucket once.
    warm_bucket_seconds = 12

    def __init__(
        self,
        checkpoint: Path,
        observe_seconds: ObserveSeconds = _ignore_seconds,
        *,
        model_factory: ModelFactory = _load_local_model,
        cuda_probe: Callable[[], None] = _require_cuda,
    ) -> None:
        self._checkpoint = checkpoint
        self._observe_seconds = observe_seconds
        self._model_factory = model_factory
        self._cuda_probe = cuda_probe
        self._model: ParakeetModel | None = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="parakeet")
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
        self._executor.submit(self._warm_buckets).result()

    def _warm_buckets(self) -> None:
        for seconds in range(1, self.warm_bucket_seconds + 1):
            transcript = self._transcribe_pcm16(bytes(self._bucket_bytes(seconds)), observe=False)
            if not isinstance(transcript, str):
                raise RuntimeError("Parakeet warmup did not return text")

    async def transcribe_partial(self, pcm16: bytes) -> str:
        return await self._run(pcm16)

    async def transcribe_final(self, pcm16: bytes) -> str:
        return await self._run(pcm16)

    async def _run(self, pcm16: bytes) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._transcribe_pcm16, pcm16)

    def close(self) -> None:
        self._executor.shutdown(wait=True)
        self._model = None

    def _bucket_bytes(self, seconds: float) -> int:
        """Byte length of the whole-second bucket that holds ``seconds``."""

        return math.ceil(max(seconds, 1.0)) * self.sample_rate * 2

    def _transcribe_pcm16(self, pcm16: bytes, *, observe: bool = True) -> str:
        import numpy as np

        if self._model is None:
            raise RuntimeError("Parakeet runtime is not loaded")
        if len(pcm16) % 2:
            raise ValueError("Parakeet input must contain complete PCM16 samples")
        padded = pcm16.ljust(self._bucket_bytes(len(pcm16) / (self.sample_rate * 2)), b"\x00")
        audio = np.frombuffer(padded, dtype="<i2").astype(np.float32) / 32768.0
        started = time.perf_counter()
        result = self._model.transcribe(audio, timestamps=False)
        if observe:
            self._observe_seconds(time.perf_counter() - started)
        if not isinstance(result, str):
            raise RuntimeError("Parakeet returned an unexpected transcription result")
        return result.strip()
