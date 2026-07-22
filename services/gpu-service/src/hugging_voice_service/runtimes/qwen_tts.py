"""One shared configurable Qwen3-TTS GGML/CUDA runtime."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Iterator
from pathlib import Path
from typing import Any, Protocol, cast

from hugging_voice_protocol.audio import OUTPUT_FRAME_BYTES, OUTPUT_SAMPLE_RATE

DEFAULT_LANGUAGE = "German"
DEFAULT_INSTRUCTIONS = (
    "A warm adult female native German speaker with authentic pronunciation and no foreign accent."
)


class QwenModel(Protocol):
    def warmup(self, *, prefill_len: int = 100) -> None: ...

    def generate_voice_design_streaming(
        self, **kwargs: Any
    ) -> Iterator[tuple[Any, int, dict[str, Any]]]: ...


QwenFactory = Callable[[Path, Path], QwenModel]


def _require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3-TTS requires CUDA; CPU fallback is disabled")


def _load_local_model(talker_path: Path, codec_path: Path) -> QwenModel:
    from faster_qwen3_tts import GGMLQwen3TTS

    return cast(
        QwenModel,
        GGMLQwen3TTS.from_gguf(
            talker_path,
            codec_path,
            use_fa=True,
        ),
    )


def _next_item(iterator: Iterator[Any]) -> tuple[bool, Any | None]:
    try:
        return True, next(iterator)
    except StopIteration:
        return False, None


class QwenTTSRuntime:
    model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    sample_rate = OUTPUT_SAMPLE_RATE

    def __init__(
        self,
        talker_path: Path,
        codec_path: Path,
        *,
        model_factory: QwenFactory = _load_local_model,
        cuda_probe: Callable[[], None] = _require_cuda,
        warmup_language: str = DEFAULT_LANGUAGE,
        warmup_instructions: str = DEFAULT_INSTRUCTIONS,
        do_sample: bool = False,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.05,
    ) -> None:
        self._talker_path = talker_path
        self._codec_path = codec_path
        self._model_factory = model_factory
        self._cuda_probe = cuda_probe
        self._warmup_language = warmup_language
        self._warmup_instructions = warmup_instructions
        self._do_sample = do_sample
        self._temperature = temperature
        self._top_k = top_k
        self._top_p = top_p
        self._repetition_penalty = repetition_penalty
        self._model: QwenModel | None = None
        self.load_count = 0

    def load(self) -> None:
        if self._model is not None:
            raise RuntimeError("Qwen3-TTS runtime is already loaded")
        self._cuda_probe()
        for path in (self._talker_path, self._codec_path):
            if not path.is_file():
                raise FileNotFoundError(f"missing local Qwen3-TTS artifact: {path}")
        self._model = self._model_factory(self._talker_path, self._codec_path)
        self.load_count += 1

    def warmup(self) -> None:
        if self._model is None:
            raise RuntimeError("Qwen3-TTS runtime is not loaded")
        self._model.warmup(prefill_len=100)
        iterator = self._raw_stream(
            "Hallo.",
            language=self._warmup_language,
            instructions=self._warmup_instructions,
        )
        try:
            chunk, sample_rate, _timing = next(iterator)
        except StopIteration as exc:
            raise RuntimeError("Qwen3-TTS warmup produced no audio") from exc
        if sample_rate != self.sample_rate or self._to_pcm16(chunk) == b"":
            raise RuntimeError("Qwen3-TTS warmup produced invalid audio")

    async def stream_pcm16_frames(
        self,
        text: str,
        *,
        language: str,
        instructions: str,
        cancelled: Callable[[], bool] = lambda: False,
    ) -> AsyncIterator[bytes]:
        if not text.strip():
            return
        iterator = self._raw_stream(
            text,
            language=language,
            instructions=instructions,
        )
        remainder = bytearray()
        while not cancelled():
            has_item, item = await asyncio.to_thread(_next_item, iterator)
            if not has_item:
                break
            if item is None:
                raise RuntimeError("Qwen3-TTS returned an empty stream item")
            chunk, sample_rate, _timing = item
            if sample_rate != self.sample_rate:
                raise RuntimeError(
                    f"Qwen3-TTS returned {sample_rate} Hz; expected {self.sample_rate} Hz"
                )
            remainder.extend(self._to_pcm16(chunk))
            while len(remainder) >= OUTPUT_FRAME_BYTES:
                if cancelled():
                    return
                yield bytes(remainder[:OUTPUT_FRAME_BYTES])
                del remainder[:OUTPUT_FRAME_BYTES]
        if remainder and not cancelled():
            remainder.extend(bytes(OUTPUT_FRAME_BYTES - len(remainder)))
            yield bytes(remainder)

    def close(self) -> None:
        self._model = None

    def _raw_stream(
        self,
        text: str,
        *,
        language: str,
        instructions: str,
    ) -> Iterator[tuple[Any, int, dict[str, Any]]]:
        if self._model is None:
            raise RuntimeError("Qwen3-TTS runtime is not loaded")
        return self._model.generate_voice_design_streaming(
            text=text,
            language=language,
            instruct=instructions,
            chunk_size=12,
            max_new_tokens=2_048,
            do_sample=self._do_sample,
            temperature=self._temperature,
            top_k=self._top_k,
            top_p=self._top_p,
            repetition_penalty=self._repetition_penalty,
        )

    @staticmethod
    def _to_pcm16(chunk: Any) -> bytes:
        import numpy as np

        audio = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if not np.all(np.isfinite(audio)):
            raise RuntimeError("Qwen3-TTS produced non-finite audio")
        return cast(
            bytes,
            np.rint(np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes(),
        )
