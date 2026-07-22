"""One shared Qwen3-TTS GGML/CUDA runtime with fixed German voice policy."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Iterator
from pathlib import Path
from typing import Any, Protocol, cast

from hugging_voice_protocol.audio import OUTPUT_FRAME_BYTES, OUTPUT_SAMPLE_RATE

PUBLIC_VOICE = "de_standard_01"
QWEN_SPEAKER = "Aiden"
QWEN_LANGUAGE = "German"
QWEN_INSTRUCTION = (
    "Sprich in klarem, ruhigem Hochdeutsch. Natürlich, freundlich und professionell. "
    "Keine übertriebene Emotionalität."
)


class QwenModel(Protocol):
    def warmup(self, *, prefill_len: int = 100) -> None: ...

    def generate_custom_voice_streaming(
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
    model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    public_voice = PUBLIC_VOICE
    speaker = QWEN_SPEAKER
    language = QWEN_LANGUAGE
    sample_rate = OUTPUT_SAMPLE_RATE

    def __init__(
        self,
        talker_path: Path,
        codec_path: Path,
        *,
        model_factory: QwenFactory = _load_local_model,
        cuda_probe: Callable[[], None] = _require_cuda,
    ) -> None:
        self._talker_path = talker_path
        self._codec_path = codec_path
        self._model_factory = model_factory
        self._cuda_probe = cuda_probe
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
        iterator = self._raw_stream("Hallo.")
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
        voice: str = PUBLIC_VOICE,
        cancelled: Callable[[], bool] = lambda: False,
    ) -> AsyncIterator[bytes]:
        if voice != self.public_voice:
            raise ValueError(f"unsupported voice {voice!r}; expected {self.public_voice!r}")
        if not text.strip():
            return
        iterator = self._raw_stream(text)
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

    def _raw_stream(self, text: str) -> Iterator[tuple[Any, int, dict[str, Any]]]:
        if self._model is None:
            raise RuntimeError("Qwen3-TTS runtime is not loaded")
        return self._model.generate_custom_voice_streaming(
            text=text,
            speaker=self.speaker,
            language=self.language,
            instruct=QWEN_INSTRUCTION,
            chunk_size=12,
            max_new_tokens=2_048,
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
