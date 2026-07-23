"""One shared configurable Qwen3-TTS GGML/CUDA runtime."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Iterator
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from hugging_voice_protocol.audio import OUTPUT_FRAME_BYTES, OUTPUT_SAMPLE_RATE

DEFAULT_LANGUAGE = "German"
DEFAULT_INSTRUCTIONS = (
    "A warm adult female native German speaker with authentic pronunciation and no foreign accent."
)

TTSMode = Literal["voice_clone", "voice_design"]

# One frozen operator-defined reference: (model language, recording, transcript).
VoiceReference = tuple[str, Path, str]


class QwenModel(Protocol):
    def warmup(self, *, prefill_len: int = 100) -> None: ...

    def generate_voice_design_streaming(
        self, **kwargs: Any
    ) -> Iterator[tuple[Any, int, dict[str, Any]]]: ...

    def generate_voice_clone_streaming(
        self, **kwargs: Any
    ) -> Iterator[tuple[Any, int, dict[str, Any]]]: ...


QwenFactory = Callable[[Path, Path], QwenModel]


def _require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3-TTS requires CUDA; CPU fallback is disabled")


# Extracted reference features land on the deployment's writable cache mount
# (tmpfs in Compose, emptyDir in Kubernetes); the container home is read-only.
VOICE_REF_CACHE_DIR = Path("/cache/qwen-tts-voice-refs")


def _load_local_model(talker_path: Path, codec_path: Path) -> QwenModel:
    from faster_qwen3_tts import GGMLQwen3TTS

    return cast(
        QwenModel,
        GGMLQwen3TTS.from_gguf(
            talker_path,
            codec_path,
            use_fa=True,
            voice_ref_cache_dir=VOICE_REF_CACHE_DIR,
        ),
    )


def _next_item(iterator: Iterator[Any]) -> tuple[bool, Any | None]:
    try:
        return True, next(iterator)
    except StopIteration:
        return False, None


def _close_iterator(iterator: Iterator[Any]) -> None:
    close = getattr(iterator, "close", None)
    if callable(close):
        close()


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
        mode: TTSMode = "voice_design",
        warmup_language: str = DEFAULT_LANGUAGE,
        warmup_instructions: str = DEFAULT_INSTRUCTIONS,
        voice_references: tuple[VoiceReference, ...] = (),
        do_sample: bool = True,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.05,
    ) -> None:
        if mode == "voice_clone" and not voice_references:
            raise ValueError("voice_clone mode requires the frozen voice references")
        self._talker_path = talker_path
        self._codec_path = codec_path
        self._model_factory = model_factory
        self._cuda_probe = cuda_probe
        self._mode: TTSMode = mode
        self._warmup_language = warmup_language
        self._warmup_instructions = warmup_instructions
        self._voice_references = voice_references
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
        for _language, recording, _text in self._voice_references:
            if not recording.is_file():
                raise FileNotFoundError(f"missing voice reference recording: {recording}")
        self._model = self._model_factory(self._talker_path, self._codec_path)
        self.load_count += 1

    def warmup(self) -> None:
        if self._model is None:
            raise RuntimeError("Qwen3-TTS runtime is not loaded")
        self._model.warmup(prefill_len=100)
        if self._mode == "voice_clone":
            # Extract every frozen reference once (speaker embedding + codec
            # tokens are cached per recording), so no session pays the
            # extraction inside its first response. The transcript itself is
            # the warmup text: it matches the recording, and a text this long
            # cannot legitimately sample an immediate end-of-speech token the
            # way a one-word warmup can.
            for language, recording, transcript in self._voice_references:
                self._pull_first_chunk(
                    self._raw_stream(
                        transcript,
                        language=language,
                        instructions="",
                        ref_audio=recording,
                        ref_text=transcript,
                    ),
                    detail=str(recording),
                )
            return
        self._pull_first_chunk(
            self._raw_stream(
                "Hallo.",
                language=self._warmup_language,
                instructions=self._warmup_instructions,
            ),
            detail="voice_design",
        )

    def _pull_first_chunk(
        self, iterator: Iterator[tuple[Any, int, dict[str, Any]]], *, detail: str
    ) -> None:
        try:
            try:
                chunk, sample_rate, _timing = next(iterator)
            except StopIteration as exc:
                raise RuntimeError(f"Qwen3-TTS warmup produced no audio ({detail})") from exc
            if sample_rate != self.sample_rate or self._to_pcm16(chunk) == b"":
                raise RuntimeError(f"Qwen3-TTS warmup produced invalid audio ({detail})")
        finally:
            _close_iterator(iterator)

    async def stream_pcm16_frames(
        self,
        text: str,
        *,
        language: str,
        instructions: str,
        ref_audio: Path | None = None,
        ref_text: str | None = None,
        cancelled: Callable[[], bool] = lambda: False,
    ) -> AsyncIterator[bytes]:
        if not text.strip():
            return
        iterator = self._raw_stream(
            text,
            language=language,
            instructions=instructions,
            ref_audio=ref_audio,
            ref_text=ref_text,
        )
        remainder = bytearray()
        try:
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
        finally:
            await asyncio.to_thread(_close_iterator, iterator)

    def close(self) -> None:
        self._model = None

    def _raw_stream(
        self,
        text: str,
        *,
        language: str,
        instructions: str,
        ref_audio: Path | None = None,
        ref_text: str | None = None,
    ) -> Iterator[tuple[Any, int, dict[str, Any]]]:
        if self._model is None:
            raise RuntimeError("Qwen3-TTS runtime is not loaded")
        sampling: dict[str, Any] = {
            "chunk_size": 12,
            "max_new_tokens": 2_048,
            "do_sample": self._do_sample,
            "temperature": self._temperature,
            "top_k": self._top_k,
            "top_p": self._top_p,
            "repetition_penalty": self._repetition_penalty,
        }
        if self._mode == "voice_clone":
            if ref_audio is None or not ref_text:
                raise RuntimeError("voice_clone synthesis requires a reference recording")
            # The frozen recording fully defines the speaker; passing an
            # additional instruct prompt would fight the cloned identity.
            return self._model.generate_voice_clone_streaming(
                text=text,
                language=language,
                ref_audio=str(ref_audio),
                ref_text=ref_text,
                **sampling,
            )
        return self._model.generate_voice_design_streaming(
            text=text,
            language=language,
            instruct=instructions,
            **sampling,
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
