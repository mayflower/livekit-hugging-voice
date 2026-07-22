"""Per-session Silero VAD using only the model bundled in the pinned package."""

from __future__ import annotations

import sys
from array import array
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast

from hugging_voice_protocol.audio import PCM16_BYTES_PER_SAMPLE


class SileroModel(Protocol):
    def __call__(self, samples: Any, sample_rate: int) -> Any: ...

    def reset_states(self) -> None: ...


SampleTensorFactory = Callable[[Any], Any]


@dataclass(frozen=True, slots=True)
class VADSignal:
    kind: Literal["speech_started", "speech_stopped"]
    sample_index: int


def _load_bundled_model() -> SileroModel:
    import torch
    from silero_vad import load_silero_vad

    torch.set_num_threads(1)
    return cast(SileroModel, load_silero_vad(onnx=False))


def _to_torch_tensor(samples: Any) -> Any:
    import torch

    return torch.tensor(samples, dtype=torch.float32).div_(32768.0)


class SessionVAD:
    """Stateful 512-sample VAD context owned by exactly one session."""

    sample_rate = 16_000
    window_samples = 512
    window_bytes = window_samples * PCM16_BYTES_PER_SAMPLE

    def __init__(
        self,
        *,
        threshold: float = 0.6,
        min_speech_ms: int = 384,
        min_speech_continuation_ms: int = 192,
        min_silence_ms: int = 500,
        speech_pad_ms: int = 30,
        model_factory: Callable[[], SileroModel] = _load_bundled_model,
        sample_tensor_factory: SampleTensorFactory = _to_torch_tensor,
    ) -> None:
        self._threshold = threshold
        self._min_speech_samples = self._ms_to_samples(min_speech_ms)
        self._continuation_samples = self._ms_to_samples(min_speech_continuation_ms)
        self._min_silence_samples = self._ms_to_samples(min_silence_ms)
        self._speech_pad_samples = self._ms_to_samples(speech_pad_ms)
        self._model = model_factory()
        self._sample_tensor_factory = sample_tensor_factory
        self._remainder = bytearray()
        self._processed_samples = 0
        self._candidate_start: int | None = None
        self._candidate_silence_start: int | None = None
        self._speech_start: int | None = None
        self._silence_start: int | None = None

    @classmethod
    def _ms_to_samples(cls, milliseconds: int) -> int:
        return cls.sample_rate * milliseconds // 1_000

    @property
    def buffered_bytes(self) -> int:
        return len(self._remainder)

    @property
    def speaking(self) -> bool:
        return self._speech_start is not None

    def process_pcm16(self, payload: bytes) -> list[VADSignal]:
        if len(payload) % PCM16_BYTES_PER_SAMPLE:
            raise ValueError("Silero input must contain complete PCM16 samples")
        self._remainder.extend(payload)
        signals: list[VADSignal] = []
        while len(self._remainder) >= self.window_bytes:
            window = bytes(self._remainder[: self.window_bytes])
            del self._remainder[: self.window_bytes]
            probability = self._speech_probability(window)
            signals.extend(self._advance(probability))
            self._processed_samples += self.window_samples
        return signals

    def flush(self) -> list[VADSignal]:
        signals: list[VADSignal] = []
        if self._speech_start is not None:
            signals.append(VADSignal("speech_stopped", self._processed_samples))
        self._clear_turn_state()
        self._remainder.clear()
        self._model.reset_states()
        return signals

    def reset(self) -> None:
        self._processed_samples = 0
        self._remainder.clear()
        self._clear_turn_state()
        self._model.reset_states()

    def configure(
        self,
        *,
        threshold: float,
        min_speech_ms: int,
        min_speech_continuation_ms: int,
        min_silence_ms: int,
        speech_pad_ms: int,
    ) -> None:
        if not 0.1 <= threshold <= 0.95:
            raise ValueError("Silero threshold is outside the supported range")
        self._threshold = threshold
        self._min_speech_samples = self._ms_to_samples(min_speech_ms)
        self._continuation_samples = self._ms_to_samples(min_speech_continuation_ms)
        self._min_silence_samples = self._ms_to_samples(min_silence_ms)
        self._speech_pad_samples = self._ms_to_samples(speech_pad_ms)
        self.reset()

    def _speech_probability(self, pcm16: bytes) -> float:
        samples = array("h")
        samples.frombytes(pcm16)
        if sys.byteorder != "little":
            samples.byteswap()
        tensor = self._sample_tensor_factory(samples)
        probability = float(self._model(tensor, self.sample_rate).item())
        if not 0.0 <= probability <= 1.0:
            raise RuntimeError(f"Silero returned invalid probability {probability}")
        return probability

    def _advance(self, probability: float) -> list[VADSignal]:
        window_start = self._processed_samples
        window_end = window_start + self.window_samples
        is_speech = probability >= self._threshold
        signals: list[VADSignal] = []

        if self._speech_start is None:
            if is_speech:
                if self._candidate_start is None:
                    self._candidate_start = window_start
                self._candidate_silence_start = None
                if window_end - self._candidate_start >= self._min_speech_samples:
                    self._speech_start = max(0, self._candidate_start - self._speech_pad_samples)
                    signals.append(VADSignal("speech_started", self._speech_start))
            elif self._candidate_start is not None:
                if self._candidate_silence_start is None:
                    self._candidate_silence_start = window_start
                if window_end - self._candidate_silence_start > self._continuation_samples:
                    self._candidate_start = None
                    self._candidate_silence_start = None
            return signals

        if is_speech:
            self._silence_start = None
            return signals
        if self._silence_start is None:
            self._silence_start = window_start
        if window_end - self._silence_start >= self._min_silence_samples:
            speech_end = min(window_end, self._silence_start + self._speech_pad_samples)
            signals.append(VADSignal("speech_stopped", speech_end))
            self._clear_turn_state()
        return signals

    def _clear_turn_state(self) -> None:
        self._candidate_start = None
        self._candidate_silence_start = None
        self._speech_start = None
        self._silence_start = None
