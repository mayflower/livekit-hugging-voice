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
VADConfiguration = tuple[float, int, int, int, int]


@dataclass(frozen=True, slots=True)
class VADSignal:
    kind: Literal["speech_started", "speech_stopped"]
    sample_index: int


def _load_bundled_model() -> SileroModel:
    """Load the ONNX build, which holds the GIL once instead of per operator.

    Every 40 ms chunk runs this model through ``asyncio.to_thread``, on the same
    executor as final STT. The JIT build is many small torch ops, each cycling
    the GIL, so a Parakeet call — itself a long chain of GIL round trips around
    CUDA syncs — queues behind them. Measured in the deployed pod: 3 s of speech
    transcribes in 60 ms alone, but one VAD stream at 25 Hz pushed the same call
    to 1497 ms while the median stayed at 69 ms. ONNX Runtime releases the GIL
    for the whole graph and brought that peak back to 114 ms.

    Both builds ship inside the pinned ``silero-vad`` package, so this stays
    offline and within the verified revision.
    """

    import torch
    from silero_vad import load_silero_vad

    torch.set_num_threads(1)
    return cast(SileroModel, load_silero_vad(onnx=True))


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
        min_speech_ms: int = 160,
        min_speech_continuation_ms: int = 192,
        min_silence_ms: int = 500,
        speech_pad_ms: int = 200,
        model_factory: Callable[[], SileroModel] = _load_bundled_model,
        sample_tensor_factory: SampleTensorFactory = _to_torch_tensor,
    ) -> None:
        self._configuration: VADConfiguration = (
            threshold,
            min_speech_ms,
            min_speech_continuation_ms,
            min_silence_ms,
            speech_pad_ms,
        )
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

    @property
    def speech_candidate_active(self) -> bool:
        """Whether post-stop audio may be the beginning of resumed speech."""

        return self._candidate_start is not None

    @property
    def configuration(self) -> VADConfiguration:
        return self._configuration

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
    ) -> bool:
        if not 0.1 <= threshold <= 0.95:
            raise ValueError("Silero threshold is outside the supported range")
        configuration = (
            threshold,
            min_speech_ms,
            min_speech_continuation_ms,
            min_silence_ms,
            speech_pad_ms,
        )
        if configuration == self._configuration:
            return False
        self._configuration = configuration
        self._threshold = threshold
        self._min_speech_samples = self._ms_to_samples(min_speech_ms)
        self._continuation_samples = self._ms_to_samples(min_speech_continuation_ms)
        self._min_silence_samples = self._ms_to_samples(min_silence_ms)
        self._speech_pad_samples = self._ms_to_samples(speech_pad_ms)
        self.reset()
        return True

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
