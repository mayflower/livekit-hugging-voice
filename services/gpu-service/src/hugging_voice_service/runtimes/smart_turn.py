"""Pinned local Smart Turn v3.2 inference over bounded 16-kHz PCM audio."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

SMART_TURN_SAMPLE_RATE = 16_000
SMART_TURN_WINDOW_SECONDS = 8
SMART_TURN_WINDOW_SAMPLES = SMART_TURN_SAMPLE_RATE * SMART_TURN_WINDOW_SECONDS


@dataclass(frozen=True, slots=True)
class SmartTurnResult:
    probability: float


class SmartTurnRuntime:
    """One shared CPU-only ONNX runtime loaded once by the service lifecycle."""

    def __init__(self, model_path: Path) -> None:
        self._model_path = model_path
        self._session: Any | None = None
        self._feature_extractor: Any | None = None
        self.load_count = 0

    def load(self) -> None:
        if self._session is not None:
            return
        import onnxruntime as ort  # type: ignore[import-untyped]
        from transformers import WhisperFeatureExtractor

        options = ort.SessionOptions()
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            str(self._model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        if self._session.get_providers() != ["CPUExecutionProvider"]:
            raise RuntimeError("Smart Turn must use only CPUExecutionProvider")
        self._feature_extractor = WhisperFeatureExtractor(  # type: ignore[no-untyped-call]
            chunk_length=SMART_TURN_WINDOW_SECONDS
        )
        self.load_count += 1

    def warmup(self) -> None:
        self.predict_pcm16(bytes(SMART_TURN_SAMPLE_RATE * 2))

    def predict_pcm16(self, audio: bytes) -> SmartTurnResult:
        if self._session is None or self._feature_extractor is None:
            raise RuntimeError("Smart Turn runtime is not loaded")
        if len(audio) % 2:
            raise ValueError("Smart Turn requires complete PCM16 samples")

        import numpy as np

        samples = np.frombuffer(audio, dtype="<i2").astype(np.float32) / 32_768.0
        if samples.size > SMART_TURN_WINDOW_SAMPLES:
            samples = samples[-SMART_TURN_WINDOW_SAMPLES:]
        elif samples.size < SMART_TURN_WINDOW_SAMPLES:
            samples = np.pad(
                samples,
                (SMART_TURN_WINDOW_SAMPLES - samples.size, 0),
                mode="constant",
            )
        inputs = self._feature_extractor(
            samples,
            sampling_rate=SMART_TURN_SAMPLE_RATE,
            return_tensors="np",
            padding="max_length",
            max_length=SMART_TURN_WINDOW_SAMPLES,
            truncation=True,
            do_normalize=True,
        )
        features = np.asarray(inputs.input_features, dtype=np.float32)
        outputs = self._session.run(None, {"input_features": features})
        probability = float(np.asarray(outputs[0]).reshape(-1)[0])
        if not 0.0 <= probability <= 1.0:
            raise RuntimeError("Smart Turn returned an invalid probability")
        return SmartTurnResult(probability=probability)

    def close(self) -> None:
        self._session = None
        self._feature_extractor = None
