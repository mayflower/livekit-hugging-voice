from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hugging_voice_protocol.events import SpeechStoppedEvent
from hugging_voice_service.config import SemanticTurnSettings, SpeechSettings
from hugging_voice_service.pipeline import VoicePipeline
from hugging_voice_service.runtimes.smart_turn import (
    SMART_TURN_WINDOW_SAMPLES,
    SmartTurnResult,
    SmartTurnRuntime,
)
from hugging_voice_service.schedulers.turn import (
    TurnJob,
    TurnScheduler,
    TurnSchedulerFullError,
)
from hugging_voice_service.telemetry import ServiceTelemetry
from test_pipeline import (
    ImmediateGemma,
    ImmediateTTS,
    UnusedSTT,
    make_state,
)


class ImmediateTurnScheduler:
    def __init__(self, probability: float) -> None:
        self.probability = probability
        self.jobs: list[TurnJob] = []

    async def submit(self, job: TurnJob) -> SmartTurnResult | None:
        self.jobs.append(job)
        if job.is_stale():
            return None
        return SmartTurnResult(probability=self.probability)

    async def cancel_session(self, session_id: str) -> None:
        del session_id

    async def wait_session_idle(self, session_id: str) -> None:
        del session_id


def semantic_pipeline(
    probability: float,
    *,
    fallback_silence_ms: int = 500,
) -> tuple[VoicePipeline, ImmediateTurnScheduler, Any, Any]:
    state, transport = make_state()
    scheduler = ImmediateTurnScheduler(probability)
    pipeline = VoicePipeline(
        state,
        stt=UnusedSTT(),
        tts=ImmediateTTS(),
        gemma=ImmediateGemma(),
        speech=SpeechSettings(),
        semantic_turn=SemanticTurnSettings(
            mode="smart_turn_v3",
            fallback_silence_ms=fallback_silence_ms,
        ),
        turn_scheduler=scheduler,
        telemetry=ServiceTelemetry(),
    )
    state.current_turn_id = "turn_semantic"
    state.current_turn_revision = 0
    state.speech_start_sample = 0
    state.input_audio_buffer.append(bytes(16_000 * 2))
    return pipeline, scheduler, state, transport


@pytest.mark.asyncio
async def test_complete_semantic_candidate_commits_immediately() -> None:
    pipeline, scheduler, state, transport = semantic_pipeline(0.9)
    await pipeline._speech_stop_candidate(16_000)
    task = pipeline._turn_candidate_task
    assert task is not None
    await task
    await pipeline.wait_turns_idle()

    assert len(scheduler.jobs) == 1
    assert len(scheduler.jobs[0].audio) == 16_000 * 2
    assert state.speech_start_sample is None
    assert len([event for event in transport.events if isinstance(event, SpeechStoppedEvent)]) == 1
    await pipeline.drain()


@pytest.mark.asyncio
async def test_incomplete_candidate_keeps_turn_open_and_resume_cancels_fallback() -> None:
    pipeline, _, state, transport = semantic_pipeline(0.1)
    await pipeline._speech_stop_candidate(16_000)
    task = pipeline._turn_candidate_task
    assert task is not None
    await task
    assert pipeline._turn_fallback_task is not None

    await pipeline._speech_resumed()
    await asyncio.sleep(0.3)

    assert state.current_turn_id == "turn_semantic"
    assert state.speech_start_sample == 0
    assert not any(isinstance(event, SpeechStoppedEvent) for event in transport.events)
    await pipeline.drain()


@pytest.mark.asyncio
async def test_incomplete_candidate_uses_bounded_hard_silence_fallback() -> None:
    pipeline, _, state, transport = semantic_pipeline(0.1)
    await pipeline._speech_stop_candidate(16_000)
    task = pipeline._turn_candidate_task
    assert task is not None
    await task
    state.vad._candidate_start = 16_000
    await asyncio.sleep(0.3)
    await pipeline.wait_turns_idle()

    assert state.speech_start_sample is None
    assert len([event for event in transport.events if isinstance(event, SpeechStoppedEvent)]) == 1
    await pipeline.drain()


@pytest.mark.asyncio
async def test_fallback_does_not_cut_a_pending_resumption_candidate() -> None:
    pipeline, _, state, transport = semantic_pipeline(0.1)
    await pipeline._speech_stop_candidate(16_000)
    task = pipeline._turn_candidate_task
    assert task is not None
    await task
    state.vad._candidate_start = 16_000

    await asyncio.sleep(0.05)
    assert not any(isinstance(event, SpeechStoppedEvent) for event in transport.events)

    await pipeline._speech_resumed()
    await asyncio.sleep(0.3)
    assert not any(isinstance(event, SpeechStoppedEvent) for event in transport.events)
    await pipeline.drain()


def test_smart_turn_runtime_normalizes_truncates_and_uses_pinned_input_contract(
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}

    class Features:
        input_features = np.ones((1, 80, 800), dtype=np.float32)

    class Extractor:
        def __call__(self, samples: Any, **kwargs: Any) -> Features:
            captured["samples"] = samples
            captured["kwargs"] = kwargs
            return Features()

    class Session:
        def run(self, outputs: Any, inputs: dict[str, Any]) -> list[np.ndarray]:
            del outputs
            captured["features"] = inputs["input_features"]
            return [np.asarray([[0.75]], dtype=np.float32)]

    runtime = SmartTurnRuntime(tmp_path / "model.onnx")
    runtime._feature_extractor = Extractor()
    runtime._session = Session()
    audio = np.arange(SMART_TURN_WINDOW_SAMPLES + 100, dtype=np.int16).tobytes()

    result = runtime.predict_pcm16(audio)

    assert result.probability == pytest.approx(0.75)
    assert captured["samples"].shape == (SMART_TURN_WINDOW_SAMPLES,)
    assert captured["kwargs"]["sampling_rate"] == 16_000
    assert captured["features"].shape == (1, 80, 800)


def test_smart_turn_runtime_rejects_partial_pcm_sample(tmp_path: Path) -> None:
    runtime = SmartTurnRuntime(tmp_path / "model.onnx")
    runtime._feature_extractor = object()
    runtime._session = object()
    with pytest.raises(ValueError, match="complete PCM16"):
        runtime.predict_pcm16(b"\x00")


@pytest.mark.asyncio
async def test_turn_scheduler_is_bounded_and_round_robin_fair() -> None:
    entered = threading.Event()
    release = threading.Event()

    class Runtime:
        def __init__(self) -> None:
            self.order: list[int] = []

        def predict_pcm16(self, audio: bytes) -> SmartTurnResult:
            marker = audio[0]
            self.order.append(marker)
            if marker == 1:
                entered.set()
                release.wait(timeout=1.0)
            return SmartTurnResult(probability=1.0)

    def job(session: str, marker: int) -> TurnJob:
        return TurnJob(
            session_id=session,
            turn_id=f"turn_{marker}",
            turn_revision=0,
            candidate_id=marker,
            audio=bytes([marker, 0]),
            is_stale=lambda: False,
        )

    runtime = Runtime()
    scheduler = TurnScheduler(runtime, telemetry=ServiceTelemetry(), max_jobs=3)
    first = asyncio.create_task(scheduler.submit(job("session_x", 1)))
    await asyncio.wait_for(asyncio.to_thread(entered.wait), timeout=1.0)
    a1 = asyncio.create_task(scheduler.submit(job("session_a", 2)))
    a2 = asyncio.create_task(scheduler.submit(job("session_a", 3)))
    b1 = asyncio.create_task(scheduler.submit(job("session_b", 4)))
    await asyncio.sleep(0)
    with pytest.raises(TurnSchedulerFullError):
        await scheduler.submit(job("session_c", 5))
    release.set()
    await asyncio.gather(first, a1, a2, b1)
    await scheduler.aclose()

    assert runtime.order == [1, 2, 4, 3]


@pytest.mark.asyncio
async def test_cancelling_queued_turn_job_releases_scheduler_capacity() -> None:
    entered = threading.Event()
    release = threading.Event()

    class Runtime:
        def __init__(self) -> None:
            self.order: list[int] = []

        def predict_pcm16(self, audio: bytes) -> SmartTurnResult:
            marker = audio[0]
            self.order.append(marker)
            if marker == 1:
                entered.set()
                release.wait(timeout=1.0)
            return SmartTurnResult(probability=1.0)

    def job(session: str, marker: int) -> TurnJob:
        return TurnJob(
            session_id=session,
            turn_id=f"turn_{marker}",
            turn_revision=0,
            candidate_id=marker,
            audio=bytes([marker, 0]),
            is_stale=lambda: False,
        )

    runtime = Runtime()
    scheduler = TurnScheduler(runtime, telemetry=ServiceTelemetry(), max_jobs=1)
    active = asyncio.create_task(scheduler.submit(job("session_a", 1)))
    await asyncio.wait_for(asyncio.to_thread(entered.wait), timeout=1.0)
    cancelled = asyncio.create_task(scheduler.submit(job("session_b", 2)))
    await asyncio.sleep(0)
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled

    replacement = asyncio.create_task(scheduler.submit(job("session_b", 3)))
    await asyncio.sleep(0)
    release.set()
    await asyncio.gather(active, replacement)
    await scheduler.aclose()

    assert runtime.order == [1, 3]
