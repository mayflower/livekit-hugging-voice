from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable

import pytest
from hugging_voice_service.cancellation import GenerationToken
from hugging_voice_service.schedulers.stt import (
    SchedulerFullError as STTFullError,
)
from hugging_voice_service.schedulers.stt import (
    STTJob,
    STTScheduler,
)
from hugging_voice_service.schedulers.tts import (
    SchedulerFullError as TTSFullError,
)
from hugging_voice_service.schedulers.tts import (
    TTSJob,
    TTSScheduler,
)
from hugging_voice_service.telemetry import ServiceTelemetry


class RecordingSTT:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def transcribe_partial(self, pcm16: bytes) -> str:
        value = pcm16.decode()
        self.calls.append(f"partial:{value}")
        return value

    async def transcribe_final(self, pcm16: bytes) -> str:
        value = pcm16.decode()
        self.calls.append(f"final:{value}")
        return value


class RecordingTTS:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def stream_pcm16_frames(
        self,
        text: str,
        *,
        voice: str,
        cancelled: Callable[[], bool],
    ) -> AsyncIterator[bytes]:
        assert voice == "de_standard_01"
        self.calls.append(text)
        if not cancelled():
            yield bytes(960)


def stt_job(session: str, value: str, *, final: bool) -> STTJob:
    return STTJob(
        session_id=session,
        turn_id=f"turn_{value}",
        turn_revision=0,
        audio=value.encode(),
        final=final,
        is_stale=lambda: False,
    )


def token(session: str, generation: str) -> GenerationToken:
    return GenerationToken(
        session_id=session,
        turn_id=f"turn_{generation}",
        turn_revision=0,
        generation_id=f"gen_{generation}",
        epoch=1,
    )


def tts_job(session: str, value: str, frames: list[bytes] | None = None) -> TTSJob:
    async def record(frame: bytes) -> None:
        if frames is not None:
            frames.append(frame)

    return TTSJob(
        token=token(session, value),
        text=value,
        voice="de_standard_01",
        is_current=lambda: True,
        on_frame=record,
    )


@pytest.mark.asyncio
async def test_stt_is_final_first_and_round_robin_within_each_class() -> None:
    runtime = RecordingSTT()
    scheduler = STTScheduler(runtime, telemetry=ServiceTelemetry())
    tasks = [
        asyncio.create_task(scheduler.submit_partial(stt_job("session_a", "pa", final=False))),
        asyncio.create_task(scheduler.submit_partial(stt_job("session_b", "pb", final=False))),
        asyncio.create_task(scheduler.submit_final(stt_job("session_a", "fa", final=True))),
        asyncio.create_task(scheduler.submit_final(stt_job("session_b", "fb", final=True))),
    ]
    await asyncio.sleep(0)
    await scheduler.start()
    try:
        assert await asyncio.gather(*tasks) == ["pa", "pb", "fa", "fb"]
        assert runtime.calls == ["final:fa", "final:fb", "partial:pa", "partial:pb"]
    finally:
        await scheduler.aclose()


@pytest.mark.asyncio
async def test_stt_drops_only_partials_but_rejects_final_queue_overflow() -> None:
    scheduler = STTScheduler(RecordingSTT(), telemetry=ServiceTelemetry(), max_jobs=1)
    first_partial = asyncio.create_task(
        scheduler.submit_partial(stt_job("session_a", "pa", final=False))
    )
    await asyncio.sleep(0)
    assert await scheduler.submit_partial(stt_job("session_a", "new", final=False)) is None
    with pytest.raises(STTFullError):
        await scheduler.submit_final(stt_job("session_b", "fb", final=True))
    await scheduler.aclose()
    with pytest.raises(asyncio.CancelledError):
        await first_partial


@pytest.mark.asyncio
async def test_tts_is_segment_round_robin_and_queue_overflow_is_explicit() -> None:
    runtime = RecordingTTS()
    scheduler = TTSScheduler(runtime, telemetry=ServiceTelemetry(), max_jobs=3)
    emitted: dict[str, list[bytes]] = {value: [] for value in ("a1", "a2", "b1")}
    tasks = [
        asyncio.create_task(scheduler.synthesize(tts_job("session_a", "a1", emitted["a1"]))),
        asyncio.create_task(scheduler.synthesize(tts_job("session_a", "a2", emitted["a2"]))),
        asyncio.create_task(scheduler.synthesize(tts_job("session_b", "b1", emitted["b1"]))),
    ]
    await asyncio.sleep(0)
    with pytest.raises(TTSFullError):
        await scheduler.synthesize(tts_job("session_b", "overflow"))
    await scheduler.start()
    try:
        await asyncio.gather(*tasks)
        assert all(result == [bytes(960)] for result in emitted.values())
        assert runtime.calls == ["a1", "b1", "a2"]
    finally:
        await scheduler.aclose()


@pytest.mark.asyncio
async def test_tts_cancellation_removes_only_the_matching_queued_generation() -> None:
    runtime = RecordingTTS()
    scheduler = TTSScheduler(runtime, telemetry=ServiceTelemetry())
    old_job = tts_job("session_a", "old")
    new_job = tts_job("session_a", "new")
    old = asyncio.create_task(scheduler.synthesize(old_job))
    new = asyncio.create_task(scheduler.synthesize(new_job))
    await asyncio.sleep(0)
    await scheduler.cancel_generation(old_job.token)
    await scheduler.start()
    try:
        with pytest.raises(asyncio.CancelledError):
            await old
        await new
        assert runtime.calls == ["new"]
    finally:
        await scheduler.aclose()


@pytest.mark.asyncio
async def test_tts_forwards_each_frame_before_segment_generation_finishes() -> None:
    first_generated = asyncio.Event()
    continue_generation = asyncio.Event()
    first_forwarded = asyncio.Event()

    class StreamingTTS(RecordingTTS):
        async def stream_pcm16_frames(
            self,
            text: str,
            *,
            voice: str,
            cancelled: Callable[[], bool],
        ) -> AsyncIterator[bytes]:
            del text, voice, cancelled
            yield b"first"
            first_generated.set()
            await continue_generation.wait()
            yield b"second"

    emitted: list[bytes] = []

    async def record(frame: bytes) -> None:
        emitted.append(frame)
        first_forwarded.set()

    job = TTSJob(
        token=token("session_a", "stream"),
        text="stream",
        voice="de_standard_01",
        is_current=lambda: True,
        on_frame=record,
    )
    scheduler = TTSScheduler(StreamingTTS(), telemetry=ServiceTelemetry())
    await scheduler.start()
    task = asyncio.create_task(scheduler.synthesize(job))
    try:
        await asyncio.wait_for(first_generated.wait(), timeout=1.0)
        await asyncio.wait_for(first_forwarded.wait(), timeout=1.0)
        assert emitted == [b"first"]
        assert not task.done()
        continue_generation.set()
        await asyncio.wait_for(task, timeout=1.0)
        assert emitted == [b"first", b"second"]
    finally:
        continue_generation.set()
        await scheduler.aclose()
