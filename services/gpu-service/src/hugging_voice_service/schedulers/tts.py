"""Segment-round-robin scheduler for one non-reentrant Qwen runtime."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from functools import partial
from typing import Protocol

from ..cancellation import GenerationToken
from ..telemetry import ServiceTelemetry


class TTSRuntime(Protocol):
    def stream_pcm16_frames(
        self,
        text: str,
        *,
        voice: str,
        cancelled: Callable[[], bool],
    ) -> AsyncIterator[bytes]: ...


class SchedulerFullError(RuntimeError):
    pass


@dataclass(slots=True)
class TTSJob:
    token: GenerationToken
    text: str
    voice: str
    is_current: Callable[[], bool]
    on_frame: Callable[[bytes], Awaitable[None]]
    enqueued_at: float = field(default_factory=time.monotonic)
    future: asyncio.Future[None] | None = None


def _job_cancelled(job: TTSJob) -> bool:
    return not job.is_current()


class TTSScheduler:
    def __init__(
        self,
        runtime: TTSRuntime,
        *,
        telemetry: ServiceTelemetry,
        max_jobs: int = 16,
    ) -> None:
        self._runtime = runtime
        self._telemetry = telemetry
        self._max_jobs = max_jobs
        self._queues: dict[str, deque[TTSJob]] = {}
        self._condition = asyncio.Condition()
        self._worker: asyncio.Task[None] | None = None
        self._active: TTSJob | None = None
        self._last_session: str | None = None
        self._closed = False

    async def start(self) -> None:
        if self._worker is not None:
            raise RuntimeError("TTS scheduler already started")
        self._worker = asyncio.create_task(self._run())

    async def synthesize(self, job: TTSJob) -> None:
        async with self._condition:
            if self._closed:
                raise RuntimeError("TTS scheduler is closed")
            if self._queued_count() >= self._max_jobs:
                raise SchedulerFullError("TTS segment queue is full")
            job.future = asyncio.get_running_loop().create_future()
            self._queues.setdefault(job.token.session_id, deque()).append(job)
            self._condition.notify_all()
        await job.future

    async def cancel_generation(self, token: GenerationToken) -> None:
        async with self._condition:
            jobs = self._queues.get(token.session_id, deque())
            retained: deque[TTSJob] = deque()
            for job in jobs:
                if job.token == token:
                    if job.future is not None and not job.future.done():
                        job.future.cancel()
                else:
                    retained.append(job)
            if retained:
                self._queues[token.session_id] = retained
            else:
                self._queues.pop(token.session_id, None)
            self._condition.notify_all()

    async def wait_session_idle(self, session_id: str) -> None:
        async with self._condition:
            await self._condition.wait_for(
                lambda: (self._active is None or self._active.token.session_id != session_id)
                and not self._queues.get(session_id)
            )

    async def aclose(self) -> None:
        async with self._condition:
            self._closed = True
            for jobs in self._queues.values():
                for job in jobs:
                    if job.future is not None and not job.future.done():
                        job.future.cancel()
            self._queues.clear()
            self._condition.notify_all()
        if self._worker is not None:
            self._worker.cancel()
            await asyncio.gather(self._worker, return_exceptions=True)
            self._worker = None

    async def _run(self) -> None:
        while True:
            async with self._condition:
                await self._condition.wait_for(lambda: self._closed or self._queued_count() > 0)
                if self._closed:
                    return
                job = self._next_job()
                self._active = job
            self._telemetry.tts_jobs_active.inc()
            self._telemetry.tts_queue_seconds.observe(time.monotonic() - job.enqueued_at)
            started = time.monotonic()
            first_audio_at: float | None = None
            frames_emitted = 0
            try:
                if job.is_current():
                    async for frame in self._runtime.stream_pcm16_frames(
                        job.text,
                        voice=job.voice,
                        cancelled=partial(_job_cancelled, job),
                    ):
                        if not job.is_current():
                            self._telemetry.stale_chunks_dropped.inc()
                            break
                        if first_audio_at is None:
                            first_audio_at = time.monotonic()
                            self._telemetry.tts_ttfa_seconds.observe(first_audio_at - started)
                        await job.on_frame(frame)
                        frames_emitted += 1
                if job.future is not None and not job.future.done():
                    job.future.set_result(None)
            except asyncio.CancelledError:
                if job.future is not None and not job.future.done():
                    job.future.cancel()
                raise
            except Exception as exc:
                if job.future is not None and not job.future.done():
                    job.future.set_exception(exc)
            finally:
                duration = time.monotonic() - started
                self._telemetry.tts_duration_seconds.observe(duration)
                self._telemetry.tts_audio_seconds.observe(frames_emitted * 0.02)
                self._telemetry.tts_jobs_active.dec()
                async with self._condition:
                    self._active = None
                    self._condition.notify_all()

    def _next_job(self) -> TTSJob:
        sessions = sorted(session for session, jobs in self._queues.items() if jobs)
        if not sessions:
            raise RuntimeError("scheduler selected from empty queues")
        if self._last_session not in sessions:
            session = sessions[0]
        else:
            session = sessions[(sessions.index(self._last_session) + 1) % len(sessions)]
        self._last_session = session
        job = self._queues[session].popleft()
        if not self._queues[session]:
            del self._queues[session]
        return job

    def _queued_count(self) -> int:
        return sum(len(jobs) for jobs in self._queues.values())
