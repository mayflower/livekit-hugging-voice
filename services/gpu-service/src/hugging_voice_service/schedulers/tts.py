"""Bounded fair scheduler for a small pool of non-reentrant Qwen runtimes."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Literal, Protocol

from ..cancellation import GenerationToken
from ..telemetry import ServiceTelemetry

TTSPriority = Literal["final_response", "filler_or_explicit_say"]


class TTSRuntime(Protocol):
    def stream_pcm16_frames(
        self,
        text: str,
        *,
        language: str,
        instructions: str,
        ref_audio: Path | None,
        ref_text: str | None,
        cancelled: Callable[[], bool],
    ) -> AsyncIterator[bytes]: ...


class SchedulerFullError(RuntimeError):
    pass


@dataclass(slots=True)
class TTSJob:
    token: GenerationToken
    text: str
    language: str
    instructions: str
    is_current: Callable[[], bool]
    on_frame: Callable[[bytes], Awaitable[None]]
    ref_audio: Path | None = None
    ref_text: str | None = None
    priority: TTSPriority = "final_response"
    enqueued_at: float = field(default_factory=time.monotonic)
    future: asyncio.Future[None] | None = None
    worker_id: int | None = None


def _job_cancelled(job: TTSJob) -> bool:
    return not job.is_current()


class TTSScheduler:
    def __init__(
        self,
        runtimes: TTSRuntime | Sequence[TTSRuntime],
        *,
        telemetry: ServiceTelemetry,
        max_jobs: int = 16,
    ) -> None:
        if isinstance(runtimes, Sequence):
            resolved = tuple(runtimes)
        else:
            resolved = (runtimes,)
        if not 1 <= len(resolved) <= 4:
            raise ValueError("TTS scheduler requires between 1 and 4 runtimes")
        if max_jobs < len(resolved):
            raise ValueError("TTS max_jobs cannot be smaller than the worker count")
        self._runtimes = resolved
        self._telemetry = telemetry
        self._max_jobs = max_jobs
        self._queues: dict[str, deque[TTSJob]] = {}
        self._session_order: deque[str] = deque()
        self._active_sessions: set[str] = set()
        self._active: dict[int, TTSJob] = {}
        self._condition = asyncio.Condition()
        self._workers: list[asyncio.Task[None]] = []
        self._closed = False

    @property
    def worker_count(self) -> int:
        return len(self._runtimes)

    async def start(self) -> None:
        if self._workers:
            raise RuntimeError("TTS scheduler already started")
        self._workers = [
            asyncio.create_task(self._run_worker(worker_id, runtime))
            for worker_id, runtime in enumerate(self._runtimes)
        ]

    async def synthesize(self, job: TTSJob) -> None:
        async with self._condition:
            if self._closed:
                raise RuntimeError("TTS scheduler is closed")
            if self._job_count() >= self._max_jobs:
                raise SchedulerFullError("TTS segment queue is full")
            job.future = asyncio.get_running_loop().create_future()
            session_id = job.token.session_id
            if session_id not in self._queues:
                self._queues[session_id] = deque()
                self._session_order.append(session_id)
            self._queues[session_id].append(job)
            self._refresh_queue_metrics()
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
                self._drop_session_queue(token.session_id)
            self._refresh_queue_metrics()
            self._condition.notify_all()

    async def wait_session_idle(self, session_id: str) -> None:
        async with self._condition:
            await self._condition.wait_for(
                lambda: session_id not in self._active_sessions and not self._queues.get(session_id)
            )

    async def aclose(self) -> None:
        async with self._condition:
            if self._closed:
                return
            self._closed = True
            for jobs in self._queues.values():
                for job in jobs:
                    if job.future is not None and not job.future.done():
                        job.future.cancel()
            self._queues.clear()
            self._session_order.clear()
            self._refresh_queue_metrics()
            self._condition.notify_all()
        for worker in self._workers:
            worker.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

    async def _run_worker(self, worker_id: int, runtime: TTSRuntime) -> None:
        while True:
            async with self._condition:
                await self._condition.wait_for(lambda: self._closed or self._has_runnable_job())
                if self._closed:
                    return
                job = self._next_job()
                job.worker_id = worker_id
                session_id = job.token.session_id
                self._active_sessions.add(session_id)
                self._active[worker_id] = job
                self._refresh_queue_metrics()
            try:
                await self._execute_job(worker_id, runtime, job)
            finally:
                async with self._condition:
                    self._active.pop(worker_id, None)
                    self._active_sessions.discard(session_id)
                    if not self._queues.get(session_id):
                        self._drop_session_queue(session_id)
                    self._refresh_queue_metrics()
                    self._condition.notify_all()

    async def _execute_job(
        self,
        worker_id: int,
        runtime: TTSRuntime,
        job: TTSJob,
    ) -> None:
        wait_seconds = time.monotonic() - job.enqueued_at
        self._telemetry.tts_queue_seconds.observe(wait_seconds)
        self._telemetry.tts_fairness_wait_seconds.observe(wait_seconds)
        self._telemetry.tts_worker_busy.labels(worker=str(worker_id)).set(1)
        self._telemetry.tts_worker_jobs_total.labels(worker=str(worker_id)).inc()
        self._telemetry.tts_job_worker.labels(worker=str(worker_id)).inc()
        self._telemetry.tts_jobs_active.inc()
        started = time.monotonic()
        first_audio_at: float | None = None
        frames_emitted = 0
        try:
            if job.is_current():
                async for frame in runtime.stream_pcm16_frames(
                    job.text,
                    language=job.language,
                    instructions=job.instructions,
                    ref_audio=job.ref_audio,
                    ref_text=job.ref_text,
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
            self._telemetry.tts_worker_busy.labels(worker=str(worker_id)).set(0)

    def _has_runnable_job(self) -> bool:
        return any(
            jobs and session_id not in self._active_sessions
            for session_id, jobs in self._queues.items()
        )

    def _next_job(self) -> TTSJob:
        for priority in ("final_response", "filler_or_explicit_say"):
            count = len(self._session_order)
            for _ in range(count):
                session_id = self._session_order[0]
                self._session_order.rotate(-1)
                jobs = self._queues.get(session_id)
                if (
                    jobs
                    and session_id not in self._active_sessions
                    and jobs[0].priority == priority
                ):
                    return jobs.popleft()
        raise RuntimeError("scheduler selected from no runnable session")

    def _drop_session_queue(self, session_id: str) -> None:
        self._queues.pop(session_id, None)
        try:
            self._session_order.remove(session_id)
        except ValueError:
            pass

    def _queued_count(self) -> int:
        return sum(len(jobs) for jobs in self._queues.values())

    def _job_count(self) -> int:
        return self._queued_count() + len(self._active)

    def _refresh_queue_metrics(self) -> None:
        self._telemetry.tts_queue_depth.set(self._queued_count())
        self._telemetry.tts_sessions_waiting.set(
            sum(
                bool(jobs)
                for session_id, jobs in self._queues.items()
                if session_id not in self._active_sessions
            )
        )
        self._telemetry.tts_active_sessions.set(len(self._active_sessions))
