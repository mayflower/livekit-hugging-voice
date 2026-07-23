"""Final-first, session-round-robin scheduler for one Parakeet runtime."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from ..telemetry import ServiceTelemetry


class STTRuntime(Protocol):
    async def transcribe_partial(self, pcm16: bytes) -> str: ...

    async def transcribe_final(self, pcm16: bytes) -> str: ...


class SchedulerFullError(RuntimeError):
    pass


@dataclass(slots=True)
class STTJob:
    session_id: str
    turn_id: str
    turn_revision: int
    audio: bytes
    final: bool
    is_stale: Callable[[], bool]
    enqueued_at: float = field(default_factory=time.monotonic)
    future: asyncio.Future[str | None] | None = None


class STTScheduler:
    def __init__(
        self,
        runtime: STTRuntime,
        *,
        telemetry: ServiceTelemetry,
        max_jobs: int = 32,
    ) -> None:
        self._runtime = runtime
        self._telemetry = telemetry
        self._max_jobs = max_jobs
        self._final: dict[str, deque[STTJob]] = {}
        self._partial: dict[str, deque[STTJob]] = {}
        self._condition = asyncio.Condition()
        self._worker: asyncio.Task[None] | None = None
        self._closed = False
        self._active: STTJob | None = None
        self._last_final_session: str | None = None
        self._last_partial_session: str | None = None

    async def start(self) -> None:
        if self._worker is not None:
            raise RuntimeError("STT scheduler already started")
        self._worker = asyncio.create_task(self._run())

    async def submit_final(self, job: STTJob) -> str:
        job.final = True
        result = await self._enqueue_and_wait(job)
        if result is None:
            raise asyncio.CancelledError("final STT job became stale")
        return result

    async def submit_partial(self, job: STTJob) -> str | None:
        job.final = False
        async with self._condition:
            if (
                self._closed
                or self._active is not None
                or self._queued_count() >= self._max_jobs
                or any(self._final.values())
                or any(self._partial.get(job.session_id, ()))
            ):
                return None
            job.future = asyncio.get_running_loop().create_future()
            self._partial.setdefault(job.session_id, deque()).append(job)
            self._condition.notify_all()
        return await job.future

    async def cancel_session(self, session_id: str) -> None:
        async with self._condition:
            for queues in (self._final, self._partial):
                jobs = queues.pop(session_id, deque())
                for job in jobs:
                    if job.future is not None and not job.future.done():
                        job.future.cancel()
            self._condition.notify_all()

    async def wait_session_idle(self, session_id: str) -> None:
        async with self._condition:
            await self._condition.wait_for(
                lambda: (
                    (self._active is None or self._active.session_id != session_id)
                    and not self._final.get(session_id)
                    and not self._partial.get(session_id)
                )
            )

    async def aclose(self) -> None:
        async with self._condition:
            self._closed = True
            for queues in (self._final, self._partial):
                for jobs in queues.values():
                    for job in jobs:
                        if job.future is not None and not job.future.done():
                            job.future.cancel()
                queues.clear()
            self._condition.notify_all()
        if self._worker is not None:
            self._worker.cancel()
            await asyncio.gather(self._worker, return_exceptions=True)
            self._worker = None

    async def _enqueue_and_wait(self, job: STTJob) -> str | None:
        async with self._condition:
            if self._closed:
                raise RuntimeError("STT scheduler is closed")
            if self._queued_count() >= self._max_jobs:
                raise SchedulerFullError("final STT queue is full")
            job.future = asyncio.get_running_loop().create_future()
            self._final.setdefault(job.session_id, deque()).append(job)
            self._condition.notify_all()
        return await job.future

    async def _run(self) -> None:
        while True:
            async with self._condition:
                await self._condition.wait_for(lambda: self._closed or self._queued_count() > 0)
                if self._closed:
                    return
                job = self._next_job()
                self._active = job
            self._telemetry.stt_jobs_active.inc()
            self._telemetry.stt_queue_seconds.observe(time.monotonic() - job.enqueued_at)
            started = time.monotonic()
            try:
                if job.is_stale():
                    result = None
                elif job.final:
                    result = await self._runtime.transcribe_final(job.audio)
                else:
                    result = await self._runtime.transcribe_partial(job.audio)
                if job.is_stale():
                    result = None
                if job.future is not None and not job.future.done():
                    job.future.set_result(result)
            except asyncio.CancelledError:
                if job.future is not None and not job.future.done():
                    job.future.cancel()
                raise
            except Exception as exc:
                if job.future is not None and not job.future.done():
                    job.future.set_exception(exc)
            finally:
                self._telemetry.stt_inference_seconds.observe(time.monotonic() - started)
                self._telemetry.stt_jobs_active.dec()
                async with self._condition:
                    self._active = None
                    self._condition.notify_all()

    def _next_job(self) -> STTJob:
        if any(self._final.values()):
            session = self._next_session(self._final, self._last_final_session)
            self._last_final_session = session
            return self._pop(self._final, session)
        session = self._next_session(self._partial, self._last_partial_session)
        self._last_partial_session = session
        return self._pop(self._partial, session)

    @staticmethod
    def _next_session(queues: dict[str, deque[STTJob]], last: str | None) -> str:
        sessions = sorted(session for session, jobs in queues.items() if jobs)
        if not sessions:
            raise RuntimeError("scheduler selected from empty queues")
        if last not in sessions:
            return sessions[0]
        return sessions[(sessions.index(last) + 1) % len(sessions)]

    @staticmethod
    def _pop(queues: dict[str, deque[STTJob]], session: str) -> STTJob:
        job = queues[session].popleft()
        if not queues[session]:
            del queues[session]
        return job

    def _queued_count(self) -> int:
        return sum(len(jobs) for queues in (self._final, self._partial) for jobs in queues.values())
