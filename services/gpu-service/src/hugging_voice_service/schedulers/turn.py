"""Bounded fair scheduler for the shared non-reentrant Smart Turn runtime."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from ..runtimes.smart_turn import SmartTurnResult
from ..telemetry import ServiceTelemetry


class TurnRuntime(Protocol):
    def predict_pcm16(self, audio: bytes) -> SmartTurnResult: ...


class TurnSchedulerFullError(RuntimeError):
    pass


@dataclass(slots=True)
class TurnJob:
    session_id: str
    turn_id: str
    turn_revision: int
    candidate_id: int
    audio: bytes
    is_stale: Callable[[], bool]
    enqueued_at: float = field(default_factory=time.monotonic)
    future: asyncio.Future[SmartTurnResult | None] | None = None


class TurnScheduler:
    def __init__(
        self,
        runtime: TurnRuntime,
        *,
        telemetry: ServiceTelemetry,
        max_jobs: int,
    ) -> None:
        if max_jobs < 1:
            raise ValueError("Smart Turn scheduler capacity must be positive")
        self._runtime = runtime
        self._telemetry = telemetry
        self._max_jobs = max_jobs
        self._queues: dict[str, deque[TurnJob]] = {}
        self._active: TurnJob | None = None
        self._last_session: str | None = None
        self._condition = asyncio.Condition()
        self._worker: asyncio.Task[None] | None = None
        self._closed = False

    async def submit(self, job: TurnJob) -> SmartTurnResult | None:
        async with self._condition:
            if self._closed:
                raise RuntimeError("Smart Turn scheduler is closed")
            if self._queued_count() >= self._max_jobs:
                raise TurnSchedulerFullError("Smart Turn scheduler queue is full")
            job.future = asyncio.get_running_loop().create_future()
            self._queues.setdefault(job.session_id, deque()).append(job)
            self._telemetry.turn_queue_depth.set(self._queued_count())
            if self._worker is None or self._worker.done():
                self._worker = asyncio.create_task(self._run())
            self._condition.notify_all()
        try:
            return await job.future
        except asyncio.CancelledError:
            async with self._condition:
                queue = self._queues.get(job.session_id)
                if queue is not None:
                    for index, queued_job in enumerate(queue):
                        if queued_job is job:
                            del queue[index]
                            break
                    if not queue:
                        self._queues.pop(job.session_id, None)
                self._telemetry.turn_queue_depth.set(self._queued_count())
                self._condition.notify_all()
            raise

    async def cancel_session(self, session_id: str) -> None:
        async with self._condition:
            queued = self._queues.pop(session_id, deque())
            for job in queued:
                if job.future is not None and not job.future.done():
                    job.future.cancel()
            self._telemetry.turn_queue_depth.set(self._queued_count())
            self._condition.notify_all()

    async def wait_session_idle(self, session_id: str) -> None:
        async with self._condition:
            await self._condition.wait_for(
                lambda: (
                    session_id not in self._queues
                    and (self._active is None or self._active.session_id != session_id)
                )
            )

    async def aclose(self) -> None:
        async with self._condition:
            self._closed = True
            for queue in self._queues.values():
                for job in queue:
                    if job.future is not None and not job.future.done():
                        job.future.cancel()
            self._queues.clear()
            self._telemetry.turn_queue_depth.set(0)
            self._condition.notify_all()
        if self._worker is not None:
            await self._worker
            self._worker = None

    async def _run(self) -> None:
        while True:
            async with self._condition:
                if not self._queues:
                    if self._closed:
                        return
                    self._worker = None
                    return
                job = self._next_job()
                self._active = job
                self._telemetry.turn_queue_depth.set(self._queued_count())
            future = job.future
            if future is None:
                raise RuntimeError("Smart Turn job has no future")
            try:
                if job.is_stale():
                    result = None
                    self._telemetry.turn_stale_results.inc()
                else:
                    self._telemetry.turn_queue_seconds.observe(time.monotonic() - job.enqueued_at)
                    started = time.monotonic()
                    self._telemetry.turn_jobs_active.inc()
                    try:
                        result = await asyncio.to_thread(
                            self._runtime.predict_pcm16,
                            job.audio,
                        )
                    finally:
                        self._telemetry.turn_jobs_active.dec()
                    self._telemetry.turn_inference_seconds.observe(time.monotonic() - started)
                    if job.is_stale():
                        result = None
                        self._telemetry.turn_stale_results.inc()
                if not future.done():
                    future.set_result(result)
            except Exception as exc:
                if not future.done():
                    future.set_exception(exc)
            finally:
                async with self._condition:
                    self._active = None
                    self._condition.notify_all()

    def _next_job(self) -> TurnJob:
        sessions = sorted(session for session, jobs in self._queues.items() if jobs)
        if not sessions:
            raise RuntimeError("Smart Turn scheduler selected from empty queues")
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
