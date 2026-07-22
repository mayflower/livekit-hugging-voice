"""Atomic no-queue admission and drain/quarantine slot state."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from enum import StrEnum

from .telemetry import ServiceTelemetry


class SlotState(StrEnum):
    IDLE = "idle"
    ACTIVE = "active"
    DRAINING = "draining"
    STUCK = "stuck"


@dataclass(slots=True)
class SessionSlot:
    index: int
    state: SlotState = SlotState.IDLE
    session_id: str | None = None
    claimed_at: float | None = None
    released_at: float | None = None
    stuck_at: float | None = None


class CapacityManager:
    def __init__(
        self,
        max_sessions: int,
        *,
        telemetry: ServiceTelemetry,
    ) -> None:
        if max_sessions not in {1, 2}:
            raise ValueError("capacity must contain one or two session slots")
        self._slots = [SessionSlot(index=index) for index in range(max_sessions)]
        self._lock = asyncio.Lock()
        self._draining = False
        self._telemetry = telemetry
        self._refresh_metrics()

    @property
    def draining(self) -> bool:
        return self._draining

    async def claim(self, session_id: str) -> SessionSlot | None:
        async with self._lock:
            if self._draining:
                return None
            slot = next(
                (candidate for candidate in self._slots if candidate.state is SlotState.IDLE),
                None,
            )
            if slot is None:
                self._telemetry.sessions_rejected.inc()
                return None
            slot.state = SlotState.ACTIVE
            slot.session_id = session_id
            slot.claimed_at = time.monotonic()
            slot.released_at = None
            slot.stuck_at = None
            self._refresh_metrics()
            return slot

    async def begin_release(self, slot: SessionSlot) -> None:
        async with self._lock:
            if slot.state is SlotState.ACTIVE:
                slot.state = SlotState.DRAINING
                slot.released_at = time.monotonic()
                self._refresh_metrics()

    async def complete_release(self, slot: SessionSlot, *, drained: bool) -> None:
        async with self._lock:
            if drained:
                slot.state = SlotState.IDLE
                slot.session_id = None
                slot.claimed_at = None
                slot.released_at = None
                slot.stuck_at = None
            else:
                slot.state = SlotState.STUCK
                slot.stuck_at = time.monotonic()
            self._refresh_metrics()

    async def begin_service_drain(self) -> None:
        async with self._lock:
            self._draining = True
            self._refresh_metrics()

    async def report(self) -> dict[str, int]:
        async with self._lock:
            active = sum(slot.state is SlotState.ACTIVE for slot in self._slots)
            draining = sum(slot.state is SlotState.DRAINING for slot in self._slots)
            stuck = sum(slot.state is SlotState.STUCK for slot in self._slots)
            idle = sum(slot.state is SlotState.IDLE for slot in self._slots)
            return {
                "total": len(self._slots),
                "active": active,
                "draining": draining,
                "stuck": stuck,
                "available": 0 if self._draining else idle,
            }

    async def pool_report(self) -> list[dict[str, object]]:
        async with self._lock:
            now = time.monotonic()
            return [
                {
                    "slot": slot.index,
                    "state": slot.state,
                    "session_id": slot.session_id,
                    "connected_seconds": (
                        None if slot.claimed_at is None else round(now - slot.claimed_at, 3)
                    ),
                    "draining_seconds": (
                        None if slot.released_at is None else round(now - slot.released_at, 3)
                    ),
                }
                for slot in self._slots
            ]

    def _refresh_metrics(self) -> None:
        active = sum(slot.state is SlotState.ACTIVE for slot in self._slots)
        draining = sum(slot.state is SlotState.DRAINING for slot in self._slots)
        stuck = sum(slot.state is SlotState.STUCK for slot in self._slots)
        idle = sum(slot.state is SlotState.IDLE for slot in self._slots)
        self._telemetry.sessions_active.set(active)
        self._telemetry.sessions_draining.set(draining)
        self._telemetry.sessions_stuck.set(stuck)
        self._telemetry.sessions_available.set(0 if self._draining else idle)
