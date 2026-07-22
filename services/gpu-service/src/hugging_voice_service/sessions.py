"""Isolated per-session state and bounded audio storage."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

from hugging_voice_protocol.audio import PCM16_BYTES_PER_SAMPLE
from hugging_voice_protocol.events import ServerEvent

from .cancellation import GenerationCancellation
from .capacity import CapacityManager, SessionSlot
from .conversation import Conversation
from .runtimes.silero import SessionVAD


class SessionLifecycle(StrEnum):
    ACTIVE = "active"
    DRAINING = "draining"
    STUCK = "stuck"
    CLOSED = "closed"


class SessionTransport(Protocol):
    async def send(self, event: ServerEvent) -> None: ...

    async def cancel_generation(self, generation_id: str) -> int: ...

    async def close(self, *, code: int, reason: str) -> None: ...


class BoundedAudioBuffer:
    def __init__(self, *, max_seconds: int = 60, sample_rate: int = 16_000) -> None:
        self._max_bytes = max_seconds * sample_rate * PCM16_BYTES_PER_SAMPLE
        self._data = bytearray()
        self._first_sample = 0
        self._end_sample = 0

    @property
    def first_sample(self) -> int:
        return self._first_sample

    @property
    def end_sample(self) -> int:
        return self._end_sample

    @property
    def size_bytes(self) -> int:
        return len(self._data)

    def append(self, payload: bytes) -> None:
        if len(payload) % PCM16_BYTES_PER_SAMPLE:
            raise ValueError("audio buffer requires complete PCM16 samples")
        if len(self._data) + len(payload) > self._max_bytes:
            raise BufferError("input audio buffer capacity exceeded")
        self._data.extend(payload)
        self._end_sample += len(payload) // PCM16_BYTES_PER_SAMPLE

    def slice_samples(self, start: int, end: int) -> bytes:
        if start < self._first_sample or end > self._end_sample or end < start:
            raise ValueError(
                f"audio slice [{start}, {end}) outside [{self._first_sample}, {self._end_sample})"
            )
        offset = (start - self._first_sample) * PCM16_BYTES_PER_SAMPLE
        length = (end - start) * PCM16_BYTES_PER_SAMPLE
        return bytes(self._data[offset : offset + length])

    def discard_before(self, sample: int) -> None:
        if not self._first_sample <= sample <= self._end_sample:
            raise ValueError("discard point is outside audio buffer")
        count = (sample - self._first_sample) * PCM16_BYTES_PER_SAMPLE
        del self._data[:count]
        self._first_sample = sample

    def clear(self) -> None:
        self._data.clear()
        self._first_sample = 0
        self._end_sample = 0


@dataclass(slots=True)
class SessionState:
    session_id: str
    slot: SessionSlot
    transport: SessionTransport
    vad: SessionVAD
    instructions: str = ""
    conversation: Conversation = field(default_factory=Conversation)
    input_audio_buffer: BoundedAudioBuffer = field(default_factory=BoundedAudioBuffer)
    current_turn_id: str | None = None
    current_turn_revision: int = -1
    current_generation_id: str | None = None
    current_response_id: str | None = None
    speech_start_sample: int | None = None
    vad_enabled: bool = True
    transcription_enabled: bool = True
    interrupt_response: bool = True
    next_audio_sequence: int = 0
    last_partial_at: float = 0.0
    partial_epoch: int = 0
    cancellation: GenerationCancellation = field(init=False)
    connected_at: float = field(default_factory=time.monotonic)
    last_activity_at: float = field(default_factory=time.monotonic)
    lifecycle: SessionLifecycle = SessionLifecycle.ACTIVE
    pipeline: Any | None = None

    def __post_init__(self) -> None:
        self.cancellation = GenerationCancellation(self.session_id)


class SessionRegistry:
    def __init__(self, capacity: CapacityManager, *, drain_timeout: float) -> None:
        self.capacity = capacity
        self._drain_timeout = drain_timeout
        self._sessions: dict[str, SessionState] = {}
        self._releases: dict[str, asyncio.Task[bool]] = {}
        self._lock = asyncio.Lock()
        self._changed = asyncio.Condition(self._lock)

    async def create(
        self,
        *,
        session_id: str,
        transport: SessionTransport,
        vad: SessionVAD,
    ) -> SessionState | None:
        slot = await self.capacity.claim(session_id)
        if slot is None:
            return None
        state = SessionState(session_id=session_id, slot=slot, transport=transport, vad=vad)
        async with self._changed:
            self._sessions[session_id] = state
            self._changed.notify_all()
        return state

    async def get(self, session_id: str) -> SessionState | None:
        async with self._lock:
            return self._sessions.get(session_id)

    async def release(self, state: SessionState) -> bool:
        async with self._lock:
            task = self._releases.get(state.session_id)
            if task is None:
                task = asyncio.create_task(self._release_once(state))
                self._releases[state.session_id] = task
        return await asyncio.shield(task)

    async def _release_once(self, state: SessionState) -> bool:
        state.lifecycle = SessionLifecycle.DRAINING
        await self.capacity.begin_release(state.slot)
        drained = False
        try:
            if state.pipeline is not None:
                await asyncio.wait_for(state.pipeline.drain(), timeout=self._drain_timeout)
            drained = True
        except TimeoutError:
            state.lifecycle = SessionLifecycle.STUCK
        await self.capacity.complete_release(state.slot, drained=drained)
        if drained:
            state.lifecycle = SessionLifecycle.CLOSED
            async with self._changed:
                self._sessions.pop(state.session_id, None)
                self._changed.notify_all()
        else:
            async with self._changed:
                self._changed.notify_all()
        return drained

    async def states(self) -> tuple[SessionState, ...]:
        async with self._lock:
            return tuple(self._sessions.values())

    async def wait_connections_drained(self) -> None:
        async with self._changed:
            await self._changed.wait_for(
                lambda: all(
                    state.lifecycle in {SessionLifecycle.STUCK, SessionLifecycle.CLOSED}
                    for state in self._sessions.values()
                )
            )
