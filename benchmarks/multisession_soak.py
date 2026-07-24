#!/usr/bin/env python3
"""Run N real isolated service sessions and record raw latency observations."""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import os
import platform
import random
import subprocess
import time
import uuid
import wave
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import aiohttp
from hugging_voice_protocol.events import parse_server_event_json

SUBPROTOCOL = "hugging-voice-livekit.v2"
FRAME_BYTES = 1_280  # 40 ms, mono PCM16 at 16 kHz


def event_id() -> str:
    return f"evt_{uuid.uuid4().hex}"


def read_pcm16(path: Path) -> bytes:
    with wave.open(str(path), "rb") as source:
        if source.getparams()[:3] != (1, 2, 16_000) or source.getcomptype() != "NONE":
            raise ValueError(f"{path} must be uncompressed mono 16 kHz PCM16 WAV")
        audio = source.readframes(source.getnframes())
    if not audio:
        raise ValueError(f"{path} contains no audio")
    return audio


def websocket_url(service_url: str) -> str:
    parts = urlsplit(service_url)
    scheme = {"http": "ws", "https": "wss", "ws": "ws", "wss": "wss"}.get(parts.scheme)
    if scheme is None or not parts.netloc:
        raise ValueError("service URL must use http, https, ws, or wss")
    return urlunsplit((scheme, parts.netloc, "/v1/realtime", "", ""))


def http_url(service_url: str, path: str) -> str:
    parts = urlsplit(service_url)
    scheme = {"ws": "http", "wss": "https"}.get(parts.scheme, parts.scheme)
    if scheme not in {"http", "https"} or not parts.netloc:
        raise ValueError("service URL must use http, https, ws, or wss")
    return urlunsplit((scheme, parts.netloc, path, "", ""))


def command_output(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def build_canaries(session_count: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    return [f"HV{index:02d}{rng.randrange(0, 16**12):012X}" for index in range(session_count)]


def select_turn_type(workload: str, session_index: int, turn: int) -> str:
    if workload == "mixed":
        return "tool" if (turn + session_index) % 2 == 0 else "normal"
    if workload not in {"normal", "tool"}:
        raise ValueError(f"unsupported workload: {workload}")
    return workload


class JsonlWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = asyncio.Lock()

    async def write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        async with self._lock:
            with self.path.open("a", encoding="utf-8") as destination:
                destination.write(line + "\n")


@dataclass(slots=True)
class TurnResult:
    profile_id: str
    session_concurrency: int
    arrival_mode: str
    workload: str
    session_index: int
    session_label: str
    session_id: str
    turn_index: int
    turn_type: str
    llama_slot_id: int
    tts_worker_id: int | None
    audio_chunk_count: int
    audio_before_tool_result: bool
    duplicate_tool_execution: bool
    transcript: str
    response_text: str
    status: str
    isolation_canary: str
    metrics: dict[str, float]
    tool_timings: dict[str, float | int | bool | None] = field(default_factory=dict)

    def record(self) -> dict[str, Any]:
        record = {
            "record_type": "turn",
            "timestamp": datetime.now(UTC).isoformat(),
            "profile_id": self.profile_id,
            "session_concurrency": self.session_concurrency,
            "arrival_mode": self.arrival_mode,
            "workload": self.workload,
            "session_index": self.session_index,
            "session_label": self.session_label,
            "session_id": self.session_id,
            "turn_index": self.turn_index,
            "turn_type": self.turn_type,
            "llama_slot_id": self.llama_slot_id,
            "tts_worker_id": self.tts_worker_id,
            "audio_chunk_count": self.audio_chunk_count,
            "transcript_chars": len(self.transcript),
            "response_chars": len(self.response_text),
            "status": self.status,
            "isolation_canary": self.isolation_canary,
            "cross_session_leak": False,
            "audio_before_tool_result": self.audio_before_tool_result,
            "duplicate_tool_execution": self.duplicate_tool_execution,
            "unknown_or_mismatched_tool_result": False,
            "stale_final_response": False,
            "non_finite_audio": False,
            "metrics": self.metrics,
        }
        record.update(self.tool_timings)
        return record


class SoakSession:
    def __init__(
        self,
        *,
        profile_id: str,
        session_concurrency: int,
        arrival_mode: str,
        workload: str,
        session_index: int,
        label: str,
        session: aiohttp.ClientSession,
        url: str,
        token: str,
        audio: bytes,
        realtime_audio: bool,
        canary: str,
        forbidden_canaries: frozenset[str],
        tool_delay_seconds: float,
    ) -> None:
        self.profile_id = profile_id
        self.session_concurrency = session_concurrency
        self.arrival_mode = arrival_mode
        self.workload = workload
        self.session_index = session_index
        self.label = label
        self._http = session
        self._url = url
        self._token = token
        self._audio = audio
        self._realtime_audio = realtime_audio
        self._canary = canary
        self._forbidden_canaries = forbidden_canaries
        self._tool_delay_seconds = tool_delay_seconds
        self.ws: aiohttp.ClientWebSocketResponse | None = None
        self.session_id = ""
        self.slot_id = -1
        self._audio_sequence = 0
        self._session_config: dict[str, Any] = {}

    async def connect(self) -> None:
        self.ws = await self._http.ws_connect(
            self._url,
            headers={"Authorization": f"Bearer {self._token}"},
            protocols=[SUBPROTOCOL],
            heartbeat=20.0,
            max_msg_size=1_000_000,
        )
        event = await self._receive_event(timeout_seconds=20.0)
        if event.type != "session.created":
            raise RuntimeError(f"{self.label}: expected session.created, got {event.type}")
        self.session_id = event.session_id
        self.slot_id = event.llama_slot_id
        self._audio_sequence = 0
        session_config: dict[str, Any] = {
            "instructions": (
                "Antworte auf Deutsch. Beende jede vollständige Antwort exakt mit "
                f"dem isolierten Marker {self._canary}. Verwende keine anderen "
                "Benchmark-Marker."
            )
        }
        update: dict[str, Any] = {
            "type": "session.update",
            "event_id": event_id(),
            "protocol_version": 2,
            "session_id": self.session_id,
            "session": session_config,
        }
        if self.workload in {"tool", "mixed"}:
            session_config["instructions"] += (
                " Wenn add_numbers verlangt wird, rufe es mit a=19 und b=23 auf."
            )
            session_config["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": "add_numbers",
                        "strict": True,
                        "description": "Add two integers.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "a": {"type": "integer"},
                                "b": {"type": "integer"},
                            },
                            "required": ["a", "b"],
                            "additionalProperties": False,
                        },
                    },
                }
            ]
            session_config["tool_choice"] = "auto"
        self._session_config = session_config
        await self._send(update)
        while True:
            ack = await self._receive_event(timeout_seconds=20.0)
            if ack.type == "session.updated" and ack.source_event_id == update["event_id"]:
                break
            if ack.type == "error":
                raise RuntimeError(f"{self.label}: session update failed: {ack.error.message}")

    async def close(self) -> None:
        if self.ws is not None:
            await self.ws.close()
            self.ws = None

    async def reconnect(self) -> None:
        await self.close()
        await self.connect()

    async def run_turn(
        self,
        turn_index: int,
        *,
        turn_type: str,
        cancel_after_audio: bool,
        commit_barrier: asyncio.Barrier | None = None,
    ) -> TurnResult:
        if self.ws is None:
            raise RuntimeError("session is not connected")
        if turn_type not in {"normal", "tool"}:
            raise ValueError(f"unsupported turn type: {turn_type}")
        await self._set_tool_choice("required" if turn_type == "tool" else "none")
        await self._send_audio(self._audio)
        if commit_barrier is not None:
            await commit_barrier.wait()
        await self._send(
            {
                "type": "input_audio_buffer.commit",
                "event_id": event_id(),
                "protocol_version": 2,
                "session_id": self.session_id,
            }
        )

        transcript = ""
        response_text = ""
        first_transcript_at: float | None = None
        first_text_at: float | None = None
        first_audio_at: float | None = None
        audio_chunk_count = 0
        tts_worker_id: int | None = None
        audio_before_tool_result = False
        duplicate_tool_execution = False
        response_created_at: float | None = None
        speech_started_at: float | None = None
        speech_stopped_at: float | None = None
        cancelled = False
        status = "unknown"
        tool_call: Any | None = None
        tool_timings: dict[str, float | int | bool | None] = {}
        while True:
            event = await self._receive_event(timeout_seconds=180.0)
            now = time.monotonic()
            if event.type == "error":
                raise RuntimeError(f"{self.label}: {event.error.code}: {event.error.message}")
            if event.type == "input_audio_buffer.speech_started":
                speech_started_at = now
            elif event.type == "input_audio_buffer.speech_stopped":
                speech_stopped_at = now
            elif event.type == "conversation.item.input_audio_transcription.completed":
                transcript = event.transcript
                first_transcript_at = now
            elif event.type == "response.created":
                response_created_at = now
                if turn_type == "tool" and "llm_tool_request_started_at" not in tool_timings:
                    tool_timings["llm_tool_request_started_at"] = now
            elif event.type == "response.output_function_call.done":
                duplicate_tool_execution = duplicate_tool_execution or tool_call is not None
                tool_call = event
                tool_timings["tool_call_emitted_at"] = now
                tool_timings["call_size"] = len(event.arguments)
            elif event.type == "response.output_text.delta":
                response_text += event.delta
                first_text_at = first_text_at or now
            elif event.type == "response.output_audio.delta":
                first_audio_at = first_audio_at or now
                if tts_worker_id is None:
                    tts_worker_id = event.tts_worker_id
                audio_chunk_count += 1
                if turn_type == "tool" and "tool_result_ack_at" not in tool_timings:
                    audio_before_tool_result = True
                cancelled = await self._cancel_after_audio_if_requested(
                    event,
                    enabled=cancel_after_audio,
                    already_cancelled=cancelled,
                )
            elif event.type == "response.done":
                if event.reason.value == "tool_call":
                    if tool_call is None:
                        raise RuntimeError("tool response completed without a function call")
                    tool_timings.update(await self._complete_tool_call(tool_call))
                    await self._send(
                        {
                            "type": "response.create",
                            "event_id": event_id(),
                            "protocol_version": 2,
                            "session_id": self.session_id,
                            "tool_choice": "none",
                        }
                    )
                    continue
                status = event.status.value
                done_at = now
                break

        if speech_started_at is None or speech_stopped_at is None:
            raise RuntimeError(f"{self.label}: turn completed without measured VAD boundaries")
        speech_stop = speech_stopped_at
        metrics: dict[str, float] = {}
        if first_transcript_at is not None:
            metrics["speech_stop_to_final_transcript_seconds"] = first_transcript_at - speech_stop
        if first_text_at is not None:
            metrics["speech_stop_to_first_text_delta_seconds"] = first_text_at - speech_stop
        if first_audio_at is not None:
            metrics["speech_stop_to_first_audio_frame_seconds"] = first_audio_at - speech_stop
            metrics["speech_stop_to_final_first_audio_seconds"] = first_audio_at - speech_stop
        if response_created_at is not None:
            metrics["response_duration_seconds"] = done_at - response_created_at
        call_at = tool_timings.get("tool_call_emitted_at")
        ack_at = tool_timings.get("tool_result_ack_at")
        tool_start = tool_timings.get("tool_execution_started_at")
        tool_finish = tool_timings.get("tool_execution_finished_at")
        if isinstance(call_at, float):
            metrics["speech_stop_to_tool_call_seconds"] = call_at - speech_stop
        if isinstance(tool_start, float) and isinstance(tool_finish, float):
            metrics["tool_duration_seconds"] = tool_finish - tool_start
        if isinstance(ack_at, float) and first_text_at is not None:
            metrics["tool_result_ack_to_final_first_text_seconds"] = first_text_at - ack_at
        if isinstance(ack_at, float) and first_audio_at is not None:
            metrics["tool_result_ack_to_final_first_audio_seconds"] = first_audio_at - ack_at
        tool_timings["speech_started_at"] = speech_started_at
        tool_timings["speech_stopped_at"] = speech_stopped_at
        tool_timings["final_transcript_at"] = first_transcript_at
        tool_timings["final_llm_first_text_at"] = first_text_at
        tool_timings["final_tts_first_audio_at"] = first_audio_at
        tool_timings["response_done_at"] = done_at
        tool_timings["session_slot"] = self.slot_id
        tool_timings["cancelled"] = status == "cancelled"
        self._assert_isolation(response_text, require_own=status == "completed")
        return TurnResult(
            profile_id=self.profile_id,
            session_concurrency=self.session_concurrency,
            arrival_mode=self.arrival_mode,
            workload=self.workload,
            session_index=self.session_index,
            session_label=self.label,
            session_id=self.session_id,
            turn_index=turn_index,
            turn_type=turn_type,
            llama_slot_id=self.slot_id,
            tts_worker_id=tts_worker_id,
            audio_chunk_count=audio_chunk_count,
            audio_before_tool_result=audio_before_tool_result,
            duplicate_tool_execution=duplicate_tool_execution,
            transcript=transcript,
            response_text=response_text,
            status=status,
            isolation_canary=self._canary,
            metrics=metrics,
            tool_timings=tool_timings,
        )

    async def _cancel_after_audio_if_requested(
        self,
        event: Any,
        *,
        enabled: bool,
        already_cancelled: bool,
    ) -> bool:
        if not enabled or already_cancelled:
            return already_cancelled
        await self._send(
            {
                "type": "response.cancel",
                "event_id": event_id(),
                "protocol_version": 2,
                "session_id": self.session_id,
                "response_id": event.response_id,
                "generation_id": event.generation_id,
            }
        )
        return True

    async def _complete_tool_call(self, tool_call: Any) -> dict[str, float | int]:
        arguments = json.loads(tool_call.arguments)
        started_at = time.monotonic()
        if self._tool_delay_seconds:
            await asyncio.sleep(self._tool_delay_seconds)
        output = str(int(arguments["a"]) + int(arguments["b"]))
        finished_at = time.monotonic()
        output_event_id = event_id()
        await self._send(
            {
                "type": "conversation.item.create",
                "event_id": output_event_id,
                "protocol_version": 2,
                "session_id": self.session_id,
                "item": {
                    "type": "function_call_output",
                    "id": f"item_tool_output_{uuid.uuid4().hex}",
                    "call_id": tool_call.call_id,
                    "name": tool_call.name,
                    "output": output,
                    "is_error": False,
                    "turn_id": tool_call.turn_id,
                    "turn_revision": tool_call.turn_revision,
                    "generation_id": tool_call.generation_id,
                    "response_id": tool_call.response_id,
                },
            }
        )
        while True:
            ack = await self._receive_event(timeout_seconds=30.0)
            if ack.type == "conversation.item.created" and ack.source_event_id == output_event_id:
                break
            if ack.type == "error":
                raise RuntimeError(f"tool output rejected: {ack.error.message}")
        return {
            "tool_execution_started_at": started_at,
            "tool_execution_finished_at": finished_at,
            "tool_result_ack_at": time.monotonic(),
            "result_size": len(output),
        }

    async def run_barge_in_probe(self) -> TurnResult:
        """Interrupt audible generation with real input audio and finish the new turn."""
        await self._send_audio(self._audio)
        await self._commit()
        old_generation: str | None = None
        old_response: str | None = None
        old_turn: str | None = None
        while old_generation is None:
            event = await self._receive_event(timeout_seconds=180.0)
            if event.type == "error":
                raise RuntimeError(f"{self.label}: {event.error.code}: {event.error.message}")
            if event.type == "response.output_audio.delta":
                old_generation = event.generation_id
                old_response = event.response_id
                old_turn = event.turn_id

        interrupt_at = time.monotonic()
        await self._send_audio(self._audio)
        speech_stop = time.monotonic()
        await self._commit()
        transcript = ""
        response_text = ""
        first_transcript_at: float | None = None
        first_text_at: float | None = None
        first_audio_at: float | None = None
        audio_chunk_count = 0
        tts_worker_id: int | None = None
        new_response_at: float | None = None
        last_old_audio_at: float | None = None
        old_cancelled = False
        status = "unknown"
        while True:
            event = await self._receive_event(timeout_seconds=180.0)
            now = time.monotonic()
            if event.type == "error":
                raise RuntimeError(f"{self.label}: {event.error.code}: {event.error.message}")
            generation = getattr(event, "generation_id", None)
            if event.type == "response.output_audio.delta" and generation == old_generation:
                last_old_audio_at = now
            elif event.type == "response.done" and generation == old_generation:
                if event.response_id != old_response or event.reason.value != "barge_in":
                    raise RuntimeError("old response did not terminate with the barge_in reason")
                old_cancelled = True
            elif (
                event.type == "conversation.item.input_audio_transcription.completed"
                and event.turn_id != old_turn
            ):
                transcript = event.transcript
                first_transcript_at = now
            elif (
                event.type == "response.created"
                and generation != old_generation
                and event.turn_id != old_turn
            ):
                new_response_at = now
            elif (
                event.type == "response.output_text.delta"
                and generation != old_generation
                and event.turn_id != old_turn
            ):
                response_text += event.delta
                first_text_at = first_text_at or now
            elif (
                event.type == "response.output_audio.delta"
                and generation != old_generation
                and event.turn_id != old_turn
            ):
                first_audio_at = first_audio_at or now
                if tts_worker_id is None:
                    tts_worker_id = event.tts_worker_id
                audio_chunk_count += 1
            elif (
                event.type == "response.done"
                and generation != old_generation
                and event.turn_id != old_turn
            ):
                status = event.status.value
                done_at = now
                break
        if not old_cancelled:
            raise RuntimeError(
                "new response completed before the interrupted response was cancelled"
            )
        metrics = {
            "barge_in_to_last_old_audio_frame_seconds": max(
                0.0, (last_old_audio_at or interrupt_at) - interrupt_at
            )
        }
        if first_transcript_at is not None:
            metrics["speech_stop_to_final_transcript_seconds"] = first_transcript_at - speech_stop
        if first_text_at is not None:
            metrics["speech_stop_to_first_text_delta_seconds"] = first_text_at - speech_stop
        if first_audio_at is not None:
            metrics["speech_stop_to_first_audio_frame_seconds"] = first_audio_at - speech_stop
        if new_response_at is not None:
            metrics["response_duration_seconds"] = done_at - new_response_at
        self._assert_isolation(response_text, require_own=status == "completed")
        return TurnResult(
            profile_id=self.profile_id,
            session_concurrency=self.session_concurrency,
            arrival_mode=self.arrival_mode,
            workload=self.workload,
            session_index=self.session_index,
            session_label=self.label,
            session_id=self.session_id,
            turn_index=0,
            turn_type="barge_in",
            llama_slot_id=self.slot_id,
            tts_worker_id=tts_worker_id,
            audio_chunk_count=audio_chunk_count,
            audio_before_tool_result=False,
            duplicate_tool_execution=False,
            transcript=transcript,
            response_text=response_text,
            status=status,
            isolation_canary=self._canary,
            metrics=metrics,
        )

    def _assert_isolation(self, response_text: str, *, require_own: bool) -> None:
        normalized = response_text.upper()
        leaked = sorted(canary for canary in self._forbidden_canaries if canary in normalized)
        if leaked:
            raise RuntimeError(f"{self.label}: response contains foreign isolation marker")
        if require_own and self._canary not in normalized:
            raise RuntimeError(f"{self.label}: completed response omitted its isolation marker")

    async def _set_tool_choice(self, choice: str) -> None:
        if self.workload == "normal":
            return
        update_id = event_id()
        session = {**self._session_config, "tool_choice": choice}
        self._session_config = session
        await self._send(
            {
                "type": "session.update",
                "event_id": update_id,
                "protocol_version": 2,
                "session_id": self.session_id,
                "session": session,
            }
        )
        while True:
            ack = await self._receive_event(timeout_seconds=20.0)
            if ack.type == "session.updated" and ack.source_event_id == update_id:
                return
            if ack.type == "error":
                raise RuntimeError(f"{self.label}: tool choice update failed: {ack.error.message}")

    async def _send_audio(self, audio: bytes) -> None:
        for offset in range(0, len(audio), FRAME_BYTES):
            frame = audio[offset : offset + FRAME_BYTES]
            await self._send(
                {
                    "type": "input_audio_buffer.append",
                    "event_id": event_id(),
                    "protocol_version": 2,
                    "session_id": self.session_id,
                    "sequence": self._audio_sequence,
                    "audio": base64.b64encode(frame).decode("ascii"),
                }
            )
            self._audio_sequence += 1
            if self._realtime_audio:
                await asyncio.sleep(len(frame) / 2 / 16_000)

    async def _commit(self) -> None:
        await self._send(
            {
                "type": "input_audio_buffer.commit",
                "event_id": event_id(),
                "protocol_version": 2,
                "session_id": self.session_id,
            }
        )

    async def _send(self, payload: dict[str, Any]) -> None:
        if self.ws is None:
            raise RuntimeError("session is not connected")
        await self.ws.send_json(payload)

    async def _receive_event(self, *, timeout_seconds: float) -> Any:
        if self.ws is None:
            raise RuntimeError("session is not connected")
        message = await asyncio.wait_for(self.ws.receive(), timeout=timeout_seconds)
        if message.type is not aiohttp.WSMsgType.TEXT:
            raise RuntimeError(f"{self.label}: WebSocket ended with {message.type}: {message.data}")
        return parse_server_event_json(message.data)


async def fetch_text(
    session: aiohttp.ClientSession, service_url: str, path: str, token: str | None = None
) -> str:
    headers = {"Authorization": f"Bearer {token}"} if token else None
    async with session.get(
        http_url(service_url, path),
        headers=headers,
        timeout=aiohttp.ClientTimeout(total=10.0),
    ) as response:
        response.raise_for_status()
        return await response.text()


async def run(args: argparse.Namespace) -> int:
    token = args.token_file.read_text(encoding="utf-8").strip()
    if not token or any(character.isspace() for character in token):
        raise ValueError("token file must contain one non-empty bearer token")
    audio_paths: list[Path] = args.wav
    audio_inputs = [read_pcm16(path) for path in audio_paths]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite measured data: {args.output}")
    writer = JsonlWriter(args.output)

    connector = aiohttp.TCPConnector(limit=max(8, args.sessions * 2))
    async with aiohttp.ClientSession(connector=connector) as http:
        models = json.loads(await fetch_text(http, args.service_url, "/v1/models", token))
        profile_id_value = models.get("profile_id")
        if not isinstance(profile_id_value, str) or not profile_id_value:
            raise RuntimeError("/v1/models omitted the required profile_id")
        profile_id = profile_id_value
        configuration_fingerprint = hashlib.sha256(
            json.dumps(models, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        gpu = command_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ]
        )
        metadata = {
            "started_at": datetime.now(UTC).isoformat(),
            "profile_id": profile_id,
            "configuration_fingerprint": configuration_fingerprint,
            "requested_duration_seconds": args.duration,
            "session_concurrency": args.sessions,
            "arrival_mode": args.arrival,
            "workload": args.workload,
            "tool_delay_ms": args.tool_delay_ms,
            "seed": args.seed,
            "host": platform.node(),
            "platform": platform.platform(),
            "gpu": gpu,
            "cuda_runtime": command_output(
                ["python", "-c", "import torch; print(torch.version.cuda or 'unknown')"]
            ),
            "git_commit": command_output(["git", "rev-parse", "HEAD"]),
            "git_status": command_output(["git", "status", "--short"]),
            "container_image_digest": os.environ.get("HV_CONTAINER_IMAGE_DIGEST"),
            "service_models": models,
            "wav_sha256": [command_output(["sha256sum", str(path)]) for path in audio_paths],
        }
        await writer.write({"record_type": "metadata", "metadata": metadata})
        canaries = build_canaries(args.sessions, args.seed)
        sessions = []
        for index in range(args.sessions):
            canary = canaries[index]
            sessions.append(
                SoakSession(
                    profile_id=profile_id,
                    session_concurrency=args.sessions,
                    arrival_mode=args.arrival,
                    workload=args.workload,
                    session_index=index,
                    label=f"session-{index:02d}",
                    session=http,
                    url=websocket_url(args.service_url),
                    token=token,
                    audio=audio_inputs[index % len(audio_inputs)],
                    realtime_audio=not args.no_realtime_audio,
                    canary=canary,
                    forbidden_canaries=frozenset(canaries) - {canary},
                    tool_delay_seconds=args.tool_delay_ms / 1_000.0,
                )
            )
        await asyncio.gather(*(session.connect() for session in sessions))
        if len({session.session_id for session in sessions}) != len(sessions):
            raise RuntimeError("service returned a duplicate session_id")
        if len({session.slot_id for session in sessions}) != len(sessions):
            raise RuntimeError("service assigned a duplicate llama.cpp slot")

        if not args.skip_barge_in_probe:
            probe_record = (await sessions[0].run_barge_in_probe()).record()
            probe_record["record_type"] = "probe"
            await writer.write(probe_record)

        await writer.write(
            {
                "record_type": "prometheus",
                "phase": "before",
                "text": await fetch_text(http, args.service_url, "/metrics"),
            }
        )
        measurement_started_at = time.monotonic()
        deadline = time.monotonic() + args.duration

        def turn_type(client: SoakSession, turn: int) -> str:
            return select_turn_type(str(args.workload), client.session_index, turn)

        async def run_one(
            client: SoakSession, turn: int, barrier: asyncio.Barrier | None = None
        ) -> None:
            selected_turn_type = turn_type(client, turn)
            try:
                result = await client.run_turn(
                    turn,
                    turn_type=selected_turn_type,
                    cancel_after_audio=args.cancel_every > 0 and turn % args.cancel_every == 0,
                    commit_barrier=barrier,
                )
                await writer.write(result.record())
                if args.reconnect_every > 0 and turn % args.reconnect_every == 0:
                    previous_session_id = client.session_id
                    previous_slot_id = client.slot_id
                    await client.reconnect()
                    await writer.write(
                        {
                            "record_type": "reconnect",
                            "timestamp": datetime.now(UTC).isoformat(),
                            "session_index": client.session_index,
                            "previous_session_id": previous_session_id,
                            "previous_llama_slot_id": previous_slot_id,
                            "session_id": client.session_id,
                            "llama_slot_id": client.slot_id,
                        }
                    )
            except Exception as exc:
                await writer.write(
                    {
                        "record_type": "error",
                        "timestamp": datetime.now(UTC).isoformat(),
                        "profile_id": profile_id,
                        "session_concurrency": args.sessions,
                        "arrival_mode": args.arrival,
                        "workload": args.workload,
                        "session_index": client.session_index,
                        "session_label": client.label,
                        "turn_index": turn,
                        "turn_type": selected_turn_type,
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                        "tool_call_error": selected_turn_type == "tool",
                    }
                )
                raise

        async def staggered_worker(client: SoakSession) -> None:
            if client.session_index:
                await asyncio.sleep(min(args.pause, 0.25) * client.session_index)
            turn = 0
            while time.monotonic() < deadline:
                turn += 1
                await run_one(client, turn)
                await asyncio.sleep(args.pause)

        try:
            if args.arrival == "barrier":
                turn = 0
                while time.monotonic() < deadline:
                    turn += 1
                    barrier = asyncio.Barrier(args.sessions)
                    await asyncio.gather(*(run_one(session, turn, barrier) for session in sessions))
                    await asyncio.sleep(args.pause)
            else:
                await asyncio.gather(*(staggered_worker(session) for session in sessions))
        finally:
            await asyncio.gather(*(session.close() for session in sessions))
        await writer.write(
            {
                "record_type": "prometheus",
                "phase": "after",
                "text": await fetch_text(http, args.service_url, "/metrics"),
            }
        )
        await writer.write(
            {
                "record_type": "run_complete",
                "timestamp": datetime.now(UTC).isoformat(),
                "actual_duration_seconds": time.monotonic() - measurement_started_at,
            }
        )
    print(args.output)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--service-url", default=os.environ.get("HV_GPU_SERVICE_URL", "http://127.0.0.1:8765")
    )
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--wav", type=Path, action="append", required=True)
    parser.add_argument("--sessions", type=int, default=2)
    parser.add_argument("--duration", type=float, default=1_800.0)
    parser.add_argument("--arrival", choices=("staggered", "barrier"), default="staggered")
    parser.add_argument("--workload", choices=("normal", "tool", "mixed"), default="mixed")
    parser.add_argument("--tool-delay-ms", type=int, default=0)
    parser.add_argument("--pause", type=float, default=0.25)
    parser.add_argument("--cancel-every", type=int, default=7)
    parser.add_argument("--reconnect-every", type=int, default=11)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--no-realtime-audio", action="store_true")
    parser.add_argument("--skip-barge-in-probe", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(f"benchmarks/reports/soak-{datetime.now(UTC):%Y%m%dT%H%M%SZ}.raw.jsonl"),
    )
    args = parser.parse_args()
    if not 1 <= args.sessions <= 16:
        parser.error("--sessions must be in the range 1..16")
    if (
        args.duration <= 0
        or args.pause < 0
        or args.cancel_every < 0
        or args.reconnect_every < 0
        or args.tool_delay_ms < 0
    ):
        parser.error("duration must be positive; delays and cadences cannot be negative")
    if args.workload in {"tool", "mixed"}:
        args.skip_barge_in_probe = True
    return args


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run(parse_args())))
