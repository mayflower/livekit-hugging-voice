#!/usr/bin/env python3
"""Run two real isolated service sessions and record raw latency observations."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import platform
import subprocess
import time
import uuid
import wave
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import aiohttp
from hugging_voice_protocol.events import parse_server_event_json

SUBPROTOCOL = "hugging-voice-livekit.v1"
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
    session_label: str
    session_id: str
    turn_index: int
    transcript: str
    response_text: str
    status: str
    isolation_canary: str
    metrics: dict[str, float]

    def record(self) -> dict[str, Any]:
        return {
            "record_type": "turn",
            "timestamp": datetime.now(UTC).isoformat(),
            "session_label": self.session_label,
            "session_id": self.session_id,
            "turn_index": self.turn_index,
            "transcript_chars": len(self.transcript),
            "response_chars": len(self.response_text),
            "status": self.status,
            "isolation_canary": self.isolation_canary,
            "cross_session_leak": False,
            "metrics": self.metrics,
        }


class SoakSession:
    def __init__(
        self,
        *,
        label: str,
        session: aiohttp.ClientSession,
        url: str,
        token: str,
        audio: bytes,
        realtime_audio: bool,
        canary: str,
        forbidden_canary: str,
    ) -> None:
        self.label = label
        self._http = session
        self._url = url
        self._token = token
        self._audio = audio
        self._realtime_audio = realtime_audio
        self._canary = canary
        self._forbidden_canary = forbidden_canary
        self.ws: aiohttp.ClientWebSocketResponse | None = None
        self.session_id = ""
        self._audio_sequence = 0

    async def connect(self) -> None:
        self.ws = await self._http.ws_connect(
            self._url,
            headers={"Authorization": f"Bearer {self._token}"},
            protocols=[SUBPROTOCOL],
            heartbeat=20.0,
            timeout=10.0,
            max_msg_size=1_000_000,
        )
        event = await self._receive_event(timeout_seconds=20.0)
        if event.type != "session.created":
            raise RuntimeError(f"{self.label}: expected session.created, got {event.type}")
        self.session_id = event.session_id
        self._audio_sequence = 0
        await self._send(
            {
                "type": "session.update",
                "event_id": event_id(),
                "protocol_version": 1,
                "session_id": self.session_id,
                "session": {
                    "instructions": (
                        "Antworte auf Deutsch. Beende jede vollständige Antwort exakt mit "
                        f"dem isolierten Marker {self._canary}. Verwende niemals den Marker "
                        f"{self._forbidden_canary}."
                    )
                },
            }
        )

    async def close(self) -> None:
        if self.ws is not None:
            await self.ws.close()
            self.ws = None

    async def reconnect(self) -> None:
        await self.close()
        await self.connect()

    async def run_turn(self, turn_index: int, *, cancel_after_audio: bool) -> TurnResult:
        if self.ws is None:
            raise RuntimeError("session is not connected")
        await self._send_audio(self._audio)
        speech_stop = time.monotonic()
        await self._send(
            {
                "type": "input_audio_buffer.commit",
                "event_id": event_id(),
                "protocol_version": 1,
                "session_id": self.session_id,
            }
        )

        transcript = ""
        response_text = ""
        first_transcript_at: float | None = None
        first_text_at: float | None = None
        first_audio_at: float | None = None
        response_created_at: float | None = None
        cancelled = False
        status = "unknown"
        while True:
            event = await self._receive_event(timeout_seconds=180.0)
            now = time.monotonic()
            if event.type == "error":
                raise RuntimeError(f"{self.label}: {event.error.code}: {event.error.message}")
            if event.type == "conversation.item.input_audio_transcription.completed":
                transcript = event.transcript
                first_transcript_at = now
            elif event.type == "response.created":
                response_created_at = now
            elif event.type == "response.output_text.delta":
                response_text += event.delta
                first_text_at = first_text_at or now
            elif event.type == "response.output_audio.delta":
                first_audio_at = first_audio_at or now
                if cancel_after_audio and not cancelled:
                    await self._send(
                        {
                            "type": "response.cancel",
                            "event_id": event_id(),
                            "protocol_version": 1,
                            "session_id": self.session_id,
                            "response_id": event.response_id,
                            "generation_id": event.generation_id,
                        }
                    )
                    cancelled = True
            elif event.type == "response.done":
                status = event.status.value
                done_at = now
                break

        metrics: dict[str, float] = {}
        if first_transcript_at is not None:
            metrics["speech_stop_to_final_transcript_seconds"] = first_transcript_at - speech_stop
        if first_text_at is not None:
            metrics["speech_stop_to_first_text_delta_seconds"] = first_text_at - speech_stop
        if first_audio_at is not None:
            metrics["speech_stop_to_first_audio_frame_seconds"] = first_audio_at - speech_stop
        if response_created_at is not None:
            metrics["response_duration_seconds"] = done_at - response_created_at
        self._assert_isolation(response_text, require_own=status == "completed")
        return TurnResult(
            session_label=self.label,
            session_id=self.session_id,
            turn_index=turn_index,
            transcript=transcript,
            response_text=response_text,
            status=status,
            isolation_canary=self._canary,
            metrics=metrics,
        )

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
            session_label=self.label,
            session_id=self.session_id,
            turn_index=0,
            transcript=transcript,
            response_text=response_text,
            status=status,
            isolation_canary=self._canary,
            metrics=metrics,
        )

    def _assert_isolation(self, response_text: str, *, require_own: bool) -> None:
        normalized = response_text.upper()
        if self._forbidden_canary in normalized:
            raise RuntimeError(
                f"{self.label}: response contains the other session's isolation marker"
            )
        if require_own and self._canary not in normalized:
            raise RuntimeError(f"{self.label}: completed response omitted its isolation marker")

    async def _send_audio(self, audio: bytes) -> None:
        for offset in range(0, len(audio), FRAME_BYTES):
            frame = audio[offset : offset + FRAME_BYTES]
            await self._send(
                {
                    "type": "input_audio_buffer.append",
                    "event_id": event_id(),
                    "protocol_version": 1,
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
                "protocol_version": 1,
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
    async with session.get(http_url(service_url, path), headers=headers, timeout=10.0) as response:
        response.raise_for_status()
        return await response.text()


async def run(args: argparse.Namespace) -> int:
    token = args.token_file.read_text(encoding="utf-8").strip()
    if not token or any(character.isspace() for character in token):
        raise ValueError("token file must contain one non-empty bearer token")
    audio_a = read_pcm16(args.wav_a)
    audio_b = read_pcm16(args.wav_b)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite measured data: {args.output}")
    writer = JsonlWriter(args.output)

    connector = aiohttp.TCPConnector(limit=8)
    async with aiohttp.ClientSession(connector=connector) as http:
        models = json.loads(await fetch_text(http, args.service_url, "/v1/models", token))
        gpu = command_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ]
        )
        metadata = {
            "started_at": datetime.now(UTC).isoformat(),
            "requested_duration_seconds": args.duration,
            "host": platform.node(),
            "platform": platform.platform(),
            "gpu": gpu,
            "cuda_runtime": command_output(
                ["python", "-c", "import torch; print(torch.version.cuda or 'unknown')"]
            ),
            "git_commit": command_output(["git", "rev-parse", "HEAD"]),
            "container_image_digest": os.environ.get("HV_CONTAINER_IMAGE_DIGEST"),
            "service_models": models,
            "wav_a_sha256": command_output(["sha256sum", str(args.wav_a)]),
            "wav_b_sha256": command_output(["sha256sum", str(args.wav_b)]),
        }
        await writer.write({"record_type": "metadata", "metadata": metadata})
        await writer.write(
            {
                "record_type": "prometheus",
                "phase": "before",
                "text": await fetch_text(http, args.service_url, "/metrics"),
            }
        )

        sessions = [
            SoakSession(
                label="alpha",
                session=http,
                url=websocket_url(args.service_url),
                token=token,
                audio=audio_a,
                realtime_audio=not args.no_realtime_audio,
                canary="ALPHAEINS",
                forbidden_canary="BETAZWEI",
            ),
            SoakSession(
                label="beta",
                session=http,
                url=websocket_url(args.service_url),
                token=token,
                audio=audio_b,
                realtime_audio=not args.no_realtime_audio,
                canary="BETAZWEI",
                forbidden_canary="ALPHAEINS",
            ),
        ]
        await asyncio.gather(*(session.connect() for session in sessions))
        if sessions[0].session_id == sessions[1].session_id:
            raise RuntimeError("service returned the same session_id to two connections")

        if not args.skip_barge_in_probe:
            await writer.write((await sessions[0].run_barge_in_probe()).record())

        deadline = time.monotonic() + args.duration

        async def worker(client: SoakSession) -> None:
            turn = 0
            while time.monotonic() < deadline:
                turn += 1
                try:
                    result = await client.run_turn(
                        turn,
                        cancel_after_audio=args.cancel_every > 0 and turn % args.cancel_every == 0,
                    )
                    await writer.write(result.record())
                    if args.reconnect_every > 0 and turn % args.reconnect_every == 0:
                        await client.reconnect()
                except Exception as exc:
                    await writer.write(
                        {
                            "record_type": "error",
                            "timestamp": datetime.now(UTC).isoformat(),
                            "session_label": client.label,
                            "turn_index": turn,
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                        }
                    )
                    raise
                await asyncio.sleep(args.pause)

        try:
            await asyncio.gather(*(worker(session) for session in sessions))
        finally:
            await asyncio.gather(*(session.close() for session in sessions))
        await writer.write(
            {
                "record_type": "prometheus",
                "phase": "after",
                "text": await fetch_text(http, args.service_url, "/metrics"),
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
    parser.add_argument("--wav-a", type=Path, required=True)
    parser.add_argument("--wav-b", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=1_800.0)
    parser.add_argument("--pause", type=float, default=0.25)
    parser.add_argument("--cancel-every", type=int, default=7)
    parser.add_argument("--reconnect-every", type=int, default=11)
    parser.add_argument("--no-realtime-audio", action="store_true")
    parser.add_argument("--skip-barge-in-probe", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            f"benchmarks/reports/two-session-{datetime.now(UTC):%Y%m%dT%H%M%SZ}.raw.jsonl"
        ),
    )
    args = parser.parse_args()
    if args.duration <= 0 or args.pause < 0 or args.cancel_every < 0 or args.reconnect_every < 0:
        parser.error("duration must be positive; pause/cadences cannot be negative")
    return args


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run(parse_args())))
