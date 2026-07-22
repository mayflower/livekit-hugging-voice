from __future__ import annotations

import asyncio
import os
import time
import wave
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import pytest
from livekit import rtc
from livekit.agents import Agent, AgentSession
from livekit.agents.voice.io import AudioInput, AudioOutput, AudioOutputCapabilities
from livekit.plugins.hugging_voice import RealtimeModel


def external_assets() -> tuple[Path, Path]:
    if os.environ.get("HV_RUN_GPU_TESTS") != "1":
        pytest.skip("set HV_RUN_GPU_TESTS=1 to run the real AgentSession GPU test")
    token = os.environ.get("HV_GPU_TOKEN_FILE")
    wav = os.environ.get("HV_GPU_WAV_A")
    if not token or not wav:
        pytest.skip("HV_GPU_TOKEN_FILE and HV_GPU_WAV_A are required")
    paths = Path(token), Path(wav)
    absent = [str(path) for path in paths if not path.is_file()]
    if absent:
        pytest.skip(f"external GPU assets are absent: {', '.join(absent)}")
    return paths


def realtime_url(service_url: str) -> str:
    parts = urlsplit(service_url)
    scheme = {"http": "ws", "https": "wss", "ws": "ws", "wss": "wss"}.get(parts.scheme)
    if scheme is None or not parts.netloc:
        raise ValueError("HV_GPU_SERVICE_URL must use http, https, ws, or wss")
    return urlunsplit((scheme, parts.netloc, "/v1/realtime", "", ""))


def wav_frames(path: Path) -> list[rtc.AudioFrame]:
    with wave.open(str(path), "rb") as source:
        if source.getparams()[:3] != (1, 2, 16_000) or source.getcomptype() != "NONE":
            raise ValueError(f"{path} must be uncompressed mono 16 kHz PCM16 WAV")
        audio = source.readframes(source.getnframes())
    frame_bytes = 1_280
    padded = audio + bytes((-len(audio)) % frame_bytes)
    padded += bytes(frame_bytes * 15)  # 600 ms of real silence for server VAD.
    return [
        rtc.AudioFrame(
            data=padded[offset : offset + frame_bytes],
            sample_rate=16_000,
            num_channels=1,
            samples_per_channel=640,
        )
        for offset in range(0, len(padded), frame_bytes)
    ]


class QueueAudioInput(AudioInput):
    def __init__(self) -> None:
        super().__init__(label="real-gpu-input")
        self._queue: asyncio.Queue[rtc.AudioFrame | None] = asyncio.Queue(maxsize=64)

    async def __anext__(self) -> rtc.AudioFrame:
        frame = await self._queue.get()
        if frame is None:
            raise StopAsyncIteration
        return frame

    async def push(self, frame: rtc.AudioFrame) -> None:
        await self._queue.put(frame)

    async def close(self) -> None:
        while self._queue.full():
            self._queue.get_nowait()
        self._queue.put_nowait(None)


class CapturingAudioOutput(AudioOutput):
    def __init__(self) -> None:
        super().__init__(
            label="real-gpu-output",
            capabilities=AudioOutputCapabilities(pause=False),
            sample_rate=24_000,
        )
        self.frames: list[rtc.AudioFrame] = []
        self.first_frame = asyncio.Event()
        self._segment_duration = 0.0

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        if not self._segment_duration:
            self.on_playback_started(created_at=time.time())
        await super().capture_frame(frame)
        self.frames.append(frame)
        self._segment_duration += frame.duration
        self.first_frame.set()

    def flush(self) -> None:
        super().flush()
        if self._segment_duration:
            self.on_playback_finished(
                playback_position=self._segment_duration,
                interrupted=False,
            )
        self._segment_duration = 0.0

    def clear_buffer(self) -> None:
        super().flush()
        if self._segment_duration:
            self.on_playback_finished(
                playback_position=self._segment_duration,
                interrupted=True,
            )
        self.frames.clear()
        self._segment_duration = 0.0


@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.asyncio
async def test_external_agent_session_builtin_transcription_and_audio() -> None:
    token_path, wav_path = external_assets()
    token = token_path.read_text(encoding="utf-8").strip()
    if not token or any(character.isspace() for character in token):
        raise ValueError("token file must contain one non-empty bearer token")

    model = RealtimeModel(
        base_url=realtime_url(os.environ.get("HV_GPU_SERVICE_URL", "http://127.0.0.1:8765")),
        token=token,
    )
    session: AgentSession[dict[str, Any]] = AgentSession(llm=model)
    audio_input = QueueAudioInput()
    audio_output = CapturingAudioOutput()
    session.input.audio = audio_input
    session.output.audio = audio_output
    loop = asyncio.get_running_loop()
    final_transcript: asyncio.Future[str] = loop.create_future()

    @session.on("user_input_transcribed")
    def on_user_input_transcribed(event: Any) -> None:
        if event.is_final and event.transcript and not final_transcript.done():
            final_transcript.set_result(event.transcript)

    try:
        await session.start(
            agent=Agent(instructions="Antworte kurz, natürlich und ausschließlich auf Deutsch."),
            record=False,
        )
        for frame in wav_frames(wav_path):
            await asyncio.wait_for(audio_input.push(frame), timeout=5.0)
            await asyncio.sleep(frame.duration)
        transcript, _ = await asyncio.wait_for(
            asyncio.gather(final_transcript, audio_output.first_frame.wait()),
            timeout=240.0,
        )
        assert transcript.strip()
        assert audio_output.frames
        assert all(frame.sample_rate == 24_000 for frame in audio_output.frames)
        assert session.llm is model
        assert session.stt is None
        assert session.tts is None
    finally:
        await audio_input.close()
        await session.aclose()
        await model.aclose()
