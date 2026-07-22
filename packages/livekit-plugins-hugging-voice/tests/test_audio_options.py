from __future__ import annotations

import asyncio
import sys
from array import array
from pathlib import Path

import pytest
from livekit import rtc
from livekit.agents import APIConnectOptions
from livekit.agents.llm import RealtimeError
from livekit.plugins.hugging_voice.audio import InputAudioProcessor
from livekit.plugins.hugging_voice.options import resolve_base_urls, resolve_token
from livekit.plugins.hugging_voice.realtime import RealtimeModel


def pcm16(values: list[int]) -> bytes:
    samples = array("h", values)
    if sys.byteorder != "little":
        samples.byteswap()
    return samples.tobytes()


def samples(frames: list[rtc.AudioFrame]) -> list[int]:
    result: list[int] = []
    for frame in frames:
        values = array("h")
        values.frombytes(bytes(frame.data))
        if sys.byteorder != "little":
            values.byteswap()
        result.extend(values)
    return list(result)


def test_audio_processor_downmixes_48khz_stereo_and_frames_16khz_mono() -> None:
    processor = InputAudioProcessor()
    interleaved = [value for _ in range(4_800) for value in (1_000, 3_000)]
    frame = rtc.AudioFrame(
        data=pcm16(interleaved),
        sample_rate=48_000,
        num_channels=2,
        samples_per_channel=4_800,
    )
    frames = processor.push(frame) + processor.flush()

    assert frames
    assert all(output.sample_rate == 16_000 for output in frames)
    assert all(output.num_channels == 1 for output in frames)
    assert all(output.samples_per_channel == 640 for output in frames[:-1])
    converted = samples(frames)
    assert 1_590 <= len(converted) <= 1_610
    assert all(abs(value - 2_000) <= 2 for value in converted[100:-100])


def test_audio_processor_preserves_exact_16khz_mono_frame() -> None:
    processor = InputAudioProcessor()
    payload = pcm16([index - 320 for index in range(640)])
    output = processor.push(
        rtc.AudioFrame(
            data=payload,
            sample_rate=16_000,
            num_channels=1,
            samples_per_channel=640,
        )
    )
    assert len(output) == 1
    assert bytes(output[0].data) == payload
    assert processor.flush() == []


def test_base_url_and_token_resolution_are_strict(tmp_path: Path) -> None:
    secret = tmp_path / "token"
    secret.write_text("secret\n", encoding="utf-8")
    assert resolve_token(token=None, token_file=secret) == "secret"
    assert resolve_base_urls(base_url="ws://gpu:8765", base_urls=None) == (
        "ws://gpu:8765/v1/realtime",
    )
    assert resolve_base_urls(
        base_url=None,
        base_urls=["wss://one/v1/realtime", "wss://one/v1/realtime", "ws://two:8765"],
    ) == ("wss://one/v1/realtime", "ws://two:8765/v1/realtime")
    with pytest.raises(ValueError):
        resolve_base_urls(base_url="https://gpu", base_urls=None)
    with pytest.raises(ValueError):
        resolve_base_urls(base_url="ws://gpu", base_urls=["ws://other"])
    with pytest.raises(ValueError):
        resolve_token(token="inline", token_file=secret)


def test_capabilities_are_truthful_and_fixed() -> None:
    model = RealtimeModel(base_url="ws://127.0.0.1:1", token="secret")
    assert model.model == "hugging-voice-gemma4-parakeet-qwen3-tts"
    assert model.provider == "hugging-voice"
    assert model.capabilities.turn_detection
    assert model.capabilities.user_transcription
    assert model.capabilities.audio_output
    assert model.capabilities.mutable_instructions
    assert not model.capabilities.message_truncation
    assert not model.capabilities.mutable_tools
    assert not model.capabilities.manual_function_calls


@pytest.mark.asyncio
async def test_unsupported_tools_video_and_truncation_are_rejected() -> None:
    model = RealtimeModel(
        base_url="ws://127.0.0.1:1",
        token="secret",
        conn_options=APIConnectOptions(max_retry=0, timeout=0.01),
    )
    session = model.session()
    try:
        with pytest.raises(RealtimeError):
            session.push_video(
                rtc.VideoFrame(
                    width=1,
                    height=1,
                    type=rtc.VideoBufferType.RGBA,
                    data=bytes(4),
                )
            )
        with pytest.raises(RealtimeError):
            session.truncate(message_id="item_x", modalities=["audio"], audio_end_ms=0)
        with pytest.raises(RealtimeError):
            session.update_options(tool_choice="auto")
        await session.update_tools([])
        await asyncio.sleep(0)
    finally:
        await model.aclose()
