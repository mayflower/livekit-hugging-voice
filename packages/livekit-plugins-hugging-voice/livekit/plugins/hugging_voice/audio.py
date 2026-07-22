"""Bounded input audio conversion for the LiveKit adapter."""

from __future__ import annotations

import sys
from array import array

from livekit import rtc
from livekit.agents.utils.audio import AudioByteStream

INPUT_SAMPLE_RATE = 16_000
INPUT_FRAME_SAMPLES = 640


class InputAudioProcessor:
    """Downmix, resample, and frame one session's serial audio stream."""

    def __init__(self) -> None:
        self._source_rate: int | None = None
        self._resampler: rtc.AudioResampler | None = None
        self._framer = AudioByteStream(
            sample_rate=INPUT_SAMPLE_RATE,
            num_channels=1,
            samples_per_channel=INPUT_FRAME_SAMPLES,
        )

    def push(self, frame: rtc.AudioFrame) -> list[rtc.AudioFrame]:
        if frame.num_channels not in {1, 2}:
            raise ValueError("Hugging Voice accepts only mono or stereo LiveKit audio")
        if frame.sample_rate <= 0:
            raise ValueError("LiveKit audio sample rate must be positive")
        if self._source_rate is not None and frame.sample_rate != self._source_rate:
            raise ValueError("input sample rate changed during an audio stream")
        self._source_rate = frame.sample_rate
        mono = self._to_mono(frame)
        if frame.sample_rate == INPUT_SAMPLE_RATE:
            resampled = [mono]
        else:
            if self._resampler is None:
                self._resampler = rtc.AudioResampler(
                    input_rate=frame.sample_rate,
                    output_rate=INPUT_SAMPLE_RATE,
                    num_channels=1,
                    quality=rtc.AudioResamplerQuality.MEDIUM,
                )
            resampled = self._resampler.push(mono)
        output: list[rtc.AudioFrame] = []
        for converted in resampled:
            output.extend(self._framer.push(converted.data))
        return output

    def flush(self) -> list[rtc.AudioFrame]:
        output: list[rtc.AudioFrame] = []
        if self._resampler is not None:
            for frame in self._resampler.flush():
                output.extend(self._framer.push(frame.data))
        output.extend(self._framer.flush())
        self.reset()
        return output

    def clear(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._source_rate = None
        self._resampler = None
        self._framer.clear()

    @staticmethod
    def _to_mono(frame: rtc.AudioFrame) -> rtc.AudioFrame:
        if frame.num_channels == 1:
            return frame
        samples = array("h")
        samples.frombytes(bytes(frame.data))
        if sys.byteorder != "little":
            samples.byteswap()
        mono = array(
            "h",
            (
                int((int(samples[index]) + int(samples[index + 1])) / 2)
                for index in range(0, len(samples), 2)
            ),
        )
        if sys.byteorder != "little":
            mono.byteswap()
        return rtc.AudioFrame(
            data=mono.tobytes(),
            sample_rate=frame.sample_rate,
            num_channels=1,
            samples_per_channel=frame.samples_per_channel,
        )
