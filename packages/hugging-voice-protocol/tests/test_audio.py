import base64

import pytest
from hugging_voice_protocol.audio import (
    INPUT_FRAME_BYTES,
    MAX_AUDIO_BYTES,
    OUTPUT_FRAME_BYTES,
    AudioValidationError,
    decode_pcm16_base64,
    encode_pcm16_base64,
    pcm16_frame_bytes,
)


def test_fixed_frame_sizes() -> None:
    assert INPUT_FRAME_BYTES == 1_280
    assert OUTPUT_FRAME_BYTES == 960


def test_pcm16_frame_requires_mono_and_whole_samples() -> None:
    with pytest.raises(AudioValidationError, match="mono"):
        pcm16_frame_bytes(sample_rate=16_000, frame_ms=40, channels=2)
    with pytest.raises(AudioValidationError, match="whole number"):
        pcm16_frame_bytes(sample_rate=44_100, frame_ms=1)


def test_base64_roundtrip_is_strict_and_bounded() -> None:
    payload = bytes(INPUT_FRAME_BYTES)
    assert decode_pcm16_base64(encode_pcm16_base64(payload)) == payload
    with pytest.raises(AudioValidationError, match="canonical"):
        decode_pcm16_base64("not base64")
    with pytest.raises(AudioValidationError, match="complete"):
        decode_pcm16_base64(base64.b64encode(b"x").decode())
    with pytest.raises(AudioValidationError, match="exceeds"):
        encode_pcm16_base64(bytes(MAX_AUDIO_BYTES + 2))
