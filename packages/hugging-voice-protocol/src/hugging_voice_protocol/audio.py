"""PCM16 framing and strict base64 helpers shared by both transports."""

from __future__ import annotations

import base64
import binascii

PCM16_BYTES_PER_SAMPLE = 2
INPUT_SAMPLE_RATE = 16_000
OUTPUT_SAMPLE_RATE = 24_000
INPUT_FRAME_MS = 40
OUTPUT_FRAME_MS = 20
MAX_AUDIO_BYTES = 64 * 1024
MAX_AUDIO_BASE64_CHARS = ((MAX_AUDIO_BYTES + 2) // 3) * 4


class AudioValidationError(ValueError):
    """Raised when a protocol audio payload is not valid bounded PCM16."""


def pcm16_frame_bytes(*, sample_rate: int, frame_ms: int, channels: int = 1) -> int:
    """Return the exact PCM16 byte length for a whole-millisecond frame."""
    if sample_rate <= 0:
        raise AudioValidationError("sample_rate must be positive")
    if frame_ms <= 0:
        raise AudioValidationError("frame_ms must be positive")
    if channels != 1:
        raise AudioValidationError("only mono PCM16 is supported")
    samples_numerator = sample_rate * frame_ms
    if samples_numerator % 1000:
        raise AudioValidationError("frame duration does not contain a whole number of samples")
    return samples_numerator // 1000 * channels * PCM16_BYTES_PER_SAMPLE


def decode_pcm16_base64(value: str, *, max_bytes: int = MAX_AUDIO_BYTES) -> bytes:
    """Strictly decode one bounded, sample-aligned base64 PCM16 payload."""
    if not value:
        raise AudioValidationError("audio payload must not be empty")
    try:
        payload = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise AudioValidationError("audio payload is not canonical base64") from exc
    if len(payload) > max_bytes:
        raise AudioValidationError(f"audio payload exceeds {max_bytes} decoded bytes")
    if len(payload) % PCM16_BYTES_PER_SAMPLE:
        raise AudioValidationError("PCM16 payload must contain complete 16-bit samples")
    return payload


def encode_pcm16_base64(payload: bytes, *, max_bytes: int = MAX_AUDIO_BYTES) -> str:
    """Validate and encode a bounded, sample-aligned PCM16 payload."""
    if not payload:
        raise AudioValidationError("audio payload must not be empty")
    if len(payload) > max_bytes:
        raise AudioValidationError(f"audio payload exceeds {max_bytes} bytes")
    if len(payload) % PCM16_BYTES_PER_SAMPLE:
        raise AudioValidationError("PCM16 payload must contain complete 16-bit samples")
    return base64.b64encode(payload).decode("ascii")


INPUT_FRAME_BYTES = pcm16_frame_bytes(
    sample_rate=INPUT_SAMPLE_RATE,
    frame_ms=INPUT_FRAME_MS,
)
OUTPUT_FRAME_BYTES = pcm16_frame_bytes(
    sample_rate=OUTPUT_SAMPLE_RATE,
    frame_ms=OUTPUT_FRAME_MS,
)
