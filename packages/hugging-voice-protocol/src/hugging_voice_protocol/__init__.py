"""Versioned internal protocol for livekit-hugging-voice."""

from .audio import (
    INPUT_FRAME_BYTES,
    INPUT_FRAME_MS,
    INPUT_SAMPLE_RATE,
    MAX_AUDIO_BYTES,
    OUTPUT_FRAME_BYTES,
    OUTPUT_FRAME_MS,
    OUTPUT_SAMPLE_RATE,
    AudioValidationError,
    decode_pcm16_base64,
    encode_pcm16_base64,
    pcm16_frame_bytes,
)
from .errors import CloseCode, ErrorCode
from .events import (
    CLIENT_EVENT_ADAPTER,
    SERVER_EVENT_ADAPTER,
    ClientEvent,
    ServerEvent,
    parse_client_event_json,
    parse_server_event_json,
)
from .version import PROTOCOL_VERSION, WEBSOCKET_SUBPROTOCOL, __version__

__all__ = [
    "CLIENT_EVENT_ADAPTER",
    "INPUT_FRAME_BYTES",
    "INPUT_FRAME_MS",
    "INPUT_SAMPLE_RATE",
    "MAX_AUDIO_BYTES",
    "OUTPUT_FRAME_BYTES",
    "OUTPUT_FRAME_MS",
    "OUTPUT_SAMPLE_RATE",
    "PROTOCOL_VERSION",
    "SERVER_EVENT_ADAPTER",
    "WEBSOCKET_SUBPROTOCOL",
    "AudioValidationError",
    "ClientEvent",
    "CloseCode",
    "ErrorCode",
    "ServerEvent",
    "__version__",
    "decode_pcm16_base64",
    "encode_pcm16_base64",
    "parse_client_event_json",
    "parse_server_event_json",
    "pcm16_frame_bytes",
]
