"""LiveKit plugin namespace for the local Hugging Voice service."""

from .endpoint_resolver import CapacitySnapshot, EndpointResolver
from .realtime import (
    ClientEventQueued,
    PartialTranscription,
    RealtimeModel,
    RealtimeSession,
    ServerEventReceived,
)
from .version import __version__

__all__ = [
    "CapacitySnapshot",
    "ClientEventQueued",
    "EndpointResolver",
    "PartialTranscription",
    "RealtimeModel",
    "RealtimeSession",
    "ServerEventReceived",
    "__version__",
]
