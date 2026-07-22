"""Wire error codes and WebSocket close codes."""

from __future__ import annotations

from enum import IntEnum, StrEnum


class CloseCode(IntEnum):
    PROTOCOL_ERROR = 4400
    AUTHENTICATION_ERROR = 4401
    SESSION_CONFLICT = 4409
    SESSION_LIMIT_REACHED = 4429
    SERVICE_FAILURE = 4500
    SERVICE_RESTART = 1012


class ErrorCode(StrEnum):
    INVALID_EVENT = "invalid_event"
    INVALID_CONFIGURATION = "invalid_configuration"
    AUTHENTICATION_FAILED = "authentication_failed"
    SESSION_STATE_CONFLICT = "session_state_conflict"
    SESSION_LIMIT_REACHED = "session_limit_reached"
    QUEUE_OVERFLOW = "queue_overflow"
    MODEL_FAILURE = "model_failure"
    SERVICE_DRAINING = "service_draining"
    UNSUPPORTED_FEATURE = "unsupported_feature"
