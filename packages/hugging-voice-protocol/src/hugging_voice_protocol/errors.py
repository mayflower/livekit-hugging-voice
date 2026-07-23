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
    INVALID_TOOL_CONFIGURATION = "invalid_tool_configuration"
    UNSUPPORTED_TOOL_TYPE = "unsupported_tool_type"
    TOOL_SCHEMA_TOO_LARGE = "tool_schema_too_large"
    INVALID_TOOL_CHOICE = "invalid_tool_choice"
    UNKNOWN_TOOL_NAME = "unknown_tool_name"
    MALFORMED_TOOL_ARGUMENTS = "malformed_tool_arguments"
    MULTIPLE_TOOL_CALLS_NOT_SUPPORTED = "multiple_tool_calls_not_supported"
    MIXED_MESSAGE_AND_TOOL_OUTPUT = "mixed_message_and_tool_output"
    TOOL_CALL_STATE_CONFLICT = "tool_call_state_conflict"
    UNKNOWN_TOOL_CALL_OUTPUT = "unknown_tool_call_output"
    DUPLICATE_TOOL_CALL_OUTPUT = "duplicate_tool_call_output"
    STALE_TOOL_CALL_OUTPUT = "stale_tool_call_output"
    CONVERSATION_ACK_TIMEOUT = "conversation_ack_timeout"
    SESSION_UPDATE_ACK_TIMEOUT = "session_update_ack_timeout"
    MODEL_TOOL_CALL_FAILURE = "model_tool_call_failure"
