from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import cast

import pytest
from hugging_voice_protocol.audio import MAX_AUDIO_BYTES
from hugging_voice_protocol.events import (
    CLIENT_EVENT_ADAPTER,
    MAX_ALL_TOOL_SCHEMAS_BYTES,
    MAX_CONTEXT_ITEM_CHARS,
    MAX_INSTRUCTIONS_CHARS,
    MAX_TOOL_ARGUMENTS_CHARS,
    MAX_TOOL_OUTPUT_CHARS,
    SERVER_EVENT_ADAPTER,
    FunctionCallConversationItem,
    FunctionCallOutputConversationItem,
    FunctionDefinition,
    FunctionTool,
    NamedToolChoice,
    NamedToolChoiceFunction,
    ResponseCreateEvent,
    SessionConfig,
)
from pydantic import ValidationError

FIXTURE_ROOT = Path(__file__).parents[3] / "tests" / "fixtures" / "protocol"


@pytest.mark.parametrize(
    "fixture", sorted(FIXTURE_ROOT.glob("client_*.json")), ids=lambda p: p.stem
)
def test_client_fixture_roundtrip(fixture: Path) -> None:
    raw = fixture.read_text(encoding="utf-8")
    event = CLIENT_EVENT_ADAPTER.validate_json(raw)
    assert CLIENT_EVENT_ADAPTER.validate_json(event.model_dump_json()) == event


@pytest.mark.parametrize(
    "fixture", sorted(FIXTURE_ROOT.glob("server_*.json")), ids=lambda p: p.stem
)
def test_server_fixture_roundtrip(fixture: Path) -> None:
    raw = fixture.read_text(encoding="utf-8")
    event = SERVER_EVENT_ADAPTER.validate_json(raw)
    assert SERVER_EVENT_ADAPTER.validate_json(event.model_dump_json()) == event


def load_client(name: str) -> dict[str, object]:
    return cast(dict[str, object], json.loads((FIXTURE_ROOT / name).read_text(encoding="utf-8")))


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("session", "input_audio_format", "sample_rate"), 48_000),
        (("session", "output_audio_format", "sample_rate"), 16_000),
    ],
)
def test_session_update_rejects_fixed_audio_contract_changes(
    path: tuple[str, ...], value: object
) -> None:
    payload = load_client("client_session_update.json")
    target: dict[str, object] = payload
    for key in path[:-1]:
        target = target[key]  # type: ignore[assignment]
    target[path[-1]] = value
    with pytest.raises(ValidationError):
        CLIENT_EVENT_ADAPTER.validate_python(payload)


def test_session_update_accepts_valid_language_voice_and_style() -> None:
    payload = load_client("client_session_update.json")
    session = payload["session"]
    assert isinstance(session, dict)
    session.update(
        language="en-US",
        voice="clear_female",
        voice_instructions="Speak warmly and at a relaxed pace.",
    )
    event = CLIENT_EVENT_ADAPTER.validate_python(payload)
    assert event.session.language == "en-US"  # type: ignore[union-attr]
    assert event.session.voice == "clear_female"  # type: ignore[union-attr]


def test_omitted_speech_options_defer_to_server_defaults() -> None:
    payload = load_client("client_session_update.json")
    session = payload["session"]
    assert isinstance(session, dict)
    session.pop("language")
    session.pop("voice")
    event = CLIENT_EVENT_ADAPTER.validate_python(payload)
    assert event.session.language is None  # type: ignore[union-attr]
    assert event.session.voice is None  # type: ignore[union-attr]


@pytest.mark.parametrize(("field", "value"), [("language", "../en"), ("voice", "bad voice")])
def test_session_update_rejects_unsafe_speech_identifiers(field: str, value: str) -> None:
    payload = load_client("client_session_update.json")
    session = payload["session"]
    assert isinstance(session, dict)
    session[field] = value
    with pytest.raises(ValidationError):
        CLIENT_EVENT_ADAPTER.validate_python(payload)


def test_session_update_rejects_model_fields_and_large_instructions() -> None:
    for field in ("model", "speaker", "reference_audio"):
        payload = load_client("client_session_update.json")
        session = payload["session"]
        assert isinstance(session, dict)
        session[field] = "forbidden"
        with pytest.raises(ValidationError):
            CLIENT_EVENT_ADAPTER.validate_python(payload)

    payload = load_client("client_session_update.json")
    session = payload["session"]
    assert isinstance(session, dict)
    session["instructions"] = "x" * (MAX_INSTRUCTIONS_CHARS + 1)
    with pytest.raises(ValidationError):
        CLIENT_EVENT_ADAPTER.validate_python(payload)


def test_context_and_audio_are_bounded_and_unknown_events_fail() -> None:
    payload = load_client("client_conversation_item_create.json")
    item = payload["item"]
    assert isinstance(item, dict)
    item["content"] = "x" * (MAX_CONTEXT_ITEM_CHARS + 1)
    with pytest.raises(ValidationError):
        CLIENT_EVENT_ADAPTER.validate_python(payload)

    payload = load_client("client_audio_append.json")
    payload["audio"] = "not-base64"
    with pytest.raises(ValidationError):
        CLIENT_EVENT_ADAPTER.validate_python(payload)

    payload = load_client("client_audio_append.json")
    payload["audio"] = base64.b64encode(bytes(MAX_AUDIO_BYTES + 2)).decode("ascii")
    with pytest.raises(ValidationError):
        CLIENT_EVENT_ADAPTER.validate_python(payload)

    payload = load_client("client_audio_commit.json")
    payload["type"] = "unknown.event"
    with pytest.raises(ValidationError):
        CLIENT_EVENT_ADAPTER.validate_python(payload)


def test_protocol_version_and_extra_fields_are_strict() -> None:
    payload = load_client("client_audio_commit.json")
    payload["protocol_version"] = 1
    with pytest.raises(ValidationError):
        CLIENT_EVENT_ADAPTER.validate_python(payload)


def test_tool_choice_schema_and_payload_bounds_are_strict() -> None:
    tool = FunctionTool(
        function=FunctionDefinition(
            name="lookup_order",
            description="Look up one order.",
            parameters={"type": "object", "properties": {}},
        )
    )
    assert SessionConfig(tools=(tool,), tool_choice="required").tools == (tool,)
    named = NamedToolChoice(function=NamedToolChoiceFunction(name="lookup_order"))
    assert (
        ResponseCreateEvent(
            event_id="evt_tool_choice",
            session_id="session_alpha",
            tools=(tool,),
            tool_choice=named,
        ).tool_choice
        == named
    )
    with pytest.raises(ValidationError):
        SessionConfig(tools=(), tool_choice="required")
    with pytest.raises(ValidationError):
        SessionConfig(
            tools=(tool,),
            tool_choice=NamedToolChoice(function=NamedToolChoiceFunction(name="unknown_tool")),
        )
    with pytest.raises(ValidationError):
        FunctionDefinition(
            name="bad tool",
            parameters={"type": "object"},
        )
    with pytest.raises(ValidationError):
        FunctionCallConversationItem(
            id="item_call",
            call_id="call_1",
            name="lookup_order",
            arguments="x" * (MAX_TOOL_ARGUMENTS_CHARS + 1),
            turn_id="turn_1",
            turn_revision=0,
            generation_id="gen_1",
            response_id="resp_1",
        )
    with pytest.raises(ValidationError):
        FunctionCallOutputConversationItem(
            id="item_output",
            call_id="call_1",
            name="lookup_order",
            output="x" * (MAX_TOOL_OUTPUT_CHARS + 1),
            is_error=False,
            turn_id="turn_1",
            turn_revision=0,
            generation_id="gen_1",
            response_id="resp_1",
        )
    large_tools = tuple(
        FunctionTool(
            function=FunctionDefinition(
                name=f"large_{index}",
                parameters={"type": "object", "description": "x" * 13_000},
            )
        )
        for index in range(6)
    )
    with pytest.raises(ValidationError):
        SessionConfig(tools=large_tools)
    assert MAX_ALL_TOOL_SCHEMAS_BYTES == 64 * 1024

    payload = load_client("client_audio_commit.json")
    payload["unexpected"] = True
    with pytest.raises(ValidationError):
        CLIENT_EVENT_ADAPTER.validate_python(payload)

    payload = load_client("client_response_create.json")
    payload["model"] = "forbidden"
    with pytest.raises(ValidationError):
        CLIENT_EVENT_ADAPTER.validate_python(payload)
