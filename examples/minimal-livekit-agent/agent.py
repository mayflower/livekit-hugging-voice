"""Minimal native LiveKit Agent using only Hugging Voice realtime inference."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal

from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    RunContext,
    cli,
    function_tool,
)
from livekit.plugins import hugging_voice

server = AgentServer()
TOOL_EVENT_TOPIC: Final = "hugging_voice.tool_call"
MAX_TOOL_INTEGER: Final = 1_000_000_000
MAX_COUNTED_TEXT_LENGTH: Final = 500
MAX_TOOL_EVENT_BYTES: Final = 4_096


@dataclass(frozen=True, slots=True)
class DemoContext:
    room: rtc.Room


async def _publish_tool_event(
    context: RunContext[DemoContext],
    *,
    status: Literal["running", "completed", "failed"],
    arguments: Mapping[str, int | str],
    result: str | None = None,
) -> None:
    event = {
        "version": 1,
        "call_id": context.function_call.call_id,
        "name": context.function_call.name,
        "status": status,
        "arguments": arguments,
    }
    if result is not None:
        event["result"] = result
    payload = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
    if len(payload.encode("utf-8")) > MAX_TOOL_EVENT_BYTES:
        raise ValueError("tool display event exceeds its size limit")
    await context.userdata.room.local_participant.publish_data(
        payload,
        reliable=True,
        topic=TOOL_EVENT_TOPIC,
    )


@function_tool
async def add_numbers(context: RunContext[DemoContext], a: int, b: int) -> str:
    """Add two integers and return the exact result."""

    in_range = abs(a) <= MAX_TOOL_INTEGER and abs(b) <= MAX_TOOL_INTEGER
    arguments: dict[str, int | str] = {
        "a": a if abs(a) <= MAX_TOOL_INTEGER else "out_of_range",
        "b": b if abs(b) <= MAX_TOOL_INTEGER else "out_of_range",
    }
    await _publish_tool_event(context, status="running", arguments=arguments)
    if not in_range:
        message = f"integers must be between {-MAX_TOOL_INTEGER} and {MAX_TOOL_INTEGER}"
        await _publish_tool_event(context, status="failed", arguments=arguments, result=message)
        raise ValueError(message)
    result = str(a + b)
    await _publish_tool_event(context, status="completed", arguments=arguments, result=result)
    return result


@function_tool
async def count_characters(context: RunContext[DemoContext], text: str) -> str:
    """Count every Unicode character in a short text, including spaces."""

    text_is_bounded = len(text) <= MAX_COUNTED_TEXT_LENGTH
    arguments = {"text": text if text_is_bounded else f"{text[:MAX_COUNTED_TEXT_LENGTH]}…"}
    await _publish_tool_event(context, status="running", arguments=arguments)
    if not text_is_bounded:
        message = f"text must contain at most {MAX_COUNTED_TEXT_LENGTH} characters"
        await _publish_tool_event(context, status="failed", arguments=arguments, result=message)
        raise ValueError(message)
    result = str(len(text))
    await _publish_tool_event(context, status="completed", arguments=arguments, result=result)
    return result


@function_tool
async def check_service_status(context: RunContext[DemoContext], service: str) -> str:
    """Perform a deliberately slow demo status check for a named service."""

    bounded_service = service.strip()[:100]
    if not bounded_service:
        raise ValueError("service must not be empty")
    await _publish_tool_event(
        context,
        status="running",
        arguments={"service": bounded_service},
    )
    context.session.say("Ich prüfe das kurz.")
    await asyncio.sleep(1.0)
    result = f"{bounded_service}: operational"
    await _publish_tool_event(
        context,
        status="completed",
        arguments={"service": bounded_service},
        result=result,
    )
    return result


@dataclass(frozen=True, slots=True)
class SpeechOptions:
    language: str | None
    voice: str | None
    voice_instructions: str | None


def speech_options(metadata: str) -> SpeechOptions:
    configured = {
        "language": os.getenv("HUGGING_VOICE_LANGUAGE") or None,
        "voice": os.getenv("HUGGING_VOICE_VOICE") or None,
        "voice_instructions": os.getenv("HUGGING_VOICE_VOICE_INSTRUCTIONS") or None,
    }
    if metadata:
        parsed = json.loads(metadata)
        if not isinstance(parsed, dict):
            raise ValueError("agent dispatch metadata must be a JSON object")
        unknown = set(parsed) - set(configured)
        if unknown:
            raise ValueError(f"unknown speech option in dispatch metadata: {sorted(unknown)[0]}")
        for key, value in parsed.items():
            if value is not None and not isinstance(value, str):
                raise ValueError(f"dispatch speech option {key!r} must be a string or null")
            configured[key] = value
    return SpeechOptions(**configured)


@server.rtc_session(agent_name=os.getenv("HUGGING_VOICE_AGENT_NAME", "hugging-voice"))
async def entrypoint(ctx: JobContext) -> None:
    options = speech_options(ctx.job.metadata)
    model = hugging_voice.RealtimeModel(
        language=options.language,
        voice=options.voice,
        voice_instructions=options.voice_instructions,
        instructions=os.getenv(
            "HUGGING_VOICE_MODEL_INSTRUCTIONS",
            "You are a helpful voice assistant. Keep answers brief and natural.",
        ),
    )
    session: AgentSession[DemoContext] = AgentSession(
        llm=model,
        userdata=DemoContext(room=ctx.room),
    )
    await session.start(
        room=ctx.room,
        agent=Agent(
            instructions=os.getenv(
                "HUGGING_VOICE_AGENT_INSTRUCTIONS",
                "Use add_numbers for additions, count_characters for text lengths, and "
                "check_service_status for status checks. Answer briefly in German. "
                "Do not use Markdown.",
            ),
            tools=[add_numbers, count_characters, check_service_status],
        ),
    )


def main() -> None:
    cli.run_app(server)


if __name__ == "__main__":
    main()
