"""Minimal native LiveKit Agent using only Hugging Voice realtime inference."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli
from livekit.plugins import hugging_voice

server = AgentServer()


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
    session: AgentSession[dict[str, object]] = AgentSession(llm=model)
    await session.start(
        room=ctx.room,
        agent=Agent(
            instructions=os.getenv(
                "HUGGING_VOICE_AGENT_INSTRUCTIONS",
                "Have a friendly conversation. Do not use tools or Markdown.",
            )
        ),
    )


def main() -> None:
    cli.run_app(server)


if __name__ == "__main__":
    main()
