"""Minimal native LiveKit Agent using only Hugging Voice realtime inference."""

from __future__ import annotations

from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli
from livekit.plugins import hugging_voice

server = AgentServer()


@server.rtc_session(agent_name="hugging-voice-german")
async def entrypoint(ctx: JobContext) -> None:
    model = hugging_voice.RealtimeModel(
        instructions=(
            "Du bist ein hilfreicher Sprachassistent. Antworte kurz, natürlich und auf Deutsch."
        )
    )
    session: AgentSession[dict[str, object]] = AgentSession(llm=model)
    await session.start(
        room=ctx.room,
        agent=Agent(
            instructions=(
                "Führe ein freundliches Gespräch. Nutze keine Werkzeuge und keine Markdown-Ausgabe."
            )
        ),
    )


def main() -> None:
    cli.run_app(server)


if __name__ == "__main__":
    main()
