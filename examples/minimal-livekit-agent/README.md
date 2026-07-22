# Minimal LiveKit Agent

This example uses one native `AgentSession(llm=RealtimeModel(...))`. It does not
configure separate STT, LLM, or TTS providers and has no cloud fallback.

Copy `.env.example`, provide the LiveKit credentials plus the internal service
token, start LiveKit and the GPU service, then run:

```bash
uv run python examples/minimal-livekit-agent/agent.py dev
```

Required environment variables:

- `LIVEKIT_URL`, `LIVEKIT_API_KEY`, and `LIVEKIT_API_SECRET` for the LiveKit room;
- `HUGGING_VOICE_BASE_URL`, normally
  `ws://127.0.0.1:8765/v1/realtime`;
- exactly one of `HUGGING_VOICE_TOKEN` or `HUGGING_VOICE_TOKEN_FILE`.

The fixed service language is `de`; the only public voice is
`de_standard_01`. Server VAD and built-in Parakeet transcription flow through
the realtime model, so no additional LiveKit VAD/STT/TTS configuration belongs
in this example.
