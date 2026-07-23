# Minimal LiveKit Agent

The worker registers two real LiveKit FunctionTools: `add_numbers` adds two
integers and `count_characters` counts the characters in a short text. Ask for
either operation to exercise the silent call, LiveKit execution, ACK, and final
voice. The browser shows each call's bounded arguments, status, and result in a
small tool-call history delivered over a reliable LiveKit data message.

This example uses one native `AgentSession(llm=RealtimeModel(...))`. It does not
configure separate STT, LLM, or TTS providers and has no cloud fallback.

It also contains a small browser client and token server. The page creates one
private LiveKit room, explicitly dispatches the named worker, publishes the
browser microphone, plays the agent's remote audio track, and displays LiveKit
transcription text streams.

Copy `.env.example`, provide the LiveKit credentials plus the internal service
token, start LiveKit and the GPU service, then run:

```bash
uv run python examples/minimal-livekit-agent/agent.py dev
```

Run the browser server in another terminal:

```bash
LIVEKIT_API_KEY=devkey \
LIVEKIT_API_SECRET=secret \
LIVEKIT_INTERNAL_URL=ws://127.0.0.1:7880 \
uv run python examples/minimal-livekit-agent/web.py
```

Then open <http://127.0.0.1:3000>. The browser SDK is loaded from a pinned jsDelivr
URL, so the browser needs Internet access for that one static dependency.

Required environment variables:

- `LIVEKIT_URL`, `LIVEKIT_API_KEY`, and `LIVEKIT_API_SECRET` for the LiveKit room;
- `HUGGING_VOICE_BASE_URL`, normally
  `ws://127.0.0.1:8765/v1/realtime`;
- exactly one of `HUGGING_VOICE_TOKEN` or `HUGGING_VOICE_TOKEN_FILE`.
- `HUGGING_VOICE_LANGUAGE`, `HUGGING_VOICE_VOICE`, and optional
  `HUGGING_VOICE_VOICE_INSTRUCTIONS` select the session speech configuration.
- `HUGGING_VOICE_AGENT_NAME` names both worker registration and explicit browser
  dispatch; it defaults to `hugging-voice`.

The web process requires `LIVEKIT_API_KEY` and `LIVEKIT_API_SECRET` to mint
short-lived, room-scoped participant tokens. `LIVEKIT_INTERNAL_URL` is the
server-side signaling URL. If `LIVEKIT_PUBLIC_URL` is omitted, the browser uses
the web page's own origin and `/rtc` is proxied to LiveKit. This lets one trusted
HTTPS reverse proxy cover both the page and signaling.

Empty language/voice environment values inherit the service defaults (shipped as
`de` and `warm_female`). Explicit IDs must exist in the GPU service's
`speech.languages` and `speech.voices` maps. Server VAD and built-in Parakeet transcription flow through
the realtime model, so no additional LiveKit VAD/STT/TTS configuration belongs in
this example.

Do not publish this development token server with the shipped `devkey`/`secret`
credentials. Browser microphone APIs work on localhost or a trusted HTTPS origin,
not an ordinary remote HTTP address.
