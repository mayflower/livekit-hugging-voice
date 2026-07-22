# livekit-plugins-hugging-voice

This package provides the native LiveKit Agents `RealtimeModel` adapter for the
authenticated Hugging Voice WebSocket protocol. It does not wrap the OpenAI
plugin and installs no model, ML, or CUDA dependencies.

```python
from livekit.agents import AgentSession
from livekit.plugins.hugging_voice import RealtimeModel

session = AgentSession(
    llm=RealtimeModel(
        base_url="ws://127.0.0.1:8765/v1/realtime",
        token="internal-secret",
        language="en",
        voice="warm_female",
        voice_instructions="Speak warmly and at a relaxed pace.",
    )
)
```

The constructor also accepts a bounded `base_urls` list, `token_file`, configurable
`language`, `voice`, optional voice-style and model instructions, an injected
`aiohttp.ClientSession`, and LiveKit `APIConnectOptions`. Language and voice IDs are
validated authoritatively by the service against its configured maps. Omitting
language or voice inherits the defaults advertised by the connected service.
Environment fallbacks are
`HUGGING_VOICE_BASE_URL` plus exactly one of `HUGGING_VOICE_TOKEN` or
`HUGGING_VOICE_TOKEN_FILE`.

For multi-pod Kubernetes, use `headless_dns` (plus optional port/TLS settings).
The bounded resolver supports A/AAAA records, authenticated capacity ordering,
short caching, and authoritative 4429 retry without migrating connected sessions.

Each LiveKit realtime session owns one bounded WebSocket transport, resamples
mono/stereo input to 16 kHz mono in a worker thread, emits 40 ms service frames,
maps built-in VAD/final transcription/text/audio events, and reports real service
text-token metrics. Reconnect drops in-flight audio and generations, then replays
only confirmed append-only text context and current instructions. Tools, video,
truncation, arbitrary models, voice cloning, and cloud endpoints are rejected.
