# hugging-voice-gpu-service

The GPU service hosts the fixed Silero -> Smart Turn -> Parakeet -> LLM -> Qwen pipeline,
with a bounded operator-configured number of isolated sessions and one copy of
each expensive runtime.

The service lifecycle loads the mounted bearer secret, verifies the model lock
offline, requires CUDA, starts
one pinned loopback `llama-server`, loads one shared CPU-only Smart Turn runtime,
one shared Parakeet, and the configured Qwen runtime pool, performs actual warmups,
and only then becomes ready. It has no model fallback and does not invoke a Hub
download helper.

The shipped profiles enable `semantic_turn.mode: smart_turn_v3` with a 250 ms
Silero candidate pause and a 2.5 second bounded fallback. The generic
`fixed_silence` default remains 500 ms. The service advertises the effective VAD
configuration in `session.created`; the LiveKit plugin returns it unchanged during
initial configuration and reconnect.

Partial Parakeet transcription is enabled every 500 ms over a bounded trailing
four-second window. Partial jobs are optional and droppable; final STT always has
priority. Only a final transcript is committed to conversation state or passed to
the LLM.

Install the locked runtime dependencies without downloading weights:

```bash
uv sync --all-packages --extra gpu --frozen
```

The service starts with `uv run hugging-voice-service --config
services/gpu-service/config/default.yaml`. `/health/live` is available while the
background lifecycle loads; `/health/ready` stays unavailable until auth, lock,
CUDA, llama.cpp generation, Parakeet, Qwen, Gemma warmup, and realtime scheduler
startup all succeed.

`WS /v1/realtime` requires `Authorization: Bearer …` and subprotocol
`hugging-voice-livekit.v2`. The same token protects `/v1/models`, `/v1/capacity`,
`/v1/pool`, and `/v1/usage`; health and metrics remain available to internal
probes. Admission has no wait queue: at most `server.max_sessions` sessions are
active or draining, and the next connection receives structured
`session_limit_reached` followed by close 4429. The default is two;
`models.llama_parallel_slots` must be at least that value and
`models.llama_context_size` is the total context shared across the slots.

`/metrics` reports Smart Turn queue and inference time as well as estimated speech
endpoint latency and speech-end-to-first-audio latency. The latter includes
endpointing, STT, LLM, and TTS rather than starting only at the emitted
`speech_stopped` event.
