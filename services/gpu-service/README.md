# hugging-voice-gpu-service

The GPU service will host the fixed Silero -> Parakeet -> Gemma -> Qwen pipeline,
with at most two isolated sessions and one copy of each expensive runtime.

The service lifecycle loads the mounted bearer secret, verifies the model lock
offline, requires CUDA, starts
one pinned loopback `llama-server`, loads one shared Parakeet and one shared Qwen
runtime, performs actual warmups, and only then becomes ready. It has no CPU or
model fallback and does not invoke a Hub download helper.

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
`hugging-voice-livekit.v1`. The same token protects `/v1/models`, `/v1/capacity`,
`/v1/pool`, and `/v1/usage`; health and metrics remain available to internal
probes. Admission has no wait queue: at most two sessions are active or draining,
and the third receives structured `session_limit_reached` followed by close 4429.
