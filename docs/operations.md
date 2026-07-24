# Operations

## Health and capacity

`/health/live`, `/health/ready`, and `/metrics` are probe-safe internal endpoints.
All `/v1/*` operations require the bearer token. Monitor at least readiness,
active/available/draining/stuck sessions, rejections, runtime load counters,
scheduler wait/inference durations, LLM/TTS first-output latency, stale chunks,
WebSocket failures, and GPU memory.

The runtime load counters for the selected LLM and Parakeet must remain exactly
one per pod lifecycle. The Qwen load counter must equal the configured worker
count (one or two in production). All admitted sessions share those runtimes but own separate VAD,
audio, conversation, cancellation, ID, output state, and llama.cpp slot.

## Drain and shutdown

Kubernetes sends SIGTERM and grants 60 seconds. Uvicorn stops new connections;
the application revokes readiness/admission, allows active sessions up to its
30-second drain deadline, then cancels and closes remaining connections before
stopping the loopback llama-server. There is deliberately no sleep-only preStop.

A disconnecting slot remains `draining` until its pipeline and shared-scheduler
work are proven idle. If cleanup times out it becomes `stuck` and is quarantined.
Do not patch it back to idle. Drain traffic from the pod, preserve logs/metrics,
and restart the entire pod so all GPU runtime state is reconstructed safely.

## Scaling and failures

Scale replicas only to available GPUs. The headless service exposes ready pods;
the plugin’s capacity probe reduces avoidable 4429 races, while admission remains
authoritative. Existing WebSockets remain on their original pod.

Per-pod capacity defaults to two and is controlled by `server.max_sessions`,
`models.llama_parallel_slots`, and `models.llama_context_size`. Session admission
must not exceed the slot count, and total context must provide at least 2048 tokens
per slot. Increasing these limits without measured VRAM and latency evidence can
cause startup failure, OOM, or unacceptable realtime latency.

An unexpected llama-server exit, missing model/hash, missing token, CUDA failure,
or failed warmup makes readiness red. There is no CPU/cloud fallback. Conversation
state is ephemeral and is not persisted across reconnect beyond the bounded,
confirmed chat replay held by the LiveKit plugin.

## Incident evidence

Collect pod events, structured service logs, `/metrics`, `/v1/capacity`,
`/v1/pool`, the model report, image digest, driver/CUDA versions, and `nvidia-smi`.
Logs and pool reports contain correlation IDs and timings, never audio, bearer
tokens, full prompts, or full transcripts.
