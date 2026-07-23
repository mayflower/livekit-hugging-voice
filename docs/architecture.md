# Architecture

## Scope

The end device continues to use WebRTC with LiveKit. The Python LiveKit Agent uses
a native `livekit.agents.llm.RealtimeModel` plugin, which opens an authenticated
WebSocket to the GPU service. The GPU service does not implement another WebRTC
hop.

One GPU-service pod contains one Python ASGI process and one loopback-bound
`llama-server` child managed by that Python process. The pod loads one shared
Parakeet runtime, one shared Qwen runtime, and one Gemma 4 31B model in llama.cpp
with two sequence slots. Each of the at most two admitted sessions owns its VAD,
audio remainder, conversation, IDs, cancellation generation, lifecycle, and
transport.

The concrete startup lifecycle is deliberately ordered: verify every local byte
and the pinned Silero package offline, require CUDA, start the exact llama.cpp
commit and complete a real generation probe, load/warm Parakeet from its local
`.nemo`, load/warm Qwen from explicit talker/codec GGUF paths, and complete a
visible Gemma warmup. Partial failure unwinds already-created resources and leaves
readiness red. Unexpected llama-server exit also revokes readiness immediately.

The Gemma readiness probe is two-step: the pinned stack must emit a structured
`add_numbers(19, 23)` call, then consume the fixed compatibility result `42` in a
second generation. This probe is not a production tool executor.

Gemma requests prepend the operator-configured speech system prompt and the selected
language's response instruction, cap output at 256 tokens, and pass
`chat_template_kwargs.enable_thinking=false`. Reasoning fields are suppressed, and
a defensive streaming filter quarantines a leading thinking block rather than
exposing it to later TTS stages.

Shared STT and TTS runtimes are protected by small bounded fair schedulers. Gemma
allows two text generations while keeping request and cancellation state isolated.
Every output is generation-tagged so cancellation can discard stale text and audio
without suppressing a subsequent response.

Gemma may instead produce one structured function call. The service ends that
generation silently, and the plugin exposes it through LiveKit's `function_stream`
only after `response.done`. LiveKit is the sole executor and returns a bounded
`FunctionCallOutput`; after the service ACKs the atomic call/result exchange,
LiveKit requests the final Gemma response and only that response reaches Qwen.
Both generations retain the session's fixed llama.cpp slot and prompt cache.

WebSocket authentication and exact subprotocol validation happen before atomic
slot claim. Each connection has bounded inbound and outbound queues and one
serialized sender. Final STT runs before partial STT; both STT classes and TTS
segments are round-robin across sessions. Only opportunistic partial transcription
may be dropped. Session release cancels the active generation, drains shared-worker
work for that session, clears VAD/audio/conversation state, and only then returns a
slot to `idle`.

## Configurable speech policy

- Public language codes map to Qwen model-language names and LLM response
  instructions in `speech.languages`.
- Public voice IDs map to fixed Qwen VoiceDesign descriptions in `speech.voices`.
- A session selects a configured language and voice and may append a bounded style
  instruction. The mandatory native-language design remains operator-controlled.
- The shipped defaults are `de` and `warm_female`; four languages and five profiles
  are allowlisted.

## Fixed decisions

- Models: Silero VAD, Parakeet TDT 0.6B v3, Gemma 4 31B IT, Qwen3-TTS 1.7B
  VoiceDesign.
- Internal transport: authenticated bounded WebSocket carrying JSON and PCM16.
- Packaging: a small protocol library, the LiveKit plugin wheel, and the GPU service
  image.
- Deployment: Docker Compose for local operation and Kustomize for Kubernetes.
- State: ephemeral and per session; no database, Redis, broker, or operator.

## Exclusions

No UI, extra WebRTC stack, cloud model, service-side tool execution, arbitrary
model selection, voice cloning, reference audio, audio enhancement, production
fake backend, silent fallback, or runtime model download is part of version 0.2.
