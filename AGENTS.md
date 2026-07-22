# Repository instructions

These instructions are normative for every change in this repository. Read
`prompts.md` before implementing a wave. Wave 0 records the pinned upstream
baseline once in `docs/upstream-baseline.md`; later waves inspect only files
relevant to their concrete change and do not repeat a general upstream survey.

## Product boundary

Build one focused local multilingual speech-to-speech path for German, English,
French, and Italian:

`LiveKit RealtimeModel plugin -> authenticated WebSocket -> Silero VAD -> shared
Parakeet TDT 0.6B v3 -> local Gemma 4 31B IT in llama.cpp -> shared Qwen3-TTS
1.7B VoiceDesign`.

- Service, plugin, tests, and utilities are Python-only.
- The plugin must implement LiveKit Agents' `RealtimeModel` and
  `RealtimeSession` directly. It must not configure, subclass, fork, or disguise
  the OpenAI realtime plugin.
- The agent-to-service transport is an internal authenticated WebSocket. Do not
  add WebRTC, `aiortc`, ICE, STUN/TURN, browser capture, or a web UI to the GPU
  service.
- Version 1 has no tools/function calling, MCP, mAIstack, FastEnhancer,
  DeepFilterNet, voice cloning, reference audio, arbitrary client-provided voice
  designs, camera, web search,
  database, Redis, broker, operator, Helm, service-mesh requirement, or generic
  provider/backend registry.
- There is no cloud LLM, cloud fallback, silent CPU fallback, model downgrade,
  runtime download, `torch.hub` access, or movable model/Git/image pin.
- Gemma is `google/gemma-4-31B-it`, locally quantized for llama.cpp. Never replace
  it with E4B. Parakeet is `nvidia/parakeet-tdt-0.6b-v3`; TTS is
  `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`.
- Public languages are `de`, `en`, `fr`, and `it`. Public voices are five fixed,
  operator-defined VoiceDesign profiles: `warm_female`, `clear_female`,
  `warm_male`, `clear_male`, and `friendly_neutral`. Clients may add bounded style
  instructions but cannot submit model names, paths, reference audio, or arbitrary
  base voice designs.

## Runtime invariants

- A GPU pod has one Python service controlling one loopback-only `llama-server`
  child process.
- A service lifecycle loads exactly one Parakeet runtime, one Gemma runtime (one
  llama-server with two sequence slots), and one Qwen runtime. Two concurrent
  sessions must never be implemented as two complete model pipelines. Per-session
  stateful Silero VAD instances are allowed.
- At most two connected sessions are admitted. A third is rejected immediately
  and explicitly; there is no user queue.
- Conversation, VAD state, audio remainder, turn/revision, cancellation, IDs, and
  output channels are isolated per session and remain ephemeral.
- Every work item carries `session_id`, `turn_id`, `turn_revision`, and
  `generation_id` where applicable. Cancellation is generation-tagged; stale text
  and audio never cross into a later generation.
- STT and TTS use small bounded fair schedulers around the shared non-reentrant
  runtimes. Final STT outranks optional/droppable partial STT. Blocking ML work
  never blocks the asyncio event loop.
- Disconnect and shutdown drain the complete handler chain before a slot is
  reusable. Timed-out drain is quarantined/stuck and remains occupied.
- All queues, retries, reconnects, timeouts, headers, JSON messages, audio messages,
  and conversation history are bounded. Audio and final work are never silently
  dropped.
- Errors are structured and observable. Do not add broad exception suppression.

## Delivery and verification

- Model fetching is an explicit prefetch operation. Runtime verifies an exact
  revision/size/SHA-256 lock and operates with Hugging Face/Transformers offline.
- Model weights stay outside the service image. The runtime image is non-root,
  has a read-only model mount, and contains no compiler toolchain.
- Kubernetes uses Kustomize, requests one NVIDIA GPU per pod, supports drain, and
  introduces no central session state.
- Production code contains no dummy, fake, mock, or in-memory model/service path.
  Test doubles live only below `tests/` and are explicitly injected.
- GPU results must be measured on real hardware. If no NVIDIA GPU is available,
  mark GPU tests skipped/open and never simulate benchmarks or success.
- Keep Ruff, formatting, Mypy, CPU tests, package builds, container validation, and
  Kubernetes rendering green as their waves add them.
- Do not push, publish, create a remote repository, or open a pull request without
  a separate explicit request.
