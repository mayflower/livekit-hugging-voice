# Repository instructions — version 0.3

These instructions are normative for every change in this repository. Wave 0
records the pinned upstream baseline once in `docs/upstream-baseline.md`; later
changes inspect only files relevant to their concrete scope and do not repeat a
general upstream survey.

## Product boundary

Build one focused local multilingual speech-to-speech path for German, English,
French, and Italian. Version 0.3 retains the compatibility baseline and adds a
small set of measured startup-only performance profiles:

`LiveKit RealtimeModel plugin -> authenticated WebSocket -> per-session Silero
VAD -> shared CPU-only Smart Turn v3.2 -> shared Parakeet TDT 0.6B v3 -> one
local llama.cpp chat model -> bounded shared Qwen3-TTS runtime pool`.

The compatibility profile remains Gemma 4 31B IT with Qwen3-TTS 1.7B. Candidate
multi-session profiles are limited to Gemma 4 26B A4B IT or
Qwen3-30B-A3B-Instruct-2507 in llama.cpp with Qwen3-TTS 0.6B Base through the
CUDA-graph runtime. Exactly one complete profile is selected and loaded at
process startup. There is no hot swap or automatic fallback.

- Service, plugin, tests, and utilities are Python-only.
- The plugin must implement LiveKit Agents' `RealtimeModel` and
  `RealtimeSession` directly. It must not configure, subclass, fork, or disguise
  the OpenAI realtime plugin.
- The agent-to-service transport is an internal authenticated WebSocket. Do not
  add WebRTC, `aiortc`, ICE, STUN/TURN, browser capture, or a web UI to the GPU
  service.
- Version 0.3 supports native function calling through LiveKit Agents. The selected LLM
  decides whether to call a function and produces its name and JSON arguments;
  LiveKit Agents is the only tool executor. Python FunctionTools, Toolsets, and
  MCPToolsets live in the LiveKit worker. The GPU service never executes tools,
  opens MCP connections, or receives tool credentials.
- There is no second tool executor, planner, router, tool LLM, built-in
  `llama.cpp` tool, `llama.cpp` MCP proxy, or parsing of visible text, XML, or
  Markdown as a substitute for structured function calls.
- There is no mAIstack, FastEnhancer, DeepFilterNet, client-provided reference
  audio, arbitrary client-provided voice design, camera, service-side web search,
  database, Redis, broker, operator, Helm, service-mesh requirement, or generic
  provider/backend registry. Voice cloning exists only in the operator-defined
  form described below: frozen reference recordings shipped with the service.
- There is no cloud LLM, cloud fallback, silent CPU fallback, model downgrade,
  runtime download, `torch.hub` access, or movable model/Git/image pin.
- The compatibility LLM is `google/gemma-4-31B-it`, locally quantized for
  llama.cpp. The only performance candidates are `google/gemma-4-26B-A4B-it`
  and `Qwen/Qwen3-30B-A3B-Instruct-2507`. Parakeet remains
  `nvidia/parakeet-tdt-0.6b-v3`. Compatibility TTS is
  `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`; the only performance TTS candidate is
  `Qwen/Qwen3-TTS-12Hz-0.6B-Base`. A candidate may become a production default
  only with exact artifact pins, the real readiness tool probe, multilingual
  voice checks, tool-evaluation gates, and real NVIDIA multi-session results.
- Public languages are `de`, `en`, `fr`, and `it`. Public voices are five fixed,
  operator-defined profiles: `warm_female`, `clear_female`, `warm_male`,
  `clear_male`, and `friendly_neutral`. The default `voice_clone` TTS mode speaks
  every profile through the Qwen3-TTS base talker from one frozen, operator-defined
  reference recording per voice and language, so the perceived speaker stays
  identical across segments and sessions; the recordings were rendered once from
  the VoiceDesign descriptions and are packaged with the service. The `voice_design`
  mode rebuilds each voice from its description on every segment. Clients may add
  bounded style instructions (honored only in `voice_design` mode) but cannot
  submit model names, paths, reference audio, or arbitrary base voice designs.

## Runtime invariants

- A GPU pod has one Python service controlling one loopback-only `llama-server`
  child process.
- A service lifecycle loads exactly one Parakeet runtime, one selected LLM runtime
  (one llama-server with a bounded operator-configured sequence-slot count), and
  a bounded pool of one or two selected Qwen runtimes. When semantic endpointing
  is enabled it also loads exactly one shared CPU-only Smart Turn v3.2 ONNX
  runtime. Three or four TTS workers are benchmark-only in version 0.3.
  Concurrent sessions must never be implemented
  as complete model pipelines. Per-session stateful Silero VAD instances are
  allowed.
- Silero produces an endpoint candidate after the configured short pause.
  Smart Turn decides whether the utterance is complete from at most the last
  eight seconds of audio. Incomplete turns stay open until speech resumes or the
  bounded hard-silence fallback expires. Candidate jobs are bounded, fair,
  revision-aware, and never block the asyncio event loop.
- Connected-session admission, llama.cpp sequence slots, and total llama.cpp
  context are bounded operator configuration with compatibility defaults of two
  sessions, two slots, and 32768 tokens. These defaults are not a measured
  throughput or stability recommendation. `max_sessions` must not exceed the
  sequence-slot count, and the total context must provide at least 2048 tokens per
  slot.
- At least two concurrent sessions remain a supported compatibility target.
  Four sessions are the multi-session target and six may be measured when VRAM
  permits. `max_sessions=1` is not an acceptable product default or performance
  solution. The production default may move from two to four only after the
  documented real-hardware correctness, latency, fairness, tool-accuracy, and
  VRAM gates pass.
- A connection beyond the configured session limit is rejected immediately and
  explicitly; there is no user queue.
- Conversation, VAD state, audio remainder, turn/revision, cancellation, IDs, and
  output channels are isolated per session and remain ephemeral. Session tools,
  tool choice, pending calls, call/result correlations, and tool timing state are
  likewise isolated and ephemeral.
- Every work item carries `session_id`, `turn_id`, `turn_revision`, and
  `generation_id` where applicable. Cancellation is generation-tagged; stale text,
  audio, calls, and tool results never cross into a later generation.
- Each session is permanently assigned to its admitted `llama.cpp` sequence slot.
  Every generation uses that `id_slot` and `cache_prompt=true`, including a
  continuation after a tool result.
- At most one structured tool call is allowed per LLM generation and
  `parallel_tool_calls` is always false. Sequential calls may continue through
  LiveKit's existing `max_tool_steps` loop.
- Tools are materialized through LiveKit's strict OpenAI-compatible FunctionTool
  representation, sorted deterministically, bounded, and frozen after the first
  model generation. An identical update is idempotent; a semantic change after
  freeze is rejected.
- A response is either a message response or a tool-call response. A tool-call
  generation emits no visible assistant text and starts no TTS. Mixed text and
  function-call output is a model/protocol error.
- A model call remains pending until LiveKit returns its matching result. The GPU
  service commits the call and result atomically and never starts the subsequent
  reply automatically; LiveKit starts it after the result ACK.
- STT and TTS use small bounded fair schedulers around the shared non-reentrant
  runtimes. Final STT outranks optional/droppable partial STT. Blocking ML work
  never blocks the asyncio event loop.
- A multi-worker TTS scheduler preserves FIFO order within a session, schedules
  sessions fairly, and never runs two segments of the same session concurrently.
  A TTS pool is not permission to replicate Parakeet, the LLM, or complete
  pipelines.
- Disconnect and shutdown drain the complete handler chain before a slot is
  reusable. Timed-out drain is quarantined/stuck and remains occupied.
- All queues, retries, reconnects, timeouts, headers, JSON messages, audio messages,
  and conversation history are bounded. Audio and final work are never silently
  dropped.
- Errors are structured and observable. Do not add broad exception suppression.

## Tool-calling and protocol invariants

- Version 0.2 uses WebSocket subprotocol `hugging-voice-livekit.v2` and
  `protocol_version: 2` on the existing `/v1/realtime` path. Do not retain a
  permanent v1 compatibility path; reject v1 clients explicitly.
- `session.update` carries bounded tools and a default tool choice and is
  acknowledged with `session.updated`. Every `conversation.item.create` is
  acknowledged with `conversation.item.created`. Bootstrap and reconnect become
  ready only after configuration and replay ACKs have completed.
- `response.create` may override tools and tool choice for that response. Tool
  choice is exactly `auto`, `required`, `none`, or a named offered function.
  `required` with no tools and a named unknown function are invalid.
- Conversation items are typed messages, function calls, or function-call
  outputs. Context updates are append-only. A function result must match a known
  pending `call_id`, name, turn, revision, generation, and response; unknown,
  duplicate, conflicting, or stale results are rejected.
- The plugin buffers and validates a model FunctionCall until the matching
  `response.done(reason=tool_call)`. It then appends the call to its local chat
  context before exposing it through `GenerationCreatedEvent.function_stream`.
- LiveKit executes the tool and appends `FunctionCallOutput`. The plugin sends the
  output to the GPU service and waits for its ACK before `update_chat_ctx()` may
  succeed or a final reply may start. `auto_tool_reply_generation` remains false.
- Tool schemas, arguments, outputs, pending ACKs, pending calls, queues, and wire
  messages are strictly bounded by the protocol constants and configuration.
  Current protocol maxima are 32 tools, 16 KiB per schema, 64 KiB across schemas,
  16,000 characters for arguments, 16,000 characters for output, and one pending
  tool call. JSON is canonical, finite, and object-valued where required.
- Tool arguments, tool outputs, credentials, audio, and conversation content are
  never logged. External tool output is untrusted data, never a system
  instruction, and cannot replace the fixed system or voice rules.
- Disconnect, reconnect, cancellation, and barge-in must not duplicate tool
  execution or produce a stale final voice response. A non-cancellable external
  tool may finish, but its stale result must not resume an obsolete generation.

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
- Runtime readiness includes a real two-step structured selected-LLM tool-call probe
  against the pinned GGUF/Jinja/`llama.cpp` stack. It uses a fixed internal result
  only as a startup compatibility probe, never as a production tool executor.
- Contract coverage must prove the real LiveKit lifecycle:
  `function_stream -> LiveKit ToolExecutor -> FunctionCallOutput -> acknowledged
  update_chat_ctx() -> final generate_reply()`. Do not claim tool calling complete
  without that integration test.
- GPU E2E tests must prove that no audio is emitted before the tool result, the
  final result produces 24-kHz audio, Parakeet and the selected LLM remain
  single-loaded, and Qwen load count equals the selected bounded worker count.
  Performance claims require raw measurements and full provenance.
- Keep Ruff, formatting, Mypy, CPU tests, package builds, container validation, and
  Kubernetes rendering green as their waves add them.
- Do not push, publish, create a remote repository, or open a pull request without
  a separate explicit request.
