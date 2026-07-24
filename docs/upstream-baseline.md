# Pinned upstream baseline

This document is the one-time Wave 0 upstream inventory. Later waves use this
record and the repository's implemented contracts rather than repeating a general
upstream survey.

## Exact sources

| Project | Commit | Version at commit |
|---|---|---:|
| Hugging Face `speech-to-speech` | `c766ba1edf0023fba514571a4c1b4e05e344929f` | `0.2.11` |
| LiveKit `agents` | `c67c44e607f1fe60bfa312853c9f8c91235d5015` | `1.6.6` |

The relevant upstream implementation files were inspected from detached checkouts
at those exact commits. No moving branch was used as evidence.

## Hugging Face speech-to-speech findings

### Pipeline and turn flow

`s2s_pipeline.py` constructs a queue-connected VAD -> STT -> language-model -> TTS
chain. Its realtime construction gives each pipeline unit its own queues, events,
handlers, and `CancelScope`; upstream also has a pool-oriented mode that duplicates
whole units. This project keeps the proven staged flow but deliberately does not
copy the whole-pipeline-per-session pool: Parakeet, Gemma, and Qwen are shared once
per pod behind explicit fair schedulers.

`api/openai_realtime/service.py` owns ephemeral session/conversation state and maps
the limited OpenAI-style client events into input-buffer, conversation, response,
and cancellation behavior. `websocket_router.py` claims capacity atomically,
dispatches events, streams assistant text/audio, and releases a unit only after a
sentinel traverses the handler chain. The release path reports a long drain and
quarantines a unit that has not proven clean; it never immediately hands stale
queues to another client.

### Cancellation and stale output

`pipeline/cancel_scope.py` uses a monotonically changing generation value. Workers
capture the current generation and treat output as stale after `cancel()` advances
it. A separate discard window protects the asynchronous sender until the cancelled
response has drained. `new_response()` clears that discard state, preventing the
next response from being swallowed. `websocket_router.py` applies the same
generation test to assistant text and audio and ignores late completion markers
that do not belong to the live generation.

The local design generalizes each work item to `session_id`, `turn_id`,
`turn_revision`, and `generation_id`, and makes cancellation state per session.

### VAD

`VAD/vad_iterator.py` wraps stateful Silero inference and maintains threshold,
minimum silence, padding, and recurrent model state. `VAD/vad_handler.py` windows
audio, generates speech-start/stop information, performs minimum-duration and
continuation handling, and coordinates interruption. The useful behavior retained
here is server VAD, 512-sample windows at 16 kHz, pre-speech padding, minimum speech
and silence, partial/final turn boundaries, and barge-in. Model state must be
separate per session; no `torch.hub` or audio-enhancement path is retained.

### STT and TTS

`STT/parakeet_tdt_handler.py` performs blocking Parakeet inference away from the
transport path, emits transcription updates/finals, and supports cancellation and
turn bookkeeping. The upstream model/settings are not copied blindly: this project
fixes German-capable `nvidia/parakeet-tdt-0.6b-v3`, CUDA float16, one model per pod,
priority for final jobs, and droppable opportunistic partials.

`TTS/qwen3_tts_handler.py` loads Qwen3-TTS, consumes ordered text segments, streams
audio chunks, and observes the cancellation generation. This project fixes
`Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` and one shared non-reentrant runtime while
letting an operator map bounded public voice/language IDs to fixed design
descriptions. It excludes client-provided reference audio and raw client-selected
descriptions or path input; since version 0.2 the operator-defined `voice_clone`
mode (frozen packaged reference recordings) is the default. These are later
product decisions; the pinned upstream behavior analysis itself is unchanged.

### Upstream behavior intentionally excluded

The demo/browser UI, Orb, OAuth, public queue, camera, web search, WebRTC transport,
DeepFilterNet, old raw socket protocol, E4B compose model, runtime `torch.hub`
downloads, and complete OpenAI protocol compatibility are not product features.

## LiveKit Agents findings

### Core abstract API

At 1.6.6, `livekit.agents.llm.realtime.RealtimeModel` exposes immutable
`RealtimeCapabilities`, `model`, `provider`, `session()`, and `aclose()`.
`RealtimeSession` is an abstract event emitter whose required surface is:

- `chat_ctx` and `tools`;
- `update_instructions`, `update_chat_ctx`, `update_tools`, and `update_options`;
- synchronous `push_audio`, `push_video`, `generate_reply`, `commit_audio`,
  `clear_audio`, `interrupt`, and `truncate`;
- asynchronous `aclose`.

Server VAD maps to `InputSpeechStartedEvent` and `InputSpeechStoppedEvent`. Final
transcription uses `InputTranscriptionCompleted(item_id, transcript, is_final,
confidence)`. Reply creation resolves to `GenerationCreatedEvent`, whose
`response_id` correlates provider metrics and whose `message_stream` yields
`MessageGeneration` objects containing asynchronous text/audio streams plus a
modalities awaitable.

Errors are emitted as `RealtimeModelError` with a timestamp, label, exception, and
recoverability. Reconnect has a dedicated `session_reconnected` event. The base
session can report connection-acquire timing through `RealtimeModelMetrics`.

### Metrics

`metrics/base.py` defines `RealtimeModelMetrics` with response/request ID,
timestamp, duration, session duration, audio TTFT, cancellation state, text/audio
token detail structures, acquire time, connection reuse, and model/provider
metadata. Local integration must use service-reported text usage, leave unknown
audio token counts at zero, and must not invent values.

### Provider implementations used only as lifecycle references

The pinned OpenAI realtime plugin demonstrates bounded asynchronous connection
management, response/channel correlation, event-to-LiveKit mapping, reconnect,
error propagation, and exactly-once generation finalization. The NVIDIA
experimental realtime plugin provides a smaller direct `RealtimeModel` example
with optional injected `aiohttp.ClientSession`, audio adaptation, send/receive
tasks, generation state, and metrics.

They are references for LiveKit lifecycle conventions only. The product implements
its own protocol and classes and does not subclass or configure either provider.

## Local architectural consequence

The retained design is a narrow OpenAI-style event vocabulary over an authenticated
internal WebSocket, not an OpenAI-provider adapter. It combines upstream's
generation-safe cancellation and drain proof with LiveKit's native session/event
contract, while replacing duplicated upstream pipeline units with per-session
lightweight state and shared expensive model runtimes.
