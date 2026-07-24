# Internal realtime protocol v2

## Transport and authentication

The LiveKit plugin connects to `WS /v1/realtime` with subprotocol
`hugging-voice-livekit.v2` and `Authorization: Bearer <token>`. The token is never
accepted in a query string. JSON control events and base64 PCM16 payloads share the
WebSocket; this is not a public OpenAI endpoint and does not attempt complete
OpenAI compatibility.

Every event has `type`, an `evt_`-prefixed `event_id`, `protocol_version: 2`, and a
`session_`-prefixed `session_id`. Turn events additionally correlate `turn_id` and
`turn_revision`; response events add `generation_id`, `response_id`, and `item_id`.
Unknown event types and fields are rejected.

## Fixed media contract

| Direction | Encoding | Rate | Channels | Framing |
|---|---|---:|---:|---:|
| plugin -> service | signed little-endian PCM16 | 16 kHz | 1 | normally 40 ms / 1,280 bytes |
| service -> plugin | signed little-endian PCM16 | 24 kHz | 1 | 20 ms / 960 bytes |

Each base64 audio event is strictly decoded, sample-aligned, and limited to 64 KiB
of decoded data. Inbound and outbound transport queues are bounded now; overflow
is a structured terminal error, never an implicit audio drop. Empty, malformed,
stereo, unsupported-rate, and oversized audio is invalid.

## Client events

- `session.update`: bounded instructions, language code, public voice ID, optional
  speaking-style instructions (applied in `voice_design` mode; the default
  `voice_clone` mode's frozen speaker identity ignores them), fixed audio, bounded
  server-VAD values, interruption and transcription switches. The service
  validates language and voice against its configured maps; null or omitted
  language/voice values inherit its advertised defaults, plus bounded strict
  function schemas and the session tool choice.
  Model, raw speaker, reference-audio, path, and cloud fields are impossible.
- `input_audio_buffer.append`: ordered PCM16 chunk with a non-negative sequence.
- `input_audio_buffer.commit` and `input_audio_buffer.clear`: explicit server-known
  buffer operations.
- `conversation.item.create`: a typed message, function call, or function output;
  the service ACKs it with `conversation.item.created`.
- `response.create`: request a response with optional tools and `auto`, `required`,
  `none`, or named tool choice.
- `response.speak`: speak 1–500 plain-text characters with the session's fixed
  language and voice. SSML, model, voice, and reference fields are rejected. It
  uses the normal response lifecycle without an LLM inference.
- `response.cancel`: identify the exact response and generation to cancel.

## Server events

- `session.created` reports fixed model IDs, exact local revisions, configured
  default language/voice, supported language/voice IDs, sample rates, and the
  active `tts_mode` plus the session's bounded operator-configured llama.cpp slot
  (voice-style instructions are honored only in `voice_design`).
- `session.updated` acknowledges the exact `session.update` source event.
- `conversation.item.created` acknowledges durable in-session context acceptance.
- `error` contains a bounded structured code/message, retryability, and optional
  source event ID.
- `input_audio_buffer.speech_started` / `.speech_stopped` define server-VAD turns.
- `conversation.item.input_audio_transcription.delta` / `.completed` carry partial
  and final multilingual transcription. Only completed text becomes conversation state.
- `response.created` establishes the response/generation/item correlation.
- `response.output_text.delta` / `.done` stream visible assistant text.
- `response.output_audio.delta` / `.done` stream ordered PCM16 and close audio.
- `response.output_function_call.done` carries one canonical structured call. It
  is released to LiveKit only with the matching `response.done(reason=tool_call)`.
- `response.done` terminates exactly one response with status, reason, and real
  text-token usage. Unknown audio token usage is not represented or invented.

Canonical examples for protocol events live under `tests/fixtures/protocol` and are
round-tripped by the protocol package tests.

## Bounds and close codes

Instructions are capped at 8,000 characters, voice-style instructions at 2,000, a
replay item at 16,000, a text delta at 4,096, an error message at 2,048, and IDs at
96–100 characters with fixed prefixes. There are at most 32 tools, 16 KiB per
schema, 64 KiB total schemas, and 16,000 characters each for arguments and output.
Pydantic models use `extra="forbid"`.

| Code | Meaning |
|---:|---|
| 4400 | protocol/configuration error |
| 4401 | missing or invalid bearer token |
| 4409 | session state conflict |
| 4429 | all configured session slots occupied |
| 4500 | model/service failure |
| 1012 | service draining or restarting |

When the connection remains usable, the service sends a structured `error` before
closing. Input/output queues are bounded; audio and final work are never silently
dropped. Partial STT is the only deliberately droppable work class.

Admission is authoritative at the WebSocket. Authentication and subprotocol
validation precede slot claim. A slot remains occupied during disconnect cleanup;
it returns to `idle` only after generation cancellation and STT/TTS drain, or is
quarantined as `stuck` after the configured timeout.
