# Changelog

## 0.3.0 - unreleased

- Generalize the measured soak driver to 1–16 isolated sessions with barrier,
  staggered, normal, tool, and mixed workloads plus p50/p95/p99 and fairness.
- Add bounded VAD/STT/segmentation/prefix-prefill controls and raw llama.cpp
  metrics without changing the unmeasured compatibility defaults.
- Add the pinned Qwen3-TTS 0.6B CUDA-graph candidate and a fair one- or two-worker
  TTS pool; compatibility keeps the 1.7B GGML runtime.
- Add three closed llama.cpp LLM profiles and a reproducible 200-case multilingual
  tool-evaluation corpus. No candidate is selected without real GPU measurements.
- Add native bounded `RealtimeSession.say()` through `response.speak`, using TTS
  directly without a second LLM inference.

## 0.2.0 - unreleased

- Make `voice_clone` the default TTS mode: the Qwen3-TTS base talker speaks the
  five public voice profiles from frozen, operator-defined reference recordings
  (one per voice and language, packaged with the service), keeping each speaker
  identity stable across segments and sessions. `speech.tts_mode: voice_design`
  restores the previous description-driven behavior.
- Default TTS decoding to sampling (`speech.generation.do_sample: true`),
  matching the upstream Qwen3-TTS `generation_config.json`; greedy decoding
  drifted into near-silent output and missed the end-of-speech token on long
  generations.
- Add `benchmarks/generate_voice_refs.py` to render, check, and document the
  frozen reference recordings with full provenance.
- Advertise the active `tts_mode` in `session.created` so clients can tell
  that voice-style instructions apply only to `voice_design`.
- Upgrading requires re-running the model prefetch: the lock must include the
  new `qwen-talker-1.7b-base-BF16.gguf` before the service can start. The
  VoiceDesign talker left the shipped manifest; operators who run
  `voice_design` add its file entry back and prefetch again.

- Add native LiveKit Agents function calling with LiveKit as the sole tool executor.
- Add strict WebSocket protocol v2 schemas, per-response tool choice, typed
  call/result items, acknowledgements, fixed llama.cpp slot affinity, and cache reuse.
- Keep pure tool generations silent and add a two-step structured readiness probe.

## 0.1.0 - unreleased

- Add the strict authenticated Hugging Voice realtime protocol package.
- Add the native LiveKit Agents `RealtimeModel` plugin with bounded audio handling,
  cancellation, reconnection, capacity-aware endpoint discovery, and builtin
  transcription mapping.
- Add the offline CUDA GPU service for shared Parakeet, Gemma 4 31B via llama.cpp,
  Qwen3-TTS, and per-session Silero VAD, with two-session admission and drain.
- Add exact model delivery locks, Docker/Compose delivery, Kustomize demo and
  production overlays, operations documentation, CPU contracts, real-GPU opt-in
  tests, soak tooling, voice audition, and honest benchmark reporting.

No artifact has been pushed or published.
