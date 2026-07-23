# Changelog

## 0.2.0 - unreleased

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
