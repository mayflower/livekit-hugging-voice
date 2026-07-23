# Performance status

Tool turns add a silent Gemma decision and external execution before the final
text/TTS generation. Both generations use the session's fixed llama.cpp slot and
`cache_prompt=true`. Measure decision, execution, ACK-to-text, and ACK-to-audio
separately from normal voice turns.

No real latency, throughput, or VRAM result is available in this checkout. The host
exposes NVIDIA GPUs, but `.models/` and `models/manifest.lock.json` are absent, so
model verification, warmup, GPU E2E, voice audition, and the 30-minute run cannot be
executed honestly.

Use [benchmarks.md](benchmarks.md) to produce a provenance-labelled report. Do not
copy latency objectives into this file as measured values. Once a complete run exists,
record p50/p95/p99 external turn latencies, scheduler/LLM/TTS metrics, TTS RTF, idle
and one-/two-session GPU memory, errors/OOMs, runtime load counts, hardware, driver,
CUDA, image digest, model revisions, quantization, commit, and duration.
