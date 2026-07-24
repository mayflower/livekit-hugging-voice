# Performance status

Tool turns add a silent Gemma decision and external execution before the final
text/TTS generation. Both generations use the session's fixed llama.cpp slot and
`cache_prompt=true`. Measure decision, execution, ACK-to-text, and ACK-to-audio
separately from normal voice turns.

No real latency, throughput, or VRAM winner is available in this checkout. The
host exposes NVIDIA GPUs and the local `.models` directory passes its legacy
VoiceDesign lock, but lacks the Base talker selected by the reviewed
`voice_clone` compatibility profile. No executable pinned `llama-server` or
candidate model bundle is available, and the Docker daemon is inaccessible.
Warmup, candidate GPU E2E, voice audition, and the 30-minute multi-session matrix
therefore cannot yet be executed honestly from this worktree. The complete verification status is in
[`performance/final-report.md`](performance/final-report.md).

Use [benchmarks.md](benchmarks.md) to produce a provenance-labelled report. Do not
copy latency objectives into this file as measured values. Once a complete run exists,
record p50/p95/p99 external turn latencies, scheduler/LLM/TTS metrics, TTS RTF,
idle and 1/2/4-session GPU memory (plus 6 sessions when feasible), fairness,
errors/OOMs, runtime load counts, hardware, driver, CUDA, image digest, model
revisions, quantization, commit, profile, arrival mode, workload, and duration.

The compatibility defaults remain two sessions, two llama.cpp slots, 32768 total
context tokens, 500 ms minimum silence, and the existing model stack. Four sessions
are a target, not a claim. The default changes only after the correctness and
performance gates in
[`performance/multisession-selection.md`](performance/multisession-selection.md)
are supported by raw real-hardware reports.
