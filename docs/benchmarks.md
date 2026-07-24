# N-session benchmark method

Tool-turn records add decision, call, execution, result-ACK, final text/audio,
call/result size, slot, error, cancellation, and stale timestamps. The N-session
matrix separates concurrency 1, 2, 4 and optionally 6; normal, tool and mixed
workloads; and staggered versus barrier arrivals. Summaries retain p50/p95/p99
overall and per session, fairness, turns/minute, queue maxima, correctness
violations, and full provenance.

The benchmark driver is `benchmarks/multisession_soak.py`. It deterministically
rotates repeated `--wav` inputs across 1–16 requested sessions.

Benchmark reports are evidence, not configuration targets. They are generated only
from real service events and real NVIDIA telemetry. The repository intentionally
contains no populated performance report because the current checkout has no locked
model volume.

The complete acceptance matrix uses identical WAV hashes and seeds for each profile:

```text
sessions: 1, 2, 4 (6 only when VRAM permits)
arrival: staggered, barrier
workload: normal, tool, mixed
smoke duration: 3–5 minutes
acceptance duration: 30 minutes
```

Start the verified GPU service, prepare at least two different mono 16 kHz PCM16
WAV files, and record GPU state in another terminal. Raw reports must identify the
selected profile, concurrency, arrival mode, workload, commit, image digest, model
revisions, quantizations, runtime versions, hardware, driver, CUDA, WAV hashes,
seed, requested duration, actual duration, slots and TTS workers.

```bash
uv run python benchmarks/gpu_memory.py \
  --phase four_sessions --output benchmarks/reports/gpu-memory-four.csv --duration 1900
```

Run one session and then two overlapping sessions for 30 minutes each:

```bash
uv run python benchmarks/multisession_soak.py \
  --token-file deploy/docker/secrets/token \
  --wav /path/to/alpha-de.wav --sessions 1 --arrival staggered \
  --workload mixed --duration 1800 \
  --output benchmarks/reports/compat-s1-staggered-mixed.raw.jsonl
uv run python benchmarks/multisession_soak.py \
  --token-file deploy/docker/secrets/token \
  --wav /path/to/alpha-de.wav --wav /path/to/beta-de.wav \
  --sessions 4 --arrival barrier --workload mixed --duration 1800 \
  --output benchmarks/reports/compat-s4-barrier-mixed.raw.jsonl
uv run python benchmarks/summarize.py benchmarks/reports/one-session.raw.jsonl \
  --gpu-csv benchmarks/reports/gpu-memory-one.csv
uv run python benchmarks/summarize.py \
  benchmarks/reports/compat-s4-barrier-mixed.raw.jsonl \
  --gpu-csv benchmarks/reports/gpu-memory-four.csv
```

The runner refuses to overwrite raw data. It records model revisions,
quantization, llama.cpp pin, host/GPU/driver data, Git commit, WAV hashes, requested
duration, per-turn external latency, errors, and Prometheus snapshots. Set
`HV_CONTAINER_IMAGE_DIGEST` to the digest of the image actually under test. Missing
provenance remains `null`; it is never guessed.

Each session receives a different system-instruction canary. Every completed response
must contain its own canary and no response may contain the other session's canary;
the runner aborts on a violation and records successful checks without persisting the
response text. The external GPU tests also require all three shared model-load counters
to remain exactly one.

The opt-in GPU suite additionally starts a native LiveKit `AgentSession` with only the
Hugging Voice `RealtimeModel`, streams the German WAV through LiveKit audio frames, and
requires both a final builtin-transcription event and completed 24 kHz response audio,
including more than the first second of the first answer. The 30-minute external soak
paces input audio in real time, cancels every seventh turn, and reconnects every
eleventh turn so the scheduled run exercises the lifecycle paths instead of measuring
only uninterrupted steady state.
Its self-hosted runner must configure `HV_GPU_MODEL_ROOT` and `HV_GPU_MODEL_LOCK` as
absolute paths outside the Actions checkout. Checkout cleanup therefore cannot remove
the large verified model cache or its generated lock; the workflow verifies both before
starting Docker.

The service Prometheus series provide STT/TTS queue and inference time, LLM TTFT,
prompt evaluation, token rate and cache reuse where exposed, TTS TTFA and
generated-audio duration, cancellations, stale drops, runtime load counts, busy
slots, decode concurrency and worker activity. Preserve raw Prometheus records
with the client summary.
Derive TTS RTF from measured TTS generation seconds divided by measured generated
audio seconds. Do not compare runs unless hardware, image digest, model revisions,
quantization, audio inputs, and duration are all present.

Repeat `gpu_memory.py` with phases `idle`, `warm`, and `one_session` in their real
states and pass each CSV through another `--gpu-csv` option. Phase labels are
operator assertions about the state under test; keep the matching service log.

Large raw reports, GPU CSVs, and audition WAVs are ignored by Git. A release report
may be deliberately added only after reviewing its provenance and actual values.
