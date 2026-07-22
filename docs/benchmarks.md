# Benchmark method

Benchmark reports are evidence, not configuration targets. They are generated only
from real service events and real NVIDIA telemetry. The repository intentionally
contains no populated performance report because the current checkout has no locked
model volume.

Start the verified GPU service, prepare two different German mono 16 kHz PCM16 WAV
files, and record GPU state in one terminal:

```bash
uv run python benchmarks/gpu_memory.py \
  --phase two_sessions --output benchmarks/reports/gpu-memory-two.csv --duration 1900
```

Run two overlapping sessions for 30 minutes in another terminal:

```bash
uv run python benchmarks/two_session_soak.py \
  --token-file deploy/docker/secrets/token \
  --wav-a /path/to/alpha-de.wav --wav-b /path/to/beta-de.wav \
  --duration 1800
uv run python benchmarks/summarize.py benchmarks/reports/two-session-*.raw.jsonl \
  --gpu-csv benchmarks/reports/gpu-memory-two.csv
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
requires both a final builtin-transcription event and captured 24 kHz response audio.
Its self-hosted runner must configure `HV_GPU_MODEL_ROOT` and `HV_GPU_MODEL_LOCK` as
absolute paths outside the Actions checkout. Checkout cleanup therefore cannot remove
the large verified model cache or its generated lock; the workflow verifies both before
starting Docker.

The service Prometheus series provide STT/TTS queue and inference time, LLM TTFT and
token rate, TTS TTFA and generated-audio duration, cancellations, stale drops, and
runtime load counts. Preserve the raw Prometheus records with the client summary.
Derive TTS RTF from measured TTS generation seconds divided by measured generated
audio seconds. Do not compare runs unless hardware, image digest, model revisions,
quantization, audio inputs, and duration are all present.

Repeat `gpu_memory.py` with phases `idle`, `warm`, and `one_session` in their real
states and pass each CSV through another `--gpu-csv` option. Phase labels are
operator assertions about the state under test; keep the matching service log.

Large raw reports, GPU CSVs, and audition WAVs are ignored by Git. A release report
may be deliberately added only after reviewing its provenance and actual values.
