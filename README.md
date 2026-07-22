# livekit-hugging-voice

`livekit-hugging-voice` is a local, GPU-hosted German speech-to-speech service and
a native LiveKit Agents realtime-model plugin. The intended path is:

```text
LiveKit Agent
  -> livekit-plugins-hugging-voice
  -> authenticated internal WebSocket
  -> Silero VAD
  -> nvidia/parakeet-tdt-0.6b-v3
  -> google/gemma-4-31B-it through a local llama-server
  -> Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
  -> 24 kHz PCM16 German audio
```

The service is deliberately not a generic voice platform. A GPU pod will admit at
most two isolated sessions while loading Parakeet, Gemma, and Qwen exactly once.
The public language and voice are fixed to `de` and `de_standard_01`.

## Status

Version 0.1.0 is implemented locally: strict protocol, offline CUDA service,
native LiveKit plugin, bounded two-session pipeline, exact model delivery,
Docker/Compose, Kustomize, capacity-aware pod discovery, CPU contracts, and
reproducible GPU/benchmark tooling.

A local RTX A6000 smoke test has verified model hashing, full GPU warmup, LiveKit
worker registration, built-in transcription, and a 24 kHz audio response through a
native `AgentSession`. This is functional verification, not a latency or throughput
benchmark; see [`docs/performance.md`](docs/performance.md) for the measurement
contract.

## Quickstart

Install the locked CPU development environment and run its complete checks:

```bash
uv sync --all-packages --frozen
make check
```

For the local GPU demo you also need Docker Compose, NVIDIA Container Toolkit, a
CUDA 12.8-compatible GPU, and enough disk space for roughly 23 GiB of model files.
Explicitly fetch the models, create the bearer secret, select a GPU, and build the
images:

```bash
make models
install -m 640 /dev/null deploy/docker/secrets/token
python -c "import secrets; print(secrets.token_urlsafe(48))" > deploy/docker/secrets/token
export HUGGING_VOICE_SECRET_GID="$(id -g)"
export HUGGING_VOICE_GPU_DEVICE=0
docker compose \
  -f deploy/docker/compose.yaml \
  -f deploy/docker/compose.livekit.yaml \
  build
docker compose \
  -f deploy/docker/compose.yaml \
  -f deploy/docker/compose.livekit.yaml \
  up -d --no-build
curl --fail http://127.0.0.1:8765/health/ready
```

The final command starts the GPU service, pinned LiveKit development server, and
minimal worker. The worker registers as `hugging-voice-german` against
`ws://127.0.0.1:7880`. No browser UI is included.

To run a real native-agent speech smoke, provide an uncompressed mono 16 kHz PCM16
WAV file:

```bash
HV_RUN_GPU_TESTS=1 \
HV_GPU_TOKEN_FILE=deploy/docker/secrets/token \
HV_GPU_WAV_A=/absolute/path/to/speech.wav \
HV_GPU_SERVICE_URL=http://127.0.0.1:8765 \
uv run pytest -q -s \
  packages/livekit-plugins-hugging-voice/tests/test_external_gpu_agent_session.py
```

Stop the complete local stack with `make docker-down`. Configuration and
operational details are in
[`docs/docker.md`](docs/docker.md), [`docs/kubernetes.md`](docs/kubernetes.md), and
[`docs/operations.md`](docs/operations.md).

## Development

Python 3.11 and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync --all-packages --frozen
uv run ruff check .
uv run ruff format --check .
uv run mypy packages services examples
uv run pytest -q
```

Model weights are never downloaded at service startup and are not stored in the
service image or Git repository. Use `make models` for the explicit prefetch and
`make models-verify` for offline verification; see
[`docs/model-delivery.md`](docs/model-delivery.md).

The service's GPU dependencies are locked but optional for CPU development. The GPU
image installs only the service workspace package with its `gpu` extra; it does not
install the LiveKit agent or fetch model weights.

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — process and isolation design
- [`docs/protocol.md`](docs/protocol.md) — authenticated WebSocket version 1
- [`docs/model-delivery.md`](docs/model-delivery.md) — exact offline model lock
- [`docs/security.md`](docs/security.md) — trust and secret boundary
- [`docs/benchmarks.md`](docs/benchmarks.md) — measured two-session methodology
- [`docs/model-selection.md`](docs/model-selection.md) — German voice audition

Pinned source analysis is recorded in
[`docs/upstream-baseline.md`](docs/upstream-baseline.md), and the durable
architecture and safety constraints are documented in [`AGENTS.md`](AGENTS.md).
