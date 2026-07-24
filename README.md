# livekit-hugging-voice

[![CPU CI](https://github.com/mayflower/livekit-hugging-voice/actions/workflows/ci.yaml/badge.svg?branch=main&event=push)](https://github.com/mayflower/livekit-hugging-voice/actions/workflows/ci.yaml)
[![Security](https://github.com/mayflower/livekit-hugging-voice/actions/workflows/security.yaml/badge.svg?branch=main)](https://github.com/mayflower/livekit-hugging-voice/actions/workflows/security.yaml)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/mayflower/livekit-hugging-voice/badge)](https://scorecard.dev/viewer/?uri=github.com/mayflower/livekit-hugging-voice)
[![Real GPU E2E](https://github.com/mayflower/livekit-hugging-voice/actions/workflows/gpu-e2e.yaml/badge.svg)](https://github.com/mayflower/livekit-hugging-voice/actions/workflows/gpu-e2e.yaml)
[![Dependabot](https://img.shields.io/badge/Dependabot-enabled-025E8C?logo=dependabot)](https://github.com/mayflower/livekit-hugging-voice/security/dependabot)
[![Security policy](https://img.shields.io/badge/security-policy-2F81F7)](SECURITY.md)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

`livekit-hugging-voice` is a local, GPU-hosted multilingual speech-to-speech service and
a native LiveKit Agents realtime-model plugin. The intended path is:

```text
LiveKit Agent
  -> livekit-plugins-hugging-voice
  -> authenticated internal WebSocket
  -> Silero VAD
  -> nvidia/parakeet-tdt-0.6b-v3
  -> google/gemma-4-31B-it through a local llama-server
  -> Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
  -> 24 kHz PCM16 audio in the configured language and voice
```

Each GPU pod loads Parakeet and one selected LLM once, plus the configured bounded
pool of one or two Qwen runtimes in production. Session capacity is
operator-configurable; the default is two for compatibility. This default is not
a performance guarantee.

The shipped catalog provides German, English, French, and Italian plus five fixed
voice profiles. `de` and `warm_female` are the defaults. The default
`voice_clone` mode uses frozen operator-provided reference recordings;
`voice_design` creates the voice from its configured description.

## Status

Version 0.3.0 includes the native LiveKit plugin and tool calling, the offline CUDA
service, configurable bounded sessions, exact model delivery, Docker Compose,
Kustomize, capacity-aware pod discovery, CPU contracts, and reproducible GPU
benchmark tooling.

Repository safeguards include GitHub CodeQL, dependency review, OpenSSF Scorecard,
Dependabot version and security updates, secret scanning with push protection, and
SHA-pinned Actions. Results are available through the
[Actions](https://github.com/mayflower/livekit-hugging-voice/actions) and
[Security](https://github.com/mayflower/livekit-hugging-voice/security) dashboards.
Vulnerabilities must be reported privately according to [`SECURITY.md`](SECURITY.md).

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

The final command starts the GPU service, pinned LiveKit development server,
minimal worker, and browser voice demo. Open <http://127.0.0.1:3000>, select a
language and voice, and grant microphone access. The web server creates a private
room with an explicit dispatch to the `hugging-voice` worker; speech choices are
carried in that dispatch and apply only to that room.

Browser microphone access requires a secure context. Loopback is accepted for
local development; a UI exposed to another machine needs a trusted HTTPS endpoint.
See [`docs/docker.md`](docs/docker.md) for bind addresses, LiveKit media ports, and
remote-access guidance.

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

## Speech configuration

The service config owns the allowed language and voice maps. Public language codes
map to Qwen language names plus an LLM response instruction; public voice IDs map to
fixed voice profiles: a VoiceDesign description rendered for the selected native
language plus, for the default `voice_clone` mode, one frozen reference recording
per language. Edit
`services/gpu-service/config/default.yaml` for Docker-image defaults or the
Kubernetes ConfigMap for a deployment. `HV_SPEECH__DEFAULT_LANGUAGE` and
`HV_SPEECH__DEFAULT_VOICE` can override the two scalar defaults.

Each `RealtimeModel` selects a configured pair:

```python
RealtimeModel(
    language="en",
    voice="warm_female",
    voice_instructions="Speak warmly and at a relaxed pace.",
)
```

Omit `voice_instructions` to use the operator-configured voice default. The service
advertises its accepted IDs in `session.created.supported_languages` and
`supported_voices` and rejects unknown IDs during `session.update`. Omitting
`language` or `voice` inherits the service defaults, so changing those defaults
does not require a client-code change.

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
- [`docs/protocol.md`](docs/protocol.md) — authenticated WebSocket protocol v2
- [`docs/model-delivery.md`](docs/model-delivery.md) — exact offline model lock
- [`docs/security.md`](docs/security.md) — trust and secret boundary
- [`docs/benchmarks.md`](docs/benchmarks.md) — N-session measurement methodology
- [`docs/model-selection.md`](docs/model-selection.md) — German voice audition

Pinned source analysis is recorded in
[`docs/upstream-baseline.md`](docs/upstream-baseline.md), and the durable
architecture and safety constraints are documented in [`AGENTS.md`](AGENTS.md).
