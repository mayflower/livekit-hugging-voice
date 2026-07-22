# Docker and local RTX 6000 demo

## Prerequisites

- Linux with an NVIDIA RTX 6000 Ada 48 GB or another tested CUDA 12.8 GPU;
- current NVIDIA driver plus NVIDIA Container Toolkit;
- Docker Engine with Compose v2;
- enough local disk for the locked model artifacts.

No VRAM requirement or latency result is claimed until measured by the commands
below.

## Offline model delivery

Model weights are not copied into either image. Fetch and verify them explicitly:

```bash
make models
make models-verify
```

This creates `.models` and `models/manifest.lock.json`. Compose mounts the model
directory at `/models` and the generated lock separately at
`/etc/hugging-voice/manifest.lock.json`, both read-only; a missing lock, file, or
matching hash makes startup fail before CUDA/model construction. Runtime sets `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1`, and neither entrypoint invokes a downloader.

Create the mounted bearer secret without committing it:

```bash
install -m 640 /dev/null deploy/docker/secrets/token
python -c "import secrets; print(secrets.token_urlsafe(48))" \
  > deploy/docker/secrets/token
export HUGGING_VOICE_SECRET_GID="$(id -g)"
```

## Build and run

Select GPU device 0 (the default), render the Compose contract, build, and start:

```bash
export HUGGING_VOICE_GPU_DEVICE=0
export HUGGING_VOICE_SECRET_GID="$(id -g)"
docker compose -f deploy/docker/compose.yaml config
make docker-build
make docker-up
curl --fail http://127.0.0.1:8765/health/live
curl --fail http://127.0.0.1:8765/health/ready
```

The build verifies and compiles llama.cpp commit
`3ce7da2c852c538c4c5f9806da27029cf8c9cc4a`. The runtime image has CUDA/CuDNN,
Python 3.11, the frozen GPU environment, and the one `llama-server` binary, but no
compiler, Git checkout, model weights, or supervisor. The Python ASGI service is
PID 1 and owns child shutdown. `llama-server` binds only to loopback.

Start the pinned LiveKit development server and minimal worker without a Web UI:

```bash
make demo-agent
```

The optional overlay pins `livekit/livekit-server:v1.13.4`. For a release, resolve
and record manifest digests before build; do not replace pinned tags with `latest`:

```bash
make image-digests
```

This writes `dist/IMAGE_DIGESTS.json` and refuses to overwrite prior evidence.
Apply the recorded architecture-appropriate digests to a release build or retain
the file beside the image export; tag resolution alone is not an immutable release.

Compose implements file-backed secrets as bind mounts, so it cannot remap their
ownership. The token remains owner-writable and group-readable on the host, while
`HUGGING_VOICE_SECRET_GID` adds only that numeric host group to the two non-root
containers. Do not make the token world-readable.

## Failure smokes

Each prerequisite fails closed:

```bash
# Missing models/lock: temporarily point MODEL_ROOT at an empty directory.
docker run --rm --gpus device=0 \
  -v "$(mktemp -d):/models:ro" \
  -v "$PWD/deploy/docker/secrets/token:/run/secrets/hugging_voice_token:ro" \
  livekit-hugging-voice:local

# Missing token: omit the /run/secrets mount; startup fails in loading_auth.
# Missing GPU: omit --gpus; startup fails in checking_cuda with no CPU fallback.
```

An unexpected llama-server exit is monitored by the Python lifecycle, immediately
revokes readiness, and exposes a lifecycle failure metric. Container logs never
include the bearer header, audio bytes, full prompts, or complete transcripts.

## Measure, do not estimate

Record real GPU memory through warmup and two sessions:

```bash
mkdir -p benchmarks/reports
nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu \
  --format=csv -l 1 > benchmarks/reports/nvidia-smi-two-session.csv
```

In another terminal, start the service and the two-session soak. Preserve the
service Prometheus snapshot with the report. Only measured results belong in
`docs/performance.md`.
