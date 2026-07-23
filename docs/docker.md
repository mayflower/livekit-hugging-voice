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

The service binds to loopback by default. To expose its authenticated HTTP and
WebSocket endpoint on the host network, start Compose with
`HUGGING_VOICE_BIND_ADDRESS=0.0.0.0`; use `HUGGING_VOICE_PORT` to select the
published TCP port. Do not expose it without the mounted bearer-token secret.

The build verifies and compiles llama.cpp commit
`3ce7da2c852c538c4c5f9806da27029cf8c9cc4a`. The runtime image has CUDA/CuDNN,
Python 3.11, the frozen GPU environment, and the one `llama-server` binary, but no
compiler, Git checkout, model weights, or supervisor. The Python ASGI service is
PID 1 and owns child shutdown. `llama-server` binds only to loopback.

Start the pinned LiveKit development server, minimal worker, and browser voice UI:

```bash
make demo-agent
```

Open <http://127.0.0.1:3000>, choose the language and voice, and select **Start
conversation**. The page asks its same-origin token endpoint for a short-lived,
room-scoped LiveKit credential and an explicit `hugging-voice` agent dispatch. It
then publishes microphone audio and attaches the agent's remote audio track.

The demo worker reads `HUGGING_VOICE_LANGUAGE`, `HUGGING_VOICE_VOICE`, and
`HUGGING_VOICE_VOICE_INSTRUCTIONS`; they default to `de`, `warm_female`, and
the service-configured voice style. The service's allowed language/voice maps live
under `speech` in `services/gpu-service/config/default.yaml`. Rebuild after editing
that image default. Scalar service defaults can also be overridden with
`HV_SPEECH__DEFAULT_LANGUAGE` and `HV_SPEECH__DEFAULT_VOICE`.

### Browser and network access

The web UI and LiveKit signaling bind to loopback by default. For LAN or managed
reverse-proxy access, publish them and tell LiveKit which host address clients can
reach:

```bash
export HUGGING_VOICE_WEB_BIND_ADDRESS=0.0.0.0
export LIVEKIT_BIND_ADDRESS=0.0.0.0
export LIVEKIT_NODE_IP=10.0.0.25  # replace with this host's reachable address
make demo-agent
```

The UI uses TCP 3000. LiveKit media uses TCP 7881 and UDP 7882 in this development
configuration; firewalls must allow the client to reach those ports. You normally
do not need to expose TCP 7880 because the web process proxies `/rtc` signaling on
the page's origin.

Microphone capture is allowed by browsers only on localhost or a trustworthy HTTPS
origin. For development from another machine, a localhost tunnel is the smallest
option:

```bash
ssh -L 3000:127.0.0.1:3000 user@voice-host
```

Then browse to <http://localhost:3000>. The browser must still be able to reach the
advertised LiveKit media address and ports. For Internet-facing use, terminate a
trusted certificate on one public hostname, proxy the page and WebSocket upgrade
to port 3000, set `LIVEKIT_NODE_IP` appropriately, and deploy LiveKit with its
production TLS/TURN configuration. The included `--dev` server and fixed
`devkey`/`secret` are local-development settings, not a public deployment.

Relevant Compose controls are:

- `HUGGING_VOICE_WEB_BIND_ADDRESS` and `HUGGING_VOICE_WEB_PORT` for the UI;
- `LIVEKIT_BIND_ADDRESS` for direct signaling exposure when needed;
- `LIVEKIT_RTC_BIND_ADDRESS` for media port binds;
- `LIVEKIT_NODE_IP` for the media address advertised to browsers;
- `LIVEKIT_PUBLIC_URL` to bypass the same-origin signaling proxy and give the UI
  an existing `ws://` or `wss://` LiveKit endpoint.

The optional overlay pins the amd64 `livekit/livekit-server:v1.13.4` manifest by
digest. For a release, resolve and record every architecture-specific manifest
digest before build; never replace digest-bound references with movable tags:

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
