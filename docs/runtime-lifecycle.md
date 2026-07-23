# Runtime lifecycle

## Exact runtime pins

- llama.cpp: `3ce7da2c852c538c4c5f9806da27029cf8c9cc4a`
- `silero-vad==6.2.1`
- `nano-parakeet==0.2.1`
- `faster-qwen3-tts[ggml]==0.3.2`
- `qwentts-cpp-python==0.3.1` (resolved by the locked extra)
- `torch==2.10.0` and `torchaudio==2.10.0` with the CUDA 12.8 lock set

CPU development does not install the `gpu` extra. GPU tests are marked `gpu` and
require `HV_RUN_GPU_TESTS=1`, a real lock, local verified model files, the optional
dependencies, and visible CUDA. Absence of any prerequisite is reported as a skip,
not a simulated result.

## llama.cpp build

Clone or otherwise obtain the exact commit, then build the server with the Python
helper (the helper never fetches source):

```bash
python services/gpu-service/scripts/build_llama_cpp.py \
  --source /path/to/llama.cpp \
  --build /path/to/llama.cpp/build \
  --jobs 8
```

The helper verifies `git rev-parse HEAD` and configures CUDA, a static release
build, server-only target, no curl model fetcher, and no tests/examples. Runtime
arguments bind `127.0.0.1`, load the local Gemma Q4_0 path, allocate exactly two
slots and 32,768 total context, offload all possible layers, disable the Web UI,
and extract any reasoning separately.

The process manager forwards child output through structured Python logging,
polls `/health`, submits a real chat-completion probe with thinking disabled,
monitors unexpected exit, sends SIGTERM first, and escalates after the configured
shutdown timeout.

## Readiness

`/health/ready` is true only while `ServiceLifecycle` is in `ready`, the child
process remains ready, and both shared realtime schedulers are running. The bearer
secret is validated before model verification so a pod with a missing or malformed
secret never becomes ready. Authenticated `/v1/models` reports exact lock entries,
Q4_0, and the llama.cpp commit. `/metrics` exposes lifecycle, session, turn,
scheduler, latency, cancellation, stale-chunk, WebSocket, and GPU-memory series.

The ASGI service becomes live before background loading completes. A bad hash,
missing CUDA, child failure, model-constructor failure, failed synthesis, or empty
Gemma warmup leaves it unready and releases resources already started. Shutdown
atomically revokes admission, turns readiness red, allows existing connections to
finish until the configured deadline, then closes remaining sessions with 1012.
A disconnect keeps its slot in `draining` until the pipeline and both schedulers
confirm quiescence. A timeout quarantines the slot as `stuck`; it is never reused.
