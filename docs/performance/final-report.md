# Version 0.3 verification report

Status: **CPU/contract implementation complete; production GPU selection open**.

## Provenance

- Baseline and current committed revision:
  `9692abbe930c320ed1cd0b3e91088b543d153512`
- Working branch: `v3`; the version 0.3 implementation is intentionally
  uncommitted until separately requested.
- Service/package version: `0.3.0`
- llama.cpp commit: `3ce7da2c852c538c4c5f9806da27029cf8c9cc4a`
- Compatibility profile: `compat_gemma31_qwen17_ggml`
- Compatibility capacity: 2 sessions, 2 slots, 32,768 total context tokens,
  1 TTS worker
- Detected GPUs: Tesla P40, 23,040 MiB; NVIDIA RTX A6000, 49,140 MiB
- NVIDIA driver: 550.163.01
- Image digest: unavailable; local Docker daemon access is denied
- Candidate artifact revisions, sizes, hashes, and quantizations:
  `llm-candidate-artifacts.md` and the locks under `models/profiles/`
- TTS runtime source: `faster-qwen3-tts==0.3.2`, source commit
  `a70afc0f81f7f5f8801c3227968f1102f43f211c`

## Completed verification

- `uv sync --all-packages --frozen`
- `make check`: 247 passed, 6 real-GPU/external-service tests skipped
- `make packages`: all four 0.3.0 sdists and wheels built
- `docker compose -f deploy/docker/compose.yaml config`: passed
- `uv sync --all-packages --extra gpu --frozen`
- `make models-verify`: the local legacy VoiceDesign lock verified its own bytes
- Ruff, Ruff formatting, Mypy, protocol fixtures, profile parsing,
  profile/lock mismatch, scheduler fairness, cancellation, tool lifecycle, and
  direct `say()` contracts passed

`kubectl kustomize` could not be invoked because `kubectl`/`kustomize` is not
installed. The repository's deployment-contract tests parse and validate the
base and production resources, including one GPU, non-root/read-only settings,
profiles, slots, and worker configuration.

## Measurements and gates

No p50/p95/p99, TTFA, RTF, WER, VRAM-load phase, fairness, 30-minute soak, or
tool-accuracy value is reported. The local legacy model lock verifies its own
bytes, but the volume lacks the Base talker required by the reviewed
compatibility profile. This host also has no executable pinned `llama-server`,
the candidate model bundles are not present, and the Docker daemon is
inaccessible. Simulating those measurements would violate the acceptance contract.

Therefore:

- the four-session production gates remain open;
- both faster LLM candidates remain unselected;
- the two-worker CUDA TTS profile remains an explicit candidate;
- the production default remains the two-session compatibility profile;
- TensorRT-LLM is not implemented because its entry conditions are unmet;
- maximum sustainable sessions and measured VRAM remain unknown.

Raw future measurements belong under `benchmarks/reports/` and must include the
commit, clean/dirty state, image digest, model locks, hardware, driver, CUDA,
profile, sessions, slots, context, TTS workers, WAV hashes, workload, p50/p95/p99,
correctness failures, and gate decisions.
