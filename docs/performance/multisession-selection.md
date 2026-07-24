# Multi-session profile selection

Status: **no measured winner yet**

## Baseline

- Repository baseline: `9692abbe930c320ed1cd0b3e91088b543d153512`
- Compatibility profile: Gemma 4 31B Q4_0, Qwen3-TTS 1.7B GGML
- Compatibility capacity defaults: 2 sessions, 2 llama.cpp slots, 32768 total
  context tokens
- Raw report path convention:
  `benchmarks/reports/<profile>-s<sessions>-<arrival>-<workload>.raw.jsonl`
- Selection report: `benchmarks/reports/model-selection.json`

No populated performance report is committed. A report is valid only when its
raw records contain complete provenance and were produced by real NVIDIA hardware
with the exact verified model lock.

## Hardware and runtime provenance

Every candidate row must include:

- GPU model and VRAM;
- NVIDIA driver and CUDA versions;
- host and container image digest;
- Git commit and dirty-state flag;
- profile ID, model IDs, revisions, artifact sizes, SHA-256 values and
  quantizations;
- llama.cpp commit and arguments;
- Python, Torch, TTS runtime and service versions;
- session, slot, total-context and TTS-worker counts;
- the SHA-256 configuration fingerprint derived from the complete authenticated
  `/v1/models` report;
- WAV hashes, seed, workload, arrival mode and actual duration.

## Candidate profiles

| Profile | LLM | TTS | Measurement status |
|---|---|---|---|
| `compat_gemma31_qwen17_ggml` | Gemma 4 31B | Qwen3-TTS 1.7B GGML | open |
| `multisession_gemma_a4b_qwen06_cuda` | Gemma 4 26B A4B | Qwen3-TTS 0.6B CUDA graph | open |
| `multisession_qwen_a3b_qwen06_cuda` | Qwen3 30B A3B 2507 | Qwen3-TTS 0.6B CUDA graph | open |

Exactly one profile is loaded per service process. Upstream benchmark numbers do
not fill any local result field.

## Required correctness gates

All measured target configurations require zero cross-session leaks, stale final
responses, audio before tool results, duplicate tool executions, unknown or
mismatched tool results, runtime model reloads, unhandled errors, OOMs and
non-finite audio. Every completed voice response must contain more than one audio
chunk; tool generations must contain no audio; each session must retain its
assigned llama.cpp slot.

## Four-session default gates

The production default may move to four sessions only when:

1. Four barrier-loaded sessions pass a 30-minute soak.
2. Four-session p95 `speech_stop_to_first_audio` is at most 1.75 times the
   one-session p95 for the same profile.
3. Four-session p95 improves by at least 20% over the compatibility profile or
   reaches the documented absolute target.
4. The slowest per-session p95 is at most 25% above the median session p95.
5. Mean TTS RTF is below 1 and the TTS queue does not grow monotonically.
6. Tool evaluation loses no more than two percentage points from the Gemma 31B
   baseline and meets every absolute tool gate.

If these gates are not proven, the default remains two sessions. Reducing it to
one is not a permitted performance solution.
