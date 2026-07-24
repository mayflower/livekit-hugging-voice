# Reproducible model delivery

Model acquisition is an explicit deployment step and is never part of service
startup. `models/manifest.yaml` fixes each Hugging Face repository to a 40-character
commit and allowlists every required file. It currently selects
`gemma-4-31B-it-Q4_0.gguf` from the 31B repository—not E4B—and records Silero as the
pinned `silero-vad==6.2.1` package. Parakeet is delivered as the exact `.nemo`
checkpoint consumed by the local `nano-parakeet` loader. Qwen uses exact BF16
GGUF conversions from `Serveurperso/Qwen3-TTS-GGUF`: the base talker (default
`voice_clone` mode) plus the shared 12 Hz codec; the VoiceDesign talker is
fetched only when an operator adds it to the manifest for `voice_design` mode.
The logical model is `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`.
Smart Turn uses the exact quantized v3.2 CPU ONNX artifact from
`pipecat-ai/smart-turn-v3`; it is prefetched and locked like every other weight.

## Prefetch

From the repository root, with network access and any required Hugging Face token
already configured in the standard Hugging Face environment:

```bash
uv run hugging-voice-model-prefetch \
  --manifest models/manifest.yaml \
  --output .models \
  --lock models/manifest.lock.json
```

The prefetcher asks Hugging Face for the specified revision and refuses a different
resolved commit. Only allowlisted files are downloaded. It measures byte size and
SHA-256 from the local artifact, builds a complete lock in memory, fsyncs a temporary
file, and atomically replaces the destination lock only after every entry succeeds.
A failed or partial download therefore cannot publish a successful lock.

Reviewed complete delivery locks under `models/profiles/` record exact artifact
sizes and SHA-256 values. The root `manifest.lock.json` remains the ignored
operator-generated active lock because an existing checkout may hold a legacy
prefetch. Every manifest and lock carries one bounded `profile_id`; startup
requires an exact match with the selected service configuration before hashing
or loading a model. Component audit locks are not silently combined at runtime.

## Offline verification

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
uv run hugging-voice-model-verify \
  --lock models/manifest.lock.json \
  --root .models
```

Verification performs no network operation. It rejects missing files, changed
sizes, hash mismatches, unsafe paths, missing Silero installation, or a Silero
version other than 6.2.1. Smart Turn is loaded only from its verified local ONNX
path with `CPUExecutionProvider`. Verification is a startup prerequisite before
any model constructor runs.

Parakeet's checkpoint is opened directly. Compatibility Qwen uses
`GGMLQwen3TTS.from_gguf` with verified files; the CUDA candidate calls
`FasterQwen3TTS.from_pretrained` only with the verified local model directory
while both offline flags are already set. `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1` remain mandatory at runtime.

Model weights remain outside Git and outside the service image. Docker mounts the
verified directory read-only; Kubernetes uses the same prefetch CLI in an explicit
network-enabled job and mounts the resulting PVC read-only into offline service
pods.

Version 0.3 candidate sources and hashes are recorded in
`docs/performance/llm-candidate-artifacts.md`. Selecting a candidate requires a
complete profile lock containing that LLM, Smart Turn, Parakeet, the selected TTS
artifacts, and pinned Python packages.
