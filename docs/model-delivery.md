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

The real lock is not present in this Wave 1 checkout because the full model set was
not downloaded. No placeholder sizes or hashes are supplied.

## Offline verification

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
uv run hugging-voice-model-verify \
  --lock models/manifest.lock.json \
  --root .models
```

Verification performs no network operation. It rejects missing files, changed
sizes, hash mismatches, unsafe paths, missing Silero installation, or a Silero
version other than 6.2.1. Wave 2 makes this verification a startup prerequisite
before any model constructor runs.

Neither runtime calls a `from_pretrained` Hub resolver: Parakeet's checkpoint is
opened directly and Qwen uses `GGMLQwen3TTS.from_gguf` with both verified paths.
`HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1` remain mandatory at runtime.

Model weights remain outside Git and outside the service image. Docker mounts the
verified directory read-only; Kubernetes uses the same prefetch CLI in an explicit
network-enabled job and mounts the resulting PVC read-only into offline service
pods.
