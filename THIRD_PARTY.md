# Third-party components

This inventory records the version-locked direct runtime components for 0.1.0.
Transitive Python versions are fixed in `uv.lock`; image manifest digests are
generated with `make image-digests`. Model weights are fetched separately and are
not present in Git or the service image.

| Component | Pin | Use/license | Distribution status |
|---|---|---|---|
| LiveKit Agents | 1.6.6 | plugin API; Apache-2.0 | Python dependency |
| aiohttp | 3.14.2 | HTTP/WebSocket client; Apache-2.0 AND MIT | Python dependency |
| FastAPI | 0.139.2 | internal ASGI API; MIT | service Python dependency |
| Uvicorn | 0.51.0 | ASGI server; BSD-3-Clause | service Python dependency |
| Pydantic / pydantic-settings | lockfile / 2.10.1 | validation/configuration; MIT | Python dependencies |
| Prometheus client | 0.25.0 | metrics exposition; Apache-2.0 | service Python dependency |
| Hugging Face Hub | 0.36.0 | explicit prefetch only; Apache-2.0 | service Python dependency |
| PyYAML | 6.0.3 | configuration/manifests; MIT | service Python dependency |
| Gemma 4 31B IT GGUF | revision in `models/manifest.yaml` | language model; Apache-2.0 | weights fetched separately |
| Qwen3-TTS CustomVoice GGUF | revision in `models/manifest.yaml` | synthesis; Apache-2.0 | weights fetched separately |
| NVIDIA Parakeet TDT 0.6B v3 | revision in `models/manifest.yaml` | transcription; CC-BY-4.0 | weights fetched separately |
| Silero VAD | 6.2.1 | voice activity detection; MIT | optional GPU dependency |
| llama.cpp | `3ce7da2c852c538c4c5f9806da27029cf8c9cc4a` | loopback model server; MIT | binary built in image |
| nano-parakeet | 0.2.1 | Parakeet inference; MIT | optional GPU dependency |
| faster-qwen3-tts | 0.3.2 | Qwen streaming runtime; MIT | optional GPU dependency |
| PyTorch / Torchaudio | 2.8.0 | CUDA tensor/audio runtime; BSD-style | optional GPU dependencies |
| NVIDIA CUDA/CuDNN image | 12.8.1 / Ubuntu 24.04 | GPU runtime; NVIDIA terms | container base |
| Python image | 3.11.13 slim Bookworm | language runtime; PSF and bundled terms | container base |
| LiveKit Server | 1.13.4 | optional local development server; Apache-2.0 | separate container |
