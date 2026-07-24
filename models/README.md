# Model manifest

`manifest.yaml` is the reviewed allowlist of logical models, exact upstream
revisions, and files. The Q4_0 Gemma artifact is the initial verified-4-bit
candidate mandated by the product specification; benchmark-driven selection of a
different documented 4-bit quantization requires an explicit manifest change.

`manifest.lock.json` is the operator-generated active lock and remains ignored
because a local checkout may contain an older prefetched bundle. The reviewed
version 0.3 compatibility delivery lock is
`profiles/compat_gemma31_qwen17_ggml.lock.json`; its exact Base-talker metadata
comes from the immutable Hugging Face revision. The service refuses startup
until the selected lock verifies offline and its `profile_id` matches the
configuration.

The Qwen logical model is delivered as the exact BF16 qwentts.cpp conversion from
`Serveurperso/Qwen3-TTS-GGUF`; the runtime receives both locked GGUF paths directly
and never invokes its Hub resolver. Parakeet uses the locked `.nemo` checkpoint
because `nano-parakeet`'s local loader consumes that format.

Silero is delivered by the exactly pinned `silero-vad==6.2.1` Python package. Its
lock entry is verified against installed package metadata rather than fetched with
`torch.hub`.

Smart Turn is delivered as the pinned `smart-turn-v3.2-cpu.onnx` artifact from
`pipecat-ai/smart-turn-v3`. The service verifies its exact revision, byte size,
and SHA-256 before constructing one CPU-only ONNX Runtime session. There is no
runtime Hub resolution or GPU execution-provider fallback.

Version 0.3 complete candidate startup manifests and locks live under
`profiles/`:

- `multisession_gemma_a4b_qwen06_cuda`
- `multisession_qwen_a3b_qwen06_cuda`

Each contains exactly one LLM, the shared Smart Turn and Parakeet models,
Qwen3-TTS 0.6B CUDA artifacts, and pinned runtime packages. The smaller component
locks in that directory are provenance inputs only; runtime never combines locks
or falls back.
