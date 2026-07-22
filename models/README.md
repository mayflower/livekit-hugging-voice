# Model manifest

`manifest.yaml` is the reviewed allowlist of logical models, exact upstream
revisions, and files. The Q4_0 Gemma artifact is the initial verified-4-bit
candidate mandated by the product specification; benchmark-driven selection of a
different documented 4-bit quantization requires an explicit manifest change.

`manifest.lock.json` is intentionally not committed yet. It is generated only by
downloading the real multi-gigabyte artifacts with the explicit prefetch command;
sizes and SHA-256 values are calculated from those bytes and are never guessed.
The service must refuse startup until that real lock exists and verifies.

The Qwen logical model is delivered as the exact BF16 qwentts.cpp conversion from
`Serveurperso/Qwen3-TTS-GGUF`; the runtime receives both locked GGUF paths directly
and never invokes its Hub resolver. Parakeet uses the locked `.nemo` checkpoint
because `nano-parakeet`'s local loader consumes that format.

Silero is delivered by the exactly pinned `silero-vad==6.2.1` Python package. Its
lock entry is verified against installed package metadata rather than fetched with
`torch.hub`.
