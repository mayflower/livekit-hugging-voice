# faster-qwen3-tts CUDA API audit

The CUDA candidate was implemented against `faster-qwen3-tts==0.3.2`, source
commit `a70afc0f81f7f5f8801c3227968f1102f43f211c`. This was a narrow source audit
of the APIs used by the service.

- `FasterQwen3TTS.from_pretrained()` accepts a local model directory, torch
  backend, CUDA device, dtype, attention implementation and
  `local_files_only`. The service supplies only a verified local directory and
  also runs with Hugging Face and Transformers offline.
- `warmup(prefill_len=...)` captures predictor and talker CUDA graphs once.
  Startup remains unready if capture or the subsequent real streaming probe
  fails.
- `generate_voice_clone_streaming()` is under `torch.inference_mode()`, accepts
  `chunk_size`, and accepts a precomputed `voice_clone_prompt`.
- The upstream torch wrapper caches reference preparation internally. The
  service additionally prepares every frozen operator voice/language reference
  at startup and retains the resulting prompt per runtime. It never accepts a
  client reference.
- The streaming API returns a Python generator. Closing that generator runs its
  cleanup path; the service closes it in `finally`, including cancellation and
  early consumer exit.
- CUDA graph objects and static decode buffers are mutable and are not treated
  as reentrant. Each scheduler worker owns one runtime and a session can be
  active on only one worker.

The compatibility GGML runtime and the 0.6B CUDA runtime are distinct startup
profiles. A process constructs only the selected profile.
