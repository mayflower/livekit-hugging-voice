# TensorRT-LLM experiment decision

Status: **not required; entry conditions are not met**.

Wave 6 requires all of the following evidence before code or an image profile is
added:

1. `qwen3_30b_a3b_2507` wins the real quality and tool evaluation.
2. Its pinned llama.cpp path misses the four-session p95 gates.
3. Supported target NVIDIA hardware has sufficient measured VRAM.
4. The additional deployment complexity remains explicitly in scope.

There is currently no measured model-selection winner and no valid four-session
candidate report. Implementing a TensorRT-LLM branch now would add an unused
runtime path and violate the conditional wave. The experiment remains closed;
there is no fallback, image, dependency, or dead configuration for it.
