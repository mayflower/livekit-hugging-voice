# LLM candidate artifact audit

This is the narrow version 0.3 delivery audit. It records artifact facts, not
local performance or quality results.

| Profile | Source artifact | Revision | Quantization | Bytes | SHA-256 |
| --- | --- | --- | --- | ---: | --- |
| `compat_gemma31` | `ggml-org/gemma-4-31B-it-GGUF/gemma-4-31B-it-Q4_0.gguf` | `83233b01829859b51252aa4fd1de1cf9e1cf91cc` | Q4_0 | 17,992,313,088 | `031dc1c5fa9c5a0abbf3c39c5173fb2af65f5ac2dc2a090268561d3c72dcd834` |
| `gemma4_26b_a4b` | `ggml-org/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-Q4_0.gguf` | `3d3dca2094ff8112005fd10fc7a8e30cf4f45b56` | Q4_0 | 14,618,145,824 | `d208665ab1cd3a69f7a9a4bc59430e8448c8093d9b06334f566ac59d6d504a03` |
| `qwen3_30b_a3b_2507` | `bartowski/Qwen_Qwen3-30B-A3B-Instruct-2507-GGUF/Qwen_Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf` | `6c6e8692f43e4ca663f7ece8229a1361090d3a4c` | Q4_K_M | 18,632,183,808 | `382b4f5a164d200f93790ee0e339fae12852896d23485cfb203ce868fea33a95` |

The Gemma repository is the `ggml-org` conversion of the named Google source
model. The Qwen artifact is Bartowski's llama.cpp-compatible quantization of
`Qwen/Qwen3-30B-A3B-Instruct-2507`. Both are Apache-2.0. The manifests pin the
repository commit and exact file; runtime conversion is forbidden.

Qwen Instruct-2507 is configured as non-thinking. Its chat-template kwargs are
empty: `enable_thinking` is deliberately not passed as a substitute.

No candidate has passed the repository's 200-case multilingual tool evaluation
or real four-session matrix yet. Therefore neither is selected as default.
