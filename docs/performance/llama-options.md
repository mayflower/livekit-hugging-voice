# Pinned llama.cpp option audit

The service accepts only a narrow set of llama-server performance options. The
allowlist was checked against the pinned commit
`3ce7da2c852c538c4c5f9806da27029cf8c9cc4a`, not against a moving release.

At that commit, `common/arg.cpp` defines `--flash-attn [on|off|auto]`,
`--cont-batching`, `--batch-size`, `--ubatch-size`, `--cache-type-k`,
`--cache-type-v`, `--cache-reuse`, and `--metrics`. `common/common.h` records
the upstream defaults: flash attention `auto`, continuous batching enabled,
batch size 2048, ubatch size 512, K/V cache `f16`, cache reuse 0, and the
metrics endpoint disabled.

Version 0.3 explicitly enables continuous batching and metrics. It preserves
the other upstream defaults in the baseline. Candidate profiles may select
only flash attention `auto`/`on`, K/V cache `f16`/`q8_0`, bounded batch and
ubatch sizes, and bounded cache reuse. No arbitrary llama-server argument list
is accepted. A candidate cannot become the default without measured latency,
memory, quality, and structured tool-calling acceptance.

## `--swa-full`

Added to the allowlist as `models.llama_swa_full`, default `false`, which is the
upstream default and therefore leaves the baseline unchanged.

Verified against the binary the pinned commit produces rather than against the
source tree: `llama-server --help` in the running image reports
`--swa-full  use full-size SWA cache (default: false)`, alongside
`-ctxcp, --ctx-checkpoints, --swa-checkpoints N` and `--cache-reuse N`.

It is a latency option here, not a memory option, and the reason is measured.
Gemma 4 uses interleaved sliding-window attention. With the window active,
llama-server cannot reuse the cached prompt prefix and re-processes the whole
prompt on every request. From one real call: `llamacpp:prompt_tokens_total`
29738 over `llm_ttft_seconds_count` 20 text generations is **1487 processed
prompt tokens per generation**, or 1293 across all 23 LLM requests — the divisor
matters, the order of magnitude does not. At `llamacpp:prompt_tokens_seconds`
311 that is about 3.7 s of prompt processing per request, which accounts for the
observed time to first token almost entirely. How many tokens were *new* per
turn is not derivable from these counters; a `/slots` read during the call
showed 65 against a 1388-token prompt. The upstream report is
ggml-org/llama.cpp#21831, reproduced on Gemma 4.

The price is the KV cache: without the flag only the sliding layers keep a small
window, with it every layer holds the full context. On the deployment this
service runs on that works out to roughly 31 GiB at 32768 tokens and about half
at 16384, on a GPU shared with five other tenants. `ModelSettings` therefore
rejects `llama_swa_full` together with a context above 16384 rather than leaving
that coupling to a comment.

Still outstanding before the flag becomes a profile default, per the rule above:
prefill latency and time to first token from `benchmarks/multisession_soak.py`
with two sessions in both arms, the VRAM footprint from
`benchmarks/gpu_memory.py`, `llamacpp:predicted_tokens_seconds` as the decode
control — ggml-org/llama.cpp#24628 reports a decode penalty at deep context —
and one tool-calling call. The success criterion is unambiguous: processed
prompt tokens per generation must fall from ~1487 to roughly the size of one turn.
