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
