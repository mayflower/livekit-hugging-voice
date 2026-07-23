# Security policy

## Supported versions

Security fixes are applied to the current `0.2.x` development line on `main`. Older
development snapshots are not maintained.

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability. Use
[GitHub private vulnerability reporting](https://github.com/mayflower/livekit-hugging-voice/security/advisories/new)
to send a private report to the maintainers.

Include the affected revision, deployment shape, impact, reproduction conditions, and
any proposed mitigation. Do not include active credentials, private audio, prompts,
transcripts, model weights, or unrelated personal data. The maintainers will
acknowledge the report in the private advisory and coordinate validation, remediation,
and disclosure there.

Operational hardening and the service trust boundary are documented in
[`docs/security.md`](docs/security.md).

## Dependency advisory boundaries

The service upgrades vulnerable dependencies whenever the supported CUDA and model
stack has a compatible fixed release. A Dependabot alert may instead be closed as
`vulnerable_code_not_used` only when a repository contract proves that its trigger is
outside the product boundary:

- The service never trains models or loads Trainer checkpoints, so the Transformers
  `Trainer._load_rng_state` path from `GHSA-69w3-r845-3855` is unreachable.
- The service does not include or load LightGlue, so `GHSA-fgcw-684q-jj6r` is
  unreachable.
- Clients cannot select models or model configuration. Runtime model loading is
  offline and accepts only operator-prefetched files covered by the exact
  revision/size/SHA-256 lock. Consequently an attacker cannot provide the malicious
  Hub configuration required by `GHSA-29pf-2h5f-8g72`.
- Production code does not call `torch.jit.script` or accept code to compile through
  it, so the trigger for `GHSA-rrmf-rvhw-rf47` is outside the service boundary.

These assertions are enforced by the forbidden-path and deployment-contract tests.
They must be revisited before adding training, arbitrary model loading, runtime
downloads, LightGlue, or TorchScript compilation.
