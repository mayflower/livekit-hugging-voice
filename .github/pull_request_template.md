## Summary

Describe the user-visible or operational change and its reason.

## Verification

- [ ] Ruff and formatting pass.
- [ ] Mypy passes.
- [ ] CPU tests and the coverage gate pass.
- [ ] Package, Compose, container, or Kubernetes contracts affected by this change pass.
- [ ] Real GPU results are attached, or GPU validation is explicitly marked not run.

## Security and lifecycle

- [ ] No secret, model weight, transcript, prompt, or audio payload is committed or logged.
- [ ] New dependencies, Actions, images, Git commits, and model revisions remain immutable.
- [ ] Queues, retries, timeouts, history, and payload sizes remain bounded.
- [ ] Cancellation, stale-generation filtering, session isolation, drain, and shutdown remain correct.
