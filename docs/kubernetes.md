# Kubernetes delivery

## Cluster prerequisites

- NVIDIA drivers and NVIDIA Device Plugin exposing `nvidia.com/gpu`;
- a storage class that can provide the model PVC;
- nodes sized from measurements for the pinned 31B/Qwen/Parakeet combination;
- a registry containing the exact service image digest.

The CPU/memory/storage values in the base carry explicit sizing annotations and
are starting placeholders, not benchmark results. Validate them on the target
GPU/node before production use.

## Model prefetch and secret

The service pod is offline and mounts its PVC read-only. `secret.yaml` is an
unapplied shape example; the Kustomization never installs its placeholder. Create
the base resources/PVC and real Secret, then run the one-shot prefetch job:

```bash
kubectl apply -k deploy/kubernetes/overlays/demo
kubectl create secret generic hugging-voice-token \
  --from-literal=token="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')" \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f deploy/kubernetes/base/model-prefetch-job.yaml
kubectl wait --for=condition=complete job/hugging-voice-model-prefetch --timeout=60m
```

The job alone explicitly enables Hugging Face network access, writes exact revisions
and hashes to the PVC, and is intentionally excluded from every Kustomization. Delete
it after success. A rollout never downloads or mutates model files.

## Demo and production

Render before applying:

```bash
kubectl kustomize deploy/kubernetes/overlays/demo
kubectl apply -k deploy/kubernetes/overlays/demo
```

The demo is one `Recreate` pod and at most two sessions. It has no PDB or HPA.
The base ConfigMap's `speech` section defines the public language and voice IDs,
their Qwen mappings, the response-language instructions, and default speaking
styles. Update that ConfigMap and roll the pod to change the deployment catalog.

Production starts as an explicit three-replica example with preferred node
anti-affinity. Set the immutable registry/digest and a replica count no larger
than real GPU capacity:

```bash
cd deploy/kubernetes/overlays/production
kustomize edit set image livekit-hugging-voice=registry.internal/voice/livekit-hugging-voice@sha256:REPLACE
kustomize edit set replicas hugging-voice=3
kubectl apply -k .
```

Each pod still owns one GPU and admits at most two sessions. There is no CPU HPA,
database, Redis, operator, or cross-pod session migration. Keep `Recreate` unless
a rollout is proven to have spare GPU capacity; changing rollout strategy is an
operator decision, not an overlay default.

## Plugin discovery

Within the same namespace:

```python
RealtimeModel(
    headless_dns="hugging-voice-headless.default.svc.cluster.local",
    token_file="/run/secrets/hugging_voice_token",
)
```

The resolver handles A and AAAA answers, queries authenticated `/v1/capacity` with
a short bounded cache, and randomizes equally free pods. Capacity lookup is only
an optimization: the atomic WebSocket slot claim is authoritative, and 4429 makes
the plugin try the next endpoint. A connected session is never migrated.
