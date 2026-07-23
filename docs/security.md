# Security boundary

Native tools execute only in the LiveKit worker. The GPU service receives bounded
schemas, calls, and serialized results, but no credentials or MCP connections.
Tool results remain untrusted data-role context and cannot replace fixed system
or voice instructions; argument and output bodies are not logged.

The GPU service is an internal authenticated endpoint, not an Internet-facing media
server. Every `/v1/*` HTTP request and WebSocket upgrade requires the bearer token;
health and low-cardinality Prometheus endpoints carry no conversation data. Put TLS
and network policy at the trusted cluster boundary. Never place credentials in URLs,
query strings, logs, images, ConfigMaps, or Git.

Model artifacts are fetched only by the explicit prefetch command or Kubernetes Job
and verified against an exact revision, size, and SHA-256 lock. The runtime operates
offline, mounts models and the token read-only, runs as a non-root user with no Linux
capabilities, and has a read-only root filesystem plus bounded writable scratch
mounts. There is no cloud fallback, runtime downloader, database, broker, tool
execution, browser transport, or persisted conversation state.

Admission is limited atomically to two sessions. Each connection owns its VAD,
conversation, IDs, cancellation generation, and bounded channels; expensive model
runtimes are shared. Capacity discovery is advisory and leaks no user data. Treat a
quarantined slot or unexpected llama-server exit as an incident: readiness is revoked,
new work is refused, and the pod must be drained and replaced after evidence capture.

Rotate the bearer token by creating a new Secret and restarting the pod. The process
reads the token once during startup so a half-rotated pod cannot silently accept two
credentials. Logs and reports must contain neither audio nor complete prompts or
transcripts.

For local Compose only, keep the file-backed token mode at `0640` and set
`HUGGING_VOICE_SECRET_GID` to its host group. Compose cannot remap bind-mounted secret
ownership; the supplemental group lets the fixed non-root container user read the
file without making it world-readable. Kubernetes uses its native Secret volume and
pod security context instead.
