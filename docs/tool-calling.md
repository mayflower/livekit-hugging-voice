# Native tool calling

Version 0.3 keeps tool execution in the LiveKit worker. The selected LLM receives bounded
function schemas and may return one structured function call. The GPU service
finishes that generation without text or audio; the plugin exposes the call only
after `response.done(reason=tool_call)`. LiveKit executes the registered Python
tool, Toolset, or MCPToolset. The plugin appends its `FunctionCallOutput`, waits
for `conversation.item.created`, and only then permits the final spoken response.

```text
Gemma FunctionCall -> function_stream -> LiveKit ToolExecutor
  -> FunctionCallOutput -> acknowledged context update
  -> Gemma final text -> Qwen3-TTS audio
```

The service never receives credentials and never opens MCP or business-system
connections. It sees only names, descriptions, JSON Schemas, canonical argument
JSON, and bounded serialized results. Tool output is untrusted conversation data,
not a system instruction.

The example worker registers an `add_numbers` FunctionTool. A tool generation
itself is silent; only Gemma's answer after the acknowledged result is synthesized.

A tool may explicitly call `run_ctx.session.say("Ich prüfe das kurz.")`.
The plugin sends bounded `response.speak`; the service performs TTS directly with
the `filler_or_explicit_say` priority and no LLM inference. The service never
chooses such filler text itself, and queued final answers retain priority.

Protocol v2 uses the existing `/v1/realtime` URL with WebSocket subprotocol
`hugging-voice-livekit.v2`. It intentionally has no v1 compatibility mode.

## Capabilities and limits

| Capability | Value |
| --- | --- |
| mutable tools | reported false; initial updates accepted until first generation |
| per-response tool choice | yes |
| automatic provider reply | no; LiveKit owns the loop |
| parallel calls | no; one call per generation |
| Python tools, Toolsets, MCPToolsets | yes, through LiveKit |
| provider-native tools | no |

Tools are limited to 32; names to 64 characters; descriptions to 2,048
characters; each schema to 16 KiB and all schemas to 64 KiB. Arguments and
outputs are each limited to 16,000 characters. Only one call may be pending.
For voice latency and model accuracy, keep roughly 5–15 relevant tools active;
selecting the Toolset remains the LiveKit agent's job and there is no hidden router.

## Cancellation and reconnect

Calls carry session, turn, revision, generation, response, item, and call IDs. A
cancelled generation cannot consume a late result. Reconnect replays configuration
and confirmed append-only context and waits for every ACK before becoming ready.

## Security and troubleshooting

External output can contain prompt injection. It remains a bounded tool-role
message below fixed system instructions and is never logged. If a tool does not
run, verify it is registered on `Agent`, is not provider-native, and its strict
schema fits the limits. An ACK timeout deliberately blocks the final generation.
