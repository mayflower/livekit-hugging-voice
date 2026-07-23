# Codex-Implementierungsprompt: Native Tool Calls für `livekit-hugging-voice`

**Repository:** `https://github.com/mayflower/livekit-hugging-voice`  
**Erwartete Ausgangsbasis:** `main` auf Commit `4253a642a44903b1bb26291a3d82abb55f7b06f8` (`Initial local German voice agent`)  
**Zielversion:** `0.2.0`  
**Ziel:** vollständige, native und getestete Tool-Calling-Unterstützung für das eigene LiveKit-Agents-`RealtimeModel`, ohne einen zweiten Agentenstack und ohne Tool-Ausführung im GPU-Service.

---

# Auftrag

Du arbeitest im bestehenden Repository `mayflower/livekit-hugging-voice` und erweiterst die vorhandene lokale deutsche Speech-to-Speech-Strecke um **natives agentisches Tool-Calling über LiveKit Agents**.

Die Verantwortungsgrenze ist verbindlich:

```text
Gemma 4 entscheidet, ob ein Tool gebraucht wird und erzeugt name + JSON-Argumente.
LiveKit Agents führt das FunctionTool, Toolset oder MCPToolset aus.
Der GPU-Service führt niemals externe Tools, MCP-Aufrufe oder Geschäftsfunktionen aus.
Das Tool-Ergebnis wird an Gemma zurückgegeben.
Gemma erzeugt daraus die finale kurze deutsche Antwort.
Qwen3-TTS spricht ausschließlich diese finale Antwort.
```

Das Ergebnis muss den nativen LiveKit-`RealtimeModel`-Tool-Loop verwenden:

```text
GenerationCreatedEvent.function_stream
  -> LiveKit Tool Executor
  -> FunctionCallOutput
  -> RealtimeSession.update_chat_ctx(...)
  -> RealtimeSession.generate_reply(...)
  -> finale Text- und Audioantwort
```

Implementiere dies vollständig in Protokoll, Plugin, GPU-Service, Gemma-Runtime, Pipeline, Tests, Dokumentation und Benchmarks. Erzeuge keine Stubs, keine Demo-Backends im Produktionscode und keine zweite LLM- oder Tool-Orchestrierung neben LiveKit.

---

# Arbeitsweise und Ausgangsbasis

1. Lies zuerst ausschließlich die für diese Änderung relevanten lokalen Dateien:

   ```text
   AGENTS.md
   prompts.md
   README.md
   CHANGELOG.md
   docs/architecture.md
   docs/protocol.md
   docs/security.md
   docs/performance.md
   docs/benchmarks.md
   docs/upstream-baseline.md
   packages/hugging-voice-protocol/src/hugging_voice_protocol/events.py
   packages/hugging-voice-protocol/src/hugging_voice_protocol/errors.py
   packages/livekit-plugins-hugging-voice/livekit/plugins/hugging_voice/realtime.py
   services/gpu-service/src/hugging_voice_service/pipeline.py
   services/gpu-service/src/hugging_voice_service/conversation.py
   services/gpu-service/src/hugging_voice_service/sessions.py
   services/gpu-service/src/hugging_voice_service/runtimes/gemma.py
   services/gpu-service/src/hugging_voice_service/llama_process.py
   services/gpu-service/src/hugging_voice_service/telemetry.py
   services/gpu-service/src/hugging_voice_service/text_segmenter.py
   examples/minimal-livekit-agent/agent.py
   die dazugehörigen vorhandenen Tests und Protocol-Fixtures
   ```

2. Führe **keine neue allgemeine Repository- oder Upstream-Inventur** durch. `docs/upstream-baseline.md` existiert bereits. Prüfe aus dem zu `livekit-agents==1.6.6` gehörenden Source nur die konkreten Tool-Lifecycle-Stellen:

   ```text
   livekit/agents/llm/realtime.py
   livekit/agents/llm/chat_context.py
   livekit/agents/llm/tool_context.py
   livekit/agents/voice/agent_activity.py
   livekit/agents/voice/generation.py
   livekit/agents/voice/tool_executor.py
   livekit/plugins/openai/realtime/realtime_model.py
   tests/fake_realtime.py
   relevante Realtime-Tool-Tests
   ```

3. Der bisherige Ausschluss von Tool-Calling in `AGENTS.md`, `prompts.md` und den Dokumenten war eine **Version-1-Grenze**. Für diesen Auftrag ist er aufgehoben. Ändere die lokalen normativen Dokumente so, dass keine widersprüchlichen Anweisungen zurückbleiben.

4. Falls `HEAD` neuer als der erwartete Baseline-Commit ist, setze nichts zurück. Prüfe nur die für diese Änderung relevanten Unterschiede und integriere aufwärtskompatibel in den vorhandenen Stand.

5. Arbeite linear in derselben Session. Erzeuge keine neue Wave-Inventur, kein Ticket-System, keine Ledger, keine Ownership-Manifeste und keinen zusätzlichen Prozessapparat.

6. Erstelle keinen Branch, keinen Commit, keinen Push und keinen Pull Request, sofern dies nicht separat angeordnet wurde.

7. Wenn keine NVIDIA-GPU oder kein verifiziertes Modellvolume verfügbar ist, implementiere und teste die komplette CPU-/Contract-Strecke. Weise GPU-Tests ehrlich als nicht ausgeführt aus. Simuliere keine GPU-Ergebnisse und erfinde keine Latenzen.

---

# Nicht verhandelbare Architektur

Behalte die bestehende Produktarchitektur bei:

```text
Endgerät
  <-> WebRTC
LiveKit Server
  <-> LiveKit AgentSession
Python LiveKit Agent
  - FunctionTools
  - Toolsets
  - MCPToolsets
  - Berechtigungen und Credentials
  - Tool Executor
  <-> eigenes RealtimeModel-Plugin
livekit-plugins-hugging-voice
  <-> authentisierter interner WebSocket v2
Hugging Voice GPU-Service
  - Silero VAD
  - Parakeet STT
  - Gemma 4 31B über loopback-only llama-server
  - Qwen3-TTS
```

Verbindliche Grenzen:

- Der GPU-Service erhält ausschließlich Tool-Namen, Beschreibungen, JSON-Schemas, Modell-Calls und serialisierte Tool-Ergebnisse.
- Credentials, OAuth-Tokens, MCP-Verbindungen, Datenbankzugänge und Unternehmensnetzwerkzugriffe bleiben im LiveKit-Worker.
- Kein MCP-Client im GPU-Service.
- Keine eingebauten `llama.cpp`-Tools.
- Kein `llama.cpp`-MCP-Proxy.
- Kein zweites Planner-, Router- oder Tool-LLM.
- Kein Parsing von sichtbarem Modelltext, XML-Tags oder Markdown als Ersatz für strukturierte Tool Calls.
- Kein OpenAI-Realtime-Plugin als Wrapper, Basisklasse oder versteckter Proxy.
- Kein Cloud-Fallback und kein alternatives Modell.
- Gemma bleibt `google/gemma-4-31B-it` in der exakt gelockten GGUF-Variante.
- LiveKit bleibt der einzige Tool Executor.
- Zunächst maximal **ein Tool Call pro Gemma-Generation** und `parallel_tool_calls=false`.
- Sequenzielles Chaining über mehrere LiveKit-Tool-Schritte bleibt möglich; die bestehende LiveKit-Grenze `max_tool_steps` wird respektiert.
- Ein GPU-Pod lädt Parakeet, Gemma und Qwen weiterhin jeweils genau einmal und bedient höchstens zwei Sessions.

---

# Definition of Done

Der Auftrag ist erst abgeschlossen, wenn alle folgenden Eigenschaften implementiert und getestet sind:

1. Ein normaler deutscher Voice-Turn ohne Tool funktioniert unverändert als:

   ```text
   Audio -> VAD -> Parakeet -> Gemma -> Qwen-TTS -> 24-kHz-PCM16
   ```

2. Ein Tool-Turn funktioniert vollständig als:

   ```text
   deutsches Audio
   -> finales Parakeet-Transkript
   -> Gemma FunctionCall
   -> LiveKit function_stream
   -> LiveKit Tool Executor
   -> FunctionCallOutput
   -> bestätigtes Context-Update im GPU-Service
   -> zweite Gemma-Generation
   -> finale kurze deutsche Antwort
   -> Qwen-TTS
   -> 24-kHz-PCM16
   ```

3. Eine reine Tool-Generation erzeugt **keinen sichtbaren Assistant-Text und kein Audio**.

4. Die Tool-Generation endet vollständig, bevor LiveKit das Tool ausführt. Der GPU-Service wartet in der ersten Generation niemals auf das Tool-Ergebnis.

5. Der vom Modell erzeugte `FunctionCall` wird im Plugin vor der Übergabe an LiveKit in `RealtimeSession.chat_ctx` aufgenommen. LiveKit ergänzt anschließend nur das `FunctionCallOutput`.

6. `RealtimeSession.update_chat_ctx()` wartet auf ein serverseitiges ACK für das Tool-Ergebnis. Erst nach erfolgreichem ACK darf der von LiveKit ausgelöste finale `generate_reply()` wirksam werden.

7. `response.create` unterstützt per Response:

   ```text
   tools
   tool_choice = auto | required | none | named function
   ```

8. `tool_choice="none"` verhindert zuverlässig weitere Tool Calls, insbesondere nach Erreichen der LiveKit-Tool-Step-Grenze.

9. Reconnect, Barge-in, Cancellation und verspätete Tool-Ergebnisse erzeugen weder doppelte Tool-Ausführung noch stale finale Antworten noch Call-/Result-Paare in einer falschen Generation.

10. Die beiden LiveKit-Sessions sind stabil den zwei `llama.cpp`-Slots zugeordnet und nutzen `cache_prompt=true`, sodass die zweite Gemma-Generation den gemeinsamen Prefix wiederverwenden kann.

11. Der exakt gepinnte Gemma-/GGUF-/Jinja-/`llama.cpp`-Stack muss vor Readiness einen echten strukturierten Tool-Call-Selbsttest bestehen.

12. Alle bestehenden CPU-Checks bleiben grün; neue Tests decken den vollständigen Tool-Loop ab.

---

# LiveKit-Vertrag, der exakt eingehalten werden muss

## `GenerationCreatedEvent`

Erzeuge bei `response.created` weiterhin genau ein `GenerationCreatedEvent` mit zwei offenen Streams:

```python
GenerationCreatedEvent(
    message_stream=...,
    function_stream=...,
    user_initiated=...,
    response_id=...,
)
```

Ändere aber die aktuelle Logik:

- Erzeuge bei `response.created` noch **keine** `MessageGeneration`.
- Erzeuge eine `MessageGeneration` erst beim ersten echten Text- oder Audioevent.
- Bei einer reinen Tool-Generation bleibt `message_stream` leer.
- Validiere und puffere `response.output_function_call.done`, aber gib den Call noch nicht an LiveKit frei.
- Erst nach dem passenden `response.done(reason=tool_call)` wird der Call in `_chat_ctx` aufgenommen und in `function_stream` geschrieben. Dadurch kann der LiveKit-Tool-Executor erst starten, nachdem die Modellgeneration nachweislich beendet ist.
- Schließe `message_stream` und `function_stream` bei `response.done` genau einmal.
- Erlaube keine überlappenden Responses pro Realtime-Session.

## Tool Call in lokalem Chat Context

Beim Empfang eines serverseitigen Function Calls muss die Reihenfolge sein:

```text
1. FunctionCall vollständig validieren.
2. FunctionCall mit stabiler id und call_id erzeugen.
3. Auf das passende `response.done(reason=tool_call)` warten.
4. FunctionCall in `RealtimeSession._chat_ctx` einfügen.
5. Erst danach FunctionCall in `function_stream` schreiben und den Stream abschließen.
```

Der `FunctionCall` muss mindestens enthalten:

```python
FunctionCall(
    id=item_id,
    call_id=call_id,
    name=name,
    arguments=canonical_arguments_json,
)
```

Korrelationen wie `turn_id`, `turn_revision`, `generation_id` und `response_id` dürfen in einem provider-spezifischen `extra["hugging_voice"]`-Eintrag abgelegt werden, sofern dies für Reconnect und Stale-Result-Erkennung gebraucht wird. Logge keine Argumentinhalte.

## Tool-Ausführung

Führe Tools nicht selbst aus. LiveKit muss den bestehenden `perform_tool_executions`-/`ToolExecutor`-Pfad verwenden. Python-Tools, RawFunctionTools, Toolsets und MCPToolsets müssen dadurch automatisch funktionieren.

## Tool-Ergebnis

LiveKit baut nach der Ausführung aus dem aktuellen Realtime-Context einen neuen Context und ergänzt `FunctionCallOutput`. Das Plugin muss:

- den vollständigen Item-Prefix prüfen;
- nur append-only Änderungen akzeptieren;
- das zum `call_id` gehörende bekannte `FunctionCall` finden;
- einen eventuell leeren Output-Namen aus dem Call ergänzen;
- `output` und `is_error` begrenzen und serialisieren;
- das Output-Item an den GPU-Service senden;
- auf `conversation.item.created` warten;
- erst danach erfolgreich aus `update_chat_ctx()` zurückkehren.

Wird das Ergebnis abgelehnt oder läuft das ACK in einen Timeout, muss `update_chat_ctx()` einen `RealtimeError` auslösen. Ein finaler Reply auf Basis eines nicht bestätigten Tool-Ergebnisses ist verboten.

## Automatische oder explizite Fortsetzung

Setze:

```python
auto_tool_reply_generation=False
```

Nach dem Tool-Ergebnis startet LiveKit selbst den nächsten Reply. Der GPU-Service darf nicht automatisch auf den Eingang des Tool-Ergebnisses reagieren und parallel einen Reply starten.

---

# Realtime-Capabilities

Melde zu Beginn wahrheitsgemäß diese Capabilities:

```python
RealtimeCapabilities(
    message_truncation=False,
    turn_detection=True,
    user_transcription=True,
    auto_tool_reply_generation=False,
    audio_output=True,
    manual_function_calls=False,
    mutable_chat_context=False,
    mutable_instructions=True,
    mutable_tools=False,
    per_response_tool_choice=True,
    supports_say=False,
)
```

Bedeutung der konservativen Flags:

- `manual_function_calls=False`: Behaupte nicht, dass beliebige bereits vorhandene dangling Calls aus einem fremden Context fortgesetzt werden können.
- `mutable_chat_context=False`: Die gezielte append-only Unterstützung für Textitems und Tool-Ergebnisse ist keine beliebige Mid-Session-Context-Ersetzung.
- `mutable_tools=False`: Tools werden beim Session-Start materialisiert und bleiben danach stabil. Ein Agent-Handoff mit anderem Toolset soll zu einer neuen Realtime-Session führen statt die alte Session falsch wiederzuverwenden.
- `per_response_tool_choice=True`: Dies ist nur zulässig, weil `response.create` Tools und Tool Choice vollständig unterstützt.

`update_tools()` muss die initialen Tools trotzdem akzeptieren. Da LiveKit die Realtime-Session bereits vor `_update_session()` erzeugt, darf „initial“ nicht mit „WebSocket noch nicht verbunden“ verwechselt werden. Akzeptiere Tool-Konfiguration bis zur ersten gestarteten Modellgeneration und friere sie danach ein. Spätere Änderungen müssen klar mit `RealtimeError` abgewiesen werden.

Setze einen Capability-Flag erst anders, wenn seine vollständige LiveKit-Semantik implementiert und mit einem gezielten Test bewiesen ist.

---

# Koordinierter Protokoll-v2-Cutover

Ersetze das interne Protokoll v1 koordiniert durch v2. Halte keinen dauerhaften Legacy-Doppelpfad vor.

```text
WebSocket path:        /v1/realtime
WebSocket subprotocol: hugging-voice-livekit.v2
protocol_version:      2
```

Der URL-Pfad kann aus Kompatibilitätsgründen `/v1/realtime` bleiben; die echte Drahtversion wird durch Subprotocol und Eventfeld bestimmt. Aktualisiere Plugin, Service, Fixtures, Tests und Dokumentation gemeinsam. Ein v1-Client muss eindeutig abgewiesen werden.

## Versionsanhebung

Hebe gemeinsam auf `0.2.0` an:

```text
hugging-voice-protocol
livekit-plugins-hugging-voice
hugging-voice-gpu-service
minimal-livekit-agent, sofern versioniert
interne Workspace-Abhängigkeiten
uv.lock
CHANGELOG.md
```

## Protokollgrenzen

Definiere zentrale Konstanten und teste sie:

```text
MAX_TOOLS = 32
MAX_TOOL_NAME_CHARS = 64
MAX_TOOL_DESCRIPTION_CHARS = 2_048
MAX_TOOL_SCHEMA_BYTES = 16 KiB pro Tool
MAX_ALL_TOOL_SCHEMAS_BYTES = 64 KiB
MAX_TOOL_ARGUMENTS_CHARS = 16_000
MAX_TOOL_OUTPUT_CHARS = 16_000
MAX_PENDING_TOOL_CALLS = 1 pro Session
```

Behalte die bestehende WebSocket-Nachrichtengrenze im Blick. Tool-Schemas und Ergebnisse dürfen nicht unkontrolliert die 128-KiB-Grenze überschreiten.

Toolnamen müssen einem klar dokumentierten, interoperablen Muster entsprechen, beispielsweise:

```regex
^[A-Za-z_][A-Za-z0-9_-]{0,63}$
```

JSON muss kanonisch serialisiert werden:

```python
json.dumps(
    value,
    ensure_ascii=False,
    sort_keys=True,
    separators=(",", ":"),
    allow_nan=False,
)
```

## Tool-Definition

Verwende als Drahtformat die von LiveKit erzeugte OpenAI-kompatible Function-Tool-Form:

```json
{
  "type": "function",
  "function": {
    "name": "lookup_order",
    "description": "...",
    "parameters": {
      "type": "object",
      "properties": {},
      "required": [],
      "additionalProperties": false
    }
  }
}
```

Erzeuge diese Form ausschließlich über:

```python
ToolContext(tools).parse_function_tools("openai", strict=True)
```

Sortiere die resultierenden Tools für einen stabilen Prompt-Cache deterministisch nach Function-Name. Verändere die JSON-Schemas nicht semantisch. Weise `ProviderTool`s explizit ab; sie sind keine vom LiveKit-Tool-Executor ausführbaren FunctionTools.

## Tool Choice

Modelliere strikt:

```text
"auto"
"required"
"none"
{"type":"function","function":{"name":"..."}}
```

Ein named choice muss auf ein vorhandenes effektives Tool verweisen. `required` mit leerer Toolliste ist ungültig.

## Conversation-Items

Ersetze das bisherige reine Textitem durch eine discriminated union:

### Message

```json
{
  "type": "message",
  "id": "item_...",
  "role": "user",
  "content": "..."
}
```

Rollen bleiben für normale Nachrichten:

```text
user
assistant
```

### Function Call

```json
{
  "type": "function_call",
  "id": "item_...",
  "call_id": "call_...",
  "name": "lookup_order",
  "arguments": "{\"order_id\":\"123\"}",
  "turn_id": "turn_...",
  "turn_revision": 1,
  "generation_id": "gen_...",
  "response_id": "resp_..."
}
```

### Function Call Output

```json
{
  "type": "function_call_output",
  "id": "item_...",
  "call_id": "call_...",
  "name": "lookup_order",
  "output": "...",
  "is_error": false,
  "turn_id": "turn_...",
  "turn_revision": 1,
  "generation_id": "gen_...",
  "response_id": "resp_..."
}
```

Das Plugin ergänzt die Korrelationsfelder aus dem zuvor empfangenen Call. Ein Output ohne bekannten Call oder mit widersprüchlicher Korrelation ist ein strukturierter Session-Konflikt.

## Client-Events

Erweitere mindestens:

### `session.update`

```text
session.instructions
session.tools
session.tool_choice
bestehende Audio-, VAD-, Sprach- und Voice-Felder
```

### `conversation.item.create`

Akzeptiert die drei Itemtypen `message`, `function_call` und `function_call_output`.

### `response.create`

```text
instructions: optional
tools: optional per-response override
tool_choice: optional per-response override
```

Unterscheide sauber:

- Feld nicht vorhanden: Session-Default verwenden.
- `tools=[]`: für diese Response keine Tools.
- `tool_choice="none"`: Tool Calls ausdrücklich verbieten.

### `response.cancel`

Bleibt generationskorreliert.

## Server-Events

Füge mindestens hinzu:

### `session.updated`

```json
{
  "type": "session.updated",
  "source_event_id": "evt_...",
  "session_id": "session_...",
  "protocol_version": 2
}
```

### `conversation.item.created`

```json
{
  "type": "conversation.item.created",
  "source_event_id": "evt_...",
  "session_id": "session_...",
  "item_id": "item_...",
  "protocol_version": 2
}
```

### `response.output_function_call.done`

```json
{
  "type": "response.output_function_call.done",
  "event_id": "evt_...",
  "protocol_version": 2,
  "session_id": "session_...",
  "turn_id": "turn_...",
  "turn_revision": 1,
  "generation_id": "gen_...",
  "response_id": "resp_...",
  "item_id": "item_...",
  "call_id": "call_...",
  "name": "lookup_order",
  "arguments": "{\"order_id\":\"123\"}"
}
```

Es ist für diese Version kein öffentliches Argument-Delta-Event nötig. Sammle die fragmentierten llama.cpp-SSE-Deltas intern und sende erst den vollständigen validierten Call.

### `response.done`

Erweitere `reason` um:

```text
tool_call
```

Eine Response mit `status=completed` und `reason=tool_call` darf ohne Text und ohne Audio enden.

## ACK-Semantik

- `session.update` wird mit `session.updated` bestätigt.
- Jedes `conversation.item.create` wird mit `conversation.item.created` bestätigt.
- ACKs korrelieren über `source_event_id`.
- Das Plugin hält eine begrenzte Map aus ausstehenden ACK-Futures.
- Disconnect, Shutdown und terminale Fehler schlagen alle ausstehenden ACKs genau einmal fehl.
- Unbekannte, doppelte oder verspätete ACKs werden beobachtbar behandelt und dürfen keine fremde Future auflösen.
- Bootstrap und Reconnect gelten erst als betriebsbereit, nachdem Session-Konfiguration und Context-Replay bestätigt wurden.

---

# Änderungen im LiveKit-Plugin

Arbeite primär in:

```text
packages/livekit-plugins-hugging-voice/livekit/plugins/hugging_voice/realtime.py
```

Teile die Datei nur dann in kleine konkrete Module auf, wenn dies die Zustandslogik tatsächlich klarer macht. Erzeuge keine generische Provider-Abstraktion.

## Tool-Serialisierung

Implementiere einen konkreten Helper für:

- `ToolContext`-Flattening;
- Ablehnung von ProviderTools;
- OpenAI-strikte Tool-Schemas;
- deterministische Sortierung;
- Größenprüfung;
- kanonische JSON-Repräsentation;
- validierte Tool-Choice-Normalisierung.

Cache die serialisierte Toolliste innerhalb der Session, solange das Toolset unverändert ist.

## Initiale Tools und Freeze

`update_tools()` muss:

1. initiale FunctionTools akzeptieren;
2. die Schemas materialisieren;
3. `_tools` korrekt als `ToolContext` setzen;
4. bei verbundener Session ein bestätigtes `session.update` senden;
5. nach Beginn der ersten Modellgeneration jede semantische Tooländerung mit `RealtimeError` ablehnen;
6. eine identische Wiederholung idempotent akzeptieren.

## Tool Choice

`update_options()` bleibt synchron. Es darf den Session-Default aktualisieren und ein begrenztes `session.update` einreihen. Fehler werden als Realtime-Model-Fehler emittiert. Die per-response Tool Choice aus `generate_reply()` hat Vorrang vor dem Session-Default.

## Per-Response-Tools

`generate_reply()` muss die von LiveKit übergebenen optionalen `tools` und `tool_choice` in `ResponseCreateEvent` serialisieren. Lehne keine Tools mehr pauschal ab.

## Response-Zustand

Erweitere `_ResponseState` mindestens um:

```text
output_kind: undecided | message | function_call
message_channel
function_channel
message_generation_created
function_call
first_output_at
```

Verbindliche Zustandsregeln:

- Erstes sichtbares Text- oder Audioevent setzt `output_kind=message`.
- Erstes Function-Call-Event setzt `output_kind=function_call`.
- Ein späterer Wechsel des Outputtyps ist ein fataler Modell-/Protokollfehler.
- Pro Response maximal ein Function Call.
- Ein Tool Call wird bis `response.done` nur gepuffert. Vorher darf er nicht in `function_stream` erscheinen, weil LiveKit Tools unmittelbar beim Lesen des Streams ausführen kann.
- Bei `response.done(reason=tool_call)` wird der Call zuerst in `_chat_ctx` aufgenommen, dann in `function_stream` geschrieben und anschließend werden alle Generation-Streams geschlossen.
- `response.done(reason=tool_call)` schließt Streams ohne Missing-Audio-Fehler.
- Eine normale erfolgreich abgeschlossene Message-Response ohne Audio bleibt weiterhin ein Fehler.

## Chat-Context-Diff

Ersetze die bisherige reine Nachrichtenvalidierung durch eine vollständige Itemvalidierung für:

```text
ChatMessage
FunctionCall
FunctionCallOutput
```

Unterstütze:

- initialen Text-Context;
- append-only Textnachrichten;
- den vom Plugin selbst erzeugten FunctionCall;
- das von LiveKit angehängte FunctionCallOutput;
- Reconnect-Replay abgeschlossener Call-/Output-Paare.

Lehne ab:

- Entfernen oder Ersetzen vorhandener Items;
- Umordnung;
- Output vor Call;
- doppelten Output für dieselbe `call_id`;
- Callnamen-Konflikt;
- dangling externe Calls, die nicht aus dieser Session stammen;
- unbekannte Itemtypen;
- System-/Developer-Nachrichten auf dem normalen Conversation-Drahtpfad, sofern sie nicht bereits über `instructions` abgebildet werden.

## Reconnect

Beim Reconnect:

1. authentisieren und `session.created` prüfen;
2. bestätigtes `session.update` mit identischen Tools und Tool Choice senden;
3. Context-Items in Originalreihenfolge replayen;
4. jedes Replay-Item bestätigen lassen;
5. erst dann `_connected` setzen und `session_reconnected` emittieren.

Ein Disconnect darf weder eine aktive Response noch ausstehende ACKs still weiterleben lassen. Ein während einer Tool-Ausführung unterbrochener Transport darf keine doppelte Tool-Ausführung verursachen. Nutze den bereits lokal gespeicherten `FunctionCall`; emittiere ihn nach Reconnect nicht erneut in `function_stream`.

## Cancellation und Barge-in

Lies den echten LiveKit-1.6.6-Tool-Executor-/Realtime-Cancellation-Pfad und passe dich daran an. Erfinde keinen zweiten Executor.

Garantien:

- Eine neue User-Speech darf eine alte finale Antwort nicht mehr starten.
- Ein cancellable Tool wird durch LiveKit abgebrochen.
- Ein nicht cancellable Tool darf seine externe Wirkung beenden, aber sein verspätetes Ergebnis darf keinen stale Voice-Reply für die alte Generation auslösen.
- `call_id`, `turn_id`, `turn_revision` und `generation_id` verhindern die Übernahme eines Ergebnisses in einen neueren Turn.
- Ein abgebrochener oder stale Call wird beim Reconnect nicht als aktiver Modell-Call wiederbelebt.
- Keine breite Exception-Unterdrückung.

Implementiere die kleinste konkrete Zustandslogik, die diese Garantien erfüllt. Kein allgemeines Workflow-System.

---

# Änderungen im GPU-Service

## Session-State

Erweitere `SessionState` konkret um:

```text
stabile llama_slot_id = state.slot.index
session tools
session tool_choice
configuration/tool freeze state
höchstens einen pending FunctionCall
Tool-Call-Korrelation
Zeitpunkte für Tool-Metriken
```

Tooldefinitionen und Pending Calls bleiben flüchtig und strikt pro Session.

## Conversation-Modell

Ersetze die reine deque aus `user`-/`assistant`-Text durch eine kleine typisierte Conversation-Struktur.

Speichere atomare Gruppen:

```text
MessageGroup(message)
ToolExchangeGroup(function_call, function_call_output)
```

Ein vom Modell erzeugter Call wird zunächst nur als `pending_call` gehalten. Erst beim passenden `FunctionCallOutput` wird Call + Output atomar in die dauerhafte Session-Conversation übernommen.

Trimme niemals nur eine Hälfte eines Tool-Exchanges. Die bestehenden Grenzen von maximal 30 Conversation-Items und 48.000 Zeichen dürfen beibehalten oder sinnvoll auf Gruppen abgebildet werden. Dokumentiere die exakte Zählweise.

Ein Tool-Ergebnis wird für Gemma in ein deterministisches Tool-Message-Format überführt. Bewahre `is_error` sichtbar für das Modell, beispielsweise mit einer kleinen JSON-Hülle:

```json
{"is_error":false,"output":"..."}
```

Logge den Inhalt nicht.

## Session-Update

`session.update` muss:

- Tools und Tool Choice validieren;
- Tool-Definitionen kanonisch speichern;
- identische Updates idempotent akzeptieren;
- Tooländerungen nach Freeze ablehnen;
- Instruction-Updates weiterhin erlauben, sofern kein aktiver Konflikt besteht;
- nach vollständiger Anwendung `session.updated` senden.

## Conversation-ACK

`conversation.item.create` muss erst nach vollständiger Validierung und Zustandsübernahme mit `conversation.item.created` bestätigt werden.

Für `function_call_output` gilt:

- Call muss pending und bekannt sein;
- `call_id`, Name und Korrelation müssen passen;
- Output darf nur einmal übernommen werden;
- Call + Output werden atomar committed;
- Pending Call wird gelöscht;
- es wird **kein** automatischer Reply gestartet;
- der folgende `response.create` von LiveKit startet die Fortsetzung.

## Response-Lifecycle

Eine Response ist genau eine der beiden Formen:

```text
Message response:
  response.created
  response.output_text.*
  response.output_audio.*
  response.done(reason=completed)

Tool response:
  response.created
  response.output_function_call.done
  response.done(reason=tool_call)
```

Mische die Formen nicht.

Die erste Generation endet nach dem Tool Call. Halte keinen Response-Task offen, während LiveKit das Tool ausführt.

---

# Gemma- und llama.cpp-Integration

Arbeite primär in:

```text
services/gpu-service/src/hugging_voice_service/runtimes/gemma.py
services/gpu-service/src/hugging_voice_service/llama_process.py
```

## Request-Payload

Erweitere `/v1/chat/completions` um:

```json
{
  "tools": [],
  "tool_choice": "auto",
  "parallel_tool_calls": false,
  "stream": true,
  "stream_options": {"include_usage": true},
  "cache_prompt": true,
  "id_slot": 0,
  "chat_template_kwargs": {"enable_thinking": false}
}
```

Felder werden nur gesendet, wenn sie für den konkreten Request gelten. Prüfe die exakten Feldnamen gegen den gepinnten `llama.cpp`-Commit, nicht gegen eine bewegliche spätere Version.

Verwende immer:

```text
id_slot = SessionState.slot.index
cache_prompt = true
```

Damit bleiben Session A und B auf ihren jeweiligen llama.cpp-Sequenz-Slots. Die Tool-Result-Fortsetzung muss denselben Slot verwenden.

Ändere den gepinnten `llama.cpp`-Commit nur, wenn ein reproduzierbarer Test beweist, dass genau dieser Commit den benötigten Gemma-4-Tool-Call nicht korrekt unterstützt. Eine Änderung muss auf einen exakten Commit erfolgen, alle Locks und Labels aktualisieren und im Abschlussbericht mit dem reproduzierbaren Blocker begründet werden. Kein `latest`, kein Branch-Pin.

## Sampling

Verwende ohne öffentlichen Flag-Dschungel zwei feste interne Profile:

### Tool-fähige Entscheidung

Wenn effektive Tools vorhanden sind und Tool Choice nicht `none` ist:

```text
temperature: 0.2
max_tokens: 128
parallel_tool_calls: false
enable_thinking: false
```

### Finale Voice-Antwort

Wenn Tool Choice `none` ist oder keine Tools verfügbar sind:

```text
temperature: 0.7
max_tokens: 256
enable_thinking: false
```

Diese Profile sind Implementierungsdefaults, keine öffentliche Provider-Registry.

## SSE-Tool-Call-Aggregator

Ersetze den aktuellen Fehler bei `delta.tool_calls` durch einen konkreten Aggregator:

- sammle fragmentierte Einträge nach `index`;
- sammle `id`, `function.name` und `function.arguments`;
- erlaube maximal einen Index;
- normalisiere eine fehlende oder ungültige Modell-Call-ID zu einer eigenen stabilen `call_...`-ID;
- parse Argumente erst nach Abschluss;
- Argumente müssen syntaktisch gültiges JSON und ein Objekt sein;
- serialisiere sie anschließend kanonisch;
- Toolname muss in der effektiven Toolliste enthalten sein;
- named/none/required Choice muss eingehalten werden;
- ein zweiter Call ist ein Modellfehler;
- sichtbarer Text und Tool Call in derselben Response ist ein Modellfehler;
- Reasoning-Felder bleiben verborgen und dürfen weder Toolargumente noch TTS erreichen;
- Usage muss auch für eine Tool-Generation erhalten bleiben.

Definiere konkrete Runtime-Events, beispielsweise:

```python
TextDelta
ToolCall
TextUsage
```

Keine generische Eventbus-Abstraktion.

## Gemma-Conversation-Format

Erzeuge für `llama-server` OpenAI-kompatible Messages:

```text
system
user
assistant text
assistant mit tool_calls
 tool mit tool_call_id und Ergebnis
```

Stelle sicher, dass Call-ID, Name und Argumente in der Assistant-Tool-Call-Nachricht exakt zu der folgenden Tool-Result-Nachricht passen.

## Readiness-Selbsttest

Erweitere die echte Runtime-Readiness um einen zweistufigen Tool-Test mit einem lokalen synthetischen Schema, ohne einen allgemeinen Tool Executor im Service einzubauen:

1. Definiere ein internes Readiness-Tool `add_numbers(a, b)`.
2. Fordere Gemma auf Deutsch auf, `17 + 25` zwingend mit diesem Tool zu berechnen.
3. Erwarte einen strukturierten Call mit dem richtigen Namen und den richtigen Argumenten.
4. Füge intern das fest bekannte Testergebnis `42` als Tool-Result in die Probe-History ein.
5. Starte mit `tool_choice=none` eine zweite Generation.
6. Erwarte sichtbaren nichtleeren Antworttext, der das Ergebnis korrekt wiedergibt.

Dies ist ausschließlich ein Startup-Selbsttest der Modell-/Template-/Parser-Kette. Es ist kein Produktions-Toolpfad. Wenn die Probe scheitert, bleibt Readiness rot.

---

# Pipeline-State-Machine

Ändere die Pipeline von einem impliziten Text-only-Ablauf zu einer expliziten kleinen State Machine:

```text
UNDECIDED
  -> MESSAGE beim ersten sichtbaren Textdelta
  -> TOOL_CALL beim ersten ToolCall

MESSAGE
  -> Text streamen
  -> segmentieren
  -> TTS starten
  -> ToolCall danach = Fehler

TOOL_CALL
  -> niemals Text-/Audioevents senden
  -> pending_call speichern
  -> FunctionCall-Event senden
  -> Response mit reason=tool_call finalisieren
  -> sichtbarer Text danach = Fehler
```

Verbindlich:

- Starte TTS nicht für eine Tool-Generation.
- Sende bei einer Tool-Generation keine leeren Text- oder Audio-Done-Events als künstliche Message.
- Nutze den vorhandenen generationstag-basierten Cancellation-Mechanismus.
- Eine Tool-Generation zählt als abgeschlossene Response, nicht als Fehler.
- `llm_ttft_seconds` darf für Tool Calls nicht fälschlich „first visible text“ messen. Ergänze eine separate Tool-Decision-Metrik.
- Die finale Antwort nach dem Tool-Ergebnis durchläuft wieder den normalen Text-/TTS-Pfad.

---

# Antwortgeschwindigkeit und verpflichtende Optimierungen

Tool Calling erzeugt zwingend zwei Gemma-Generationen. Vermeide daher unnötigen dritten Prefill, falsche Slot-Zuordnung und spätes TTS.

## Pflichtoptimierungen

1. **Feste llama.cpp-Slot-Affinität**

   ```text
   SessionState.slot.index -> id_slot
   ```

2. **Prompt-Cache**

   ```text
   cache_prompt=true
   ```

3. **Stabile Tool-Repräsentation**

   - kanonisches JSON;
   - deterministische Reihenfolge;
   - identischer Systemprompt;
   - identische Tool-Schemas zwischen erstem Call und Result-Fortsetzung.

4. **Kein TTS für Tool Calls**

   Die erste Generation darf keinerlei TTS-Arbeit auslösen.

5. **Frühe Satzsegmentierung normaler Antworten**

   Verbessere den bestehenden Segmenter so, dass ein vollständiges Satzende nicht allein deshalb bis zum Ende der gesamten LLM-Generation wartet, weil nach dem Satzzeichen noch kein Whitespace-Delta eingetroffen ist. Bewahre den Schutz für deutsche Abkürzungen, Initialen und Dezimalzahlen. Keine naive `split('.')`-Logik.

6. **Keine Toolkatalog-Explosion**

   Der Service unterstützt maximal 32 Tools; dokumentiere für Voice-Agenten einen empfohlenen Bereich von 5 bis 15 aktiven Tools. Implementiere in diesem Repository keinen semantischen Tool-Router. Die Auswahl des passenden Toolsets bleibt Aufgabe des aufrufenden LiveKit-Agenten.

7. **Keine ungeprüfte TTS-Änderung**

   Ändere den Qwen-`chunk_size` nicht allein aufgrund einer Vermutung. Wenn eine reale GPU verfügbar ist, erweitere den Benchmark um 6, 8 und 12 und dokumentiere TTFA/RTF. Ohne Messung bleibt der bestehende Wert erhalten.

## Erwartungs- und Rollout-Gates

Diese Werte sind Zielgates und dürfen nicht als gemessen behauptet werden:

| Messgröße | eine Session | zwei Sessions |
|---|---:|---:|
| Speech-stop bis Tool Call p50 | <= 1,5 s | <= 2,5 s |
| schnelles lokales Tool bis erstes finales Audio p50 | <= 3,5 s | <= 5,0 s |
| schnelles lokales Tool bis erstes finales Audio p95 | <= 5,5 s | <= 8,0 s |
| versehentliches Audio in Tool-Generation | 0 | 0 |
| unbekannter oder falscher Toolname in Contract-Tests | 0 | 0 |
| stale finaler Reply nach Barge-in | 0 | 0 |
| Modell-/Runtime-Load-Count | genau 1 | genau 1 |

Wenn reale Hardware vorhanden ist, miss diese Werte. Wenn nicht, dokumentiere die offenen GPU-Gates ohne erfundene Zahlen.

---

# Telemetrie

Erweitere `ServiceTelemetry` mit low-cardinality Metriken. Verwende keine Session-ID, Call-ID, Benutzer-ID oder freien Toolnamen als Prometheus-Label.

Mindestens:

```text
hugging_voice_tool_call_generations_total
hugging_voice_tool_call_parse_failures_total
hugging_voice_tool_call_rejections_total
hugging_voice_tool_decision_seconds
hugging_voice_tool_result_wait_seconds
hugging_voice_tool_result_to_first_text_seconds
hugging_voice_tool_result_to_first_audio_seconds
hugging_voice_tool_schema_bytes
```

Definitionen:

- `tool_decision_seconds`: Start der Gemma-Generation bis vollständiger validierter FunctionCall.
- `tool_result_wait_seconds`: Ausgabe des FunctionCall-Events bis bestätigter Eingang des Resultats.
- `tool_result_to_first_text_seconds`: Resultat-ACK bis erstes sichtbares finales Textdelta.
- `tool_result_to_first_audio_seconds`: Resultat-ACK bis erstes finales Audioframe.

Logge strukturiert:

```text
session lifecycle state
turn/generation/response IDs
call ID
Toolname
Argumentgröße
Resultatgröße
is_error
Dauern
```

Logge niemals:

```text
Toolargumente
Toolresultate
Credentials
Audioinhalt
Conversation-Inhalt
```

---

# Fehlersemantik

Erweitere die vorhandenen strukturierten Fehlercodes nur konkret. Keine allgemeine Fehlerplattform.

Unterscheide mindestens:

```text
invalid_tool_configuration
unsupported_tool_type
tool_schema_too_large
invalid_tool_choice
unknown_tool_name
malformed_tool_arguments
multiple_tool_calls_not_supported
mixed_message_and_tool_output
tool_call_state_conflict
unknown_tool_call_output
duplicate_tool_call_output
stale_tool_call_output
conversation_ack_timeout
session_update_ack_timeout
model_tool_call_failure
```

Ordne Fehler sinnvoll den bestehenden Close-/Error-Kategorien zu:

- Konfigurations- und Schemafehler: nicht retryable, Protokoll/Configuration.
- Session-/Call-State-Konflikte: nicht still ignorieren; strukturierter Konflikt.
- Transportverlust: nach bestehender Retry-Policy, aber ohne doppelte Tool-Ausführung.
- Modell-Parserfehler: Response failed, Session nur dann terminal, wenn Zustandskonsistenz nicht mehr beweisbar ist.

Keine breiten `except Exception: pass`-Pfade.

---

# Tests

Erweitere die Tests als vollständige Implementierung, nicht nur als Happy-Path-Smoke.

## Protocol-Paket

Erzeuge kanonische v2-Fixtures für alle Client- und Serverevents.

Teste:

- Roundtrip jedes Events;
- `extra="forbid"`;
- v1 wird abgewiesen;
- Tool-Schema-Bounds;
- Gesamtgröße aller Tools;
- Toolname-Pattern;
- Tool Choice-Union;
- named choice auf unbekanntes Tool;
- Argument- und Outputgrenzen;
- FunctionCall-/Output-Korrelation;
- ACK-Events;
- `response.done(reason=tool_call)`.

## Plugin-Unit- und Contract-Tests

Teste mindestens:

- initiale Tools werden mit `ToolContext.parse_function_tools("openai", strict=True)` materialisiert;
- FunctionTools und RawFunctionTools funktionieren;
- Toolsets werden geflattet;
- ProviderTools werden abgewiesen;
- deterministische Toolreihenfolge;
- identisches Toolset ist idempotent;
- Tooländerung nach Freeze wird abgewiesen;
- `auto`, `required`, `none` und named choice;
- per-response Tooloverride;
- `response.created` erzeugt noch keine Message;
- Text erzeugt lazy genau eine MessageGeneration;
- Tool Call erzeugt keine MessageGeneration;
- FunctionCall liegt vor Stream-Emission in `_chat_ctx`;
- FunctionCallOutput wird bestätigt;
- `response.create` wird erst nach Resultat-ACK abgesendet;
- reine Tool-Response löst keinen Missing-Audio-Fehler aus;
- normale completed Message ohne Audio bleibt ein Fehler;
- unbekannter/doppelter Call;
- ACK-Timeout;
- Disconnect schlägt Pending-ACKs fehl;
- Reconnect replayt Tools und Context in korrekter Reihenfolge;
- Call wird nach Reconnect nicht doppelt an LiveKit emittiert;
- Outbound-Queue und Function-Channel bleiben bounded;
- Cancellation finalisiert genau einmal.

## Gemma-Runtime-Tests

Mit explizit injizierten Testantworten ausschließlich unter `tests/`:

- Tool-Call in einem SSE-Delta;
- fragmentierter Name;
- fragmentierte JSON-Argumente;
- Usage vor/nach Call;
- fehlende Call-ID wird stabil normalisiert;
- malformed JSON;
- Argumente sind kein Objekt;
- unbekannter Toolname;
- zwei Tool Calls;
- Text vor Tool Call;
- Tool Call vor Text;
- Reasoning vor Tool Call bleibt unsichtbar;
- `tool_choice=none` und trotzdem Call;
- named choice mit falschem Call;
- Cancellation schließt den HTTP-Stream;
- `id_slot` und `cache_prompt` sind im Payload;
- Tools sind kanonisch und stabil.

## Conversation- und Pipeline-Tests

Teste:

- MessageGroup-Trim;
- ToolExchangeGroup wird atomar getrimmt;
- pending Call wird nicht vor Ergebnis committed;
- passendes Ergebnis committed Call + Output atomar;
- falsches Resultat wird abgewiesen;
- doppeltes Resultat wird abgewiesen;
- Tool-Generation sendet genau `response.created`, FunctionCall und `response.done`;
- Tool-Generation sendet kein Text- oder Audioevent;
- finale Generation nach Tool Result erzeugt Text und Audio;
- Toolfehler wird als `is_error=true` an Gemma sichtbar;
- sequenzielles Chaining über zwei Calls;
- `tool_choice=none` beendet das Chaining;
- Barge-in während Tool-Ausführung erzeugt keinen stale Reply;
- zwei Sessions verwenden unterschiedliche `id_slot`s;
- zwei Sessions vermischen weder Calls noch Results.

## LiveKit-AgentSession-Integrationstest

Erzeuge einen echten CPU-Contract-Test mit dem vorhandenen lokalen Contract-Server und einem realen LiveKit `AgentSession`-Tool:

```python
@function_tool
async def add_numbers(a: int, b: int) -> str:
    return str(a + b)
```

Beweise im Test:

```text
server emittiert FunctionCall
LiveKit führt add_numbers aus
Plugin erhält FunctionCallOutput
GPU-Contract-Server bestätigt das Resultat
LiveKit startet den finalen Reply
finaler Text- und Audiostream wird verarbeitet
function_tools_executed wird emittiert
```

Der Test darf keine private Ersatz-Ausführung des Tools im Plugin oder Service verwenden.

## Echter GPU-E2E-Test

Erweitere die opt-in GPU-Suite um einen echten Tool-Turn:

- echtes deutsches Audio oder deterministisch erzeugter Text-Turn über die native Realtime-Session;
- Gemma erzeugt strukturierten Tool Call;
- LiveKit führt `add_numbers` aus;
- Gemma verarbeitet `42`;
- Qwen liefert 24-kHz-Audio;
- alle drei Runtime-Load-Counter bleiben genau eins;
- kein Audio wird vor dem Tool-Ergebnis ausgegeben;
- Rohmetriken und Provenance werden gespeichert;
- ohne GPU sauber skipped, niemals simuliert.

---

# Benchmarks

Erweitere die vorhandene Benchmarkstrecke statt eine zweite zu bauen.

Rohdaten pro Tool-Turn:

```text
speech_started_at
speech_stopped_at
final_transcript_at
llm_tool_request_started_at
tool_call_emitted_at
tool_execution_started_at
tool_execution_finished_at
tool_result_ack_at
final_llm_first_text_at
final_tts_first_audio_at
response_done_at
call/result sizes
session slot
errors/cancellations
```

Erweitere `benchmarks/summarize.py` um:

```text
speech-stop -> tool-call p50/p95/p99
tool duration p50/p95/p99
tool-result-ack -> final first text p50/p95/p99
tool-result-ack -> final first audio p50/p95/p99
speech-stop -> final first audio p50/p95/p99
one-session vs two-session
Tool-Call-Fehlerquote
stale/cancelled counts
```

Bewahre die bestehenden Provenance-Anforderungen:

```text
Git commit
image digest
GPU
Treiber
CUDA
Modellrevisionen
Quantisierung
llama.cpp commit
Audio-Hashes
Dauer
```

Keine Vergleichsaussage ohne vollständige Provenance.

---

# Beispiel-Agent und Dokumentation

## Minimaler Agent

Erweitere `examples/minimal-livekit-agent/agent.py` um ein echtes LiveKit-FunctionTool wie `add_numbers`. Der Agent soll angewiesen werden, für Additionen dieses Tool zu verwenden und danach kurz auf Deutsch zu antworten.

Das Beispiel muss zeigen:

```text
Agent(tools=[add_numbers], ...)
AgentSession(llm=hugging_voice.RealtimeModel(...))
```

Keine Toolausführung im GPU-Service und kein zusätzlicher LLM.

## Dokumente

Aktualisiere mindestens:

```text
README.md
CHANGELOG.md
AGENTS.md
prompts.md
docs/architecture.md
docs/protocol.md
docs/security.md
docs/performance.md
docs/benchmarks.md
packages/livekit-plugins-hugging-voice/README.md
examples/minimal-livekit-agent/README.md
```

Füge `docs/tool-calling.md` hinzu mit:

- Verantwortungsgrenze;
- LiveKit-Loop;
- Capability-Tabelle;
- Protokollsequenz;
- Python-Tool-/Toolset-/MCPToolset-Kompatibilität;
- Cancellation und Reconnect;
- Grenzen und Performancehinweise;
- Security;
- Troubleshooting.

Ändere `prompts.md` nicht komplett historisch um. Setze am Anfang einen unübersehbaren Hinweis:

```text
Dieses Dokument ist das historische Bootstrap-Prompt-Pack für Version 0.1.
Die damaligen No-Tool-Calling-Grenzen sind für Version 0.2 durch AGENTS.md und
docs/tool-calling.md ersetzt. Andere Architekturgrenzen bleiben gültig.
```

Aktualisiere `AGENTS.md` normativ auf Version 0.2. Tool Calling ist erlaubt, aber ausschließlich mit LiveKit als Executor und unter den in diesem Auftrag festgelegten Grenzen.

---

# Sicherheitsanforderungen

- GPU-Service besitzt keine Tool-Credentials.
- Bearer-Token des internen WebSockets bleibt Pflicht.
- Toolargumente und Toolresultate werden nicht geloggt.
- JSON-Schemas, Argumente und Outputs sind strikt begrenzt.
- Unbekannte Felder bleiben verboten.
- Toolname wird gegen die angebotene Liste geprüft.
- Das Modell kann kein nicht angebotenes Tool ausführen.
- Named choice wird serverseitig geprüft.
- Kein Shell-Tool, Dateisystem-Tool oder Netzwerktool im GPU-Service.
- Keine Runtime-Downloads.
- Keine Secrets in Testfixtures.
- Toolresultate werden ausschließlich als Daten in Gemmas Kontext aufgenommen, nie als neue Systeminstruction.
- Toolresultate dürfen bestehende System- und Voice-Regeln nicht überschreiben.
- Dokumentiere Prompt-Injection-Risiken aus externen Toolresultaten und halte die feste Systeminstruction vor allen Tooldaten.

---

# Ausdrücklich nicht implementieren

- keine parallelen Tool Calls;
- keine Background-Tool-Plattform;
- kein eigener Tool-Scheduler neben LiveKit;
- kein MCP im GPU-Service;
- keine generische Provider-Registry;
- kein Browser-UI;
- kein zusätzlicher WebRTC-Hop;
- kein `supports_say` in diesem Auftrag;
- keine Voice-Zwischenmeldung während des Tools;
- keine Cloudmodelle oder Fallbacks;
- keine Tool-Control-Tags in sichtbarem Text;
- kein Regex-/XML-/Markdown-Fallback für Tool Calls;
- keine unbounded Queues oder Maps;
- kein Redis, keine Datenbank und kein Message Broker;
- keine neue Modellkopie pro Session;
- keine erfundenen Performancewerte;
- kein Dependency-Upgrade ohne konkreten reproduzierbaren Grund;
- keine v1/v2-Dauerkompatibilitätsschicht.

---

# Qualitätsgates und auszuführende Befehle

Führe nach der Implementierung mindestens aus:

```bash
uv sync --all-packages
uv lock --check
uv run ruff check .
uv run ruff format --check .
uv run mypy packages services examples
uv run pytest -q
uv run python -m build packages/hugging-voice-protocol --outdir dist
uv run python -m build packages/livekit-plugins-hugging-voice --outdir dist
uv run python -m build services/gpu-service --outdir dist
uv run python -m build examples/minimal-livekit-agent --outdir dist
```

Danach erneut mit Lock:

```bash
uv sync --all-packages --frozen
make check
make packages
```

Validiere außerdem:

```text
Dockerfile-Buildkontext
Docker-Compose-Rendering
Kustomize demo
Kustomize production
keine unpinned Images oder Gitrefs
keine neuen Runtime-Downloads
keine Produktions-Testdoubles
```

Wenn Modelle und GPU vorhanden sind:

```bash
make models-verify
HV_RUN_GPU_TESTS=1 uv run pytest -q -m gpu
```

Führe einen kurzen Ein-Session- und Zwei-Session-Tool-Benchmark aus, bevor du Performanceaussagen triffst.

Falls `uv sync --all-packages` den Lock aktualisiert, committe nichts, aber stelle sicher, dass `uv.lock` vollständig und reproduzierbar ist.

---

# Abschlussbericht

Gib am Ende einen präzisen Implementierungsbericht aus:

1. Ausgangs-Commit und tatsächlicher Endstand des Working Trees.
2. Geänderte und neue Dateien, nach Protokoll, Plugin, Service, Tests und Dokumentation gruppiert.
3. Implementierter LiveKit-Tool-Lifecycle.
4. Tatsächlich gesetzte `RealtimeCapabilities` mit Begründung.
5. Exakte Protokoll-v2-Events und Bounds.
6. Reconnect-, Cancellation- und Stale-Result-Semantik.
7. Slot-Affinität und Prompt-Cache-Nachweis in Tests.
8. Alle ausgeführten Befehle mit echten Ergebnissen.
9. GPU-Tests: ausgeführt mit Provenance oder ausdrücklich nicht ausgeführt.
10. Gemessene Latenzen nur dann, wenn echte Messdaten vorliegen.
11. Verbleibende Einschränkungen ausschließlich evidenzbasiert.

Behaupte nicht „vollständig“, wenn der echte LiveKit-FunctionCall -> ToolExecutor -> FunctionCallOutput -> bestätigter zweiter Gemma-Reply nicht in einem Integrationstest bewiesen ist.

