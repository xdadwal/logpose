# OpenTelemetry tracing to flat files

Status: design for approval, 2026-11-19

## Intent and the why

logpose runs agent loops inside small web apps. The owner wants to evaluate
agent behavior (what the model saw, what it decided, what each tool returned,
how long it took, what it cost) and later watch runs in tools like Langfuse.

Today logpose has the pieces but not the record:
- `observers` carry IDs and timings but no content, by design.
- `stream()` events carry content but mostly no IDs.
- `tool_observer` carries tool content, but a failure in it stops the run.

Nothing records exactly what was sent on each model attempt, or links a run to
the app's own context.

This change makes each run produce a standard **OpenTelemetry GenAI trace**,
written to local files. The files serve as eval data now. Later, the standard
OpenTelemetry Collector can ship them to Langfuse or similar with no logpose
changes.

## Goals

- Opt-in tracing configured on `Agent`, or by environment variable, that writes
  one OTLP/JSON file per run.
- Spans follow the OpenTelemetry GenAI semantic conventions:
  - `invoke_agent` for the run;
  - `chat` for each provider attempt;
  - `execute_tool` for each tool call.
- Content (prompts, outputs, reasoning, tool arguments and results) is recorded
  only when the user opts in, separately from turning tracing on.
- Each trace records enough to evaluate a run offline:
  - the exact request sent on each attempt, after `context_policy`;
  - the response, including reasoning and tool calls;
  - usage, finish reason, and timings;
  - tool arguments, results, errors, and timeouts;
  - gate decisions;
  - retries;
  - run outcome (completed, max iterations, run timeout, cancelled, error);
  - a configuration snapshot;
  - app-supplied metadata.
- The app can label a run: `run(..., metadata={...})`, an optional
  `run(..., run_id=...)`, and a stable `Conversation.id` that groups runs of one
  conversation.
- `Usage` gains `reasoning_tokens`, read from Chat Completions'
  `completion_tokens_details.reasoning_tokens` when present.

## Non-goals

- No network exporters, no Langfuse or other vendor adapters, no OpenTelemetry
  SDK dependency. Files are the only output for now.
- No joining of traces the app already records with the OpenTelemetry SDK
  (W3C context propagation). That comes later, as an adapter on the same hook.
- No eval scoring, dataset tooling, or replay.
- No metrics (OpenTelemetry metrics or histograms), only traces.

## Anti-goals

These would make it a failure even if it works:
- Tracing that can slow down, fail, or change a run. File I/O failures must
  never surface to the caller or alter results.
- Content written to disk when the user turned tracing on but did not opt into
  content. Credentials, API keys, and headers are never written in any mode.
- A new runtime dependency.

## Constraints

- Python 3.10+, core package only, stdlib plus existing dependencies (`pydantic`,
  `httpx`).
- `import logpose` must stay lazy: tracing code must not import `httpx` or a
  provider module.
- Must respect the web-app limits: tracing adds no unbounded waits.
- The GenAI semantic conventions are still marked Development. Pin one
  convention version, record it in every file, and keep all naming in one
  module so a later convention change touches only that module.

## Approaches considered

- **A. Core trace hook plus a built-in OTLP/JSON file writer, no dependency
  (chosen).** The loop emits content-bearing trace records to an internal hook,
  and a writer turns them into spans. This keeps logpose dependency-free and
  contains convention churn in one module. The same hook can later feed an
  OpenTelemetry SDK adapter or other formats.
- B. An optional `logpose[otel]` extra on the OpenTelemetry SDK. Rejected by the
  owner ("single package"). It also needs a custom file exporter anyway.
- C. Langfuse-specific integration. Out of scope for now.

## Decided

### Turning it on

```python
from logpose import Agent, Tracing

agent = Agent(
    "openai-compat",
    model="gpt-5.1",
    tracing=Tracing(path="traces/", capture_content=True),
)
result = await agent.run(
    "…",
    metadata={"dataset_item": "q-17", "experiment": "baseline"},
)
```

- `Tracing(path: str | os.PathLike, capture_content: bool = False,
  service_name: str = "logpose")` is a frozen dataclass.
- When `tracing=` is not passed, the environment can enable it:
  - `LOGPOSE_TRACE_PATH` turns tracing on;
  - `LOGPOSE_TRACE_CONTENT` (`1`/`true`/`yes`, case-insensitive) turns on
    content capture.
- Off by default. `Agent(tracing=None)` explicitly disables it even when the
  environment is set. The default is a sentinel that means "read the
  environment".
- Blank environment values count as unset, consistent with `LOGPOSE_*`.

### Files

- One file per run: `{path}/{UTC start time as YYYYMMDDTHHMMSSffffffZ}_{run_id}.jsonl`.
  The directory is created if missing.
- Format is OTLP/JSON, the OpenTelemetry protobuf-JSON mapping of
  `ExportTraceServiceRequest`. There's one JSON object per line, each holding
  `resourceSpans[].scopeSpans[].spans[]`, so the OpenTelemetry Collector's
  `otlpjsonfile` receiver can read the files unchanged.
- Each span is written as its own line when it ends, so a crashed run still
  leaves the spans that finished.
- The resource has `service.name`, `telemetry.sdk.name="logpose"`, and
  `telemetry.sdk.version`. The instrumentation scope is `logpose` with
  `schema_url` set to the pinned semantic-convention version.

### Spans

There's one trace per `run()` / `stream()` call.

```
invoke_agent logpose          run
├── chat {model}              each provider attempt (retries are siblings)
│     └ event: retry scheduled
├── execute_tool {tool}       each tool call
│     └ event: blocked by gate
└── chat {model}
```

**`invoke_agent` span**

| Field | Value |
|---|---|
| kind | INTERNAL |
| `gen_ai.operation.name` | `invoke_agent` |
| `gen_ai.provider.name` | the provider name |
| `gen_ai.request.model` | the model |
| `gen_ai.agent.name` | `"logpose"` |
| `gen_ai.conversation.id` | the conversation ID |
| `logpose.run.id` | the run ID |
| `logpose.run.iterations` | iterations taken |
| `logpose.run.outcome` | `completed`, `max_iterations`, `run_timeout`, `cancelled`, or `error` |
| usage | totals: `gen_ai.usage.input_tokens`, `output_tokens`, and `logpose.usage.reasoning_tokens`, `cache_read_input_tokens`, `cache_creation_input_tokens` |
| metadata | each pair as `logpose.metadata.<key>`, stringified |
| config snapshot | `logpose.config.*`: max_iterations, max_tokens, run_timeout, provider_turn_timeout, tool_timeout, max_concurrent_tools, retry policy, tool names, plus a hash of the system prompt |
| system prompt and tool schemas | only with content capture: `gen_ai.system_instructions` and `gen_ai.tool.definitions` |

**`chat` span** (one per provider attempt)

| Field | Value |
|---|---|
| kind | CLIENT |
| `gen_ai.operation.name` | `chat` |
| `gen_ai.request.model` | the model |
| `gen_ai.request.max_tokens` | the token limit |
| `gen_ai.response.finish_reasons` | the finish reason |
| usage | as above |
| `logpose.attempt.number` | the attempt number |
| `logpose.attempt.retryable` | whether the failure can be retried |
| `logpose.context.*` | the context policy's diagnostics |
| messages | only with content capture: `gen_ai.input.messages` (exactly the messages sent, after `context_policy`) and `gen_ai.output.messages` (the assembled assistant message, including reasoning and tool calls) |

**`execute_tool` span** (one per tool call)

| Field | Value |
|---|---|
| kind | INTERNAL |
| `gen_ai.operation.name` | `execute_tool` |
| `gen_ai.tool.name` | the tool name |
| `gen_ai.tool.call.id` | the call ID |
| `logpose.tool.queue_seconds` | time spent waiting for a slot |
| `logpose.tool.timed_out` | whether the tool hit its time limit |
| `logpose.tool.is_error` | whether the result is an error |
| arguments and result | only with content capture: `gen_ai.tool.call.arguments` and `gen_ai.tool.call.result` |
| gate block | a `logpose.gate.blocked` event, with the replacement text only with content capture |

**Errors:** span status is ERROR with `error.type` set to the exception type
name. Exception messages are recorded only with content capture.

### API additions

- `Agent.run(prompt=None, *, conversation=None, metadata=None, run_id=None)`,
  and the same for `stream`, `run_sync`, `stream_sync`, and `SyncAgent`.
- `metadata` is a `Mapping[str, str | int | float | bool]`. Other value types
  raise `LogposeError` before the run starts.
- `run_id` overrides the generated one. It must match `[A-Za-z0-9._-]{1,128}`
  because it appears in a filename.
- `RuntimeContext` gains `metadata`, so tools and observers can read it through
  `current_runtime_context()`.
- `Conversation` gains `id: str`, auto-generated and settable through
  `Conversation(messages, id=...)`.
- `Usage` gains `reasoning_tokens: int = 0`, and addition sums it.
- `Tracing` is exported from `logpose`.

### Engineering

- A content-bearing internal hook alongside the existing metadata-only
  `observers` (which stay unchanged). It is private for now, and the file
  writer is its only consumer.
- File writes happen off the event loop and never block the run. Any write or
  serialization error is swallowed and logged once to the `logpose.trace`
  logger at WARNING. Tracing must never raise into a run.
- Content written with capture on goes through the same credential redaction
  used for provider errors.
- One pinned semantic-convention version, recorded as `schema_url` and as
  `logpose.semconv.version` on the resource.

## Left to the builder

- The exact pinned semantic-convention version and attribute spellings. Pick the
  latest release of the GenAI conventions (in their own repository since 2026)
  and follow it, keeping all names in one module.
- The internal hook's shape and the writer's threading or queue design, within
  the anti-goals above.
- Span ID and trace ID generation (random, 16/8 bytes, hex) and clock source.
  Wall-clock nanoseconds come from `time.time_ns()`, and durations from the
  monotonic clock where possible.
- Whether sync-facade runs need extra care for flushing at close.
