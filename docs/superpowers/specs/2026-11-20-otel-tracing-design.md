# OpenTelemetry tracing (stored in the session store, exported as OTLP/JSON)

Status: design for approval, 2026-11-20. This supersedes the 2026-11-19 draft,
which was never approved.

## Intent and the why

The owner wants to evaluate agent behaviour (what the model saw and decided,
what each tool returned, cost and timing) and later view runs in tools such as
Langfuse. Today logpose has metadata-only observers, but no record of content
or of how the pieces of a run relate. This adds **OpenTelemetry GenAI traces**
for every run. They are stored in logpose's automatic SQLite store, which the
owner said should be reused for tracing, and exported as standard **OTLP/JSON
files** that the OpenTelemetry Collector reads unchanged, so third-party tools
can be wired up later without logpose changes.

## Goals

- **Opt-in tracing:** `Agent(tracing=Tracing(...))`, or `LOGPOSE_TRACE=1` and
  `LOGPOSE_TRACE_CONTENT=1`. Off by default.
- Spans follow the GenAI semantic conventions: `invoke_agent` (the run),
  `chat` (each provider attempt), and `execute_tool` (each tool call), plus
  logpose spans for `guardrail` and `compaction`.
- Spans are **written to the store** (a new table `logpose_spans`, schema
  version 3) through the `SessionStore` interface, so MySQL and Postgres can
  hold them later.
- **Export:** `await export_traces(path, *, since=None, until=None,
  session_id=None)` writes OTLP/JSON lines, one `ExportTraceServiceRequest` per
  trace.
- Content capture (prompts, outputs, reasoning, tool arguments and results) is
  separate and off unless enabled.
- Run labelling: `run(..., metadata={...}, run_id=...)`. The session ID becomes
  `gen_ai.conversation.id`.

## Non-goals

- Network exporters, Langfuse or vendor adapters, the OpenTelemetry SDK, W3C
  context propagation, and metrics.
- Eval scoring and dataset tooling.

## Anti-goals

- Tracing that slows, fails or changes a run. Write failures are swallowed and
  logged once to the `logpose.trace` logger.
- Content stored when capture is off. **Credentials, API keys and headers are
  never stored in any mode.**
- A new dependency. The store already uses `aiosqlite`.

## Constraints

- Builds on Plan 1 (store), Plan 2 (compaction) and Plan 3 (guardrails).
  Migrations go from version 2 to version 3.
- GenAI conventions are still marked "Development". Pin one version, record it
  as `schema_url` and `logpose.semconv.version`, and keep every attribute name
  in one module (`logpose/tracing/semconv.py`).
- Tracing works with or without sessions. Spans are keyed by trace ID;
  `session_id` is nullable.

## Approaches

- **Spans in the session store plus an OTLP/JSON exporter (chosen,
  Claude's call).** This reuses the store as the owner intended, gives one place
  to query, and still produces Collector-ready files on demand.
- Per-run OTLP/JSON files written directly (the 2026-11-19 draft): simpler, but
  it ignores the owner's later "reuse SQLite" intent and makes files the only
  record.
- An optional `logpose[otel]` extra on the OpenTelemetry SDK: rejected earlier
  ("single package").

## Decided

### Settings

- `Tracing(capture_content: bool = False, service_name: str = "logpose",
  content_max_chars: int = 64_000, flush_timeout: float = 1.0)`, a frozen
  dataclass.
- `Agent(tracing=...)`: `None` explicitly disables tracing; the default sentinel
  reads `LOGPOSE_TRACE` and `LOGPOSE_TRACE_CONTENT` (`1`/`true`/`yes`). Blank
  counts as unset.

### Resolutions of the three open questions from the 2026-11-19 draft (Claude's calls)

1. **Credentials:** only known secrets are scrubbed (configured API keys,
   through the existing redaction, and all headers are omitted). Captured
   content is stored as-is; the app owns what its tools return. This is
   documented.
2. **Size:** each content attribute is capped at `content_max_chars`, with the
   marker `…[truncated N chars]` and `logpose.content.truncated=true` on the
   span. Compaction already keeps requests small, so caps rarely trigger.
3. **Flushing:** spans are queued and written by a background writer. At run
   end, `run()`/`stream()` wait at most `flush_timeout` (1 s) for that run's
   spans. Anything still pending is dropped with one WARNING.

### Span model

The span model from the 2026-11-19 draft is kept: the `invoke_agent`, `chat`
and `execute_tool` attributes, usage, metadata as `logpose.metadata.*`, the
config snapshot `logpose.config.*`, gate-block events, and error status with
`error.type`. Additions:

- `gen_ai.conversation.id` = `session_id` when present.
- The `invoke_agent` outcome adds `blocked`. Usage includes `reasoning_tokens`.
- A **`guardrail` span** (INTERNAL) per guard: guard name, allowed, category,
  and scores. With capture on, it also records the input text.
- A **`compaction` span** (INTERNAL) when pointers or a summary were applied:
  pointer count, `summarized`, estimates before and after, and the budget. With
  capture on, it also records the summary text. When a summary call was made,
  the span carries a `logpose.context.summarizing` event (the runtime
  `context.summarizing` event: estimate before and budget) timed when the call
  began; the summary call itself is a sibling `chat` span.
- The input-guardrail `content_check` pseudo-guard (a non-text prompt, blocked
  as category `error`) gets a `guardrail` span like any other guard.
- `purge_expired` skips sessions under a live lease, and so keeps their spans.
- **Token usage follows the convention:** `gen_ai.usage.input_tokens` includes
  cached tokens (logpose's uncached input plus `cache_read_input_tokens`), with
  raw counts in `logpose.usage.*`. Cache counts are also emitted as
  `gen_ai.usage.cache_read.input_tokens` / `gen_ai.usage.cache_creation.input_tokens`
  (defined in the pinned 1.45.0 registry). `gen_ai.provider.name` stays the logpose
  provider name (`openai-compat`): no well-known value applies to an arbitrary
  compatible server.
- Run metadata is stored on spans as strings; `bool` values as `true`/`false`.
- Configured secrets shorter than 12 characters are not scrubbed from content.

### Storage (schema version 3)

`logpose_spans` has these columns:

| Column | Type | Notes |
|---|---|---|
| `trace_id` | TEXT | 32 hex characters |
| `span_id` | TEXT | 16 hex characters |
| `parent_span_id` | TEXT NULL | |
| `run_id` | TEXT | |
| `session_id` | TEXT NULL | |
| `name` | TEXT | |
| `kind` | INTEGER | the OTLP enum |
| `start_ns` | INTEGER | |
| `end_ns` | INTEGER | |
| `status_code` | INTEGER | |
| `status_message` | TEXT NULL | |
| `attributes` | TEXT | JSON |
| `events` | TEXT | JSON |

- Primary key `(trace_id, span_id)`; indexes on `run_id`, `session_id` and
  `start_ns`.
- `delete_session` and `purge_expired` also remove a session's spans.
- `purge_spans(older_than_ms)` is added for traces without a session.

### Interface additions (`SessionStore` and the optional `SpanStore`)

- `SessionStore` gains `async def run_id_exists(run_id: str) -> bool` (run
  records are session data, so every store answers it).
- The span methods live on an optional, runtime-checkable `SpanStore` protocol
  (like `SummaryStore`), so a custom `SessionStore` need not implement tracing.
  A store without them runs sessions normally with tracing disabled (one
  WARNING per process on `logpose.store`); `Agent.purge_traces` then returns 0
  and `export_traces` raises `LogposeError`. `SQLiteStore` implements both.
  - `async def write_spans(spans: Sequence[SpanRecord]) -> None`
  - `def read_spans(*, since_ns=None, until_ns=None, session_id=None, trace_id=None) -> AsyncIterator[SpanRecord]`
  - `def read_root_spans(*, after_seq=None, since_ns=None, until_ns=None, session_id=None) -> AsyncIterator[tuple[int, SpanRecord]]`
  - `async def purge_spans(older_than_ms: int) -> int`

### API additions

- `run`/`stream(..., metadata: Mapping[str, str | int | float | bool] | None = None, run_id: str | None = None)`, also on the sync facade.
  - `run_id` must match `[A-Za-z0-9._-]{1,128}`.
  - `RuntimeContext` gains `metadata`.
- `export_traces(path, *, since=None, until=None, session_id=None) -> int`:
  async; returns the number of traces written; writes OTLP/JSON lines for the
  Collector's `otlpjsonfile` receiver.
- `Tracing` and `export_traces` are exported. `Agent.purge_traces(older_than)`
  is a thin wrapper.

## Builder-check resolutions (Claude's calls, 2026-11-20)

1. **`run_id`:** a caller-supplied `run_id` **replaces** the generated ID
   everywhere (`RuntimeContext`, `RuntimeEvent`, `RunRecord`, spans). When
   tracing or a session is active, logpose checks the store at run start and
   raises `LogposeError` if that `run_id` already exists in `logpose_runs` or
   `logpose_spans`. Eval re-runs must use fresh IDs; there's no overwriting.
2. **Writer ownership:** the store manager owns one store **per (database
   path, event loop)**, because an aiosqlite connection is bound to its loop.
   The sync facade's private loop gets its own store. Each store owns one
   background span writer.
   - **Flushing:** wait up to `flush_timeout` at run end; `Agent.aclose()`
     flushes and closes stores this agent opened; best-effort `atexit` flush.
     Anything still unwritten is dropped with one WARNING.
3. **`chat` span content:**
   - **Input:** the full input messages are stored, following the convention, so
     Langfuse shows real prompts. Growth is bounded because compaction keeps
     each request within `input_budget`, and each attribute is capped at
     `content_max_chars`. The cap keeps the **end** (most recent context) and
     puts the `…[truncated N chars]` marker at the start.
   - **Retries:** only the first attempt of a turn stores the input. Retry
     attempts set `logpose.input.same_as_span=<span_id>` instead.
4. **`export_traces`:**
   - **Connection:** opens its own connection through the store manager (the
     same `LOGPOSE_DB_PATH`).
   - **Completed traces only:** exports only traces whose root `invoke_agent`
     span has ended; a live run is never exported half-finished.
   - **Time window:** `since` and `until` are timezone-aware `datetime`s,
     matched against the **root span's start time**.
   - **Return value:** `ExportResult(traces: int, through: datetime | None,
     cursor: int | None)`. `through` is the latest root start exported
     (informational only).
   - **Incremental export (amended 2026-11-20 after review):** a start-time or
     end-time watermark is not exactly-once, because spans are written by a
     background writer, possibly long after the run ends. So each span row
     gets a write-order number `export_seq`, assigned inside the write
     transaction. A re-write of the same span keeps its original number.
     - Export reads roots in `export_seq` order.
     - `cursor` is the largest `export_seq` exported, or the caller's `after`
       when nothing was exported; it never goes backwards.
     - Callers pass `after=result.cursor` on the next call to export each
       completed trace exactly once.
     - `since`/`until` remain plain filters on the root's start time.
   - **Service name:** `Tracing.service_name` is stored on the root span and
     becomes the resource `service.name` on export (resourceSpans are grouped
     by it).
   - **Purge:** `purge_spans` deletes whole traces whose root started before
     the cutoff, plus any orphan spans older than the cutoff.
5. **Metadata:**
   - stored on spans (`logpose.metadata.*`) and in `RuntimeContext`;
   - in `logpose_runs`, through a new `metadata TEXT` (JSON) column added by the
     version 3 migration, for linking eval runs to dataset items;
   - **not** added to `RuntimeEvent`, which stays content-free.

## Left to the builder

- The pinned convention version and exact attribute spellings, all in
  `semconv.py`.
- The internal span-builder hook's shape, and the background writer's
  queue size. If the queue is full, drop the span and log once.
- ID generation (random, hex) and clocks (`time.time_ns()` for start and end;
  monotonic for durations).
