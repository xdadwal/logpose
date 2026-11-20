# OpenTelemetry Tracing Implementation Plan (Plan 4 of 4)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Opt-in OpenTelemetry GenAI traces for every run, stored in the automatic SQLite store (schema version 3) and exportable as OTLP/JSON for the OpenTelemetry Collector.

**Architecture:**
- `logpose/tracing/semconv.py` holds every attribute name and the pinned convention version.
- `logpose/tracing/spans.py` holds the `SpanRecord` type and the builder that turns loop lifecycle points into spans.
- `logpose/tracing/writer.py` holds the background writer owned by each store.
- `logpose/tracing/export.py` converts stored spans to OTLP/JSON.
- The `Agent` loop calls a private span hook at the same points it emits `RuntimeEvent`s.

**Tech Stack:** Python 3.10+, aiosqlite (already a dependency), pytest.

**Spec:** `docs/superpowers/specs/2026-11-20-otel-tracing-design.md` (with the span model from `docs/superpowers/specs/2026-11-19-otel-tracing-design.md`). Requires Plans 1–3 to be merged.

## Global Constraints

- **Off by default.** `Agent(tracing=Tracing(...))`, or `LOGPOSE_TRACE` / `LOGPOSE_TRACE_CONTENT` (`1`/`true`/`yes`). `tracing=None` forces it off.
- **Content is stored only when `capture_content` is true.** Credentials and headers are never stored. Content attributes are capped at `content_max_chars` (64,000), keeping the end and putting `…[truncated N chars]` at the start, plus `logpose.content.truncated=true`.
- **Tracing never raises into a run and never changes results.** Writer failures are logged once to the `logpose.trace` logger at WARNING.
- **Flushing:** each run waits at most `flush_timeout` (1.0 s) for its spans.
- **Store ownership:** one store and one writer per (database path, event loop).
- **`run_id`:** a caller-supplied `run_id` replaces the generated one everywhere; a duplicate (in `logpose_runs` or `logpose_spans`) raises `LogposeError` at run start.
- **Gates:** `uv run pytest -q`, `uv run ruff check`, `uv run mypy src/`.

## Review Focus

1. **Tracing on, but the database is unwritable** (read-only directory). The run completes normally, exactly one WARNING is logged, and the result is identical to tracing off. Test owned by Task 3.
2. **Export during a live run.** The in-progress trace is not exported; after the run ends, a second export with `since=result.through` includes it exactly once. Test owned by Task 4.
3. **A provider retry inside a run.** Two sibling `chat` spans; only the first stores input content, and the second has `logpose.input.same_as_span` set to the first's span ID. Test owned by Task 2.
4. **Capture off, with a tool that returns a string containing the configured API key.** No span attribute in the database contains the key, or any prompt or tool text. Test owned by Task 2.
5. **Sync facade with tracing.** Spans land in the store opened on the facade's private loop, and `SyncAgent.close()` flushes them. Test owned by Task 3.

---

### Task 1: Semantic conventions, span records, and storage (schema version 3)

**Files:**
- Create: `src/logpose/tracing/__init__.py`, `src/logpose/tracing/semconv.py`, `src/logpose/tracing/spans.py`
- Modify: `src/logpose/store/base.py`, `src/logpose/store/sqlite.py`
- Test: `tests/test_tracing_store.py`

**Interfaces (produces):**
- `semconv.py`: `SEMCONV_VERSION`, `SCHEMA_URL`, and string constants for every attribute the spec lists (`GEN_AI_OPERATION_NAME`, `GEN_AI_REQUEST_MODEL`, …). No other module spells an attribute name.
- `spans.py`: frozen dataclass `SpanRecord` with the fields `trace_id`, `span_id`, `parent_span_id`, `run_id`, `session_id`, `name`, `kind`, `start_ns`, `end_ns`, `status_code`, `status_message`, `attributes: dict`, `events: list[dict]`; plus `new_trace_id()` and `new_span_id()`.
- **Store:**
  - the version 3 migration adds `logpose_spans` (as in the spec, with indexes) and a `metadata TEXT` column on `logpose_runs`;
  - `write_spans`, `read_spans` (async iterator) and `purge_spans`;
  - `delete_session` and `purge_expired` also remove spans;
  - `run_id_exists(run_id) -> bool`.

- [ ] **Step 1: Failing tests:**
  - `test_migrates_v2_to_v3_preserving_data`.
  - `test_write_and_read_spans_round_trip`.
  - `test_read_spans_filters`: by `session_id`, by `trace_id`, and by start range.
  - `test_purge_spans_and_session_delete_remove_spans`.
  - `test_run_id_exists`.
  - `test_semconv_is_single_source`: `grep` for the `"gen_ai.` literal in `src/` finds hits only in `semconv.py`.
- [ ] **Step 2: Run and confirm they fail.**
- [ ] **Step 3: Implement.** Pin the latest GenAI semantic-conventions release, and record its version in `semconv.py`.
- [ ] **Step 4: Run the gates.** Run: `uv run pytest -q && uv run ruff check && uv run mypy src/`. Expected: clean.
- [ ] **Step 5: Commit** with `git commit -m "feat(tracing): semconv, span records, schema v3 storage"`.

### Task 2: Span builder wired into the loop

**Files:**
- Modify: `src/logpose/agent.py`, `src/logpose/runtime.py` (`RuntimeContext.metadata`), `src/logpose/tracing/spans.py`
- Test: `tests/test_tracing_spans.py`

**Interfaces:**
- **Consumes:** Task 1, and the Plan 1 to 3 hooks (run records, compaction diagnostics, guardrail results).
- **Produces:**
  - `Tracing` dataclass (the spec's fields).
  - `Agent(tracing=...)` with the sentinel and environment resolution.
  - `run`/`stream(..., metadata=None, run_id=None)` with validation.
  - A private `_SpanBuilder` per run, which:
    - opens and closes `invoke_agent`, `guardrail`, `compaction`, `chat` (per attempt) and `execute_tool` spans with the spec's attributes;
    - emits retry and gate-block events;
    - sets error status with `error.type`;
    - converts usage so `gen_ai.usage.input_tokens` includes cached tokens, with the raw counts under `logpose.usage.*`;
    - stores metadata as `logpose.metadata.*` and in `RunRecord.metadata`.
  - Spans are handed to the store's writer (Task 3). Until Task 3 lands, the hand-off goes to an in-memory list that tests read.

- [ ] **Step 1: Failing tests** using `FakeProvider` and an in-memory span sink:
  - `test_span_tree_shape`: a run with one tool round gives a single trace: `invoke_agent` → `chat`, `execute_tool`, `chat`, with correct parents and `gen_ai.operation.name` values.
  - `test_invoke_agent_attributes`: outcome, iterations, usage (`input_tokens` includes cached), `gen_ai.conversation.id` equal to the session ID, metadata, and the config snapshot.
  - `test_retry_creates_sibling_chat_spans`: Review Focus 3.
  - `test_capture_off_stores_no_content`: Review Focus 4.
  - `test_capture_on_stores_messages_with_tail_truncation`: a 100,000-character tool result is stored at the 64,000 cap, keeping the end, with the marker and the truncated flag.
  - `test_guardrail_and_compaction_spans`: a blocked run gives `invoke_agent` (outcome `blocked`) plus a `guardrail` span; a run that hits compaction gives a `compaction` span.
  - `test_error_status_on_failures`: a provider error gives `chat` status ERROR with `error.type`.
  - `test_run_id_override_and_duplicate_rejected`: the given `run_id` appears in `RuntimeContext`, events and spans; reusing it raises `LogposeError` before any provider call.
  - `test_tracing_off_by_default_and_none_forces_off`.
- [ ] **Step 2: Run and confirm they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the gates.** Run: `uv run pytest -q && uv run ruff check && uv run mypy src/`. Expected: clean.
- [ ] **Step 5: Commit** with `git commit -m "feat(tracing): GenAI span builder in the agent loop"`.

### Task 3: Background writer, flushing, and the sync facade

**Files:**
- Create: `src/logpose/tracing/writer.py`
- Modify: `src/logpose/store/manager.py`, `src/logpose/agent.py`, `src/logpose/sync.py`
- Test: `tests/test_tracing_writer.py`

**Interfaces (produces):**
- `SpanWriter(store, *, queue_max=10_000)`, owned by each store:
  - `submit(span)` never blocks; if the queue is full it drops the span and warns once;
  - `async flush(run_id, timeout) -> bool`;
  - `async close()`.
- **Run end:** `await writer.flush(run_id, tracing.flush_timeout)`; on timeout, warn once.
- **`Agent.aclose()`** flushes and closes the stores it opened.
- **`atexit`:** a best-effort flush.
- **Sync facade:** `stream_sync` / `run_sync` / `SyncAgent` pass `metadata` and `run_id` through; `SyncAgent.close()` flushes.

- [ ] **Step 1: Failing tests:**
  - `test_unwritable_db_never_breaks_run`: Review Focus 1, using `caplog` to assert exactly one WARNING.
  - `test_flush_waits_bounded`: a slow store (monkeypatched `write_spans` sleeping 2 s) with `flush_timeout=0.2` makes the run return within about 0.5 s, and warns.
  - `test_queue_full_drops_and_warns_once`.
  - `test_spans_persisted_after_run`: after `run()` returns, `read_spans(trace_id=…)` has all spans.
  - `test_sync_facade_tracing`: Review Focus 5.
- [ ] **Step 2: Run and confirm they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the gates.** Run: `uv run pytest -q && uv run ruff check && uv run mypy src/`. Expected: clean.
- [ ] **Step 5: Commit** with `git commit -m "feat(tracing): background span writer with bounded flush"`.

### Task 4: OTLP/JSON export and docs

**Files:**
- Create: `src/logpose/tracing/export.py`
- Modify: `src/logpose/__init__.py`, `src/logpose/agent.py` (`purge_traces`), `README.md`, `CHANGELOG.md`
- Test: `tests/test_tracing_export.py`, `tests/test_public_api.py`

**Interfaces (produces):**
- `async def export_traces(path, *, since: datetime | None = None, until: datetime | None = None, session_id: str | None = None) -> ExportResult`, where `ExportResult(traces: int, through: datetime | None)`.
  - It opens a store through the manager.
  - It exports only traces whose root span has ended.
  - It filters on the root span's start time.
  - It writes one `ExportTraceServiceRequest` JSON object per line, containing `resourceSpans → scopeSpans → spans`:
    - hex IDs; `startTimeUnixNano` / `endTimeUnixNano` as strings;
    - attributes as `[{key, value: {stringValue|intValue|doubleValue|boolValue|arrayValue}}]`;
    - the resource has `service.name`, `telemetry.sdk.name="logpose"`, `telemetry.sdk.version` and `logpose.semconv.version`;
    - the scope is `logpose`, with `schemaUrl`.
- `Agent.purge_traces(older_than: timedelta) -> int`.
- Exports: `Tracing`, `export_traces`, `ExportResult`.

- [ ] **Step 1: Failing tests:**
  - `test_export_otlp_json_shape`: the parsed line matches the OTLP/JSON structure above; `intValue` is a string; parent links are preserved.
  - `test_export_skips_live_traces_and_is_incremental`: Review Focus 2.
  - `test_export_filters_by_session_and_window`.
  - `test_purge_traces`.
  - Public-API additions.
- [ ] **Step 2: Run and confirm they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Write the docs.**
  - **README "Tracing" section:**
    - enabling it, and capture-content privacy (what is and isn't scrubbed);
    - where spans live, and `export_traces`, with an OpenTelemetry Collector `otlpjsonfile` receiver config snippet;
    - the run-ID and metadata usage for evals;
    - purging;
    - a note that the conventions are still "Development", with the pinned version.
  - **CHANGELOG:** Added entries.
- [ ] **Step 5: Run the gates.** Run: `uv run pytest -q && uv run ruff check && uv run mypy src/`. Expected: clean.
- [ ] **Step 6: Commit** with `git commit -m "feat(tracing): OTLP/JSON export; docs"`.
