# Per-run `extra` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `run`/`stream` (and the sync facade) accept `extra=`, merged over `Agent(extra=)` for that run only.

**Architecture:** `run`/`stream` validate and merge once into a per-run dict carried through `_loop` (like `run_metadata`); `_request` and `_summarize` read the run's merged dict instead of `self.extra`; the span builder records only the sorted key names.

**Tech Stack:** Python 3.10+, `uv`, pytest-asyncio (auto), `FakeProvider` / `httpx.MockTransport`.

**Spec:** `docs/superpowers/specs/2026-11-23-per-run-extra-design.md`

## Global Constraints

- Merge: `{**agent.extra, **extra}`, for every model call of the run (turns, structured-output correction turns, the context-length retry).
- Per-run `extra` must be a `Mapping` with `str` keys and may not contain `model`, `messages`, `tools`, `stream`, `stream_options` → `LogposeError` when `run`/`stream` is called (eagerly, before iteration). `Agent(extra=)` behaviour unchanged.
- Merged `extra` with `response_format` + an effective `output_type` → `LogposeError` at call time.
- Summary call: merged `extra` minus `_TURN_ONLY_EXTRA`.
- Guardrail calls never receive `extra`.
- Tracing: `invoke_agent` attribute `logpose.request.extra_keys` = sorted key names of the merged `extra` (list of str), only when non-empty; never values (also with capture on). The name lives in `tracing/semconv.py`.
- Not persisted in sessions. Backward compatible. Gates: `uv run --all-extras pytest -q`, `uv run ruff check`, `uv run --all-extras mypy src/`.

## Review Focus

1. Per-run `extra` must not leak into the next run on the same shared Agent (and concurrent runs must not see each other's) — test in Task 1.
2. `extra=None` and `extra={}` behave exactly like today (request bodies identical) — test in Task 1.
3. A per-run `max_completion_tokens` replaces logpose's ceiling field in the provider body (only one field sent) — test in Task 1 via `httpx.MockTransport`.
4. A value in per-run `extra` never appears in any stored span attribute, event, or status, even with `capture_content=True` — test in Task 1.
5. Sync facade forwards `extra` (run_sync, stream_sync, SyncAgent.run/stream) — test in Task 1.

---

### Task 1: Per-run `extra` through the loop, sync facade and tracing

**Files:**
- Modify: `src/logpose/agent.py` (`run`, `stream`, `_loop`/`_traced_loop`/`_loop_body` plumbing, `_request`, `_summarize`, `_trace_config` or the span-builder call), `src/logpose/sync.py` (`run_sync`, `stream_sync`, `SyncAgent.run`, `SyncAgent.stream`), `src/logpose/tracing/spans.py` (`start_run`), `src/logpose/tracing/semconv.py`
- Test: `tests/test_per_run_extra.py` (new)

**Interfaces:**
- Produces: `Agent.run(..., extra: Mapping[str, Any] | None = None)`, `Agent.stream(..., extra: Mapping[str, Any] | None = None)`; same keyword on `run_sync`, `stream_sync`, `SyncAgent.run`, `SyncAgent.stream`. Semconv constant `LOGPOSE_REQUEST_EXTRA_KEYS = "logpose.request.extra_keys"`.

- [ ] **Step 1: Write failing tests** in `tests/test_per_run_extra.py` (use `FakeProvider` capturing `CompletionRequest`s, plus an `OpenAICompatProvider` with `httpx.MockTransport` where the wire body matters):
  - `test_run_extra_merges_over_agent_extra`: `Agent(extra={"a": 1, "b": 2})`, `run(..., extra={"b": 3, "c": 4})` → request `extra == {"a": 1, "b": 3, "c": 4}`; agent's `extra` unchanged.
  - `test_run_extra_applies_to_every_turn`: a tool round (two turns) → both requests carry `c`.
  - `test_run_extra_does_not_leak`: second `run` without `extra` → request `extra == {"a": 1, "b": 2}`; two concurrent `asyncio.gather` runs with different `extra` → each run's requests carry only its own values.
  - `test_no_extra_unchanged`: `extra=None` and `extra={}` give the same request extra as today.
  - `test_run_extra_reserved_keys_rejected`: each of `model`, `messages`, `tools`, `stream`, `stream_options` → `LogposeError` raised by `run(...)` and by calling `stream(...)` (before iteration); non-str key and non-Mapping → `LogposeError`.
  - `test_run_extra_response_format_conflicts_with_output_type`: `run(..., output_type=M, extra={"response_format": {...}})` → `LogposeError`; without `output_type` it passes through to the request.
  - `test_run_extra_ceiling_field_replaces_logposes`: wire body with `extra={"max_completion_tokens": 50}` has `max_completion_tokens == 50` and no `max_tokens`.
  - `test_summary_call_gets_merged_extra_minus_turn_only`: force a summary (small `context_budget`) with per-run `extra={"safety_identifier": "h", "tool_choice": "none"}` → summary request has `safety_identifier` and not `tool_choice`.
  - `test_guardrails_never_see_extra`: a custom async guard records its `GuardrailInput`; it has no access to extra (assert the guard call path is unchanged and the decision guard's HTTP body via MockTransport has no `safety_identifier`).
  - `test_tracing_records_extra_keys_not_values`: tracing on with `capture_content=True` (in-memory sink fixture as in tests/test_tracing_spans.py), `extra={"safety_identifier": "SECRET-VALUE-123"}` → root span `logpose.request.extra_keys == ["safety_identifier"]` (merged with agent keys, sorted); `"SECRET-VALUE-123"` appears in no attribute, event or status of any span; attribute absent when merged extra is empty.
  - `test_sync_facade_forwards_extra`: `run_sync`, `stream_sync`, `SyncAgent.run`, `SyncAgent.stream` each deliver `c` in the request.
- [ ] **Step 2: Run** `uv run --all-extras pytest -q tests/test_per_run_extra.py` — expect failures (`unexpected keyword argument 'extra'`).
- [ ] **Step 3: Implement.** Validate + merge in `run`/`stream` (one helper `_merge_run_extra(extra) -> dict[str, Any]`, reserved keys `_RESERVED_EXTRA_KEYS = frozenset({"model","messages","tools","stream","stream_options"})`); extend the existing `response_format`/`output_type` check to the merged dict; carry the merged dict as a keyword through `_loop` → `_traced_loop` → `_loop_body`; `_request(history, extra)` and `_summarize(..., extra)` use it (summary filters `_TURN_ONLY_EXTRA`); pass sorted keys to the span builder for `start_run`. Update docstrings (Args: `extra`).
- [ ] **Step 4: Run** the new tests, then the full gates — expect all pass.
- [ ] **Step 5: Commit** `git add <changed files> && git commit -m "feat: per-run extra on run/stream and the sync facade"`.

### Task 2: Docs

**Files:** `README.md` (the `extra` / Configuration reference section, Tracing section attribute list), `CHANGELOG.md` (`## [Unreleased]` → Added), `tests/test_public_api.py` only if signatures are asserted there.

- [ ] **Step 1:** README: per-run `extra` with a `safety_identifier`/`prompt_cache_key` example on a shared agent; the merge rule, reserved keys, `response_format` and ceiling rules, summary call, guardrails exclusion; Tracing: `logpose.request.extra_keys` (names only).
- [ ] **Step 2:** CHANGELOG Added entry. Run gates.
- [ ] **Step 3: Commit** `git commit -m "docs: per-run extra"`.
