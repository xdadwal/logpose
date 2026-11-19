# Input Guardrails Implementation Plan (Plan 3 of 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `Agent(input_guardrails=[...])` checks each new user message before the agent loop. It ships `pattern_guard` and the OpenAI Decisions-API `openai_decision_guard`. A blocked message ends the run with a streamed placeholder reply, and never enters history.

**Architecture:**
- `logpose/guardrails.py` holds the types, the two-tier runner, and both built-in guards.
- `Agent._loop` runs the guards before appending the prompt (blocking mode), or alongside the first provider turn while holding back output (parallel mode).
- The Decisions guard makes its own `httpx` request, outside the provider layer.

**Tech Stack:** Python 3.10+, httpx, pydantic v2, pytest.

**Spec:** `docs/superpowers/specs/2026-11-20-input-guardrails-design.md` (builds on Plan 1, `docs/superpowers/plans/2026-11-20-session-store.md`, and runs after Plan 2).

## Global Constraints

- **Guard order:** synchronous guards run first, in list order (the first block ends the run); then asynchronous guards run concurrently (the first block wins, the others are cancelled).
- **Decisions request:** `POST {base_url}/decisions`, defaults `base_url="https://api.openai.com/v1"` and `model="gpt-6-luna"`. Two predicates, named `on_topic` and `unsafe`, with the exact instruction texts from the spec.
- **Thresholds and evaluation order:**
  - block `"unsafe"` if `p(unsafe) > unsafe_max`, else block `"off_topic"` if `p(on_topic) < on_topic_min`;
  - defaults `on_topic_min=0.5`, `unsafe_max=0.5`, `history=4`, `timeout=5.0`, `on_error="block"`;
  - a refusal on `unsafe` blocks as `"unsafe"`; a refusal on `on_topic` applies `on_error`.
- **Default error reply** for a guard that raises: `"Sorry, something went wrong. Please try again."`, category `"error"`.
- **A blocked run:**
  - the prompt is not appended to history;
  - the stream yields `TextDelta(reply)` then `RunEnd`;
  - `RunResult.text == reply`, `stop_reason == "refusal"`, and `RunResult.guardrail` is set;
  - with `output_type`, it raises `GuardrailTripped` instead;
  - the session outcome is `"blocked"`.
- **Telemetry:** `guardrail.decided` events carry no message text.
- **Usage:** the Decisions usage (input, output, reasoning, cached) is added to the run's usage.
- **Gates:** `uv run pytest -q`, `uv run ruff check`, `uv run mypy src/`.

## Review Focus

1. **A run on an existing conversation is blocked.** `conv.messages` is exactly as before the run (the prompt isn't appended), and a following allowed run's request doesn't contain the blocked text. Test owned by Task 2.
2. **Parallel mode with a model that requests a tool before the guard finishes.** The tool never executes when the guard then blocks. Test owned by Task 3.
3. **An over-length message with both guards configured.** The Decisions endpoint is never called. Test owned by Task 2.
4. **The Decisions endpoint returns 500, or times out, with the default `on_error`.** The request is blocked with category `"error"`, and the response body is not in the reason or the event. Test owned by Task 1.
5. **Prior turns that contain tool calls.** Only user and assistant text from recent turns reaches the Decisions `input`. Tool-call and tool-result messages are skipped and never count toward `history`. Test owned by Task 1.

---

### Task 1: Guard types and built-in guards

**Files:**
- Create: `src/logpose/guardrails.py`
- Modify: `src/logpose/__init__.py`, `src/logpose/errors.py`
- Test: `tests/test_guardrails.py`

**Interfaces (produces):**
- Frozen dataclasses `GuardrailInput` and `GuardrailVerdict` (with `allow` and `block` classmethods) exactly as in the spec; the alias `InputGuardrail`.
- `GuardrailResult(guard: str, category: str, reason: str | None, scores: Mapping[str, float])`.
- `GuardrailTripped(LogposeError)` with a `.result` attribute.
- `guard_text(message: Message) -> str` joins the text blocks with `\n`. Messages have no image blocks, so guards are text-only.
- `pattern_guard(*, max_chars=4000, block_patterns=(), reply, category="blocked_pattern") -> InputGuardrail`: a sync function with `__name__ = "pattern_guard"`.
- `openai_decision_guard(domain, *, replies, api_key=None, base_url="https://api.openai.com/v1", model="gpt-6-luna", on_topic_min=0.5, unsafe_max=0.5, history=4, timeout=5.0, on_error="block", client: httpx.AsyncClient | None = None) -> InputGuardrail`:
  - an async callable with `.name == "openai_decision_guard"`;
  - its verdict scores include `on_topic` and `unsafe` when answered;
  - a per-call usage hook `.last_usage: Usage` (or an equivalent the runner reads).

- [ ] **Step 1: Failing tests** in `tests/test_guardrails.py`. The Decisions tests use `httpx.MockTransport` that records the request body.
  - `test_pattern_guard_blocks_too_long_and_patterns`: 5,000 characters give `block("too_long")`; a regex hit gives the configured category; clean text is allowed.
  - `test_decision_request_shape`: `model == "gpt-6-luna"`, the `questions` names are `["on_topic", "unsafe"]` with type `predicate`, the `on_topic` instructions contain the domain string, `input` is a message array ending with the new user message, at most `history` prior messages are included, and the auth header is `Bearer <key>`.
  - `test_decision_thresholds` (parametrized):
    - `(on_topic=0.9, unsafe=0.1)` → allow;
    - `(0.2, 0.1)` → `off_topic`;
    - `(0.9, 0.8)` → `unsafe`;
    - `(0.2, 0.8)` → `unsafe` (unsafe is checked first).
  - `test_refusal_on_unsafe_blocks_as_unsafe` and `test_refusal_on_on_topic_applies_on_error` (both `"block"` and `"allow"`).
  - `test_decision_http_error_and_timeout_apply_on_error`: Review Focus 4, asserting the mock's response-body marker string is absent from the reason.
  - `test_decision_usage_mapped`: a response `usage` with input 120 (cached 20), output 3, and reasoning 2 gives `Usage(input_tokens=100, cache_read_input_tokens=20, output_tokens=3, reasoning_tokens=2)`.
  - `test_recent_history_is_text_only`: Review Focus 5. With history of user text, then assistant tool call, then tool result, then assistant text, then the new user message, and `history=4`, the `input` holds only the 2 text messages plus the new one.
  - `test_replies_must_include_both_keys` and `test_threshold_range_validated`: each raises `LogposeError` at construction.
  - `test_missing_api_key_applies_on_error`: no argument and no env, so the first call gives an `"error"` block.
- [ ] **Step 2: Run and confirm they fail.** Run: `uv run pytest tests/test_guardrails.py -q`. Expected: FAIL (module missing).
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the gates.** Run: `uv run pytest -q && uv run ruff check && uv run mypy src/`. Expected: clean.
- [ ] **Step 5: Commit** with `git commit -m "feat(guardrails): guard types, pattern_guard, openai_decision_guard"`.

### Task 2: Blocking-mode integration

**Files:**
- Modify: `src/logpose/agent.py`, `src/logpose/events.py` (`RunResult.guardrail`), `src/logpose/runtime.py` (event fields), `src/logpose/store/base.py` (outcome `"blocked"`)
- Test: `tests/test_guardrails_agent.py`

**Interfaces:**
- **Consumes:** Task 1's types and guards; Plan 1's `RunRecord` and session handling.
- **Produces:**
  - `Agent(..., input_guardrails=(), guardrail_mode="blocking")`, validated (each guard callable, mode in the allowed set).
  - `RunResult.guardrail: GuardrailResult | None = None`, added as the last field.
  - `RuntimeEvent` gains optional `guard_name`, `guard_allowed`, `guard_category` and `guard_scores` (a JSON string of the scores).
  - The runner `async def _run_guardrails(inp: GuardrailInput) -> tuple[GuardrailResult | None, Usage]` implements the two tiers.

- [ ] **Step 1: Failing tests:**
  - `test_allowed_run_unchanged`: an allowing guard leaves the request and result identical to a run with no guards, except usage.
  - `test_block_streams_placeholder_and_skips_model`: the `FakeProvider` call count is 0; the events are exactly `[TextDelta(reply), RunEnd]`; `result.text == reply`; `stop_reason == "refusal"`; `result.guardrail.category == "off_topic"`.
  - `test_blocked_prompt_not_in_history`: Review Focus 1.
  - `test_sync_guard_short_circuits_async`: Review Focus 3, with a pattern block plus an async guard that records whether it was called; the async guard is not called.
  - `test_first_async_block_wins_and_others_cancelled`.
  - `test_guard_exception_blocks_with_error_reply`.
  - `test_output_type_block_raises_guardrail_tripped`.
  - `test_session_records_blocked_outcome`: with Plan 1 sessions, a `logpose_runs` row has `outcome="blocked"`, and the session messages are unchanged.
  - `test_guardrail_event_has_no_text`: the event fields contain no prompt substring.
  - `test_deadline_during_guard_is_run_timeout`: with a slow guard and `run_timeout=0.2`, `RunTimeoutError` is raised.
  - `test_guard_usage_added_to_result`.
  - `test_no_prompt_skips_guards`: `run(None, conversation=conv)` calls no guard.
- [ ] **Step 2: Run and confirm they fail.**
- [ ] **Step 3: Implement.** Run the guards in `_loop` before `history.append(prompt)`, inside `_run_budget`.
- [ ] **Step 4: Run the gates.** Run: `uv run pytest -q && uv run ruff check && uv run mypy src/`. Expected: clean.
- [ ] **Step 5: Commit** with `git commit -m "feat(agent): blocking input guardrails with placeholder replies"`.

### Task 3: Parallel mode

**Files:**
- Modify: `src/logpose/agent.py`
- Test: `tests/test_guardrails_agent.py`

**Interfaces (produces):** `guardrail_mode="parallel"`:
- the prompt is appended tentatively;
- the guards start as a task alongside the first `_provider_turn`;
- deltas are buffered until the guard task resolves;
- before executing any tools, logpose awaits the guard task;
- **on allow:** the buffered deltas are flushed in order and the run continues;
- **on block:** the provider turn is cancelled, the tentatively appended prompt and anything appended after it are removed from history, and the result is the same as a blocking-mode block.

- [ ] **Step 1: Failing tests:**
  - `test_parallel_allow_flushes_buffered_text_in_order`: the guard sleeps 0.1 s and the provider streams `"a","b","c"` immediately; events start with `"a","b","c"` only after the guard resolves; the final text is `"abc"`.
  - `test_parallel_block_cancels_provider_and_restores_history`.
  - `test_parallel_tool_waits_for_guard`: Review Focus 2. The provider requests a tool instantly, the guard blocks after 0.1 s, and the tool's handler is never called.
  - `test_parallel_block_emits_only_placeholder`: no main-model text reaches the consumer.
- [ ] **Step 2: Run and confirm they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the gates.** Run: `uv run pytest -q && uv run ruff check && uv run mypy src/`. Expected: clean.
- [ ] **Step 5: Commit** with `git commit -m "feat(agent): parallel guardrail mode with buffered output"`.

### Task 4: Sync facade, exports, docs

**Files:**
- Modify: `src/logpose/__init__.py`, `README.md`, `CHANGELOG.md`
- Test: `tests/test_public_api.py`, `tests/test_sync.py`

- [ ] **Step 1: Failing tests.**
  - Add the exports to the public-API test: `GuardrailInput`, `GuardrailVerdict`, `GuardrailResult`, `GuardrailTripped`, `InputGuardrail`, `pattern_guard`, `openai_decision_guard`.
  - `test_run_sync_blocked_returns_placeholder`.
- [ ] **Step 2: Run and confirm they fail.**
- [ ] **Step 3: Implement** the exports. The sync facade needs no new keywords, since guards are configured on the agent.
- [ ] **Step 4: Write the docs.**
  - **README "Input guardrails" section:**
    - the setup example;
    - the two-tier order;
    - blocking versus parallel, and the parallel trade-off;
    - what a block returns;
    - threshold tuning ("start at 0.5, review `guardrail.decided` scores from real traffic");
    - the OpenAI-only note for the Decisions guard;
    - **limits:** input guardrails don't see injection inside tool outputs or documents. Pair them with read-only tools, and `on_tool_call` for actions.
  - **CHANGELOG:** Added entries.
- [ ] **Step 5: Run the gates.** Run: `uv run pytest -q && uv run ruff check && uv run mypy src/`. Expected: clean.
- [ ] **Step 6: Commit** with `git commit -m "docs: input guardrails"`.
