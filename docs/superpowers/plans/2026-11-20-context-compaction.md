# Context Compaction Implementation Plan (Plan 2 of 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A built-in, on-by-default compaction path:
- tool outputs from earlier runs are always sent as pointers;
- the model gets a `recall_tool_output` tool;
- at 85% of the input budget, older turns are summarized (cached, and persisted with sessions);
- a request rejected for context length is compacted and retried once.

**Architecture:**
- `logpose/compaction.py` holds the pure transforms (estimate, pointers, turn boundaries) and the recall tool factory.
- The `Agent` loop runs the built-in path in `_prepare_request` whenever `context_policy` is the default sentinel.
- Per-conversation state lives in a private slot on `Conversation`.
- Summaries persist in a new `logpose_summaries` table (schema version 2).

**Tech Stack:** Python 3.10+, pydantic v2, aiosqlite, pytest.

**Spec:** `docs/superpowers/specs/2026-11-20-context-compaction-design.md` (depends on Plan 1: `docs/superpowers/plans/2026-11-20-session-store.md`).

## Global Constraints

- **Defaults:**
  - `context_window=128_000`
  - `context_budget=32_000` (`None` means no cap)
  - `compaction_threshold=0.85`
  - `compaction_target=0.60`
  - `pointer_min_chars=1_000`
- **Input budget:** `input_budget = min(context_window − resolved max_tokens, context_budget)`. `≤ 0` raises `LogposeError` at construction.
- **Token estimate:** characters ÷ 4, multiplied by a per-conversation calibration ratio.
- **Pointer text, exactly:** `[tool output stored: <tool_use_id>, <N> chars. Call recall_tool_output("<tool_use_id>") to read it.]`
- **Summary message, exactly:** a user-role message whose text is `"[Summary of earlier conversation]\n" + summary`. The summary call uses `max_tokens` 2048 and no tools.
- **`context_policy`:**
  - the default sentinel means the built-in path;
  - an explicit function replaces it entirely (no pointers, recall, summary, reactive retry, or event);
  - `None` sends the full history.
- **Never change** stored history, the caller's `Conversation.messages`, or `RunResult.messages`.
- **Gates:** `uv run pytest -q`, `uv run ruff check`, `uv run mypy src/`.

## Review Focus

1. **A run with several tool rounds.** All of its own tool outputs stay full on every turn of that run; only outputs from earlier runs become pointers. Test owned by Task 1.
2. **A caller edits `conversation.messages` between runs** (deletes a message). The cached summary and cut point are discarded and recomputed, not applied to the wrong messages. Test owned by Task 2.
3. **The summary call fails** (provider error). If the uncompacted request fits the window, it is sent uncompacted and the event has `summarized=False`; otherwise `ContextOverflowError` is raised. Test owned by Task 2.
4. **The agent has an app tool named `recall_tool_output`.** `LogposeError` is raised at construction. Test owned by Task 1.
5. **The first user turn is never summarized,** even when the summary span would otherwise start at index 0. Test owned by Task 2.

---

### Task 1: Pointers, the recall tool, and settings

**Files:**
- Create: `src/logpose/compaction.py`
- Modify: `src/logpose/agent.py` (settings, sentinel, `_prepare_request`, tool injection, `Conversation._compaction` slot)
- Test: `tests/test_compaction.py`

**Interfaces (produces):**
- `estimate_tokens(messages: Sequence[Message], system: str | None, tools: Sequence[ToolSpec]) -> int` returns characters of the JSON-serialized pieces ÷ 4.
- `apply_pointers(messages: Sequence[Message], *, current_start: int, min_chars: int) -> tuple[list[Message], int]` returns new message copies and the number of pointers applied. It never mutates its input.
- `make_recall_tool(history: Callable[[], Sequence[Message]]) -> ToolDef` builds the tool named `recall_tool_output`, with parameters `call_id: str` and `offset: int = 0`. It returns up to 8,000 characters plus a continuation hint, or `"No stored tool output with id '<id>'."`.
- `Agent(..., context_window=128_000, context_budget=32_000, compaction_threshold=0.85, compaction_target=0.60, pointer_min_chars=1_000)`, validated:
  - integers must be positive;
  - `0 < target < threshold ≤ 1`.
- Agent attributes `_context_policy_is_default: bool` and `input_budget: int`.
- `_loop` records `current_start` (the history length when the run started, before the prompt is appended) and passes it to `_prepare_request`.
- The recall tool is appended to the agent's tools when the default path is active and `tools` is non-empty.

- [ ] **Step 1: Failing tests:**
  - `test_pointer_replaces_long_outputs_from_earlier_runs_only`: Review Focus 1, with `FakeProvider`. Run 1 has two tool rounds with 5,000-character outputs. Run 2's first request has pointers for both of run 1's outputs, matching the exact pointer text. Run 1's second request had both outputs in full.
  - `test_short_outputs_stay_inline`: a 500-character output from an earlier run is unchanged.
  - `test_pointer_keeps_pairing`: the pointed block keeps `tool_use_id` and `is_error`.
  - `test_stored_history_is_unchanged`: after run 2, `conv.messages` holds the full 5,000-character content.
  - `test_recall_tool_reads_full_output_and_pages`: `recall_tool_output(id)` returns the first 8,000 characters plus a hint; `offset=8000` returns the rest; an unknown ID returns the message.
  - `test_recall_tool_present_only_with_tools`: an agent with tools gets `recall_tool_output` in its request tool specs from the first turn; a tool-less agent doesn't.
  - `test_recall_name_collision_rejected`: Review Focus 4.
  - `test_custom_policy_disables_builtin` and `test_none_policy_sends_full_history`.
  - `test_input_budget_validation`: `context_window=10_000` with `max_tokens=16_384` raises `LogposeError`.
- [ ] **Step 2: Run and confirm they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the gates.** Run: `uv run pytest -q && uv run ruff check && uv run mypy src/`. Expected: clean.
- [ ] **Step 5: Commit** with `git commit -m "feat(compaction): tool output pointers, recall tool, budget settings"`.

### Task 2: Calibration and the summarization fallback

**Files:**
- Modify: `src/logpose/compaction.py`, `src/logpose/agent.py`, `src/logpose/errors.py`, `src/logpose/__init__.py`
- Test: `tests/test_compaction.py`

**Interfaces (produces):**
- `ContextOverflowError(LogposeError)` with attributes `estimated_tokens` and `input_budget`; exported.
- `turn_starts(messages) -> list[int]`: indices of caller-authored user turns (a user message whose content contains a `TextBlock` and which is not a tool-result message).
- `_CompactionState` on `Conversation._compaction` with fields `ratio: float`, `summary: str | None`, `cut_index: int`, `fingerprint: str`. The ratio is updated after each turn as `0.7 × old + 0.3 × (reported input tokens ÷ estimate)`, starting at 1.0.
- **Summary request:** system, then the first user turn, then the summary message, then the messages from `cut_index` on.
- **The span** grows from just after the first user turn, by whole turns, until the estimate is ≤ `compaction_target × input_budget`.
- **The summary call** uses the agent's provider and model with the retry policy, bounded by the run deadline. Its usage is added to the run totals.
- A runtime event `context.compacted`; `RuntimeEvent` gains optional `pointers_applied`, `summarized`, `estimated_tokens_before`, `estimated_tokens_after` and `input_budget`.

- [ ] **Step 1: Failing tests:**
  - `test_no_summary_below_threshold`: a small chat makes no extra call.
  - `test_summary_at_threshold`: `context_budget=2_000`, with history long enough to exceed 85%.
    - The provider's first call is the summary call (no tools, `max_tokens == 2048`).
    - The turn request contains the first user turn, a message starting `"[Summary of earlier conversation]\n"`, and only recent turns.
    - The estimate is ≤ 60% of the budget.
  - `test_summary_is_reused_until_threshold_again`: the next run makes no summary call and the request starts the same way.
  - `test_first_user_turn_never_summarized`: Review Focus 5.
  - `test_cut_never_splits_tool_pair`: the request is valid (each `tool_use` is followed by its result).
  - `test_edited_history_invalidates_cache`: Review Focus 2.
  - `test_summary_failure_sends_uncompacted_when_it_fits` and `test_summary_failure_overflows_when_it_does_not`: Review Focus 3.
  - `test_overflow_after_summary_raises`.
  - `test_calibration_ratio_updates_from_reported_usage`.
  - `test_compacted_event_emitted` with the expected fields.
  - `test_summary_usage_counted_in_run_result`.
- [ ] **Step 2: Run and confirm they fail.**
- [ ] **Step 3: Implement.** The summary instruction must ask the model to keep facts, decisions, IDs and open questions, and to add no new information.
- [ ] **Step 4: Run the gates.** Run: `uv run pytest -q && uv run ruff check && uv run mypy src/`. Expected: clean.
- [ ] **Step 5: Commit** with `git commit -m "feat(compaction): calibrated summarization fallback at 85% of budget"`.

### Task 3: Persisted summaries and the reactive retry

**Files:**
- Modify: `src/logpose/store/base.py`, `src/logpose/store/sqlite.py` (schema version 2 migration), `src/logpose/agent.py`
- Test: `tests/test_store_sqlite.py`, `tests/test_compaction.py`

**Interfaces (produces):**
- **Schema version 2:** table `logpose_summaries(session_id TEXT, upto_seq INTEGER, text TEXT, created_at INTEGER, PRIMARY KEY (session_id, upto_seq))`; the migration runs from version 1.
- **Store methods:**
  - `save_summary(sid, upto_seq: int, text: str)`;
  - `load_summary(sid) -> tuple[int, str] | None`, returning the latest summary;
  - `delete_session` and `purge_expired` also remove summaries.
- **Session handle:** `SessionHandle.seqs: list[int]` holds the seq of each loaded message, so a stored `upto_seq` maps to `cut_index`. A loaded summary seeds `_CompactionState` on enter; a new summary is saved on exit.
- **Reactive retry:** a provider error with status 400 whose message contains `context_length`, `maximum context length` or `context window` (case-insensitive) forces a summary and retries the turn once.

- [ ] **Step 1: Failing tests:**
  - `test_schema_migrates_v1_to_v2`: a version 1 database opens to version 2 and keeps its data.
  - `test_summary_persists_across_requests`: with a session, run A summarizes; in a new process-like reset (`reset_stores_for_tests`), run B's request reuses the same summary text with no summary call.
  - `test_reactive_context_length_retry`: the provider returns 400 `"This model's maximum context length is ..."` once, then succeeds. Exactly one summary call and one retry happen, and the run completes.
  - `test_second_context_length_rejection_propagates`.
  - `test_purge_removes_summaries`.
- [ ] **Step 2: Run and confirm they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the gates.** Run: `uv run pytest -q && uv run ruff check && uv run mypy src/`. Expected: clean.
- [ ] **Step 5: Commit** with `git commit -m "feat(compaction): persisted summaries and reactive context-length retry"`.

### Task 4: Docs

**Files:**
- Modify: `README.md`, `CHANGELOG.md`

- [ ] **Step 1: README "Long conversations" section.** Cover:
  - the pointer rule and the recall tool;
  - the budget formula and its defaults;
  - the 85% summary with the 60% target;
  - the pinned first turn;
  - that stored history is never changed;
  - how to tune or disable it (`context_policy=None`, or your own function, which replaces everything);
  - the `context.compacted` event;
  - `ContextOverflowError`.
- [ ] **Step 2: CHANGELOG.** Added entries; a Changed entry noting that compaction is now on by default.
- [ ] **Step 3: Run the gates.** Run: `uv run pytest -q && uv run ruff check && uv run mypy src/`. Expected: clean.
- [ ] **Step 4: Commit** with `git commit -m "docs: long-conversation compaction"`.
