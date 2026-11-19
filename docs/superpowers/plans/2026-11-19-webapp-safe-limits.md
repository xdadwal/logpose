# Web-App Safe Limits Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bound every value that could run without limit with web-app-safe defaults, and add a whole-run deadline (`Agent(run_timeout=320)`). `None` stays available as an explicit "unlimited".

**Architecture:** Defaults change in place. `run_timeout` is enforced in `Agent._loop` / `Agent._provider_turn` from one monotonic deadline computed at run start. Each provider turn is bounded by `min(provider_turn_timeout, remaining)`, and tool execution and request preparation by `remaining`. An internal expiry becomes `RunTimeoutError` in `_loop`, where the partial history is available.

**Tech Stack:** Python 3.10+, httpx, pytest (asyncio auto), ruff, mypy.

**Spec:** `docs/superpowers/specs/2026-11-19-neutral-provider-defaults-design.md` (section "Amendment: web-app safe limits")

## Global Constraints

- Values are copied from the spec amendment table. `None` means unlimited wherever a limit is optional.
- A run-deadline expiry is never retried and never reported as a `ProviderError`.
- Gates: `uv run pytest -q`, `uv run ruff check`, `uv run mypy src/`.

## Review Focus

1. The run deadline expires during a provider retry backoff sleep. It must raise `RunTimeoutError`, not sleep past the deadline. Test owned by Task 1.
2. The run deadline is the binding limit while a turn is streaming (`provider_turn_timeout` > remaining). It must raise `RunTimeoutError`, not a retryable `turn_timeout` `ProviderError`. Test owned by Task 1.
3. `run_timeout=None` with `provider_turn_timeout=None` must behave exactly as before (no deadline at all). Test owned by Task 1.
4. A `list_dir` page cut by the character budget must give a continuation that resumes at the first omitted entry, with no entry skipped or repeated. Test owned by Task 3.
5. `read_file` with `max_lines=None` on a file larger than one page must return a cursor continuation, not the whole file. Test owned by Task 3.

---

### Task 1: Agent limits and whole-run deadline

**Files:**
- Modify: `src/logpose/agent.py` (constants, `__init__` signature, validation and docs, `_loop`, `_provider_turn`)
- Modify: `src/logpose/errors.py` (add `RunTimeoutError`), `src/logpose/__init__.py` (export it), `src/logpose/retry.py` (defaults)
- Test: `tests/test_agent.py`, `tests/test_retry.py`, `tests/test_public_api.py`

**Interfaces:**
- Produces:
  - Constants:

    | Constant | Value | Notes |
    |---|---|---|
    | `DEFAULT_MAX_ITERATIONS` | `10` | |
    | `DEFAULT_MAX_TOKENS` | `8192` | |
    | `DEFAULT_TOOL_TIMEOUT` | `30.0` | |
    | `DEFAULT_PROVIDER_TURN_TIMEOUT` | `120.0` | re-added, exported in `agent.__all__` |
    | `DEFAULT_RUN_TIMEOUT` | `320.0` | exported |

  - `Agent(..., provider_turn_timeout: float | None = DEFAULT_PROVIDER_TURN_TIMEOUT, run_timeout: float | None = DEFAULT_RUN_TIMEOUT, ...)`. Both are validated like `tool_timeout`: a positive number or `None`, with `bool` rejected.
  - `class RunTimeoutError(LogposeError)` with `__init__(self, message: str, messages: list[Message], run_timeout: float)` and attributes `message`, `messages`, `run_timeout`. Exported from `logpose`.
  - A runtime event named `"run.timed_out"`, carrying `duration_seconds` and `iterations`.
  - `RetryPolicy` defaults: `max_attempts=2`, `max_delay=4.0`, `max_retry_after=10.0`. The others are unchanged.

- [ ] **Step 1: Write the failing tests.**
  - `tests/test_agent.py`:
    - `test_web_app_defaults`: `Agent(FakeProvider(), model="m")` has `max_iterations == 10`, `tool_timeout == 30`, `provider_turn_timeout == 120`, `run_timeout == 320`. A request has `max_tokens == 8192`.
    - `test_run_timeout_raises_with_partial_history`: a provider that sleeps 0.2 s per turn and always requests a tool, with `run_timeout=0.3` and `provider_turn_timeout=None`, raises `RunTimeoutError`. `exc.run_timeout == 0.3` and `exc.messages` is non-empty.
    - `test_run_deadline_binding_during_streaming_is_not_a_provider_error`: a provider that streams one delta and then sleeps 1 s, with `provider_turn_timeout=5` and `run_timeout=0.1`, raises `RunTimeoutError`, and the provider was called exactly once (no retry).
    - `test_run_deadline_cuts_retry_backoff`: a provider that fails with a retryable `ProviderError(retry_after=5)` before any delta, with `retry_policy=RetryPolicy(max_attempts=3, max_retry_after=5)` and `run_timeout=0.2`, raises `RunTimeoutError` within 1 s of wall time.
    - `test_run_deadline_bounds_tool_execution`: an async tool that sleeps 5 s, with `tool_timeout=None` and `run_timeout=0.2`, raises `RunTimeoutError`.
    - `test_unlimited_run_and_turn_are_allowed`: `run_timeout=None, provider_turn_timeout=None` completes a normal scripted run, and `_turn_timeout()` is `None`.
    - `test_run_timeout_validation`: `0`, `-1`, `True`, and `"x"` each raise `LogposeError` matching `"run_timeout"`.
    - `test_run_timed_out_event_is_emitted`: an observer records `"run.timed_out"` for the timeout case.
  - Update tests that asserted the old defaults (`max_iterations` 25, the 32000 token default, `_turn_timeout() is None` by default, the 300 s tool timeout).
  - `tests/test_retry.py`: `test_default_policy_is_web_app_safe` asserts `RetryPolicy()` equals `RetryPolicy(max_attempts=2, initial_delay=0.5, backoff_multiplier=2.0, max_delay=4.0, jitter=0.2, max_retry_after=10.0)`. Update any test that asserted the old defaults.
  - `tests/test_public_api.py`: `"RunTimeoutError"` is in `logpose.__all__`.

- [ ] **Step 2: Run the tests and confirm they fail.**
  Run: `uv run pytest tests/test_agent.py tests/test_retry.py tests/test_public_api.py -q`
  Expected: the new tests FAIL.

- [ ] **Step 3: Implement.**
  - `_loop` computes `run_deadline = started_at + run_timeout` (or `None`). A private helper `_remaining(run_deadline) -> float | None` returns the seconds left.
  - At the top of each iteration, if the time remaining is ≤ 0, raise `RunTimeoutError`. Wrap `_prepare_request` and `_execute` in `async with _deadline(remaining)`, converting `TimeoutError` to `RunTimeoutError`.
  - Pass `run_deadline` to `_provider_turn`:
    - each attempt's deadline is `min(turn_timeout, remaining)`;
    - on `TimeoutError`, if the run deadline was the binding one, raise a private `_RunExpired` (not retried), otherwise keep the existing `turn_timeout` path;
    - before any retry sleep, if `delay >= remaining`, raise `_RunExpired`.
  - `_loop` converts `_RunExpired` and the `TimeoutError`s above into `RunTimeoutError(f"Run exceeded run_timeout={run_timeout}s.", list(history), run_timeout)` and emits `run.timed_out` first.
  - Update the docstrings for every changed default.

- [ ] **Step 4: Run the tests and confirm they pass.**
  Run: `uv run pytest -q`
  Expected: PASS.

- [ ] **Step 5: Commit.**
  ```bash
  git add src/logpose tests
  git commit -m "feat(agent)!: web-app safe limits and run_timeout deadline"
  ```

### Task 2: Provider HTTP timeout default

**Files:**
- Modify: `src/logpose/providers/openai_compat.py`
- Test: `tests/test_provider_openai_compat.py`

**Interfaces:**
- Produces:
  - `DEFAULT_HTTP_TIMEOUT = httpx.Timeout(connect=5.0, read=60.0, write=10.0, pool=5.0)`, exported.
  - `OpenAICompatProvider(timeout: float | httpx.Timeout | None = DEFAULT_HTTP_TIMEOUT)`. An explicit `None` means `httpx.Timeout(None)`.

- [ ] **Step 1: Write the tests.**
  - `test_default_http_timeout_is_web_app_safe`: `OpenAICompatProvider(base_url=BASE)._http().timeout == httpx.Timeout(connect=5.0, read=60.0, write=10.0, pool=5.0)`.
  - Keep `test_timeout_none_disables_http_timeouts`, but pass `timeout=None` explicitly.
- [ ] **Step 2: Run them and confirm they fail.** Run: `uv run pytest tests/test_provider_openai_compat.py -q`. Expected: the default-timeout test FAILS.
- [ ] **Step 3: Implement**, including the docstring update.
- [ ] **Step 4: Run and confirm they pass.** Run: `uv run pytest -q`. Expected: PASS.
- [ ] **Step 5: Commit.**
  ```bash
  git add src/logpose/providers/openai_compat.py tests/test_provider_openai_compat.py
  git commit -m "feat(providers): web-app safe default HTTP timeouts"
  ```

### Task 3: Toolkit limits

**Files:**
- Modify: `src/logpose/shell.py`, `src/logpose/workspace.py`, `src/logpose/web.py`
- Test: `tests/test_shell.py`, `tests/test_workspace.py`, `tests/test_web.py`, `tests/test_web_search_backends.py`

**Interfaces:**
- Produces:
  - `shell`:

    | Constant | Value |
    |---|---|
    | `_DEFAULT_TIMEOUT` | `30.0` |
    | `_DEFAULT_CAPTURE_LIMIT` | `1024 * 1024` |
    | `_DEFAULT_OUTPUT_CAP` | `16_000` |

  - `workspace`:
    - `_MAX_PAGE_CHARS = 16_000`;
    - `read_file_page(..., max_lines: int | None = None)`, where `None` means read until the page limit, and any `max_lines >= 1` is accepted with no upper bound;
    - `list_directory(..., max_entries: int | None = None)`, where `None` means no entry cap, with output capped at `_MAX_PAGE_CHARS` and a continuation `after=` set to the last included entry;
    - `make_grep_tool(output_char_cap: int | None = 16_000)`;
    - the model-facing tool signatures change to match: `read_file(path, start_line=1, max_lines: int | None = None, cursor=0)` and `list_dir(path=".", after="", max_entries: int | None = None)`.
  - `web`:
    - `_REQUEST_TIMEOUT = 10`, `_SEARCH_TIMEOUT = 10`;
    - `fetch_readable_text(..., max_bytes=1024 * 1024, output_char_cap: int | None = 16_000)`;
    - `make_search_tool(output_char_cap=8_000)`.

- [ ] **Step 1: Write the tests.**
  - `test_shell_defaults_are_web_app_safe`: run a command printing 20,000 characters with the default tool. The result contains `...[truncated]`, and the returned text is ≤ 16,000 + 20 characters.
  - `test_read_file_has_no_line_cap`: `read_file_page(path, base_path=root, max_lines=500)` on a 300-line file returns all 300 lines.
  - `test_read_file_page_is_char_bounded`: a 40,000-character file with `max_lines=None` returns ≤ 16,000 characters plus a `cursor=` continuation. Following the cursors rebuilds the file exactly.
  - `test_list_dir_has_no_entry_cap`: 250 short entries are all listed with `max_entries=None`.
  - `test_list_dir_char_budget_continues_without_gaps`: 3,000 entries with 20-character names. Following the `after=` continuations lists every entry exactly once.
  - `test_grep_output_is_capped_by_default`: a match-heavy tree produces output ≤ 16,000 + 20 characters.
  - `test_fetch_output_is_capped_by_default`: an injected `get` returning a 50,000-character page gives output that contains `[truncated]` (or the module's existing truncation marker) and is ≤ 16,000 + 200 characters.
  - `test_search_tool_output_cap_default`: a backend returning 20,000 characters produces output ending with `...[truncated]`.
  - Update existing tests that assert the old 200 / 1..200 validation messages.
- [ ] **Step 2: Run them and confirm they fail.** Run: `uv run pytest tests/test_shell.py tests/test_workspace.py tests/test_web.py tests/test_web_search_backends.py -q`. Expected: the new tests FAIL.
- [ ] **Step 3: Implement.** In `list_directory`, add entries in sorted order while the running length is ≤ the cap. If any remain, add the continuation with `after=` set to the last included name.
- [ ] **Step 4: Run and confirm they pass.** Run: `uv run pytest -q && uv run ruff check && uv run mypy src/`. Expected: all clean.
- [ ] **Step 5: Commit.**
  ```bash
  git add src/logpose tests
  git commit -m "feat(tools): web-app safe toolkit limits; uncap read_file lines and list_dir entries"
  ```

### Task 4: Docs and changelog

**Files:**
- Modify: `README.md`, `CHANGELOG.md`, `examples/local_docker.py`

- [ ] **Step 1: README.**
  - Replace "Provider deadlines" with a "Limits and deadlines" section holding a table of the spec amendment's values and a note that each accepts `None` for unlimited.
  - Change "Output limit" to 8192.
  - Change "Run steps" to 10.
  - Add `run_timeout` with a `RunTimeoutError` example.
  - Change the 300-second tool deadline to 30 s.
  - Add `RunTimeoutError` to the errors table.
- [ ] **Step 2: CHANGELOG `[Unreleased]`.** Under `### Changed`, add a **Breaking** bullet for the new defaults, replacing the 32000 / no-deadline bullets. Under `### Added`, add `run_timeout` and `RunTimeoutError`.
- [ ] **Step 3: Example.** `examples/local_docker.py` keeps `max_tokens=4096`, plus a comment that local models may need a larger `provider_turn_timeout` / `run_timeout`.
- [ ] **Step 4: Verify.**
  - Run: `uv run pytest -q && uv run ruff check && uv run mypy src/`. Expected: clean.
  - Run: `git grep -n -E '32000|16000-token|300-second|15 minutes|no provider deadlines' -- README.md src`. Expected: no output.
- [ ] **Step 5: Commit.**
  ```bash
  git add README.md CHANGELOG.md examples
  git commit -m "docs: document web-app safe limits and run_timeout"
  ```
