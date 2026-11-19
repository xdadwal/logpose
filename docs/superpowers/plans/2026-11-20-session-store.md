# Session Store Implementation Plan (Plan 1 of 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** logpose automatically persists sessions, lossless message history, and per-run usage in SQLite, with a per-session lease, behind a `SessionStore` interface that MySQL and Postgres can implement later.

**Architecture:**
- `logpose/store/base.py` defines the database-free interface and data types.
- `logpose/store/sqlite.py` implements it on `aiosqlite`.
- `logpose/store/manager.py` lazily creates one store per database path per process.
- `Agent` uses the store only when sessions are used.

**Tech Stack:** Python 3.10+, aiosqlite, pydantic v2, pytest (asyncio auto).

**Spec:** `docs/superpowers/specs/2026-11-20-session-store-design.md`

## Global Constraints

- **Dependencies:** `aiosqlite>=0.20` becomes a core dependency. `import logpose` must not import `aiosqlite`; import it lazily on first session use.
- **No SQL outside `logpose/store/sqlite.py`.** No public `store=` parameter anywhere.
- **Database path:** `LOGPOSE_DB_PATH` (blank counts as unset); default `logpose.db` in the current working directory; create the parent directory if missing.
- **Portable schema:** column types TEXT, INTEGER and NULL only; JSON stored as TEXT; times as integer Unix milliseconds (UTC); tables prefixed `logpose_`; deletes cascade in code, not via `ON DELETE CASCADE`.
- **Defaults:** `idle_ttl` 30 days, `lease_ttl` 60 s (renewed every `lease_ttl / 3`), `busy_timeout_ms` 5000, `session_wait` 0.0.
- **Session ID format:** `session_id` must match `[A-Za-z0-9._-]{1,128}`. Anything else raises `LogposeError`.
- **Runs without sessions are unchanged** and never touch disk.
- **Gates:** `uv run pytest -q`, `uv run ruff check`, `uv run mypy src/`.

## Review Focus

1. **Two processes take the lease on one session** (two `SQLiteStore` instances on one file). Exactly one wins; the other gets `SessionBusyError`. Test owned by Task 3.
2. **A lease is lost mid-run** (another holder took it after expiry). The stale holder's messages are not written, its run record is written, and `SessionBusyError` reaches the caller. Test owned by Task 4.
3. **The client disconnects mid-stream.** Partial history persists, the run record has `outcome="cancelled"` with real token counts, and the lease is released (a new `agent.session()` succeeds immediately). Test owned by Task 4.
4. **A database written by a newer schema version is opened.** `LogposeError` names both versions, and no data is touched. Test owned by Task 2.
5. **Stored history replays exactly:** thinking blocks with signatures, `raw` and `redacted_thinking` blocks, and tool call and result pairs reload equal (`==`) to what was appended. Test owned by Task 2.

---

### Task 1: Foundations (usage, conversation JSON, errors, dependency)

**Files:**
- Modify: `src/logpose/messages.py` (`Usage`), `src/logpose/providers/openai_compat.py` (`_map_usage`), `src/logpose/agent.py` (`Conversation`), `src/logpose/errors.py`, `src/logpose/__init__.py`, `pyproject.toml`, `uv.lock`
- Test: `tests/test_models.py`, `tests/test_provider_openai_compat.py`, `tests/test_agent.py`, `tests/test_public_api.py`

**Interfaces (produces):**
- `Usage.reasoning_tokens: int = 0`, included in `__add__`.
- `_map_usage` reads `completion_tokens_details.reasoning_tokens` when present, defaulting to 0.
- `Conversation.to_json() -> str` produces `{"version": 1, "messages": [...]}`.
- `Conversation.from_json(text: str) -> Conversation` (classmethod). An unknown `version` raises `LogposeError`.
- `class SessionBusyError(LogposeError)`, exported from `logpose` and `logpose.errors.__all__`.

- [ ] **Step 1: Failing tests.**
  - `test_usage_adds_reasoning_tokens`: `(Usage(reasoning_tokens=2) + Usage(reasoning_tokens=3)).reasoning_tokens == 5`.
  - `test_reasoning_tokens_mapped_from_completion_details`: a usage chunk with `completion_tokens_details={"reasoning_tokens": 7}` gives `done.usage.reasoning_tokens == 7`.
  - `test_conversation_json_round_trip_is_lossless`: messages with thinking (with signature), text, tool use, tool result, and a `RawBlock` reload equal via `from_json(to_json())`.
  - `test_conversation_from_json_rejects_unknown_version`: `'{"version": 99, "messages": []}'` raises `LogposeError` matching `"version"`.
  - Add `SessionBusyError` to the public-API expected exports and owners map.
- [ ] **Step 2: Run and confirm they fail.** Run: `uv run pytest tests/test_models.py tests/test_provider_openai_compat.py tests/test_agent.py tests/test_public_api.py -q`. Expected: the new tests fail.
- [ ] **Step 3: Implement.** Add `"aiosqlite>=0.20"` to `dependencies`, then run `uv lock`.
- [ ] **Step 4: Run the gates.** Run: `uv run pytest -q && uv run ruff check && uv run mypy src/`. Expected: clean.
- [ ] **Step 5: Commit** with `git commit -m "feat: reasoning token usage, Conversation JSON, SessionBusyError, aiosqlite dependency"`.

### Task 2: Store interface and SQLite schema, messages and runs

**Files:**
- Create: `src/logpose/store/__init__.py`, `src/logpose/store/base.py`, `src/logpose/store/sqlite.py`
- Test: `tests/test_store_sqlite.py`

**Interfaces (produces):**
- `base.py` (no database imports):
  - `RunRecord`: frozen dataclass with the spec's fields, all `int` except `run_id`, `session_id`, `model`, `provider`, `outcome` (`Literal["completed", "max_iterations", "run_timeout", "cancelled", "error"]`) and `error_type: str | None`.
  - `SessionHandle`: class with `id: str`, `conversation: Conversation`, `metadata: dict[str, str]`, `created: bool`, `lost: bool`.
  - `class SessionStore(Protocol)` with the methods from the spec plus `session(...)`.
  - `validate_session_id(session_id: str) -> str`.
- `sqlite.py`:
  - `SQLiteStore(path, *, idle_ttl=30 * 86400.0, lease_ttl=60.0, busy_timeout_ms=5000)`.
  - Async methods: `open()`, `close()`, `load_messages(sid) -> list[Message]`, `append_messages(sid, messages: Sequence[Message], *, start_seq: int) -> int` (returns the next seq), `record_run(record)`, `delete_session(sid)`, `purge_expired(now_ms=None) -> int`.
  - Async context-manager support.
  - Schema version 1 exactly as the spec describes; `PRAGMA journal_mode=WAL` and `busy_timeout` on open; a versioned migration list.
- This task leaves `session(...)` unimplemented; it raises `NotImplementedError` until Task 3.

- [ ] **Step 1: Failing tests** in `tests/test_store_sqlite.py` (each uses `tmp_path / "t.db"`):
  - `test_open_creates_schema_version_1`: after `open()`, `SELECT version FROM logpose_schema` returns `1`, and all four tables exist.
  - `test_open_is_idempotent`: opening twice raises no error and the version stays 1.
  - `test_newer_schema_version_is_refused`: set the version to 99, then `open()` raises `LogposeError` matching `"99"` and `"1"`.
  - `test_append_and_load_round_trip_all_blocks`: Review Focus 5.
  - `test_append_continues_seq`: two appends give seqs 0..n-1 then n..m, and the order is preserved.
  - `test_record_run_persists_all_fields`: a row read back matches the `RunRecord`.
  - `test_delete_session_removes_messages_and_runs`.
  - `test_purge_expired_counts_and_deletes`: with two sessions, one past `expires_at`, `purge_expired()` returns `1` and only the expired session's rows are gone.
  - `test_validate_session_id` (parametrized): `"abc-1.2_x"` is accepted; `""`, `"a b"`, `"x" * 129` and `"../x"` each raise `LogposeError`.
- [ ] **Step 2: Run and confirm they fail.** Run: `uv run pytest tests/test_store_sqlite.py -q`. Expected: FAIL (module missing).
- [ ] **Step 3: Implement** `base.py` and `sqlite.py` per the Interfaces. Use one `aiosqlite` connection per store, serialized with an `asyncio.Lock`.
- [ ] **Step 4: Run and confirm they pass.** Run: `uv run pytest -q && uv run ruff check && uv run mypy src/`. Expected: clean.
- [ ] **Step 5: Commit** with `git commit -m "feat(store): SessionStore interface and SQLite schema, messages, runs, purge"`.

### Task 3: Session lease and lifecycle

**Files:**
- Modify: `src/logpose/store/sqlite.py`
- Test: `tests/test_store_sqlite.py`

**Interfaces (produces):**
- `SQLiteStore.session(session_id, *, wait=0.0, metadata=None) -> AsyncContextManager[SessionHandle]`.
- **On enter:** validate the ID. In one `BEGIN IMMEDIATE` transaction:
  - treat a session past `expires_at` as gone (delete its rows);
  - create the session if missing (`created=True`, metadata set) or merge the given metadata;
  - take the lease if it is free or expired, otherwise poll every 50 ms up to `wait`, then raise `SessionBusyError`.

  Then load the messages into `handle.conversation`, remember the loaded length and next seq, and start a renewal task every `lease_ttl / 3`.
- **Renewal** extends `lease_expires_at` only when the token still matches; otherwise it sets `handle.lost = True` and stops.
- **On exit (normal or exception):** stop renewal. If the token matches, append the new messages, bump `version`, `updated_at` and `expires_at = now + idle_ttl`, and clear the lease. If the token doesn't match or `lost` is set, write nothing and raise `SessionBusyError` (chained, if exiting with another exception).

- [ ] **Step 1: Failing tests:**
  - `test_new_session_is_created_with_metadata`: `created is True`, and the metadata is persisted.
  - `test_existing_session_loads_history_and_merges_metadata`: messages appended in a first session block are loaded by a second; given metadata keys overwrite.
  - `test_messages_added_in_block_are_persisted_on_exit`.
  - `test_messages_persist_when_block_raises`: an exception inside the block still persists the appended messages and re-raises.
  - `test_second_holder_is_rejected_immediately`: Review Focus 1, using two `SQLiteStore`s on one file. The second `session()` raises `SessionBusyError` with `wait=0`.
  - `test_wait_succeeds_after_release`: the second holder, with `wait=1.0`, enters after the first exits (around 0.2 s).
  - `test_expired_lease_can_be_taken`: `lease_ttl=0.2`, and the first holder's renewal is disabled (monkeypatch). The second holder enters after 0.3 s.
  - `test_stale_holder_save_is_refused`: after the above, the first holder's exit raises `SessionBusyError`, and its messages are not stored.
  - `test_renewal_keeps_long_holder_alive`: `lease_ttl=0.3`, the holder sleeps 1.0 s inside the block, and the second store is still rejected at 0.8 s.
  - `test_expired_session_starts_fresh_on_access`: past `expires_at`, `session()` gives `created=True` and empty history.
- [ ] **Step 2: Run and confirm they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the gates.** Run: `uv run pytest -q && uv run ruff check && uv run mypy src/`. Expected: clean.
- [ ] **Step 5: Commit** with `git commit -m "feat(store): session lease with renewal, stale-save refusal, expiry on access"`.

### Task 4: Automatic management and Agent integration

**Files:**
- Create: `src/logpose/store/manager.py`
- Modify: `src/logpose/agent.py`
- Test: `tests/test_sessions.py`

**Interfaces:**
- **Consumes:** the Task 2 and 3 store APIs.
- **Produces:**
  - `manager.get_store() -> SessionStore`. Async; resolves the path from `LOGPOSE_DB_PATH` (or `logpose.db`); one opened store **per (absolute path, running event loop)**, because an aiosqlite connection is bound to its loop and the sync facade runs a private loop. Creation is guarded per key; `aiosqlite` is imported lazily. `Agent.aclose()` closes the stores this agent opened.
  - `manager.reset_stores_for_tests()`, which is private.
  - `Agent(..., session_wait: float = 0.0)`.
  - `Agent.session(session_id, *, metadata=None) -> AsyncContextManager[SessionHandle]`.
  - `run`/`stream(..., session_id: str | None = None, session_metadata: Mapping[str, str] | None = None, session: SessionHandle | None = None)`.
  - `Agent.purge_expired_sessions() -> int` and `Agent.delete_session(session_id) -> None`, both async.
- **Rules:**
  - Combining any two of `session_id`, `session` and `conversation` raises `LogposeError`.
  - A run on a session uses `handle.conversation` and writes one `RunRecord` per run with real usage and iterations on every outcome, through a run-scoped accumulator that `_loop` updates.
  - Outcomes map as: `MaxIterationsError` → `max_iterations`; `RunTimeoutError` → `run_timeout`; `CancelledError` or `GeneratorExit` → `cancelled`; any other exception → `error`, with `error_type` set to the type name.
  - Cancellation persists and releases inside a shield bounded to 5 s.
  - The `session_id` convenience path takes the lease when iteration starts.

- [ ] **Step 1: Failing tests** in `tests/test_sessions.py`. An autouse fixture sets `LOGPOSE_DB_PATH` to `tmp_path/"s.db"` and calls `reset_stores_for_tests()`.
  - `test_no_session_never_touches_disk`: run without a session; the database file does not exist and `aiosqlite` is not in `sys.modules`. Run this one in a subprocess.
  - `test_session_id_persists_history_across_runs`: two `run(..., session_id="s1")` calls; the second request carries the first exchange.
  - `test_explicit_session_raises_busy_before_streaming`: inside one `agent.session("s1")` block, a second `agent.session("s1")` raises `SessionBusyError` before any event is consumed.
  - `test_run_record_written_with_usage`: after a run, one `logpose_runs` row with `outcome="completed"` and `input_tokens` equal to the summed usage.
  - `test_error_and_timeout_outcomes_recorded`: provider failure gives `"error"` (with `error_type`); `run_timeout` gives `"run_timeout"`; both rows carry non-zero tokens when a turn completed first.
  - `test_cancel_mid_stream_persists_and_releases`: Review Focus 3, by breaking out of `stream()` after the first `TextDelta`.
  - `test_lost_lease_writes_run_but_not_messages`: Review Focus 2, by forcing `handle.lost = True` during the run.
  - `test_session_argument_conflicts` (parametrized pairs) each raise `LogposeError`.
  - `test_purge_and_delete_via_agent`.
  - `test_db_path_from_env_and_default`: a blank env var gives `Path.cwd() / "logpose.db"`.
- [ ] **Step 2: Run and confirm they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the gates.** Run: `uv run pytest -q && uv run ruff check && uv run mypy src/`. Expected: clean.
- [ ] **Step 5: Commit** with `git commit -m "feat(agent): automatic SQLite sessions with run records"`.

### Task 5: Sync facade, exports, docs

**Files:**
- Modify: `src/logpose/sync.py`, `src/logpose/__init__.py`, `README.md`, `CHANGELOG.md`
- Test: `tests/test_sync.py`, `tests/test_public_api.py`

**Interfaces (produces):**
- `run_sync`, `stream_sync`, `SyncAgent.run` and `SyncAgent.stream` accept `session_id`, `session_metadata` and `session`, forwarded only when passed (reuse the `_output_kwargs` pattern).
- `SyncAgent.session(...)` is a sync context manager over the agent's private loop.
- `SessionHandle` is exported from `logpose`.

- [ ] **Step 1: Failing tests:**
  - `test_run_sync_with_session_id_persists`.
  - `test_sync_agent_session_context`.
  - Public-API additions for `SessionHandle`.
- [ ] **Step 2: Run and confirm they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Write the docs.**
  - **README "Sessions" section:**
    - the explicit `async with agent.session(sid)` plus `stream(..., session=h)` pattern, with a FastAPI-style 409 example;
    - the `session_id=` convenience form;
    - `LOGPOSE_DB_PATH`;
    - expiry and `purge_expired_sessions()`;
    - the single-host note (several hosts need MySQL or Postgres later).
  - **CHANGELOG:** Added entries; a Changed entry for the `aiosqlite` dependency.
- [ ] **Step 5: Run the gates.** Run: `uv run pytest -q && uv run ruff check && uv run mypy src/`. Expected: clean.
- [ ] **Step 6: Commit** with `git commit -m "feat: sessions in sync facade, exports, docs"`.
