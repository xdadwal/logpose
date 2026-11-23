# Storage Configuration and Read-only Session History Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `Agent(db_path=, session_ttl=)`, `export_traces(..., db_path=)`, and a lease-free `agent.load_session(session_id)`.

**Architecture:** the agent resolves its database path once and passes it to every `store.manager` call (the per-(path, loop) cache is unchanged); the session TTL travels with each write instead of living on the store; `load_session` reads committed rows through the store without a lease.

**Tech Stack:** Python 3.10+, aiosqlite, `uv`, pytest-asyncio (auto).

**Spec:** `docs/superpowers/specs/2026-11-23-storage-config-design.md`

## Global Constraints

- Path resolution: `db_path` argument (not blank) → `LOGPOSE_DB_PATH` (not blank) → `./logpose.db`; absolute with `~` expanded. An explicit `db_path` is fixed at construction; without it the env/default is resolved at each use, as before. Type `str | os.PathLike[str] | None`.
- `session_ttl: timedelta | float | None = None` (float = seconds); default 30 days; ≤ 0 or non-finite → `LogposeError`.
- TTL rule: every write by an agent (session creation and each save) stamps `expires_at = now + that agent's session_ttl`; expiry checks and purge read the stored `expires_at`.
- `load_session(session_id) -> list[Message]`: full stored history (no summary, not compacted); unknown or expired → `[]`; never deletes/resets/extends; no lease; validates the ID (`LogposeError`); opens the store lazily; uncommitted (in-progress) messages not visible.
- Backward compatible: without the new arguments, behaviour and `LOGPOSE_DB_PATH` are unchanged.
- Gates: `uv run --all-extras pytest -q`, `uv run ruff check`, `uv run --all-extras mypy src/`.

## Review Focus

1. Two agents with different `db_path`s in one process (and one event loop) use two separate stores and never cross-read sessions, traces or run IDs — test in Task 1.
2. Two agents on the same path with different `session_ttl`s: each write stamps its own TTL; no error, no shared state — test in Task 1.
3. `load_session` on an expired-but-unpurged session returns `[]` and the rows are still there afterwards (a later `purge_expired_sessions` removes them) — test in Task 2.
4. `load_session` while another task holds the session lease (mid-stream) returns the previously saved messages, not `SessionBusyError` — test in Task 2.
5. `export_traces(db_path=X)` reads X even when `LOGPOSE_DB_PATH` points elsewhere — test in Task 1.

---

### Task 1: `db_path` and `session_ttl`

**Files:**
- Modify: `src/logpose/store/manager.py` (`db_path(explicit=None)`, `get_store(*, owner=None, path=None)`, `store_opening(*, owner, path)`), `src/logpose/store/sqlite.py` (TTL per write: `session(..., ttl=...)` / `_try_acquire` / `_release` stamp the caller's TTL; default `idle_ttl` stays the fallback), `src/logpose/store/base.py` (Protocol signature for `session` gains `ttl: float | None = None`), `src/logpose/agent.py` (`__init__` args + validation, `self.db_path: Path`, `self.session_ttl: float`; `_store()`, `store_opening`, session paths, purge/delete use them), `src/logpose/tracing/export.py` (`db_path=None` kwarg), `src/logpose/sync.py` only if it resolves paths itself.
- Test: `tests/test_storage_config.py` (new)

**Interfaces:**
- Produces: `Agent(..., db_path: str | os.PathLike[str] | None = None, session_ttl: timedelta | float | None = None)`; attributes `Agent.db_path: pathlib.Path`, `Agent.session_ttl: float` (seconds). `manager.db_path(explicit: str | os.PathLike[str] | None = None) -> Path`; `manager.get_store(*, owner: object | None = None, path: Path | None = None) -> SessionStore` (None → `db_path()`). `SQLiteStore.session(session_id, *, wait=0.0, metadata=None, ttl: float | None = None)`. `export_traces(path, *, since=None, until=None, session_id=None, after=None, db_path=None) -> ExportResult`.

- [ ] **Step 1: Write failing tests** in `tests/test_storage_config.py` (use `tmp_path`, `monkeypatch.setenv/delenv("LOGPOSE_DB_PATH")`, `reset_stores_for_tests` fixture as existing session tests do):
  - `test_db_path_argument_wins_over_env`: env → A, `Agent(db_path=B)` → `agent.db_path == B.resolve()` and a session run creates B, not A.
  - `test_db_path_env_fallback_and_default`: no arg → env; blank arg/env → `cwd/logpose.db` (chdir to tmp_path).
  - `test_db_path_expands_user_and_is_absolute`.
  - `test_two_agents_two_paths_isolated`: same loop, sessions with the same ID on two paths hold different histories; `delete_session`/`purge_expired_sessions`/`purge_traces` on one leave the other intact.
  - `test_tracing_uses_agent_db_path`: traced run with `db_path=B` → spans readable from store at B (via `export_traces(out, db_path=B)`), none at env path.
  - `test_export_traces_db_path_overrides_env`.
  - `test_session_ttl_validation`: `0`, `-1`, `float("inf")`, `float("nan")`, `timedelta(0)` → `LogposeError`; `timedelta(days=7)` → `agent.session_ttl == 604800.0`; default `== 30*86400`.
  - `test_session_ttl_stamped_per_write`: `Agent(session_ttl=7 days)` → stored `expires_at - updated_at == 7 days` (read via sqlite3) after a run; a second agent on the same path with 1 day writing the same session later restamps it to 1 day; no error.
  - `test_short_ttl_expires`: `session_ttl=0.05` → after sleeping 0.1 s the next session starts empty; `purge_expired_sessions()` returns 1 before that.
  - `test_defaults_unchanged`: no new args → same path as `manager.db_path()` and 30-day stamping.
- [ ] **Step 2: Run** them — expect failures.
- [ ] **Step 3: Implement** per Interfaces. The store cache key stays `(str(path), loop)`. All agent storage entry points pass `path=self.db_path`; every lease/save passes `ttl=self.session_ttl`. Docstrings.
- [ ] **Step 4: Run** new tests and full gates.
- [ ] **Step 5: Commit** `git commit -m "feat: Agent(db_path=, session_ttl=) and export_traces(db_path=)"`.

### Task 2: `load_session`

**Files:**
- Modify: `src/logpose/store/sqlite.py` (`load_messages` returns `[]` when the session row is missing or `expires_at < now`; no writes), `src/logpose/store/base.py` (Protocol docstring), `src/logpose/agent.py` (`async def load_session(self, session_id: str) -> list[Message]`), `src/logpose/sync.py` (`SyncAgent.load_session(self, session_id: str) -> list[Message]`)
- Test: `tests/test_storage_config.py`

**Interfaces:**
- Consumes: Task 1's `Agent.db_path` and `manager.get_store(owner=, path=)`.
- Produces: `Agent.load_session(session_id: str) -> list[Message]`; `SyncAgent.load_session(session_id: str) -> list[Message]`.

- [ ] **Step 1: Write failing tests:**
  - `test_load_session_returns_saved_history`: after two runs on session `s`, `await agent.load_session("s")` equals the stored messages (user, assistant, user, assistant), full history even when a summary exists (force compaction with a small `context_budget`: result has no summary message and includes the summarized turns).
  - `test_load_session_unknown_is_empty`.
  - `test_load_session_expired_is_empty_and_untouched`: short TTL, expire → `[]`; rows still present (sqlite3 count > 0); `expires_at` unchanged.
  - `test_load_session_does_not_take_lease`: hold `agent.session("s")` open in one task and stream a run suspended mid-turn (a `FakeProvider` gated on an `asyncio.Event`) → `load_session("s")` returns the previously saved messages (in-progress prompt not included), no `SessionBusyError`; then the run finishes normally.
  - `test_load_session_invalid_id`: `"bad id!"` → `LogposeError`.
  - `test_load_session_uses_agent_db_path`.
  - `test_sync_load_session`: `SyncAgent(agent).load_session("s")`.
- [ ] **Step 2: Run** — expect failures.
- [ ] **Step 3: Implement.** `load_session` validates, gets the store via `_store()` (lazily opened, owner=self), returns `await store.load_messages(sid)` (a new list; callers can't mutate stored state).
- [ ] **Step 4: Run** new tests and full gates.
- [ ] **Step 5: Commit** `git commit -m "feat: Agent.load_session for lease-free history reads"`.

### Task 3: Docs

**Files:** `README.md` (Sessions: `load_session`, `db_path`, `session_ttl` + the TTL rule; Tracing: `export_traces(db_path=)`; Configuration reference table), `CHANGELOG.md` (`## [Unreleased]` → Added), `tests/test_public_api.py` if it asserts signatures.

- [ ] **Step 1:** README updates incl. a FastAPI snippet passing `db_path`/`session_ttl` from a settings object and a page-load `load_session` example; note that `.env` loaded by pydantic-settings does not populate `os.environ`, so pass settings explicitly.
- [ ] **Step 2:** CHANGELOG Added entries. Gates.
- [ ] **Step 3: Commit** `git commit -m "docs: storage configuration and load_session"`.
