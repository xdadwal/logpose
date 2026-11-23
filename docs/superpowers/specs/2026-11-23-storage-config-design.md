# Storage configuration and read-only session history

Status: approved 2026-11-23 (short design; owner approved the proposed API in
conversation).

## Intent and the why

A FastAPI site runs one shared `Agent` under systemd with `ProtectSystem=strict`
and reads settings with pydantic-settings, which does not export `.env` into
`os.environ`. It needs to (1) pass the database path from its settings object,
(2) choose a shorter session retention (for example 7 days), and (3) show a
visitor's earlier messages on page load even while the same session is
streaming in another tab, without taking the write lease.

## Goals / non-goals

- `Agent(db_path=..., session_ttl=...)`, `export_traces(..., db_path=...)`.
- `await agent.load_session(session_id) -> list[Message]` and
  `SyncAgent.load_session(session_id)`.
- Backward compatible: without the new arguments, `LOGPOSE_DB_PATH` and the
  30-day default behave exactly as before.
- Non-goals: session rewind/truncate, listing sessions or run records,
  annotating runs.

## Approaches

- **Plain `Agent` arguments resolved once (chosen).** Fits the manager's
  per-(path, event loop) store cache unchanged.
- Accept a store instance: rejected — an aiosqlite connection is bound to one
  event loop, which breaks the sync facade's private loop and
  `asyncio.run`-per-request apps.
- A settings object (`StoreConfig`): rejected for now — two scalars don't need
  it.

## Decided

### `db_path`
- Resolution: `Agent(db_path=)` if given (not blank) → `LOGPOSE_DB_PATH` (not
  blank) → `./logpose.db`; made absolute (`~` expanded) once, at construction.
  Accepts `str | os.PathLike[str] | None`.
- Used everywhere the agent touches storage: the manager's per-(path, loop)
  store cache, the tracing span writer (per store), `purge_traces`,
  `purge_expired_sessions`, `delete_session`, `load_session`, sessions, and the
  run-ID duplicate check.
- `export_traces(path, ..., db_path=None)`: the same resolution without an
  agent (`path` stays the output file).

### `session_ttl`
- `Agent(session_ttl=timedelta | float seconds | None)`, default 30 days,
  must be positive → else `LogposeError`.
- **Conflict rule:** the TTL belongs to the agent that writes, not the store.
  Every write stamps the session row's `expires_at = now + that agent's
  session_ttl` (creation and each save). Purge and expiry checks read the stored
  `expires_at`. Two agents with different TTLs on one database therefore never
  conflict; a session expires `session_ttl` after its last write by whichever
  agent made it. Changing the TTL affects a session from its next write.
- The store no longer needs its own TTL; `SQLiteStore(idle_ttl=)` remains as the
  default when a caller passes none.

### `load_session`
- Returns the full stored history as `list[Message]` (not the compacted view; no
  summary).
- Unknown or expired session → `[]`. The read never deletes, resets, or extends
  anything.
- A run in progress is not visible until its messages are saved (only committed
  rows are read).
- Takes no lease; works while another request holds it.
- Validates the session ID like `session()` (`LogposeError`).
- Opens the store lazily like other session calls.
- Store: a read-only `load_messages(session_id) -> list[Message]` on the
  `SessionStore` protocol.

## Left to the builder

How the TTL reaches the store methods (argument per call vs per handle), test
layout, README wording.
