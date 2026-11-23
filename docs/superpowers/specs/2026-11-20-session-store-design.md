# Session store (SQLite default, relational-portable)

Status: design for approval, 2026-11-20

## Intent and the why

logpose powers chat features in small web apps, including public, login-free
support chats behind a stateless REST API. Today a `Conversation` lives only in
process memory, so every app re-implements session storage, history
serialization, and per-session locking, and gets the locking wrong easily
(concurrent requests on one session interleave messages and corrupt history).

This adds a session store: one abstraction for every database operation, with
SQLite as the default implementation. Its tables are relational and portable
so MySQL and Postgres implementations can be added later with the same schema,
and the later tracing work can reuse it. It also gives the next pieces of work,
per-user limits and compaction, the data they need (stored history and
per-session token usage).

## Goals

- `SessionStore`: an async interface through which all database work happens.
  No SQL outside the store module.
- `SQLiteStore`: the implementation, on `aiosqlite`. **logpose creates and
  manages it automatically (owner's decision, 2026-11-20): apps never construct
  or pass a store.** `SessionStore` stays an internal seam so MySQL and Postgres
  can be added later.
- Store sessions, their full message history (all content blocks, losslessly),
  and one record per run (usage, outcome).
- A per-session lease so two requests never run on one session at once, across
  worker processes on one host.
- Agent integration: `run`/`stream(..., session_id=...)` (or `agent.session(...)`)
  loads history, runs, persists new messages and the run record, then releases.
  Without session use, behaviour is unchanged and nothing touches disk.
- Schema versioning and migrations from day one.
- Expiry and purge of idle sessions.

## Non-goals

- MySQL and Postgres implementations (later; the schema and interface must not
  preclude them).
- Tracing tables or OpenTelemetry export (shelved; `runs` is the intended join
  point later).
- Compaction and per-user limits. They are separate next pieces that build on
  this.
- Separate tables for tool calls or content blocks. Blocks are stored as JSON
  inside each message for now.
- Encryption at rest or PII scrubbing of stored content.

## Anti-goals

- Blocking the event loop on database I/O.
- Touching disk for runs that don't use sessions. The database is created
  lazily, on first session use only.
- A stored history that can't be replayed to the model exactly as it was
  produced: tool calls and results, reasoning and signatures, and raw blocks
  must round-trip.
- Any change to runs that don't pass `session_id`.

## Constraints

- Python 3.10+. `aiosqlite` becomes a **core dependency** (the store is automatic,
  so it can't be an optional extra). Importing `logpose` still must not import
  `aiosqlite`; it is imported lazily on first session use.
- Portable schema: only TEXT, INTEGER (64-bit) and NULL column types; JSON stored
  as TEXT; timestamps as integer Unix milliseconds (UTC); no SQLite-only
  features in the table definitions (no `WITHOUT ROWID`, no `AUTOINCREMENT`
  reliance; explicit sequence columns).
- SQLite runs in WAL mode with a busy timeout. That supports several worker
  processes on one host; several hosts need a networked database later.

## Approaches

- **Stdlib-style SQL on `aiosqlite` behind a `SessionStore` interface, managed
  automatically by logpose (chosen).** Simple, one small dependency, async-native, with portability kept
  by the interface and a portable schema.
- Blocking `sqlite3` on the event loop: rejected. Lock waits and slow fsyncs
  would freeze every stream in the worker.
- `sqlite3` in `asyncio.to_thread`: viable, but the owner chose an async library.
- SQLAlchemy Core: rejected for now as too heavy. It may be revisited when MySQL
  and Postgres are added.

## Decided

### Interface (`logpose/store/base.py`, no database imports)

```python
class SessionStore(Protocol):
    async def open(self) -> None                         # create/migrate schema
    async def close(self) -> None
    def session(self, session_id: str, *, wait: float = 0.0,
                metadata: Mapping[str, str] | None = None
               ) -> AsyncContextManager[SessionHandle]   # lease (renewed) + load
    async def load_messages(self, session_id: str) -> list[Message]
    async def record_run(self, record: RunRecord) -> None
    async def delete_session(self, session_id: str) -> None
    async def purge_expired(self, now_ms: int | None = None) -> int
```

- `SessionHandle` has `id: str`, `conversation: Conversation` (loaded
  history), `metadata: dict[str, str]`, and `created: bool` (true if new).
- On normal exit of the `session(...)` block, the store appends any messages
  added to `conversation` since load, bumps `version` and `updated_at`, extends
  `expires_at`, and releases the lease.
- On an exception, it **still persists** the messages appended so far (history
  stays a valid partial record, matching how a failed run leaves its partial
  history), then releases and re-raises.
- **Metadata (Claude's call):** set when the session is created. When given on
  a later call, it is merged in (given keys overwrite), so an app can attach a
  user ID once one exists.
- **Expired but not yet purged (Claude's call):** `session()` treats a session
  past `expires_at` as gone. It deletes the old rows in the same transaction and
  starts fresh (`created=True`). Expiry is therefore enforced on access;
  `purge_expired()` only reclaims space.
- `RunRecord` (frozen dataclass) has: `run_id`, `session_id`, `model`,
  `provider`, `started_at_ms`, `ended_at_ms`, `outcome` (`completed`,
  `max_iterations`, `run_timeout`, `cancelled`, or `error`), `iterations`,
  `input_tokens`, `output_tokens`, `reasoning_tokens`,
  `cache_read_input_tokens`, `cache_creation_input_tokens`, and `error_type`
  (`str | None`, type name only).
- `SessionBusyError(LogposeError)` is raised when the lease is held and `wait`
  elapses (default `0`, reject immediately; apps map it to HTTP 409).
- `session_id` must match `[A-Za-z0-9._-]{1,128}`. Anything else raises
  `LogposeError`.

### Schema (version 1)

| Table | Columns |
|---|---|
| `logpose_schema` | `version INTEGER NOT NULL` |
| `logpose_sessions` | `id TEXT PRIMARY KEY`, `created_at INTEGER`, `updated_at INTEGER`, `expires_at INTEGER`, `version INTEGER`, `metadata TEXT` (JSON), `lease_token TEXT NULL`, `lease_expires_at INTEGER NULL` |
| `logpose_messages` | `session_id TEXT`, `seq INTEGER`, `role TEXT`, `content TEXT` (JSON list of blocks via `Message.model_dump(mode="json")`), `created_at INTEGER`; `PRIMARY KEY (session_id, seq)` |
| `logpose_runs` | `run_id TEXT PRIMARY KEY`, `session_id TEXT`, plus the `RunRecord` fields as columns; index on `(session_id, started_at)` |

- Table names are prefixed `logpose_` so the tables can share an app's database.
- Deleting a session deletes its messages and runs. This is done explicitly in
  the store, not with `ON DELETE CASCADE`, to stay portable.
- Messages are append-only: `seq` continues from the stored maximum.
  Compaction never rewrites stored rows.

### Lease semantics

**Renewal (Claude's call).** Leases are renewed while held. The holder renews
every `lease_ttl / 3` from a background task, and stops renewing on exit. So the
lease TTL is a crash-detection window, not a cap on run length. Runs with
`run_timeout=None` or long tool calls keep their lease, and a crashed worker's
lease frees within `lease_ttl`. A renewal that finds the token changed marks the
handle lost, and the final save is refused as below.
Renewal stops once the handle has had no run in progress for `3 × lease_ttl`
(the handle is then `lapsed`; a new run on it re-extends the lease only if the
stored token is still its own, else raises `SessionBusyError`), so a block that is never
exited, for example a dropped streaming response kept alive by a reference
cycle, frees its lease when it expires.


- Acquire in one transaction (`BEGIN IMMEDIATE` on SQLite):
  - if `lease_token` is null or `lease_expires_at < now`, set a fresh random
    token and `lease_expires_at = now + lease_ttl`;
  - otherwise the lease is busy, so wait in short polls up to `wait`, then
    `SessionBusyError`.
- `lease_ttl` is a store setting, default 60 s. Renewal makes it independent of
  `run_timeout`, so the protocol needs no TTL parameter.
- Release and save only succeed if the stored `lease_token` still matches. If
  the lease expired and someone else took it, the stale holder's save is
  refused with `SessionBusyError` (its messages are not written). This prevents
  lost updates.

### Expiry

- `idle_ttl` defaults to 30 days. Each successful save sets
  `expires_at = now + idle_ttl`.
- `purge_expired()` deletes sessions past `expires_at` (and their messages and
  runs) and returns the number of sessions purged. logpose never purges on its
  own; the app calls it, for example on a schedule.

### Automatic management (Claude's calls, following the owner's decision)

- **Location:** the database path comes from `LOGPOSE_DB_PATH`. Default:
  `logpose.db` in the process's working directory. Blank counts as unset,
  matching the other `LOGPOSE_*` variables. The parent directory is created if
  missing.
- **Lifecycle:** one store per (path, event loop), because an aiosqlite
  connection is bound to its loop and the sync facade runs a private loop.
  Created and opened (schema migrated) lazily on first session use, and shared
  by every `Agent` on that loop. Closed at `Agent.aclose()` of the last agent using it, or at
  interpreter exit; best effort.
- **No public store parameter.** `Agent` has no `store=` argument. Tests point
  `LOGPOSE_DB_PATH` at a temporary file. A private constructor hook for
  injecting a store may exist for tests only.
- **Maintenance surface:** `await agent.purge_expired_sessions() -> int` and
  `await agent.delete_session(session_id)`. Apps call these; logpose never
  purges on its own.
- **Settings:** the idle TTL (30 days) and lease TTL (60 s) are fixed defaults
  for now. Exposing them is a later change.

### SQLite specifics (`logpose/store/sqlite.py`)

- `SQLiteStore(path: str | os.PathLike, *, idle_ttl: float = 30 * 86400,
  lease_ttl: float = 60.0, busy_timeout_ms: int = 5000)`.
- On `open()`: `PRAGMA journal_mode=WAL`, `PRAGMA busy_timeout`,
  `PRAGMA foreign_keys` left default, create tables if missing, and run
  migrations to the current schema version. Opening a database with a newer
  schema version than the code supports raises `LogposeError`.
- One `aiosqlite` connection per store, serialized internally. This is
  sufficient for the target scale; a pool is left for later.
- Async context manager support, used internally.

### Agent integration

- `Agent(..., session_wait: float = 0.0)`. There is no store argument; the store
  is automatic (see above).
- **Explicit form (Claude's call), for streaming web apps:**
  `async with agent.session(session_id, metadata=...) as handle:` takes the
  lease and loads history, so `SessionBusyError` is raised before any HTTP
  response starts (the app can still send a 409). Then
  `agent.stream(msg, session=handle)` runs against it. The handle records each
  run's `RunRecord` and persists new messages on exit. The same `session=`
  works on `run`, `stream`, `run_sync`, `stream_sync` and `SyncAgent`.
- **Convenience form:** `run`/`stream(..., session_id: str | None = None,
  session_metadata: Mapping[str, str] | None = None)`. It is equivalent to
  wrapping one call in `agent.session(...)`. For `stream`, the lease is taken
  when iteration starts, so a busy session surfaces as `SessionBusyError` from
  the first `__anext__`; streaming apps that need a 409 use the explicit form.
- Combining any two of `session_id=`, `session=` and `conversation=` raises
  `LogposeError`.
- With `session_id`, the run:
  1. opens `store.session(session_id)`, using `conversation` from the handle;
  2. runs as today;
  3. records a `RunRecord` with the outcome, including error and cancellation
     paths;
  4. exits the session block, which persists the new messages and releases.
- Cancellation (client disconnect, or the consumer breaking out of a stream
  early) still persists the partial history and records `outcome="cancelled"`,
  within a bounded best-effort shield so a disconnect cannot leave the lease
  held until it expires.
- **Exit budgets (Claude's call).** Each exit runs within
  `SESSION_EXIT_TIMEOUT` (8 s, above the 5 s `busy_timeout`). A session block
  first closes the streams still open on its handle; each stream's exit (its run
  record) is bounded on its own, and the phase is abandoned after the timeout
  plus a 2 s margin. The block's save and release then get a fresh 8 s, so two
  busy waits in a row cannot cancel the save. Under extreme contention the
  stream's run record may be lost (logged); the messages are not.
- **Usage on every outcome (Claude's call).** The loop keeps a run-scoped
  accumulator of usage and iterations that the session wrapper reads, so
  `RunRecord` carries real token counts for `completed`, `error`,
  `max_iterations`, `run_timeout` and `cancelled` runs alike. The limits work
  depends on this not undercounting.
- **Stale lease (Claude's call).** If the final save is refused because the lease
  was lost, the `RunRecord` is still written (the tokens were spent), and
  `SessionBusyError` is raised to the caller.
- **`reasoning_tokens` (Claude's call).** `Usage` gains `reasoning_tokens: int = 0`
  (summed by `+`). The OpenAI-compatible provider fills it from
  `completion_tokens_details.reasoning_tokens` when present. That is the only
  provider, so the scope is small.
- `Conversation` gains `to_json() -> str` and `Conversation.from_json(text)`
  (with a format `version` field). They are public helpers for apps that store
  history themselves, and are used by tests.

## Left to the builder

- The polling interval for `wait` (suggest 50 ms).
- The internal migration framework shape (a list of versioned SQL scripts is
  enough).
- How `SessionHandle` tracks "messages added since load" (for example,
  remembering the loaded length).
- Exact error message wording.
