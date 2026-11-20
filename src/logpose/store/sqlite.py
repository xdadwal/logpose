"""SQLite implementation of :class:`~logpose.store.base.SessionStore`.

This is the only module that contains SQL. The schema is kept portable: TEXT,
INTEGER and NULL only, JSON as TEXT, times as integer Unix milliseconds (UTC).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import os
import random
import secrets
import sqlite3
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from types import TracebackType
from typing import TYPE_CHECKING, Any, TypeVar

from logpose.agent import Conversation
from logpose.errors import LogposeError, SessionBusyError
from logpose.messages import Message
from logpose.store.base import RunRecord, SessionHandle, validate_session_id
from logpose.tracing.spans import SpanRecord

if TYPE_CHECKING:  # pragma: no cover
    import aiosqlite

__all__ = ["SQLiteStore"]

# Versioned migrations: index i holds the script that upgrades version i to i + 1.
_MIGRATIONS: list[list[str]] = [
    [
        "CREATE TABLE logpose_sessions ("
        "id TEXT PRIMARY KEY, created_at INTEGER, updated_at INTEGER, "
        "expires_at INTEGER, version INTEGER, metadata TEXT, "
        "lease_token TEXT NULL, lease_expires_at INTEGER NULL)",
        "CREATE TABLE logpose_messages ("
        "session_id TEXT, seq INTEGER, role TEXT, content TEXT, created_at INTEGER, "
        "PRIMARY KEY (session_id, seq))",
        "CREATE TABLE logpose_runs ("
        "run_id TEXT PRIMARY KEY, session_id TEXT, model TEXT, provider TEXT, "
        "started_at INTEGER, ended_at INTEGER, outcome TEXT, iterations INTEGER, "
        "input_tokens INTEGER, output_tokens INTEGER, reasoning_tokens INTEGER, "
        "cache_read_input_tokens INTEGER, cache_creation_input_tokens INTEGER, "
        "error_type TEXT NULL)",
        "CREATE INDEX logpose_runs_session ON logpose_runs (session_id, started_at)",
    ],
    [
        "CREATE TABLE logpose_summaries ("
        "session_id TEXT, upto_seq INTEGER, text TEXT, created_at INTEGER, "
        "PRIMARY KEY (session_id, upto_seq))",
    ],
    [
        "ALTER TABLE logpose_runs ADD COLUMN metadata TEXT NULL",
        "CREATE TABLE logpose_spans ("
        "trace_id TEXT, span_id TEXT, parent_span_id TEXT NULL, run_id TEXT, "
        "session_id TEXT NULL, name TEXT, kind INTEGER, start_ns INTEGER, end_ns INTEGER, "
        "status_code INTEGER, status_message TEXT NULL, attributes TEXT, events TEXT, "
        "PRIMARY KEY (trace_id, span_id))",
        "CREATE INDEX logpose_spans_run ON logpose_spans (run_id)",
        "CREATE INDEX logpose_spans_session ON logpose_spans (session_id)",
        "CREATE INDEX logpose_spans_start ON logpose_spans (start_ns)",
    ],
]
SCHEMA_VERSION = len(_MIGRATIONS)

_SPAN_PAGE = 200
_POLL_INTERVAL = 0.05  # seconds between lease attempts while waiting

_T = TypeVar("_T")


def _json_safe(value: Any) -> Any:
    """Make ``value`` strictly JSON-serializable: non-finite floats become strings."""
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    if isinstance(value, dict):
        return {k if isinstance(k, str) else str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _dumps_safe(value: Any) -> str:
    return json.dumps(_json_safe(value), default=str, allow_nan=False)


def _now_ms() -> int:
    return int(time.time() * 1000)


def _message_row(session_id: str, seq: int, m: Message, now: int) -> tuple[str, Sequence[Any]]:
    dumped = m.model_dump(mode="json")
    return (
        "INSERT INTO logpose_messages (session_id, seq, role, content, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (session_id, seq, dumped["role"], json.dumps(dumped["content"]), now),
    )


def _refuse_newer(version: int) -> None:
    if version > SCHEMA_VERSION:
        raise LogposeError(
            f"database schema version {version} is newer than the supported "
            f"version {SCHEMA_VERSION}"
        )


async def _rollback(db: aiosqlite.Connection) -> None:
    """Leave the connection outside any transaction, without masking the caller's error.

    Always queues ``ROLLBACK`` rather than checking ``db.in_transaction``: after a
    cancellation the worker thread may still be running ``BEGIN``/``COMMIT``, and
    aiosqlite runs its queue in order, so the rollback lands after it. When no
    transaction is active, the "no transaction is active" error is suppressed.
    """
    with contextlib.suppress(Exception):
        await db.execute("ROLLBACK")


async def _fetch_messages(db: aiosqlite.Connection, session_id: str) -> list[Message]:
    async with db.execute(
        "SELECT role, content FROM logpose_messages WHERE session_id = ? ORDER BY seq",
        (session_id,),
    ) as cur:
        rows = await cur.fetchall()
    return [
        Message.model_validate({"role": role, "content": json.loads(content)})
        for role, content in rows
    ]


def _summary_row(session_id: str, upto_seq: int, text: str, now: int) -> tuple[str, Sequence[Any]]:
    return (
        "INSERT OR REPLACE INTO logpose_summaries (session_id, upto_seq, text, created_at) "
        "VALUES (?, ?, ?, ?)",
        (session_id, upto_seq, text, now),
    )


async def _fetch_summary(db: aiosqlite.Connection, session_id: str) -> tuple[int, str] | None:
    # Latest written wins: INSERT OR REPLACE gives the row a new, larger rowid,
    # and rowid order does not depend on the wall clock.
    async with db.execute(
        "SELECT upto_seq, text FROM logpose_summaries WHERE session_id = ? "
        "ORDER BY rowid DESC LIMIT 1",
        (session_id,),
    ) as cur:
        row = await cur.fetchone()
    return (int(row[0]), str(row[1])) if row is not None else None


class SQLiteStore:
    """Session store backed by one aiosqlite connection, serialized by a lock."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        idle_ttl: float = 30 * 86400.0,
        lease_ttl: float = 60.0,
        busy_timeout_ms: int = 5000,
    ) -> None:
        self._path = os.fspath(path)
        self._idle_ttl_ms = int(idle_ttl * 1000)
        self._lease_ttl = lease_ttl
        self._lease_ttl_ms = int(lease_ttl * 1000)
        self._busy_timeout_ms = int(busy_timeout_ms)
        self._db: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    async def __aenter__(self) -> SQLiteStore:
        await self.open()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()

    def _conn(self) -> aiosqlite.Connection:
        """Return the open connection. Call only while holding ``self._lock``."""
        if self._db is None:
            raise LogposeError("SQLiteStore is not open")
        return self._db

    async def open(self) -> None:
        async with self._lock:
            if self._db is not None:
                return
            import aiosqlite

            try:
                db = await aiosqlite.connect(self._path, isolation_level=None)
            except sqlite3.Error as exc:
                raise self._open_error(exc) from exc
            try:
                async with db.execute(f"PRAGMA busy_timeout={self._busy_timeout_ms}"):
                    pass
                # Refuse a newer schema before changing anything in the file.
                await self._check_schema_version(db)
                await self._enable_wal(db)
                await self._migrate(db)
            except sqlite3.Error as exc:
                await db.close()
                raise self._open_error(exc) from exc
            except BaseException:
                await db.close()
                raise
            self._db = db

    def _open_error(self, exc: sqlite3.Error) -> LogposeError:
        return LogposeError(f"Could not open session database {self._path!r}: {exc}")

    @staticmethod
    async def _check_schema_version(db: aiosqlite.Connection) -> None:
        """Raise if the file holds a schema newer than this release supports. Read-only."""
        async with db.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'logpose_schema'"
        ) as cur:
            if await cur.fetchone() is None:
                return
        async with db.execute("SELECT version FROM logpose_schema") as cur:
            row = await cur.fetchone()
        if row is not None:
            _refuse_newer(int(row[0]))

    async def _enable_wal(self, db: aiosqlite.Connection) -> None:
        """Switch to WAL, retrying while another connection holds a lock.

        ``PRAGMA journal_mode=WAL`` needs an exclusive lock and does not use the
        busy handler, so connections racing on a fresh file (several workers
        starting at once) see an immediate "database is locked". Retry with a
        short backoff for up to ``busy_timeout``; the last error propagates.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._busy_timeout_ms / 1000
        delay = 0.005
        while True:
            try:
                async with db.execute("PRAGMA journal_mode=WAL"):
                    return
            except sqlite3.OperationalError as exc:
                message = str(exc).lower()
                if "locked" not in message and "busy" not in message:
                    raise
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise
            await asyncio.sleep(min(delay * (0.5 + random.random()), remaining))
            delay = min(delay * 2, 0.1)

    async def _migrate(self, db: aiosqlite.Connection) -> None:
        try:
            await db.execute("BEGIN IMMEDIATE")
            await db.execute("CREATE TABLE IF NOT EXISTS logpose_schema (version INTEGER NOT NULL)")
            async with db.execute("SELECT version FROM logpose_schema") as cur:
                row = await cur.fetchone()
            if row is None:
                await db.execute("INSERT INTO logpose_schema (version) VALUES (0)")
                current = 0
            else:
                current = int(row[0])
            _refuse_newer(current)
            for script in _MIGRATIONS[current:]:
                for stmt in script:
                    await db.execute(stmt)
            if current < SCHEMA_VERSION:
                await db.execute("UPDATE logpose_schema SET version = ?", (SCHEMA_VERSION,))
            await db.execute("COMMIT")
        except BaseException:
            await _rollback(db)
            raise

    async def close(self) -> None:
        async with self._lock:
            db, self._db = self._db, None
            if db is not None:
                await db.close()

    async def _transaction(self, fn: Callable[[aiosqlite.Connection], Awaitable[_T]]) -> _T:
        """Run ``fn`` inside one ``BEGIN IMMEDIATE`` transaction under the store lock.

        Any failure, including in ``COMMIT``, rolls back, so the connection is never
        left inside a transaction.
        """
        async with self._lock:
            db = self._conn()
            try:
                # BEGIN is inside the try: if the caller is cancelled while it is
                # queued, the worker still runs it, and the rollback must follow.
                await db.execute("BEGIN IMMEDIATE")
                result = await fn(db)
                await db.execute("COMMIT")
            except BaseException:
                await _rollback(db)
                raise
            return result

    async def _write(self, statements: Sequence[tuple[str, Sequence[Any]]]) -> list[int]:
        """Run statements in one immediate transaction; return each rowcount."""

        async def run(db: aiosqlite.Connection) -> list[int]:
            counts = []
            for sql, params in statements:
                cur = await db.execute(sql, tuple(params))
                counts.append(cur.rowcount)
                await cur.close()
            return counts

        return await self._transaction(run)

    # --- sessions -----------------------------------------------------------------------

    def session(
        self,
        session_id: str,
        *,
        wait: float = 0.0,
        metadata: Mapping[str, str] | None = None,
    ) -> AbstractAsyncContextManager[SessionHandle]:
        """Lease ``session_id`` for the duration of an ``async with`` block.

        On enter the session is created if missing (or if past ``expires_at``), the
        given metadata is merged in, the lease is taken (polling up to ``wait``
        seconds, then :class:`SessionBusyError`), and the history is loaded. The lease
        is renewed every ``lease_ttl / 3`` while the block runs. On exit, normal or
        by exception, messages appended to ``handle.conversation`` are saved and the
        lease released, unless the lease was lost, in which case nothing is saved
        and :class:`SessionBusyError` is raised.
        """
        return self._session(session_id, wait, dict(metadata) if metadata else {})

    @asynccontextmanager
    async def _session(
        self, session_id: str, wait: float, metadata: dict[str, str]
    ) -> AsyncIterator[SessionHandle]:
        validate_session_id(session_id)
        handle, token = await self._acquire(session_id, wait, metadata)
        renewal = asyncio.create_task(self._renewal_loop(handle, token))
        try:
            yield handle
        except BaseException as exc:
            await self._stop(renewal)
            await self._release(handle, token, exc)
            raise
        await self._stop(renewal)
        await self._release(handle, token, None)

    async def _acquire(
        self, session_id: str, wait: float, metadata: dict[str, str]
    ) -> tuple[SessionHandle, str]:
        deadline = time.monotonic() + max(wait, 0.0)
        while True:
            got = await self._transaction(lambda db: self._try_acquire(db, session_id, metadata))
            if got is not None:
                return got
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise SessionBusyError(f"session {session_id!r} is in use")
            await asyncio.sleep(min(_POLL_INTERVAL, remaining))

    async def _try_acquire(
        self, db: aiosqlite.Connection, session_id: str, metadata: dict[str, str]
    ) -> tuple[SessionHandle, str] | None:
        now = _now_ms()
        async with db.execute(
            "SELECT expires_at, metadata, lease_token, lease_expires_at "
            "FROM logpose_sessions WHERE id = ?",
            (session_id,),
        ) as cur:
            row = await cur.fetchone()
        if row is not None:
            expires_at, stored_meta, lease_token, lease_expires_at = row
            if lease_token is not None and lease_expires_at >= now:
                return None  # held by a live lease
            if expires_at < now:
                # Expired but not purged: treat as gone and start fresh.
                for sql in (
                    "DELETE FROM logpose_messages WHERE session_id = ?",
                    "DELETE FROM logpose_runs WHERE session_id = ?",
                    "DELETE FROM logpose_summaries WHERE session_id = ?",
                    "DELETE FROM logpose_spans WHERE session_id = ?",
                    "DELETE FROM logpose_sessions WHERE id = ?",
                ):
                    await db.execute(sql, (session_id,))
                row = None
        token = secrets.token_hex(16)
        lease_expires = now + self._lease_ttl_ms
        if row is None:
            merged = dict(metadata)
            await db.execute(
                "INSERT INTO logpose_sessions (id, created_at, updated_at, expires_at, "
                "version, metadata, lease_token, lease_expires_at) "
                "VALUES (?, ?, ?, ?, 0, ?, ?, ?)",
                (
                    session_id,
                    now,
                    now,
                    now + self._idle_ttl_ms,
                    json.dumps(merged),
                    token,
                    lease_expires,
                ),
            )
            return (
                SessionHandle(session_id, Conversation(), merged, created=True),
                token,
            )
        merged = {**json.loads(stored_meta or "{}"), **metadata}
        await db.execute(
            "UPDATE logpose_sessions SET metadata = ?, lease_token = ?, lease_expires_at = ? "
            "WHERE id = ?",
            (json.dumps(merged), token, lease_expires, session_id),
        )
        messages = await _fetch_messages(db, session_id)
        async with db.execute(
            "SELECT COALESCE(MAX(seq) + 1, 0) FROM logpose_messages WHERE session_id = ?",
            (session_id,),
        ) as cur:
            seq_row = await cur.fetchone()
        next_seq = int(seq_row[0]) if seq_row else 0
        handle = SessionHandle(
            session_id,
            Conversation(messages),
            merged,
            created=False,
            loaded_count=len(messages),
            next_seq=next_seq,
            stored_summary=await _fetch_summary(db, session_id),
        )
        return handle, token

    async def _renewal_loop(self, handle: SessionHandle, token: str) -> None:
        """Extend the lease every ``lease_ttl / 3`` while the token still matches."""
        interval = self._lease_ttl / 3
        while True:
            await asyncio.sleep(interval)
            try:
                counts = await self._write(
                    [
                        (
                            "UPDATE logpose_sessions SET lease_expires_at = ? "
                            "WHERE id = ? AND lease_token = ?",
                            (_now_ms() + self._lease_ttl_ms, handle.id, token),
                        )
                    ]
                )
            except Exception:
                # Transient failure (busy database, store closed): try again next tick.
                # If the lease lapses meanwhile, the token check on exit refuses the save.
                continue
            if counts[0] == 0:
                handle.lost = True
                return

    @staticmethod
    async def _stop(task: asyncio.Task[None]) -> None:
        task.cancel()
        # asyncio.wait never raises the task's CancelledError, but still propagates
        # a cancellation of the caller.
        await asyncio.wait([task])

    async def _release(self, handle: SessionHandle, token: str, exc: BaseException | None) -> None:
        new = handle.new_messages()
        summary = handle.summary_to_save()

        async def save(db: aiosqlite.Connection) -> bool:
            async with db.execute(
                "SELECT lease_token FROM logpose_sessions WHERE id = ?", (handle.id,)
            ) as cur:
                row = await cur.fetchone()
            if row is None or row[0] != token:
                return False
            if handle.lost:
                # Still ours but marked lost: free the lease, write no session data.
                await db.execute(
                    "UPDATE logpose_sessions SET lease_token = NULL, lease_expires_at = NULL "
                    "WHERE id = ?",
                    (handle.id,),
                )
                return False
            now = _now_ms()
            for i, m in enumerate(new):
                await db.execute(*_message_row(handle.id, handle.next_seq + i, m, now))
            if summary is not None:
                # Same transaction as the messages it covers.
                await db.execute(*_summary_row(handle.id, *summary, now))
            await db.execute(
                "UPDATE logpose_sessions SET version = version + 1, updated_at = ?, "
                "expires_at = ?, lease_token = NULL, lease_expires_at = NULL WHERE id = ?",
                (now, now + self._idle_ttl_ms, handle.id),
            )
            return True

        if not await self._transaction(save):
            handle.lost = True
            raise SessionBusyError(
                f"lease on session {handle.id!r} was lost; its changes were not saved"
            ) from exc

    # --- messages, runs, purge ------------------------------------------------------------

    async def load_messages(self, session_id: str) -> list[Message]:
        validate_session_id(session_id)
        async with self._lock:
            return await _fetch_messages(self._conn(), session_id)

    async def append_messages(
        self, session_id: str, messages: Sequence[Message], *, start_seq: int
    ) -> int:
        validate_session_id(session_id)
        now = _now_ms()
        stmts = [_message_row(session_id, start_seq + i, m, now) for i, m in enumerate(messages)]
        if stmts:
            await self._write(stmts)
        return start_seq + len(stmts)

    async def save_summary(self, session_id: str, upto_seq: int, text: str) -> None:
        """Store a summary covering messages up to and including ``seq == upto_seq``."""
        validate_session_id(session_id)
        await self._write([_summary_row(session_id, upto_seq, text, _now_ms())])

    async def load_summary(self, session_id: str) -> tuple[int, str] | None:
        """Return the most recently saved summary as ``(upto_seq, text)``, or ``None``."""
        validate_session_id(session_id)
        async with self._lock:
            return await _fetch_summary(self._conn(), session_id)

    async def record_run(self, record: RunRecord) -> None:
        validate_session_id(record.session_id)
        try:
            await self._insert_run(record)
        except sqlite3.IntegrityError as exc:
            # Cross-process backstop: the in-process claim cannot see other
            # processes writing to the same database.
            if "logpose_runs.run_id" not in str(exc):
                raise
            raise LogposeError(
                f"run_id {record.run_id!r} already exists; every run needs a fresh run_id."
            ) from None

    async def _insert_run(self, record: RunRecord) -> None:
        await self._write(
            [
                (
                    "INSERT INTO logpose_runs (run_id, session_id, model, provider, "
                    "started_at, ended_at, outcome, iterations, input_tokens, output_tokens, "
                    "reasoning_tokens, cache_read_input_tokens, cache_creation_input_tokens, "
                    "error_type, metadata) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        record.run_id,
                        record.session_id,
                        record.model,
                        record.provider,
                        record.started_at_ms,
                        record.ended_at_ms,
                        record.outcome,
                        record.iterations,
                        record.input_tokens,
                        record.output_tokens,
                        record.reasoning_tokens,
                        record.cache_read_input_tokens,
                        record.cache_creation_input_tokens,
                        record.error_type,
                        None if record.metadata is None else json.dumps(dict(record.metadata)),
                    ),
                )
            ]
        )

    async def delete_session(self, session_id: str) -> None:
        validate_session_id(session_id)
        await self._write(
            [
                ("DELETE FROM logpose_messages WHERE session_id = ?", (session_id,)),
                ("DELETE FROM logpose_runs WHERE session_id = ?", (session_id,)),
                ("DELETE FROM logpose_summaries WHERE session_id = ?", (session_id,)),
                ("DELETE FROM logpose_spans WHERE session_id = ?", (session_id,)),
                ("DELETE FROM logpose_sessions WHERE id = ?", (session_id,)),
            ]
        )

    async def purge_expired(self, now_ms: int | None = None) -> int:
        now = _now_ms() if now_ms is None else now_ms
        expired = "SELECT id FROM logpose_sessions WHERE expires_at < ?"
        counts = await self._write(
            [
                (f"DELETE FROM logpose_messages WHERE session_id IN ({expired})", (now,)),
                (f"DELETE FROM logpose_runs WHERE session_id IN ({expired})", (now,)),
                (f"DELETE FROM logpose_summaries WHERE session_id IN ({expired})", (now,)),
                (f"DELETE FROM logpose_spans WHERE session_id IN ({expired})", (now,)),
                ("DELETE FROM logpose_sessions WHERE expires_at < ?", (now,)),
            ]
        )
        return counts[4]

    # --- spans ----------------------------------------------------------------------------

    async def write_spans(self, spans: Sequence[SpanRecord]) -> None:
        """Store spans in one transaction; a repeated ``(trace_id, span_id)`` is replaced."""
        if not spans:
            return
        await self._write(
            [
                (
                    "INSERT OR REPLACE INTO logpose_spans (trace_id, span_id, parent_span_id, "
                    "run_id, session_id, name, kind, start_ns, end_ns, status_code, "
                    "status_message, attributes, events) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        sp.trace_id,
                        sp.span_id,
                        sp.parent_span_id,
                        sp.run_id,
                        sp.session_id,
                        sp.name,
                        sp.kind,
                        sp.start_ns,
                        sp.end_ns,
                        sp.status_code,
                        sp.status_message,
                        _dumps_safe(sp.attributes),
                        _dumps_safe(sp.events),
                    ),
                )
                for sp in spans
            ]
        )

    async def read_spans(
        self,
        *,
        since_ns: int | None = None,
        until_ns: int | None = None,
        session_id: str | None = None,
        trace_id: str | None = None,
    ) -> AsyncIterator[SpanRecord]:
        """Stream spans ordered by ``(start_ns, trace_id, span_id)``; bounds are inclusive.

        Pages of ``_SPAN_PAGE`` rows are read under the store lock, which is released
        before records are yielded, so consumers may call the store inside the loop.
        """
        where: list[str] = []
        params: list[Any] = []
        for clause, value in (
            ("start_ns >= ?", since_ns),
            ("start_ns <= ?", until_ns),
            ("session_id = ?", session_id),
            ("trace_id = ?", trace_id),
        ):
            if value is not None:
                where.append(clause)
                params.append(value)
        cols = (
            "trace_id, span_id, parent_span_id, run_id, session_id, name, kind, "
            "start_ns, end_ns, status_code, status_message, attributes, events"
        )
        names = [c.strip() for c in cols.split(",")]
        base = " AND ".join(where)
        after: tuple[int, str, str] | None = None
        # Keyset pagination: the lock is held per page only, never across a yield.
        while True:
            clauses, args = base, list(params)
            if after is not None:
                keyset = (
                    "(start_ns > ? OR (start_ns = ? AND trace_id > ?) "
                    "OR (start_ns = ? AND trace_id = ? AND span_id > ?))"
                )
                clauses = f"{clauses} AND {keyset}" if clauses else keyset
                a0, a1, a2 = after
                args += [a0, a0, a1, a0, a1, a2]
            sql = (
                f"SELECT {cols} FROM logpose_spans"
                + (f" WHERE {clauses}" if clauses else "")
                + f" ORDER BY start_ns, trace_id, span_id LIMIT {_SPAN_PAGE}"
            )
            async with self._lock:
                async with self._conn().execute(sql, args) as cur:
                    rows = list(await cur.fetchall())
            for r in rows:
                d = dict(zip(names, r, strict=True))
                yield SpanRecord(
                    trace_id=d["trace_id"],
                    span_id=d["span_id"],
                    parent_span_id=d["parent_span_id"],
                    run_id=d["run_id"],
                    session_id=d["session_id"],
                    name=d["name"],
                    kind=int(d["kind"]),
                    start_ns=int(d["start_ns"]),
                    end_ns=int(d["end_ns"]),
                    status_code=int(d["status_code"]),
                    status_message=d["status_message"],
                    attributes=json.loads(d["attributes"]),
                    events=json.loads(d["events"]),
                )
            if len(rows) < _SPAN_PAGE:
                return
            last = rows[-1]
            after = (int(last[7]), str(last[0]), str(last[1]))

    async def purge_spans(self, older_than_ms: int) -> int:
        """Delete spans that started before the Unix-millisecond cutoff; return the count.

        Deletes by start time regardless of session. ``logpose_runs`` rows are kept,
        so :meth:`run_id_exists` stays true for purged runs.
        """
        counts = await self._write(
            [("DELETE FROM logpose_spans WHERE start_ns < ?", (older_than_ms * 1_000_000,))]
        )
        return counts[0]

    async def run_id_exists(self, run_id: str) -> bool:
        async with self._lock:
            db = self._conn()
            for table in ("logpose_runs", "logpose_spans"):
                async with db.execute(
                    f"SELECT 1 FROM {table} WHERE run_id = ? LIMIT 1", (run_id,)
                ) as cur:
                    if await cur.fetchone() is not None:
                        return True
        return False
