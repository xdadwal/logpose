"""SQLite implementation of :class:`~logpose.store.base.SessionStore`.

This is the only module that contains SQL. The schema is kept portable: TEXT,
INTEGER and NULL only, JSON as TEXT, times as integer Unix milliseconds (UTC).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import secrets
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from types import TracebackType
from typing import TYPE_CHECKING, Any, TypeVar

from logpose.agent import Conversation
from logpose.errors import LogposeError, SessionBusyError
from logpose.messages import Message
from logpose.store.base import RunRecord, SessionHandle, validate_session_id

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
]
SCHEMA_VERSION = len(_MIGRATIONS)

_POLL_INTERVAL = 0.05  # seconds between lease attempts while waiting

_T = TypeVar("_T")


def _now_ms() -> int:
    return int(time.time() * 1000)


def _message_row(session_id: str, seq: int, m: Message, now: int) -> tuple[str, Sequence[Any]]:
    dumped = m.model_dump(mode="json")
    return (
        "INSERT INTO logpose_messages (session_id, seq, role, content, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (session_id, seq, dumped["role"], json.dumps(dumped["content"]), now),
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

            db = await aiosqlite.connect(self._path, isolation_level=None)
            try:
                # busy_timeout first, so switching to WAL also waits for other writers.
                async with db.execute(f"PRAGMA busy_timeout={self._busy_timeout_ms}"):
                    pass
                async with db.execute("PRAGMA journal_mode=WAL"):
                    pass
                await self._migrate(db)
            except BaseException:
                await db.close()
                raise
            self._db = db

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
            if current > SCHEMA_VERSION:
                raise LogposeError(
                    f"database schema version {current} is newer than the supported "
                    f"version {SCHEMA_VERSION}"
                )
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

    async def record_run(self, record: RunRecord) -> None:
        validate_session_id(record.session_id)
        await self._write(
            [
                (
                    "INSERT INTO logpose_runs (run_id, session_id, model, provider, "
                    "started_at, ended_at, outcome, iterations, input_tokens, output_tokens, "
                    "reasoning_tokens, cache_read_input_tokens, cache_creation_input_tokens, "
                    "error_type) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                ("DELETE FROM logpose_sessions WHERE expires_at < ?", (now,)),
            ]
        )
        return counts[2]
