"""SQLite implementation of :class:`~logpose.store.base.SessionStore`.

This is the only module that contains SQL. The schema is kept portable: TEXT,
INTEGER and NULL only, JSON as TEXT, times as integer Unix milliseconds (UTC).
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from types import TracebackType
from typing import TYPE_CHECKING, Any

from logpose.errors import LogposeError
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


def _now_ms() -> int:
    return int(time.time() * 1000)


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
        self._idle_ttl = idle_ttl
        self._lease_ttl = lease_ttl
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
        if self._db is None:
            raise LogposeError("SQLiteStore is not open")
        return self._db

    async def open(self) -> None:
        if self._db is not None:
            return
        import aiosqlite

        db = await aiosqlite.connect(self._path, isolation_level=None)
        try:
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute(f"PRAGMA busy_timeout={self._busy_timeout_ms}")
            await self._migrate(db)
        except BaseException:
            await db.close()
            raise
        self._db = db

    async def _migrate(self, db: aiosqlite.Connection) -> None:
        await db.execute("BEGIN IMMEDIATE")
        try:
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
        except BaseException:
            await db.execute("ROLLBACK")
            raise
        await db.execute("COMMIT")

    async def close(self) -> None:
        db, self._db = self._db, None
        if db is not None:
            await db.close()

    def session(
        self,
        session_id: str,
        *,
        wait: float = 0.0,
        metadata: Mapping[str, str] | None = None,
    ) -> AbstractAsyncContextManager[SessionHandle]:
        raise NotImplementedError("SQLiteStore.session is implemented in a later task")

    async def _write(self, statements: Sequence[tuple[str, Sequence[Any]]]) -> list[int]:
        """Run statements in one immediate transaction; return each rowcount."""
        db = self._conn()
        async with self._lock:
            await db.execute("BEGIN IMMEDIATE")
            try:
                counts = []
                for sql, params in statements:
                    cur = await db.execute(sql, tuple(params))
                    counts.append(cur.rowcount)
                    await cur.close()
            except BaseException:
                await db.execute("ROLLBACK")
                raise
            await db.execute("COMMIT")
        return counts

    async def load_messages(self, session_id: str) -> list[Message]:
        validate_session_id(session_id)
        db = self._conn()
        async with self._lock:
            async with db.execute(
                "SELECT role, content FROM logpose_messages WHERE session_id = ? ORDER BY seq",
                (session_id,),
            ) as cur:
                rows = await cur.fetchall()
        return [
            Message.model_validate({"role": role, "content": json.loads(content)})
            for role, content in rows
        ]

    async def append_messages(
        self, session_id: str, messages: Sequence[Message], *, start_seq: int
    ) -> int:
        validate_session_id(session_id)
        now = _now_ms()
        stmts: list[tuple[str, Sequence[Any]]] = []
        seq = start_seq
        for m in messages:
            dumped = m.model_dump(mode="json")
            stmts.append(
                (
                    "INSERT INTO logpose_messages (session_id, seq, role, content, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (session_id, seq, dumped["role"], json.dumps(dumped["content"]), now),
                )
            )
            seq += 1
        if stmts:
            await self._write(stmts)
        return seq

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
