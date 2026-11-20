"""Automatic management of the session store.

One opened :class:`~logpose.store.sqlite.SQLiteStore` exists per (absolute
database path, running event loop): an aiosqlite connection is bound to the
loop that awaits it, and the sync facade runs its own private loop. Stores are
created lazily on first session use, shared by every :class:`~logpose.agent.Agent`
on that loop, and closed by :meth:`~logpose.agent.Agent.aclose` of the last
agent using them, or at interpreter exit (best effort).

Each store owns one :class:`~logpose.tracing.writer.SpanWriter`. Closing a store
closes its writer first (writing what is queued, bounded); a store whose loop is
gone has its leftover spans written with a short-lived stdlib connection, so a
run's trace survives a finished ``asyncio.run`` or interpreter exit (best effort).

``aiosqlite`` is imported only when a store is first opened, so runs without
sessions never load it and never touch disk.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import threading
import warnings
import weakref
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:  # pragma: no cover
    from logpose.store.base import SessionHandle, SessionStore
    from logpose.store.sqlite import SQLiteStore
    from logpose.tracing.spans import SpanRecord
    from logpose.tracing.writer import SpanWriter

__all__ = ["DB_PATH_ENV", "DEFAULT_DB_FILENAME", "db_path", "get_store"]

DB_PATH_ENV = "LOGPOSE_DB_PATH"
"""Environment variable naming the session database file. Blank counts as unset."""

DEFAULT_DB_FILENAME = "logpose.db"
"""Database file used, in the working directory, when ``LOGPOSE_DB_PATH`` is unset."""

_Key = tuple[str, asyncio.AbstractEventLoop]

_LOGGER = logging.getLogger("logpose.store")
_STOP_WARNED = False

_EXIT_FLUSH_TIMEOUT = 2.0
"""Seconds the exit hook waits for a writer on a still-running loop."""


class _Entry:
    """One opened store and the agents currently using it."""

    def __init__(
        self, store: SQLiteStore, writer: SpanWriter, loop: asyncio.AbstractEventLoop
    ) -> None:
        self.store = store
        self.writer = writer
        self.loop = loop
        self.owners: weakref.WeakSet[object] = weakref.WeakSet()


_GUARD = threading.Lock()  # protects the registries below across loops/threads
_ENTRIES: dict[_Key, _Entry] = {}
_CREATE_LOCKS: dict[_Key, asyncio.Lock] = {}
# Every opened, not yet closed store, with its span writer (for exit cleanup).
_OPEN: dict[SQLiteStore, _Entry] = {}
_EXIT_HOOK_INSTALLED = False

# Handles currently inside a session block, mapped to the store that leased them.
_ACTIVE_HANDLES: weakref.WeakKeyDictionary[SessionHandle, SessionStore] = (
    weakref.WeakKeyDictionary()
)
# Handles with a run in flight; one run at a time per handle.
_RUNNING: weakref.WeakSet[SessionHandle] = weakref.WeakSet()


def db_path() -> Path:
    """Return the absolute session database path.

    ``LOGPOSE_DB_PATH`` wins when set and not blank; otherwise ``logpose.db`` in
    the current working directory.
    """
    raw = os.environ.get(DB_PATH_ENV, "").strip()
    return Path(os.path.abspath(os.path.expanduser(raw or DEFAULT_DB_FILENAME)))


async def get_store(*, owner: object | None = None) -> SessionStore:
    """Return the opened store for the current path and running loop.

    Creates and opens it (migrating the schema) on first use; creation is
    serialized per key. The parent directory of the database is created if
    missing.

    Args:
        owner: The object (normally an :class:`~logpose.agent.Agent`) using the
            store. When the last owner releases it with :func:`release_stores`,
            the store is closed.
    """
    loop = asyncio.get_running_loop()
    path = db_path()
    key: _Key = (str(path), loop)
    with _GUARD:
        # A store whose loop has closed can never be matched again (for example a
        # long-lived agent driven by asyncio.run per request): stop it now rather
        # than leaking its connection and thread until aclose or exit.
        dead = [k for k, e in _ENTRIES.items() if e.loop.is_closed()]
        swept = [_ENTRIES.pop(k) for k in dead]
        for k in dead:
            _CREATE_LOCKS.pop(k, None)
        entry = _ENTRIES.get(key)
        lock = None if entry is not None else _CREATE_LOCKS.setdefault(key, asyncio.Lock())
    for stale in swept:
        await _abandon_async(stale)
    if lock is not None:
        async with lock:
            with _GUARD:
                entry = _ENTRIES.get(key)
            if entry is None:
                from logpose.store.sqlite import SQLiteStore
                from logpose.tracing.writer import SpanWriter

                path.parent.mkdir(parents=True, exist_ok=True)
                store = SQLiteStore(path)
                await store.open()
                _install_exit_hook()
                entry = _Entry(store, SpanWriter(store), loop)
                with _GUARD:
                    _ENTRIES[key] = entry
                    _OPEN[store] = entry
    assert entry is not None
    if owner is not None:
        entry.owners.add(owner)
    return entry.store


def span_writer(store: SessionStore) -> SpanWriter | None:
    """Return the span writer owned by ``store``, or ``None`` once it is closed."""
    with _GUARD:
        entry = _OPEN.get(cast("SQLiteStore", store))
    return entry.writer if entry is not None else None


async def release_stores(owner: object, run_ids: Iterable[str] = ()) -> None:
    """Drop ``owner`` from every store it used; close stores left with no owner.

    A store bound to the current loop is closed here, after its span writer has
    written what is queued (bounded by the writer's close timeout). A store
    still shared with other owners on this loop waits, bounded the same way,
    only for the spans of ``run_ids`` (the owner's own runs), never for other
    owners' spans. One bound to a loop that is running elsewhere is closed on
    that loop without waiting; one bound to a loop that is stopped or closed has
    its connection thread stopped directly and its leftover spans written
    without a loop.
    """
    from logpose.tracing.writer import CLOSE_TIMEOUT

    try:
        current: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
    except RuntimeError:  # pragma: no cover - always called from a coroutine
        current = None
    orphaned: list[_Entry] = []
    shared: list[_Entry] = []
    with _GUARD:
        for key, entry in list(_ENTRIES.items()):
            if owner in entry.owners:
                entry.owners.discard(owner)
                if not entry.owners:
                    del _ENTRIES[key]
                    _CREATE_LOCKS.pop(key, None)
                    orphaned.append(entry)
                elif entry.loop is current:
                    shared.append(entry)
    mine = list(run_ids)
    for entry in shared:
        if mine:
            await entry.writer.flush_runs(mine, CLOSE_TIMEOUT)
    for entry in orphaned:
        if entry.loop is current:
            await _close(entry)
        elif entry.loop.is_running() and not entry.loop.is_closed():
            with contextlib.suppress(RuntimeError):
                asyncio.run_coroutine_threadsafe(_close(entry), entry.loop)
        else:
            await _abandon_async(entry)


async def _close(entry: _Entry) -> None:
    """Close the writer (bounded), save any leftover spans, close the store."""
    try:
        try:
            await entry.writer.close()
        finally:
            left = entry.writer.drain()
            if left:
                await asyncio.to_thread(_save_leftovers, entry, left)
    finally:
        try:
            await entry.store.close()
        finally:
            with _GUARD:
                _OPEN.pop(entry.store, None)


async def _abandon_async(entry: _Entry) -> None:
    """:func:`_abandon` without blocking the running loop on the leftover write."""
    left = _stop_now(entry.store)
    if left:
        await asyncio.to_thread(_save_leftovers, entry, left)


def _abandon(entry: _Entry) -> None:
    """Stop a store whose loop is gone and write its leftover spans. Best effort."""
    left = _stop_now(entry.store)
    if left:
        _save_leftovers(entry, left)


def _save_leftovers(entry: _Entry, spans: list[SpanRecord]) -> None:
    """Write spans a writer never got to, with a stdlib connection. Never raises."""
    from logpose.store.sqlite import write_spans_blocking

    try:
        write_spans_blocking(entry.store._path, spans)
    except Exception as exc:  # noqa: BLE001 - best effort; reported once
        entry.writer.warn_lost(len(spans), type(exc).__name__)


def _stop_now(store: SQLiteStore) -> list[SpanRecord]:
    """Stop a store's connection thread without awaiting anything. Best effort.

    aiosqlite's worker thread is not a daemon, so an unclosed connection would
    otherwise keep the interpreter from exiting.

    Returns:
        The spans its writer still held, for a loop-free write by the caller.
    """
    with _GUARD:
        entry = _OPEN.pop(store, None)
    left = entry.writer.drain() if entry is not None else []
    db, store._db = store._db, None
    if db is None:
        return left
    # ``stop()`` exists from aiosqlite 0.22 (the declared floor); older
    # releases only have the private ``_stop_running()``.
    stop = getattr(db, "stop", None) or getattr(db, "_stop_running", None)
    error: Exception | None = None
    if stop is not None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                stop()
            return left
        except Exception as exc:  # noqa: BLE001 - best effort; reported below
            error = exc
    global _STOP_WARNED
    if not _STOP_WARNED:
        _STOP_WARNED = True
        _LOGGER.warning(
            "could not stop an aiosqlite connection thread (%s); interpreter exit may "
            "hang until it is closed. logpose requires aiosqlite>=0.22.",
            type(error).__name__ if error is not None else "no stop() method",
        )
    return left


def _shutdown() -> None:
    """Exit hook: flush writers whose loop still runs, then stop every store.

    A writer on a loop running in another thread (the sync facade's) gets a
    bounded close on that loop; anything left, and everything held by writers
    whose loop is gone, is written without a loop. Best effort, never raises.
    """
    with _GUARD:
        entries = list(_OPEN.values())
    for entry in entries:
        with contextlib.suppress(BaseException):
            _flush_on_own_loop(entry)
    for entry in entries:
        with contextlib.suppress(BaseException):
            _abandon(entry)


def _flush_on_own_loop(entry: _Entry) -> None:
    """Close ``entry``'s writer on its loop, if that loop is running in another thread."""
    loop = entry.loop
    if loop.is_closed() or not loop.is_running():
        return
    try:
        if asyncio.get_running_loop() is loop:  # pragma: no cover - not from exit hook
            return
    except RuntimeError:
        pass
    future = asyncio.run_coroutine_threadsafe(entry.writer.close(_EXIT_FLUSH_TIMEOUT), loop)
    future.result(_EXIT_FLUSH_TIMEOUT + 0.5)


def _install_exit_hook() -> None:
    """Stop open connections before the interpreter joins non-daemon threads.

    ``atexit`` handlers run only *after* that join, which an open aiosqlite
    connection would block forever, so the threading-shutdown hook is used when
    the interpreter provides it.
    """
    global _EXIT_HOOK_INSTALLED
    with _GUARD:
        if _EXIT_HOOK_INSTALLED:
            return
        _EXIT_HOOK_INSTALLED = True
    register = getattr(threading, "_register_atexit", None)
    try:
        if register is None:  # pragma: no cover - CPython 3.9+ always has it
            raise RuntimeError
        register(_shutdown)
    except RuntimeError:  # pragma: no cover - already shutting down, or no hook
        import atexit

        atexit.register(_shutdown)


def _open_count() -> int:
    """How many opened stores are not yet closed. For tests."""
    with _GUARD:
        return len(_OPEN)


def reset_stores_for_tests() -> None:
    """Forget every store, stopping open connections. Private: tests only."""
    with _GUARD:
        stores = list(_OPEN)
        _ENTRIES.clear()
        _CREATE_LOCKS.clear()
        _ACTIVE_HANDLES.clear()
        _RUNNING.clear()
    for store in stores:
        _stop_now(store)  # spans still queued are discarded: tests start clean
