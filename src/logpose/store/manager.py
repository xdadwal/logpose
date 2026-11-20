"""Automatic management of the session store.

One opened :class:`~logpose.store.sqlite.SQLiteStore` exists per (absolute
database path, running event loop): an aiosqlite connection is bound to the
loop that awaits it, and the sync facade runs its own private loop. Stores are
created lazily on first session use, shared by every :class:`~logpose.agent.Agent`
on that loop, and closed by :meth:`~logpose.agent.Agent.aclose` of the last
agent using them, or at interpreter exit (best effort).

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
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from logpose.store.base import SessionHandle, SessionStore
    from logpose.store.sqlite import SQLiteStore

__all__ = ["DB_PATH_ENV", "DEFAULT_DB_FILENAME", "db_path", "get_store"]

DB_PATH_ENV = "LOGPOSE_DB_PATH"
"""Environment variable naming the session database file. Blank counts as unset."""

DEFAULT_DB_FILENAME = "logpose.db"
"""Database file used, in the working directory, when ``LOGPOSE_DB_PATH`` is unset."""

_Key = tuple[str, asyncio.AbstractEventLoop]

_LOGGER = logging.getLogger("logpose.store")
_STOP_WARNED = False


class _Entry:
    """One opened store and the agents currently using it."""

    def __init__(self, store: SQLiteStore, loop: asyncio.AbstractEventLoop) -> None:
        self.store = store
        self.loop = loop
        self.owners: weakref.WeakSet[object] = weakref.WeakSet()


_GUARD = threading.Lock()  # protects the registries below across loops/threads
_ENTRIES: dict[_Key, _Entry] = {}
_CREATE_LOCKS: dict[_Key, asyncio.Lock] = {}
_OPEN: set[SQLiteStore] = set()  # every opened, not yet closed store (for exit cleanup)
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
        _stop_now(stale.store)
    if lock is not None:
        async with lock:
            with _GUARD:
                entry = _ENTRIES.get(key)
            if entry is None:
                from logpose.store.sqlite import SQLiteStore

                path.parent.mkdir(parents=True, exist_ok=True)
                store = SQLiteStore(path)
                await store.open()
                _install_exit_hook()
                entry = _Entry(store, loop)
                with _GUARD:
                    _ENTRIES[key] = entry
                    _OPEN.add(store)
    assert entry is not None
    if owner is not None:
        entry.owners.add(owner)
    return entry.store


async def release_stores(owner: object) -> None:
    """Drop ``owner`` from every store it used; close stores left with no owner.

    A store bound to the current loop is closed here. One bound to a loop that is
    running elsewhere is closed on that loop without waiting; one bound to a loop
    that is stopped or closed has its connection thread stopped directly.
    """
    try:
        current: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
    except RuntimeError:  # pragma: no cover - always called from a coroutine
        current = None
    orphaned: list[_Entry] = []
    with _GUARD:
        for key, entry in list(_ENTRIES.items()):
            if owner in entry.owners:
                entry.owners.discard(owner)
                if not entry.owners:
                    del _ENTRIES[key]
                    _CREATE_LOCKS.pop(key, None)
                    orphaned.append(entry)
    for entry in orphaned:
        if entry.loop is current:
            await _close(entry.store)
        elif entry.loop.is_running() and not entry.loop.is_closed():
            with contextlib.suppress(RuntimeError):
                asyncio.run_coroutine_threadsafe(_close(entry.store), entry.loop)
        else:
            _stop_now(entry.store)


async def _close(store: SQLiteStore) -> None:
    try:
        await store.close()
    finally:
        with _GUARD:
            _OPEN.discard(store)


def _stop_now(store: SQLiteStore) -> None:
    """Stop a store's connection thread without awaiting anything. Best effort.

    aiosqlite's worker thread is not a daemon, so an unclosed connection would
    otherwise keep the interpreter from exiting.
    """
    with _GUARD:
        _OPEN.discard(store)
    db, store._db = store._db, None
    if db is None:
        return
    # ``stop()`` exists from aiosqlite 0.22 (the declared floor); older
    # releases only have the private ``_stop_running()``.
    stop = getattr(db, "stop", None) or getattr(db, "_stop_running", None)
    error: Exception | None = None
    if stop is not None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                stop()
            return
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


def _shutdown() -> None:
    with _GUARD:
        stores = list(_OPEN)
    for store in stores:
        _stop_now(store)


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
        _stop_now(store)
