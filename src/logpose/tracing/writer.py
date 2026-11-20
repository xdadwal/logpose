"""The background span writer: one per opened store.

Runs hand finished spans to :meth:`SpanWriter.submit`, which never blocks and
never raises. A single task on the store's event loop drains the queue in
bounded batches through ``store.write_spans``, so the store lock is held for
one batch at a time and live sessions are never stalled behind a backlog.

Failures are isolated from runs: a write error is logged once per writer to the
``logpose.trace`` logger and the batch is dropped; a full queue drops the span
and logs once; a flush that times out logs once and returns ``False``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections import Counter, deque
from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:  # pragma: no cover
    from logpose.tracing.spans import SpanRecord

__all__ = ["CLOSE_TIMEOUT", "DEFAULT_QUEUE_MAX", "SpanWriter"]

_LOGGER = logging.getLogger("logpose.trace")

DEFAULT_QUEUE_MAX = 10_000
"""Spans a writer holds before it starts dropping new ones."""

BATCH_MAX = 256
"""Spans written per transaction (the store lock is held for one batch)."""

CLOSE_TIMEOUT = 5.0
"""Seconds :meth:`SpanWriter.close` waits for queued spans before giving up."""

TASK_NAME = "logpose-span-writer"


class _SpanStore(Protocol):
    async def write_spans(self, spans: Sequence[SpanRecord]) -> None: ...

    async def run_id_exists(self, run_id: str) -> bool: ...


class SpanWriter:
    """Queue spans and write them to ``store`` from one background task.

    Bound to the event loop of the first :meth:`submit` (the store's loop).
    Pending spans are counted per run ID, so :meth:`flush` waits only for the
    run that asks.
    """

    def __init__(self, store: _SpanStore, *, queue_max: int = DEFAULT_QUEUE_MAX) -> None:
        self._store = store
        self._queue_max = queue_max
        self._queue: deque[SpanRecord] = deque()
        self._pending: Counter[str] = Counter()  # queued + in flight, per run ID
        self._waiters: dict[str | None, list[asyncio.Future[None]]] = {}
        self._wake: asyncio.Event | None = None
        self._task: asyncio.Task[None] | None = None
        self._closed = False
        self._warned: set[str] = set()

    # -- public --------------------------------------------------------------

    @property
    def pending(self) -> int:
        """Spans queued or being written."""
        return sum(self._pending.values())

    def submit(self, span: SpanRecord) -> None:
        """Queue ``span`` for writing. Never blocks; never raises.

        Drops the span (warning once) when the queue is full or the writer is closed.
        """
        try:
            if self._closed:
                self._warn_once("closed", "span writer is closed; spans are being dropped")
                return
            if len(self._queue) >= self._queue_max:
                self._warn_once(
                    "full",
                    "span queue is full (%d spans); spans are being dropped",
                    self._queue_max,
                )
                return
            self._queue.append(span)
            self._pending[span.run_id] += 1
            self._ensure_task()
            assert self._wake is not None
            self._wake.set()
        except Exception as exc:  # noqa: BLE001 - tracing never fails a run
            self._warn_once("submit", "could not queue a span: %s", type(exc).__name__)

    async def run_id_exists(self, run_id: str) -> bool:
        """Whether ``run_id`` has spans queued here or already in the store."""
        if self._pending[run_id] > 0:
            return True
        return await self._store.run_id_exists(run_id)

    async def flush(self, run_id: str | None, timeout: float) -> bool:
        """Wait up to ``timeout`` seconds for ``run_id``'s spans (``None``: all spans).

        Returns:
            ``True`` if nothing of that run is left unwritten. On timeout the
            spans stay queued, a WARNING is logged once per writer, and
            ``False`` is returned. Never raises, except on cancellation.
        """
        if self._outstanding(run_id) == 0:
            return True
        if self._closed and self._task is None:
            return False  # closed: what is left is for drain(), not a new task
        self._ensure_task()
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._waiters.setdefault(run_id, []).append(future)
        try:
            await asyncio.wait({future}, timeout=timeout)
        finally:
            waiting = self._waiters.get(run_id, [])
            if future in waiting:
                waiting.remove(future)
                if not waiting:
                    self._waiters.pop(run_id, None)
        if future.done():
            return True
        self._warn_once(
            "flush",
            "span flush timed out after %ss; unwritten spans stay queued and may be lost",
            timeout,
        )
        return False

    async def close(self, timeout: float = CLOSE_TIMEOUT) -> None:
        """Stop accepting spans, write what is queued (bounded), stop the task.

        Spans still unwritten after ``timeout`` stay in the queue for
        :meth:`drain`, which the store manager uses as a last resort.
        """
        if self._closed and self._task is None:
            return
        self._closed = True
        try:
            if self._task is not None and not self._task.done():
                await self.flush(None, timeout)
        finally:
            task, self._task = self._task, None
            if task is not None and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task

    def drain(self) -> list[SpanRecord]:
        """Take every unwritten span out of the writer (it is closed afterwards).

        For the store manager's last-resort, loop-free write once the writer's
        loop is gone. Safe to call from any thread once that loop has stopped.
        """
        self._closed = True
        spans = list(self._queue)
        self._queue.clear()
        self._pending.clear()
        return spans

    def warn_lost(self, count: int, reason: str) -> None:
        """Log (once per writer) that ``count`` spans could not be written."""
        if count:
            self._warn_once("lost", "%d spans were not written: %s", count, reason)

    # -- internals -----------------------------------------------------------

    def _outstanding(self, run_id: str | None) -> int:
        return self.pending if run_id is None else self._pending[run_id]

    def _warn_once(self, key: str, message: str, *args: object) -> None:
        if key in self._warned:
            return
        self._warned.add(key)
        _LOGGER.warning(message, *args)

    def _ensure_task(self) -> None:
        if self._wake is None:
            self._wake = asyncio.Event()
        if self._task is None or self._task.done():
            self._task = asyncio.get_running_loop().create_task(self._run(), name=TASK_NAME)

    def _settle(self, batch: Sequence[SpanRecord]) -> None:
        """Count ``batch`` as done and wake any flush it completes."""
        for span in batch:
            left = self._pending[span.run_id] - 1
            if left > 0:
                self._pending[span.run_id] = left
            else:
                del self._pending[span.run_id]
        done_runs = {span.run_id for span in batch if self._pending[span.run_id] == 0}
        targets: list[str | None] = [r for r in done_runs if r in self._waiters]
        if not self._pending and None in self._waiters:
            targets.append(None)
        for key in targets:
            for future in self._waiters.pop(key, []):
                if not future.done():
                    future.set_result(None)

    async def _run(self) -> None:
        assert self._wake is not None
        while True:
            if not self._queue:
                self._wake.clear()
                await self._wake.wait()
                continue
            batch = [self._queue.popleft() for _ in range(min(len(self._queue), BATCH_MAX))]
            try:
                await self._store.write_spans(batch)
            except asyncio.CancelledError:
                # Keep them for a last-resort write (see drain).
                self._queue.extendleft(reversed(batch))
                raise
            except Exception as exc:  # noqa: BLE001 - tracing never fails a run
                self._warn_once(
                    "write",
                    "could not write spans to the store (%s); tracing data is being dropped",
                    type(exc).__name__,
                )
            self._settle(batch)
