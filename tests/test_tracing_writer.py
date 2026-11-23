"""The background span writer: store persistence, bounded flushing, failure isolation."""

from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
import stat
import threading
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest

from logpose import Agent, SyncAgent, run_sync, stream_sync, tool
from logpose.errors import LogposeError
from logpose.events import RunEnd
from logpose.store import manager
from logpose.store.sqlite import SQLiteStore
from logpose.tracing import semconv as sc
from logpose.tracing.spans import SpanRecord, Tracing
from logpose.tracing.writer import SpanWriter
from tests.fake_provider import FakeProvider, ScriptedTurn, tool_call


@tool
def add(a: int, b: int) -> str:
    """Add two numbers.

    Args:
        a: First.
        b: Second.
    """
    return str(a + b)


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv("LOGPOSE_TRACE", raising=False)
    monkeypatch.delenv("LOGPOSE_TRACE_CONTENT", raising=False)
    manager.reset_stores_for_tests()
    yield
    manager.reset_stores_for_tests()


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "t.db"
    monkeypatch.setenv("LOGPOSE_DB_PATH", str(path))
    return path


def _rows(path: Path, sql: str, args: Sequence[Any] = ()) -> list[tuple[Any, ...]]:
    with sqlite3.connect(path) as conn:
        return conn.execute(sql, tuple(args)).fetchall()


def _script() -> list[ScriptedTurn]:
    return [
        ScriptedTurn.tool_use(tool_call("add", {"a": 1, "b": 2}, id="c1")),
        ScriptedTurn.text("3"),
    ]


def _warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.levelno >= logging.WARNING]


def _span(run_id: str, n: int = 0) -> SpanRecord:
    return SpanRecord(
        trace_id="a" * 32,
        span_id=f"{n:016x}",
        parent_span_id=None,
        run_id=run_id,
        session_id=None,
        name="s",
        kind=1,
        start_ns=n,
        end_ns=n,
        status_code=0,
        status_message=None,
        attributes={},
        events=[],
    )


class _GatedStore:
    """A store whose writes wait on a gate; records what was written."""

    def __init__(self) -> None:
        self.gate = asyncio.Event()
        self.written: list[SpanRecord] = []

    async def write_spans(self, spans: Sequence[SpanRecord]) -> None:
        await self.gate.wait()
        self.written.extend(spans)

    async def run_id_exists(self, run_id: str) -> bool:
        return any(s.run_id == run_id for s in self.written)


def _writer_tasks() -> list[asyncio.Task[Any]]:
    return [t for t in asyncio.all_tasks() if t.get_name() == "logpose-span-writer"]


# -- Review Focus 1 ------------------------------------------------------------------


@pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0, reason="needs POSIX permissions")
async def test_unwritable_db_never_breaks_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    locked = tmp_path / "ro"
    locked.mkdir()
    locked.chmod(stat.S_IRUSR | stat.S_IXUSR)
    monkeypatch.setenv("LOGPOSE_DB_PATH", str(locked / "t.db"))
    try:
        plain = Agent(FakeProvider(_script()), model="m", tools=[add], tracing=None)
        expected = await plain.run("1+2?")
        traced = Agent(
            FakeProvider(_script() + _script()), model="m", tools=[add], tracing=Tracing()
        )
        caplog.set_level(logging.DEBUG)
        result = await traced.run("1+2?")
        again = await traced.run("1+2?")  # still untraced, not warned again
        assert again.text == "3"
        await traced.aclose()
    finally:
        locked.chmod(stat.S_IRWXU)
    assert result.text == expected.text == "3"
    assert result.stop_reason == expected.stop_reason
    assert result.iterations == expected.iterations
    assert result.usage == expected.usage
    assert [m.model_dump() for m in result.messages] == [m.model_dump() for m in expected.messages]
    (warning,) = _warnings(caplog)
    assert warning.name == "logpose.trace"
    assert not (locked / "t.db").exists()


async def test_write_failure_warns_once_and_never_breaks_runs(
    db: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    async def broken(self: SQLiteStore, spans: Sequence[SpanRecord]) -> None:
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(SQLiteStore, "write_spans", broken)
    caplog.set_level(logging.DEBUG)
    agent = Agent(FakeProvider(_script() + _script()), model="m", tools=[add], tracing=Tracing())
    first = await agent.run("1+2?")
    second = await agent.run("1+2?")
    await agent.aclose()
    assert first.text == second.text == "3"
    (warning,) = _warnings(caplog)
    assert warning.name == "logpose.trace"
    assert "OperationalError" in warning.getMessage()


# -- flushing ----------------------------------------------------------------------


async def test_flush_waits_bounded(
    db: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    original = SQLiteStore.write_spans

    async def slow(self: SQLiteStore, spans: Sequence[SpanRecord]) -> None:
        await asyncio.sleep(2)
        await original(self, spans)

    monkeypatch.setattr(SQLiteStore, "write_spans", slow)
    agent = Agent(
        FakeProvider(ScriptedTurn.text("hi")), model="m", tracing=Tracing(flush_timeout=0.2)
    )
    # Open the store first so the timing covers the run, not schema creation.
    await manager.get_store(owner=agent)
    caplog.set_level(logging.WARNING)
    began = time.monotonic()
    result = await agent.run("hello")
    elapsed = time.monotonic() - began
    assert result.text == "hi"
    assert 0.2 <= elapsed < 0.5
    assert [r.name for r in _warnings(caplog)] == ["logpose.trace"]
    assert "flush" in _warnings(caplog)[0].getMessage()
    await agent.aclose()


async def test_flush_waits_only_for_its_own_run() -> None:
    store = _GatedStore()
    writer = SpanWriter(store)  # type: ignore[arg-type]
    writer.submit(_span("busy"))
    assert await writer.flush("idle", 5.0) is True  # nothing of its own pending
    assert await writer.flush("busy", 0.05) is False
    store.gate.set()
    assert await writer.flush("busy", 5.0) is True
    assert [s.run_id for s in store.written] == ["busy"]
    await writer.close()
    assert _writer_tasks() == []


async def test_queue_full_drops_and_warns_once(caplog: pytest.LogCaptureFixture) -> None:
    store = _GatedStore()
    writer = SpanWriter(store, queue_max=3)  # type: ignore[arg-type]
    caplog.set_level(logging.WARNING)
    for n in range(10):
        writer.submit(_span("r", n))  # never blocks, never raises
    (warning,) = _warnings(caplog)
    assert warning.name == "logpose.trace"
    assert "full" in warning.getMessage()
    store.gate.set()
    assert await writer.flush("r", 5.0) is True
    assert [s.start_ns for s in store.written] == [0, 1, 2]
    writer.submit(_span("r", 11))  # room again: accepted
    assert await writer.flush("r", 5.0) is True
    assert len(store.written) == 4
    assert len(_warnings(caplog)) == 1
    await writer.close()


async def test_writer_batches_are_bounded() -> None:
    calls: list[int] = []

    class Counting(_GatedStore):
        async def write_spans(self, spans: Sequence[SpanRecord]) -> None:
            calls.append(len(spans))
            await super().write_spans(spans)

    store = Counting()
    store.gate.set()
    writer = SpanWriter(store)  # type: ignore[arg-type]
    for n in range(1000):
        writer.submit(_span("r", n))
    assert await writer.flush(None, 5.0) is True
    assert sum(calls) == 1000
    assert max(calls) <= 256
    await writer.close()


# -- persistence -------------------------------------------------------------------


async def test_spans_persisted_after_run(db: Path) -> None:
    agent = Agent(FakeProvider(_script()), model="m", tools=[add], tracing=Tracing())
    await agent.run("1+2?", run_id="persist-1", metadata={"item": "q1"})
    # Read before aclose: the run itself waited for its spans.
    rows = _rows(db, "SELECT DISTINCT trace_id FROM logpose_spans WHERE run_id = ?", ["persist-1"])
    (trace_id,) = (r[0] for r in rows)
    store = await manager.get_store()
    spans = [s async for s in store.read_spans(trace_id=trace_id)]
    assert sorted(s.name for s in spans) == [
        "chat m",
        "chat m",
        "execute_tool add",
        "invoke_agent logpose",
    ]
    (top,) = [s for s in spans if s.parent_span_id is None]
    assert top.attributes[sc.LOGPOSE_METADATA_PREFIX + "item"] == "q1"
    assert all(s.run_id == "persist-1" for s in spans)
    await agent.aclose()
    assert manager._open_count() == 0
    assert _writer_tasks() == []


async def test_duplicate_run_id_checked_against_store_without_session(db: Path) -> None:
    provider = FakeProvider(ScriptedTurn.text("ok"), repeat_last=True)
    first = Agent(provider, model="m", tracing=Tracing())
    await first.run("a", run_id="dup-store")
    await first.aclose()
    # A different agent (no shared in-memory state) still sees the stored spans.
    second = Agent(provider, model="m", tracing=Tracing())
    calls = provider.call_count
    with pytest.raises(LogposeError, match="run_id"):
        await second.run("b", run_id="dup-store")
    # A run recorded through a session (logpose_runs) is also seen.
    untraced = Agent(provider, model="m", tracing=None)
    await untraced.run("c", session_id="s1", run_id="dup-runs")
    with pytest.raises(LogposeError, match="run_id"):
        await second.run("d", run_id="dup-runs")
    assert provider.call_count == calls + 1
    await second.aclose()
    await untraced.aclose()


async def test_duplicate_run_id_sees_spans_still_queued(
    db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = SQLiteStore.write_spans
    gate = asyncio.Event()

    async def gated(self: SQLiteStore, spans: Sequence[SpanRecord]) -> None:
        await gate.wait()
        await original(self, spans)

    monkeypatch.setattr(SQLiteStore, "write_spans", gated)
    provider = FakeProvider(ScriptedTurn.text("ok"), repeat_last=True)
    agent = Agent(provider, model="m", tracing=Tracing(flush_timeout=0))
    await agent.run("a", run_id="queued-1")
    with pytest.raises(LogposeError, match="run_id"):
        await agent.run("b", run_id="queued-1")
    gate.set()
    await agent.aclose()
    assert _rows(db, "SELECT COUNT(*) FROM logpose_spans WHERE run_id = 'queued-1'") == [(2,)]


async def test_session_run_spans_are_persisted(db: Path) -> None:
    agent = Agent(FakeProvider(ScriptedTurn.text("ok")), model="m", tracing=Tracing())
    await agent.run("a", session_id="chat-1", run_id="sess-1")
    rows = _rows(
        db, "SELECT session_id, name FROM logpose_spans WHERE run_id = 'sess-1' ORDER BY name"
    )
    assert rows == [("chat-1", "chat m"), ("chat-1", "invoke_agent logpose")]
    await agent.aclose()


# -- lifecycle ---------------------------------------------------------------------


async def test_aclose_flushes_pending_spans(db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original = SQLiteStore.write_spans
    gate = asyncio.Event()

    async def gated(self: SQLiteStore, spans: Sequence[SpanRecord]) -> None:
        await gate.wait()
        await original(self, spans)

    monkeypatch.setattr(SQLiteStore, "write_spans", gated)
    agent = Agent(
        FakeProvider(ScriptedTurn.text("ok")), model="m", tracing=Tracing(flush_timeout=0)
    )
    await agent.run("a", run_id="late-1")
    assert _rows(db, "SELECT COUNT(*) FROM logpose_spans") == [(0,)]
    gate.set()
    await agent.aclose()
    assert _rows(db, "SELECT COUNT(*) FROM logpose_spans WHERE run_id = 'late-1'") == [(2,)]
    assert manager._open_count() == 0
    assert _writer_tasks() == []


def test_exit_hook_writes_spans_left_by_a_finished_loop(
    db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = SQLiteStore.write_spans

    async def stuck(self: SQLiteStore, spans: Sequence[SpanRecord]) -> None:
        await asyncio.sleep(60)
        await original(self, spans)

    monkeypatch.setattr(SQLiteStore, "write_spans", stuck)

    async def main() -> None:
        agent = Agent(
            FakeProvider(ScriptedTurn.text("ok")), model="m", tracing=Tracing(flush_timeout=0)
        )
        await agent.run("a", run_id="exit-1")
        # No aclose: asyncio.run cancels the writer when main returns.

    asyncio.run(main())
    assert _rows(db, "SELECT COUNT(*) FROM logpose_spans") == [(0,)]
    manager._shutdown()
    assert _rows(db, "SELECT COUNT(*) FROM logpose_spans WHERE run_id = 'exit-1'") == [(2,)]
    assert manager._open_count() == 0


def test_closed_loop_sweep_saves_pending_spans(db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def stuck(self: SQLiteStore, spans: Sequence[SpanRecord]) -> None:
        await asyncio.sleep(60)

    monkeypatch.setattr(SQLiteStore, "write_spans", stuck)
    agent = Agent(
        FakeProvider(ScriptedTurn.text("ok")), model="m", tracing=Tracing(flush_timeout=0)
    )
    asyncio.run(agent.run("a", run_id="sweep-1"))
    assert manager._open_count() == 1

    async def second() -> None:
        await manager.get_store(owner=agent)  # sweeps the store of the closed loop

    asyncio.run(second())
    assert _rows(db, "SELECT COUNT(*) FROM logpose_spans WHERE run_id = 'sweep-1'") == [(2,)]
    manager.reset_stores_for_tests()


# -- Review Focus 5 ------------------------------------------------------------------


def test_sync_facade_tracing(db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original = SQLiteStore.write_spans
    loops: list[asyncio.AbstractEventLoop] = []

    gate = threading.Event()  # set from the test thread; polled on the private loop

    async def slow(self: SQLiteStore, spans: Sequence[SpanRecord]) -> None:
        loops.append(asyncio.get_running_loop())
        while not gate.is_set():
            await asyncio.sleep(0.01)
        await original(self, spans)

    monkeypatch.setattr(SQLiteStore, "write_spans", slow)
    provider = FakeProvider(ScriptedTurn.text("ok"), repeat_last=True)
    agent = Agent(provider, model="m", tracing=Tracing(flush_timeout=0))
    sync_agent = SyncAgent(agent)
    result = sync_agent.run("a", run_id="sync-1", metadata={"via": "SyncAgent.run"})
    assert result.text == "ok"
    events = list(sync_agent.stream("b", run_id="sync-2", metadata={"via": "SyncAgent.stream"}))
    assert isinstance(events[-1], RunEnd)
    run_sync(agent, "c", run_id="sync-3", metadata={"via": "run_sync"})
    ends = [e for e in stream_sync(agent, "d", run_id="sync-4", metadata={"via": "stream_sync"})]
    assert isinstance(ends[-1], RunEnd)
    # flush_timeout=0 and a slow store: nothing is written before close().
    assert _rows(db, "SELECT COUNT(*) FROM logpose_spans") == [(0,)]
    with manager._GUARD:
        store_loops = {key[1] for key in manager._ENTRIES}
    assert threading.current_thread() is threading.main_thread()
    assert len(store_loops) == 1
    (private_loop,) = store_loops
    gate.set()
    sync_agent.close()
    assert loops and all(loop is private_loop for loop in loops)
    rows = _rows(
        db,
        "SELECT run_id, attributes FROM logpose_spans WHERE parent_span_id IS NULL ORDER BY run_id",
    )
    assert [r[0] for r in rows] == ["sync-1", "sync-2", "sync-3", "sync-4"]
    vias = [r[1] for r in rows]
    for via, label in zip(
        vias, ["SyncAgent.run", "SyncAgent.stream", "run_sync", "stream_sync"], strict=True
    ):
        assert f'"logpose.metadata.via": "{label}"' in via
    assert manager._open_count() == 0


# -- fix round 1 -------------------------------------------------------------------


async def test_aclose_waits_only_for_its_own_runs(
    db: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    original = SQLiteStore.write_spans
    gate = asyncio.Event()
    gate.set()

    async def gated(self: SQLiteStore, spans: Sequence[SpanRecord]) -> None:
        await gate.wait()
        await original(self, spans)

    monkeypatch.setattr(SQLiteStore, "write_spans", gated)
    provider = FakeProvider(ScriptedTurn.text("ok"), repeat_last=True)
    long_lived = Agent(provider, model="m", tracing=Tracing(flush_timeout=0.1))
    per_request = Agent(provider, model="m", tracing=Tracing())
    await long_lived.run("warm up")
    await per_request.run("request", run_id="req-1")  # written: nothing left pending
    gate.clear()
    store = await manager.get_store()
    writer = manager.span_writer(store)
    assert writer is not None
    writer.submit(_span("someone-else"))  # another agent's span, stuck behind the gate
    caplog.set_level(logging.WARNING)
    began = time.monotonic()
    await per_request.aclose()
    assert time.monotonic() - began < 0.5
    assert _warnings(caplog) == []
    # The run-end warning was not used up by the close.
    await long_lived.run("later")
    assert [r.name for r in _warnings(caplog)] == ["logpose.trace"]
    assert "flush" in _warnings(caplog)[0].getMessage()
    gate.set()
    await long_lived.aclose()


async def test_aclose_flushes_its_own_runs_on_a_shared_store(
    db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = SQLiteStore.write_spans
    gate = asyncio.Event()

    async def gated(self: SQLiteStore, spans: Sequence[SpanRecord]) -> None:
        await gate.wait()
        await original(self, spans)

    monkeypatch.setattr(SQLiteStore, "write_spans", gated)
    provider = FakeProvider(ScriptedTurn.text("ok"), repeat_last=True)
    keeper = Agent(provider, model="m", tracing=None)
    await manager.get_store(owner=keeper)  # keeps the store shared and open
    agent = Agent(provider, model="m", tracing=Tracing(flush_timeout=0))
    await agent.run("a", run_id="mine-1")
    asyncio.get_running_loop().call_later(0.1, gate.set)
    await agent.aclose()
    assert _rows(db, "SELECT COUNT(*) FROM logpose_spans WHERE run_id = 'mine-1'") == [(2,)]
    await keeper.aclose()


def _big_span(run_id: str, n: int, chars: int) -> SpanRecord:
    span = _span(run_id, n)
    span.attributes["gen_ai.input.messages"] = "x" * chars
    return span


async def test_queue_bounded_by_bytes(caplog: pytest.LogCaptureFixture) -> None:
    store = _GatedStore()
    writer = SpanWriter(store, queue_max_bytes=10_000)  # type: ignore[arg-type]
    caplog.set_level(logging.WARNING)
    for n in range(5):
        writer.submit(_big_span("r", n, 4_000))
    (warning,) = _warnings(caplog)
    assert "full" in warning.getMessage()
    store.gate.set()
    assert await writer.flush("r", 5.0) is True
    assert [s.start_ns for s in store.written] == [0, 1]
    writer.submit(_big_span("r", 9, 4_000))  # bytes were released: accepted
    assert await writer.flush("r", 5.0) is True
    assert len(store.written) == 3
    assert len(_warnings(caplog)) == 1
    await writer.close()


async def test_batches_bounded_by_bytes() -> None:
    calls: list[int] = []

    class Counting(_GatedStore):
        async def write_spans(self, spans: Sequence[SpanRecord]) -> None:
            calls.append(len(spans))
            await super().write_spans(spans)

    store = Counting()
    store.gate.set()
    writer = SpanWriter(store)  # type: ignore[arg-type]
    for n in range(10):
        writer.submit(_big_span("r", n, 1024 * 1024))
    assert await writer.flush(None, 5.0) is True
    assert sum(calls) == 10
    assert max(calls) <= 4
    await writer.close()


async def test_store_open_failure_backs_off(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    attempts: list[int] = []

    async def broken(*, owner: object | None = None, path: Path | None = None) -> Any:
        attempts.append(1)
        raise LogposeError("cannot open")

    monkeypatch.setattr(manager, "get_store", broken)
    caplog.set_level(logging.WARNING)
    agent = Agent(
        FakeProvider(ScriptedTurn.text("ok"), repeat_last=True), model="m", tracing=Tracing()
    )
    await agent.run("a")
    await agent.run("b")
    assert len(attempts) == 1  # the second run did not touch the store
    agent._trace_open_retry_at = 0.0  # the back-off has expired
    await agent.run("c")
    assert len(attempts) == 2
    assert len(_warnings(caplog)) == 1


def test_exit_hook_never_raises(db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    agent = Agent(FakeProvider(ScriptedTurn.text("ok")), model="m", tracing=Tracing())
    asyncio.run(agent.run("a"))
    assert manager._open_count() == 1

    def boom(entry: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(manager, "_abandon", boom)
    manager._shutdown()  # must not raise
    monkeypatch.undo()
    manager.reset_stores_for_tests()


def test_submit_without_a_loop_counts_nothing(caplog: pytest.LogCaptureFixture) -> None:
    writer = SpanWriter(_GatedStore())  # type: ignore[arg-type]
    caplog.set_level(logging.WARNING)
    writer.submit(_span("r"))  # no running loop: dropped, never raised
    assert writer.pending == 0
    assert len(_warnings(caplog)) == 1


# -- another process holding the database write lock --------------------------------


def _hold_write_lock(path: Path) -> sqlite3.Connection:
    """A second connection (as another process would) holding the write lock."""
    conn = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
    conn.execute("BEGIN IMMEDIATE")
    return conn


def _same_result(a: Any, b: Any) -> None:
    assert a.text == b.text
    assert a.stop_reason == b.stop_reason
    assert a.iterations == b.iterations
    assert a.usage == b.usage
    assert [m.model_dump() for m in a.messages] == [m.model_dump() for m in b.messages]


async def _until(predicate: Any, timeout: float = 8.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition not reached"
        await asyncio.sleep(0.02)


async def test_run_id_check_is_bounded_while_another_process_holds_the_lock(
    db: Path, caplog: pytest.LogCaptureFixture
) -> None:
    timeout = 0.5
    provider = FakeProvider(ScriptedTurn.text("ok"), repeat_last=True)
    expected = await Agent(provider, model="m", tracing=None).run("x")
    agent = Agent(provider, model="m", tracing=Tracing(flush_timeout=timeout))
    await agent.run("warm-up")  # the store is open and its schema checked
    caplog.set_level(logging.WARNING)
    blocker = _hold_write_lock(db)
    try:
        await agent.run("x")  # its spans' batch now waits in SQLite's busy handler
        start = time.monotonic()
        result = await agent.run("x", run_id="custom-1")
        elapsed = time.monotonic() - start
    finally:
        blocker.execute("ROLLBACK")
        blocker.close()
    assert elapsed < timeout + 0.4, elapsed
    _same_result(result, expected)
    checks = [r for r in _warnings(caplog) if "run_id" in r.getMessage()]
    assert len(checks) == 1 and checks[0].name == "logpose.trace"
    # Lock released: later runs are checked and traced normally.
    await agent.run("x", run_id="custom-2")
    with pytest.raises(LogposeError, match="run_id"):
        await agent.run("x", run_id="custom-2")
    await agent.aclose()
    assert _rows(db, "SELECT COUNT(*) FROM logpose_spans WHERE run_id = 'custom-1'") == [(2,)]
    assert _rows(db, "SELECT COUNT(*) FROM logpose_spans WHERE run_id = 'custom-2'") == [(2,)]


async def test_first_store_open_is_bounded_while_another_process_holds_the_lock(
    db: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    async with SQLiteStore(db):  # an existing database (WAL, current schema)
        pass
    opens: list[int] = []
    original = SQLiteStore.open

    async def counted(self: SQLiteStore) -> None:
        opens.append(1)
        await original(self)

    monkeypatch.setattr(SQLiteStore, "open", counted)
    timeout = 0.5
    provider = FakeProvider(ScriptedTurn.text("ok"), repeat_last=True)
    expected = await Agent(provider, model="m", tracing=None).run("x")
    agent = Agent(provider, model="m", tracing=Tracing(flush_timeout=timeout))
    caplog.set_level(logging.WARNING)
    blocker = _hold_write_lock(db)
    try:
        start = time.monotonic()
        first = await agent.run("x", run_id="locked-1")
        elapsed = time.monotonic() - start
        second = await agent.run("x", run_id="locked-2")  # still opening: untraced too
    finally:
        blocker.execute("ROLLBACK")
        blocker.close()
    assert elapsed < timeout + 0.4, elapsed
    _same_result(first, expected)
    _same_result(second, expected)
    (warning,) = _warnings(caplog)
    assert warning.name == "logpose.trace"
    # The open carried on in the background; the next run uses it.
    await _until(lambda: manager._open_count() == 1)
    await agent.run("x", run_id="after-lock")
    assert len(opens) == 1
    assert len(_warnings(caplog)) == 1
    assert _rows(db, "SELECT COUNT(*) FROM logpose_spans WHERE run_id = 'after-lock'") == [(2,)]
    assert _rows(db, "SELECT COUNT(*) FROM logpose_spans WHERE run_id LIKE 'locked-%'") == [(0,)]
    await agent.aclose()
    assert manager._open_count() == 0


async def test_slow_open_that_then_fails_still_backs_off(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    attempts: list[int] = []

    async def slow_then_broken(*, owner: object | None = None, path: Path | None = None) -> Any:
        attempts.append(1)
        await asyncio.sleep(0.4)
        raise LogposeError("cannot open")

    monkeypatch.setattr(manager, "get_store", slow_then_broken)
    caplog.set_level(logging.WARNING)
    agent = Agent(
        FakeProvider(ScriptedTurn.text("ok"), repeat_last=True),
        model="m",
        tracing=Tracing(flush_timeout=0.3),
    )
    start = time.monotonic()
    await agent.run("a")  # gives up waiting; the open goes on and fails
    assert time.monotonic() - start < 0.7
    await asyncio.sleep(0.3)
    await agent.run("b")  # sees the failure: backs off
    await agent.run("c")
    assert len(attempts) == 1
    assert [r.name for r in _warnings(caplog)] == ["logpose.trace", "logpose.trace"]


async def test_pending_slow_open_is_waited_on_only_once(
    db: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    original = manager.get_store

    async def slow(*, owner: object | None = None, path: Path | None = None) -> Any:
        await asyncio.sleep(1.0)
        return await original(owner=owner, path=path)

    monkeypatch.setattr(manager, "get_store", slow)
    caplog.set_level(logging.WARNING)
    provider = FakeProvider(ScriptedTurn.text("ok"), repeat_last=True)
    agent = Agent(provider, model="m", tracing=Tracing(flush_timeout=0.3))
    other = Agent(provider, model="m", tracing=Tracing(flush_timeout=0.3))
    times = []
    for runner in (agent, agent, agent, other):
        start = time.monotonic()
        await runner.run("x")
        times.append(time.monotonic() - start)
    assert times[0] >= 0.3
    # Later runs (any agent) do not wait on the same pending open again.
    assert max(times[1:]) < 0.15, times
    await _until(lambda: manager._open_count() == 1)
    await agent.run("x", run_id="after-open")
    await agent.aclose()
    await other.aclose()
    assert _rows(db, "SELECT COUNT(*) FROM logpose_spans WHERE run_id = 'after-open'") == [(2,)]
