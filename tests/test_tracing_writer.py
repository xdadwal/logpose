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

    async def slow(self: SQLiteStore, spans: Sequence[SpanRecord]) -> None:
        await asyncio.sleep(0.3)
        await original(self, spans)

    monkeypatch.setattr(SQLiteStore, "write_spans", slow)
    agent = Agent(
        FakeProvider(ScriptedTurn.text("ok")), model="m", tracing=Tracing(flush_timeout=0)
    )
    await agent.run("a", run_id="late-1")
    assert _rows(db, "SELECT COUNT(*) FROM logpose_spans") == [(0,)]
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

    async def slow(self: SQLiteStore, spans: Sequence[SpanRecord]) -> None:
        loops.append(asyncio.get_running_loop())
        await asyncio.sleep(0.3)
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
