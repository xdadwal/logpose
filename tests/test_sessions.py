"""Automatic SQLite sessions: the store manager and Agent integration."""

from __future__ import annotations

import asyncio
import contextlib
import os
import sqlite3
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from logpose.agent import Agent, Conversation
from logpose.errors import LogposeError, ProviderError, SessionBusyError
from logpose.events import RunEnd, TextDelta
from logpose.messages import Usage
from logpose.retry import RetryPolicy
from logpose.store import manager
from logpose.store.base import SessionHandle
from logpose.tools import tool
from tests.fake_provider import FakeProvider, ScriptedTurn, tool_call

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    path = tmp_path / "s.db"
    monkeypatch.setenv("LOGPOSE_DB_PATH", str(path))
    manager.reset_stores_for_tests()
    yield path
    manager.reset_stores_for_tests()


def _rows(path: Path, sql: str, *params: object) -> list[tuple[Any, ...]]:
    with sqlite3.connect(path) as conn:
        return list(conn.execute(sql, params))


def _usage(i: int, o: int) -> Usage:
    return Usage(input_tokens=i, output_tokens=o, reasoning_tokens=1)


@tool
async def nap(seconds: float) -> str:
    """Sleep.

    Args:
        seconds: How long.
    """
    await asyncio.sleep(seconds)
    return "rested"


def test_no_session_never_touches_disk(tmp_path: Path) -> None:
    db_path = tmp_path / "never.db"
    code = (
        "import asyncio, os, sys\n"
        "sys.path.insert(0, os.getcwd())\n"
        "from logpose import Agent\n"
        "from tests.fake_provider import FakeProvider, ScriptedTurn\n"
        "async def main():\n"
        "    agent = Agent(FakeProvider(ScriptedTurn.text('hi')), model='m')\n"
        "    await agent.run('hello')\n"
        "    await agent.aclose()\n"
        "asyncio.run(main())\n"
        "assert 'aiosqlite' not in sys.modules, 'aiosqlite leaked'\n"
        "assert 'sqlite3' not in sys.modules, 'sqlite3 leaked'\n"
        "print('ok')\n"
    )
    env = {**os.environ, "LOGPOSE_DB_PATH": str(db_path)}
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
        cwd=ROOT,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
    assert not db_path.exists()


def test_unclosed_session_store_does_not_block_interpreter_exit(tmp_path: Path) -> None:
    """aiosqlite's worker thread is non-daemon; an unclosed store must not hang exit."""
    db_path = tmp_path / "exit.db"
    code = (
        "import asyncio, os, sys\n"
        "sys.path.insert(0, os.getcwd())\n"
        "from logpose import Agent\n"
        "from tests.fake_provider import FakeProvider, ScriptedTurn\n"
        "async def main():\n"
        "    agent = Agent(FakeProvider(ScriptedTurn.text('hi')), model='m')\n"
        "    await agent.run('hello', session_id='s1')\n"
        "asyncio.run(main())\n"
        "print('ok')\n"
    )
    env = {**os.environ, "LOGPOSE_DB_PATH": str(db_path)}
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
        cwd=ROOT,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
    assert _rows(db_path, "SELECT COUNT(*) FROM logpose_messages") == [(2,)]


async def test_session_id_persists_history_across_runs(db: Path) -> None:
    provider = FakeProvider([ScriptedTurn.text("Hi Ada."), ScriptedTurn.text("You are Ada.")])
    agent = Agent(provider, model="m")
    await agent.run("I am Ada.", session_id="s1")
    result = await agent.run("Who am I?", session_id="s1")
    assert result.text == "You are Ada."
    texts = [m.text for m in provider.requests[1].messages]
    assert texts == ["I am Ada.", "Hi Ada.", "Who am I?"]
    assert _rows(db, "SELECT COUNT(*) FROM logpose_messages WHERE session_id = 's1'") == [(4,)]
    await agent.aclose()


async def test_session_metadata_is_stored(db: Path) -> None:
    agent = Agent(FakeProvider(ScriptedTurn.text("ok")), model="m")
    await agent.run("hi", session_id="s1", session_metadata={"user": "u1"})
    assert _rows(db, "SELECT metadata FROM logpose_sessions") == [('{"user": "u1"}',)]
    await agent.aclose()


async def test_explicit_session_raises_busy_before_streaming() -> None:
    provider = FakeProvider([ScriptedTurn.text("one")])
    agent = Agent(provider, model="m")
    async with agent.session("s1") as handle:
        assert isinstance(handle, SessionHandle)
        with pytest.raises(SessionBusyError):
            async with agent.session("s1"):
                pytest.fail("second lease must not be granted")  # pragma: no cover
        events = [e async for e in agent.stream("hi", session=handle)]
        assert isinstance(events[-1], RunEnd)
    assert provider.call_count == 1
    await agent.aclose()


async def test_session_id_stream_surfaces_busy_on_first_anext() -> None:
    agent = Agent(FakeProvider([ScriptedTurn.text("x")]), model="m")
    async with agent.session("s1"):
        events = agent.stream("hi", session_id="s1")  # no lease taken yet
        with pytest.raises(SessionBusyError):
            await events.__anext__()
    await agent.aclose()


async def test_explicit_session_persists_on_block_exit(db: Path) -> None:
    provider = FakeProvider([ScriptedTurn.text("a"), ScriptedTurn.text("b")])
    agent = Agent(provider, model="m")
    async with agent.session("s1", metadata={"k": "v"}) as handle:
        await agent.run("one", session=handle)
        await agent.run("two", session=handle)
        assert len(handle.conversation) == 4
    assert _rows(db, "SELECT COUNT(*) FROM logpose_messages") == [(4,)]
    assert _rows(db, "SELECT COUNT(*) FROM logpose_runs") == [(2,)]
    assert _rows(db, "SELECT lease_token FROM logpose_sessions") == [(None,)]
    await agent.aclose()


async def test_handle_rejected_after_block_exit() -> None:
    agent = Agent(FakeProvider([ScriptedTurn.text("a")]), model="m")
    async with agent.session("s1") as handle:
        pass
    with pytest.raises(LogposeError):
        agent.stream("hi", session=handle)
    await agent.aclose()


async def test_run_record_written_with_usage(db: Path) -> None:
    provider = FakeProvider(
        [
            ScriptedTurn.tool_use(tool_call("nap", {"seconds": 0}), usage=_usage(10, 3)),
            ScriptedTurn.text("done", usage=_usage(7, 2)),
        ]
    )
    agent = Agent(provider, model="m", tools=[nap])
    await agent.run("go", session_id="s1")
    rows = _rows(
        db,
        "SELECT outcome, iterations, input_tokens, output_tokens, reasoning_tokens, "
        "model, provider, error_type, session_id FROM logpose_runs",
    )
    assert rows == [("completed", 2, 17, 5, 2, "m", "fake", None, "s1")]
    await agent.aclose()


async def test_error_and_timeout_outcomes_recorded(db: Path) -> None:
    provider = FakeProvider(
        [
            ScriptedTurn.tool_use(tool_call("nap", {"seconds": 0}), usage=_usage(5, 1)),
            ScriptedTurn.failure(ProviderError("boom")),
        ]
    )
    agent = Agent(provider, model="m", tools=[nap], retry_policy=RetryPolicy(max_attempts=1))
    with pytest.raises(ProviderError):
        await agent.run("go", session_id="s1")

    slow = FakeProvider(
        [ScriptedTurn.tool_use(tool_call("nap", {"seconds": 5}), usage=_usage(9, 4))]
    )
    timed = Agent(
        slow, model="m", tools=[nap], run_timeout=0.3, provider_turn_timeout=None, tool_timeout=None
    )
    from logpose.errors import RunTimeoutError

    with pytest.raises(RunTimeoutError):
        await timed.run("go", session_id="s2")

    rows = _rows(
        db,
        "SELECT session_id, outcome, error_type, iterations, input_tokens, output_tokens "
        "FROM logpose_runs ORDER BY session_id",
    )
    assert rows == [
        ("s1", "error", "ProviderError", 2, 5, 1),
        ("s2", "run_timeout", None, 1, 9, 4),
    ]
    # Partial history is persisted, and the leases are released.
    assert _rows(db, "SELECT COUNT(*) FROM logpose_sessions WHERE lease_token IS NULL") == [(2,)]
    await agent.aclose()
    await timed.aclose()


async def test_max_iterations_outcome_recorded(db: Path) -> None:
    from logpose.errors import MaxIterationsError

    provider = FakeProvider(
        [ScriptedTurn.tool_use(tool_call("nap", {"seconds": 0}), usage=_usage(1, 1))],
        repeat_last=True,
    )
    agent = Agent(provider, model="m", tools=[nap], max_iterations=2)
    with pytest.raises(MaxIterationsError):
        await agent.run("go", session_id="s1")
    assert _rows(db, "SELECT outcome, iterations, input_tokens FROM logpose_runs") == [
        ("max_iterations", 2, 2)
    ]
    await agent.aclose()


async def test_cancel_mid_stream_persists_and_releases(db: Path) -> None:
    provider = FakeProvider(
        [ScriptedTurn.text("hello there", chunks=["hello", " there"], usage=_usage(4, 2))]
    )
    agent = Agent(provider, model="m")
    events = agent.stream("hi", session_id="s1")
    async for event in events:
        if isinstance(event, TextDelta):
            break
    await events.aclose()  # type: ignore[attr-defined]
    assert _rows(db, "SELECT outcome FROM logpose_runs") == [("cancelled",)]
    assert _rows(db, "SELECT lease_token FROM logpose_sessions") == [(None,)]
    # The user turn was appended before the provider was called.
    assert _rows(db, "SELECT role FROM logpose_messages") == [("user",)]
    # The session is immediately usable again.
    async with agent.session("s1") as handle:
        assert [m.text for m in handle.conversation] == ["hi"]
    await agent.aclose()


async def test_task_cancellation_persists_and_releases(db: Path) -> None:
    provider = FakeProvider(
        [ScriptedTurn.tool_use(tool_call("nap", {"seconds": 10}), usage=_usage(3, 3))]
    )
    agent = Agent(provider, model="m", tools=[nap], tool_timeout=None)
    task = asyncio.create_task(agent.run("go", session_id="s1"))
    while provider.call_count == 0:
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert _rows(db, "SELECT outcome, input_tokens FROM logpose_runs") == [("cancelled", 3)]
    assert _rows(db, "SELECT lease_token FROM logpose_sessions") == [(None,)]
    await agent.aclose()


async def test_second_cancel_during_exit_still_saves(
    db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A repeated cancellation while the exit runs must not skip the save."""
    from logpose.store.sqlite import SQLiteStore

    real = SQLiteStore.record_run
    entered = asyncio.Event()

    async def slow_record(self: SQLiteStore, record: Any) -> None:
        entered.set()
        await asyncio.sleep(0.2)
        await real(self, record)

    monkeypatch.setattr(SQLiteStore, "record_run", slow_record)
    provider = FakeProvider([ScriptedTurn.text("x", delay=10)])
    agent = Agent(provider, model="m", provider_turn_timeout=None)
    task = asyncio.create_task(agent.run("go", session_id="s1"))
    while provider.call_count == 0:
        await asyncio.sleep(0.01)
    task.cancel()
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert _rows(db, "SELECT outcome FROM logpose_runs") == [("cancelled",)]
    assert _rows(db, "SELECT lease_token FROM logpose_sessions") == [(None,)]
    assert _rows(db, "SELECT COUNT(*) FROM logpose_messages") == [(1,)]
    await agent.aclose()


async def test_lost_lease_writes_run_but_not_messages(db: Path) -> None:
    holder: list[SessionHandle] = []

    @tool
    def lose() -> str:
        """Simulate losing the lease."""
        holder[0].lost = True
        return "ok"

    provider = FakeProvider(
        [
            ScriptedTurn.tool_use(tool_call("lose"), usage=_usage(6, 1)),
            ScriptedTurn.text("done", usage=_usage(2, 1)),
        ]
    )
    agent = Agent(provider, model="m", tools=[lose])
    with pytest.raises(SessionBusyError):
        async with agent.session("s1") as handle:
            holder.append(handle)
            await agent.run("go", session=handle)
    assert _rows(db, "SELECT outcome, input_tokens FROM logpose_runs") == [("completed", 8)]
    assert _rows(db, "SELECT COUNT(*) FROM logpose_messages") == [(0,)]
    assert _rows(db, "SELECT lease_token FROM logpose_sessions") == [(None,)]
    await agent.aclose()


async def test_lost_lease_on_session_id_path_raises_busy(
    db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    @tool
    def lose() -> str:
        """Simulate losing the lease."""
        for h in list(manager._ACTIVE_HANDLES):
            h.lost = True
        return "ok"

    provider = FakeProvider(
        [ScriptedTurn.tool_use(tool_call("lose"), usage=_usage(6, 1)), ScriptedTurn.text("d")]
    )
    agent = Agent(provider, model="m", tools=[lose])
    with pytest.raises(SessionBusyError):
        await agent.run("go", session_id="s1")
    assert _rows(db, "SELECT outcome, input_tokens FROM logpose_runs") == [("completed", 6)]
    assert _rows(db, "SELECT COUNT(*) FROM logpose_messages") == [(0,)]
    await agent.aclose()


async def test_structured_output_rollback_persists_only_kept_messages(db: Path) -> None:
    from pydantic import BaseModel

    class Answer(BaseModel):
        n: int

    provider = FakeProvider(
        [
            ScriptedTurn.text("seeded"),
            ScriptedTurn.text("not json"),
            ScriptedTurn.text('{"n": 3}'),
        ]
    )
    agent = Agent(provider, model="m")
    await agent.run("seed", session_id="s1")
    result = await agent.run("q", session_id="s1", output_type=Answer)
    assert result.output == Answer(n=3)
    rows = _rows(db, "SELECT seq, content FROM logpose_messages ORDER BY seq")
    # The invalid answer and the correction prompt were rolled back, not saved.
    assert [r[0] for r in rows] == [0, 1, 2, 3]
    assert "not json" not in "".join(r[1] for r in rows)
    await agent.aclose()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"session_id": "s1", "conversation": "conv"},
        {"session_id": "s1", "session": "handle"},
        {"session": "handle", "conversation": "conv"},
        {"session_metadata": {"a": "b"}},
    ],
)
async def test_session_argument_conflicts(kwargs: dict[str, Any]) -> None:
    agent = Agent(FakeProvider([ScriptedTurn.text("x")]), model="m")
    async with agent.session("h1") as handle:
        resolved = {
            k: (Conversation() if v == "conv" else handle if v == "handle" else v)
            for k, v in kwargs.items()
        }
        with pytest.raises(LogposeError):
            agent.stream("hi", **resolved)
        with pytest.raises(LogposeError):
            await agent.run("hi", **resolved)
    await agent.aclose()


async def test_invalid_session_id_and_wait() -> None:
    agent = Agent(FakeProvider([ScriptedTurn.text("x")]), model="m")
    with pytest.raises(LogposeError):
        agent.stream("hi", session_id="bad id!")
    with pytest.raises(LogposeError):
        Agent(FakeProvider([]), model="m", session_wait=-1)
    await agent.aclose()


async def test_session_wait_waits_for_release() -> None:
    agent = Agent(FakeProvider([ScriptedTurn.text("x")]), model="m", session_wait=2.0)
    release = asyncio.Event()

    async def holder() -> None:
        async with agent.session("s1"):
            await release.wait()

    t = asyncio.create_task(holder())
    await asyncio.sleep(0.1)
    asyncio.get_running_loop().call_later(0.2, release.set)
    result = await agent.run("hi", session_id="s1")
    assert result.text == "x"
    await t
    await agent.aclose()


async def test_purge_and_delete_via_agent(db: Path) -> None:
    agent = Agent(FakeProvider([ScriptedTurn.text("a"), ScriptedTurn.text("b")]), model="m")
    await agent.run("one", session_id="s1")
    await agent.run("two", session_id="s2")
    await agent.delete_session("s1")
    assert _rows(db, "SELECT id FROM logpose_sessions") == [("s2",)]
    assert await agent.purge_expired_sessions() == 0
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE logpose_sessions SET expires_at = 0")
    assert await agent.purge_expired_sessions() == 1
    assert _rows(db, "SELECT COUNT(*) FROM logpose_messages") == [(0,)]
    await agent.aclose()


def test_db_path_from_env_and_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "nested" / "dir" / "x.db"
    monkeypatch.setenv("LOGPOSE_DB_PATH", str(target))
    assert manager.db_path() == target
    monkeypatch.setenv("LOGPOSE_DB_PATH", "   ")
    assert manager.db_path() == Path.cwd() / "logpose.db"
    monkeypatch.delenv("LOGPOSE_DB_PATH")
    assert manager.db_path() == Path.cwd() / "logpose.db"


async def test_parent_dir_created_and_store_shared_per_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "nested" / "dir" / "x.db"
    monkeypatch.setenv("LOGPOSE_DB_PATH", str(target))
    a = Agent(FakeProvider([ScriptedTurn.text("a")]), model="m")
    b = Agent(FakeProvider([ScriptedTurn.text("b")]), model="m")
    sa = await a._store()
    sb = await b._store()
    assert sa is sb
    assert target.exists()
    await a.aclose()
    # Still open for b.
    await b.run("hi", session_id="s1")
    await b.aclose()
    assert manager._open_count() == 0


def test_sync_facade_uses_its_own_loop_store(db: Path) -> None:
    from logpose.sync import SyncAgent

    async def outer() -> None:
        agent = Agent(FakeProvider([ScriptedTurn.text("a")]), model="m")
        await agent.run("hi", session_id="s1")
        await agent.aclose()

    asyncio.run(outer())
    with SyncAgent(Agent(FakeProvider([ScriptedTurn.text("b")]), model="m")) as sync_agent:
        result = sync_agent.run("again", session_id="s1")
        assert result.text == "b"
    assert _rows(db, "SELECT COUNT(*) FROM logpose_messages") == [(4,)]
    assert manager._open_count() == 0


def test_closed_loop_stores_are_swept(db: Path) -> None:
    """A long-lived agent driven by asyncio.run per request must not leak connections."""
    import threading

    agent = Agent(FakeProvider([ScriptedTurn.text("x")], repeat_last=True), model="m")
    asyncio.run(agent.run("hi", session_id="s1"))
    threads_after_first = threading.active_count()
    for _ in range(4):
        asyncio.run(agent.run("hi", session_id="s1"))
    assert manager._open_count() == 1
    deadline = time.monotonic() + 2
    while threading.active_count() > threads_after_first and time.monotonic() < deadline:
        time.sleep(0.01)
    assert threading.active_count() <= threads_after_first
    assert _rows(db, "SELECT COUNT(*) FROM logpose_runs") == [(5,)]

    async def close() -> None:
        await agent.aclose()  # tolerates the swept stores

    asyncio.run(close())
    assert manager._open_count() == 0


# --- every persisted tool_use is answered, whatever ends the run -------------------


def _assert_replayable(messages: list[Any], ids: list[str], text: str) -> None:
    """The history ends with one user message answering every call in ``ids``."""
    from logpose.messages import ToolResultBlock, ToolUseBlock

    assert messages[-2].role == "assistant"
    assert [b.id for b in messages[-2].content if isinstance(b, ToolUseBlock)] == ids
    last = messages[-1]
    assert last.role == "user"
    assert all(isinstance(b, ToolResultBlock) for b in last.content)
    assert [b.tool_use_id for b in last.content] == ids
    assert all(b.is_error and b.content == text for b in last.content)


_ENDED = "Run ended before this tool finished."


async def test_cancel_during_tool_persists_answered_tool_use(db: Path) -> None:
    provider = FakeProvider(
        [
            ScriptedTurn.tool_use(
                tool_call("nap", {"seconds": 10}, id="t1"),
                tool_call("nap", {"seconds": 10}, id="t2"),
            ),
            ScriptedTurn.text("recovered"),
        ]
    )
    agent = Agent(provider, model="m", tools=[nap], tool_timeout=None)
    task = asyncio.create_task(agent.run("go", session_id="s1"))
    while provider.call_count == 0:
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    async with agent.session("s1") as handle:
        _assert_replayable(list(handle.conversation), ["t1", "t2"], _ENDED)
    # The session stays usable: the next request carries the answered history.
    result = await agent.run("again", session_id="s1")
    assert result.text == "recovered"
    await agent.aclose()


async def test_consumer_break_after_tool_call_persists_answered_tool_use(db: Path) -> None:
    from logpose.events import ToolCall

    provider = FakeProvider([ScriptedTurn.tool_use(tool_call("nap", {"seconds": 0}, id="t1"))])
    agent = Agent(provider, model="m", tools=[nap])
    events = agent.stream("go", session_id="s1")
    async for event in events:
        if isinstance(event, ToolCall):
            break
    await events.aclose()  # type: ignore[attr-defined]
    async with agent.session("s1") as handle:
        _assert_replayable(list(handle.conversation), ["t1"], _ENDED)
    await agent.aclose()


async def test_raising_gate_persists_answered_tool_use(db: Path) -> None:
    def gate(call: Any) -> None:
        raise PermissionError("down")

    provider = FakeProvider([ScriptedTurn.tool_use(tool_call("nap", {"seconds": 0}, id="t1"))])
    agent = Agent(provider, model="m", tools=[nap], on_tool_call=gate)
    with pytest.raises(PermissionError):
        await agent.run("go", session_id="s1")
    async with agent.session("s1") as handle:
        _assert_replayable(list(handle.conversation), ["t1"], _ENDED)
    await agent.aclose()


async def test_conversation_answers_tool_use_on_break_and_gate_error() -> None:
    from logpose.events import ToolCall

    conv = Conversation()
    agent = Agent(
        FakeProvider([ScriptedTurn.tool_use(tool_call("nap", {"seconds": 0}, id="t1"))]),
        model="m",
        tools=[nap],
    )
    events = agent.stream("go", conversation=conv)
    async for event in events:
        if isinstance(event, ToolCall):
            break
    await events.aclose()  # type: ignore[attr-defined]
    _assert_replayable(list(conv), ["t1"], _ENDED)

    def gate(call: Any) -> None:
        raise PermissionError("down")

    conv2 = Conversation()
    gated = Agent(
        FakeProvider([ScriptedTurn.tool_use(tool_call("nap", {"seconds": 0}, id="t2"))]),
        model="m",
        tools=[nap],
        on_tool_call=gate,
    )
    with pytest.raises(PermissionError):
        await gated.run("go", conversation=conv2)
    _assert_replayable(list(conv2), ["t2"], _ENDED)


async def test_conversation_cancel_during_tool_answers_tool_use() -> None:
    conv = Conversation()
    provider = FakeProvider([ScriptedTurn.tool_use(tool_call("nap", {"seconds": 10}, id="t1"))])
    agent = Agent(provider, model="m", tools=[nap], tool_timeout=None)
    task = asyncio.create_task(agent.run("go", conversation=conv))
    while provider.call_count == 0:
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    _assert_replayable(list(conv), ["t1"], _ENDED)


async def test_run_timeout_answers_tool_use_exactly_once(db: Path) -> None:
    from logpose.errors import RunTimeoutError

    provider = FakeProvider([ScriptedTurn.tool_use(tool_call("nap", {"seconds": 5}, id="t1"))])
    agent = Agent(
        provider,
        model="m",
        tools=[nap],
        run_timeout=0.3,
        provider_turn_timeout=None,
        tool_timeout=None,
    )
    with pytest.raises(RunTimeoutError) as info:
        await agent.run("go", session_id="s1")
    deadline = "Run deadline exceeded before this tool finished."
    _assert_replayable(info.value.messages, ["t1"], deadline)
    async with agent.session("s1") as handle:
        stored = list(handle.conversation)
    assert [m.role for m in stored] == ["user", "assistant", "user"]
    _assert_replayable(stored, ["t1"], deadline)
    await agent.aclose()


# --- aiosqlite worker threads are always stopped, or the failure is logged ---------


def test_aiosqlite_floor_has_connection_stop() -> None:
    """0.20/0.21 lack ``Connection.stop()``; an unstopped worker hangs interpreter exit."""
    text = (ROOT / "pyproject.toml").read_text()
    assert '"aiosqlite>=0.22"' in text
    assert '"aiosqlite>=0.20"' not in text


class _OldConnection:
    """aiosqlite <0.22: no ``stop()``, only the private ``_stop_running()``."""

    def __init__(self) -> None:
        self.stopped = False

    def _stop_running(self) -> None:
        self.stopped = True


class _Unstoppable:
    pass


def _fake_store(db: object) -> Any:
    from logpose.store.sqlite import SQLiteStore

    store = SQLiteStore.__new__(SQLiteStore)
    store._db = db  # type: ignore[assignment]
    return store


def test_stop_now_falls_back_to_stop_running() -> None:
    conn = _OldConnection()
    manager._stop_now(_fake_store(conn))
    assert conn.stopped


def test_stop_now_logs_once_when_thread_cannot_be_stopped(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(manager, "_STOP_WARNED", False, raising=False)
    with caplog.at_level("WARNING", logger="logpose"):
        manager._stop_now(_fake_store(_Unstoppable()))
        manager._stop_now(_fake_store(_Unstoppable()))
    warned = [r for r in caplog.records if "could not stop" in r.getMessage()]
    assert len(warned) == 1


# --- the README's streaming-endpoint pattern ----------------------------------------


def _streaming_endpoint(agent: Agent) -> Any:
    """Mirror of the README FastAPI streaming example, minus FastAPI.

    Returns ``(409, None, None)`` for a busy session, else ``(200, body, background)``
    where ``background`` is what the README passes as ``BackgroundTask(stack.aclose)``.
    """
    from contextlib import AsyncExitStack

    async def chat_stream(sid: str, text: str) -> tuple[int, Any, Any]:
        stack = AsyncExitStack()
        try:
            handle = await stack.enter_async_context(agent.session(sid))
        except SessionBusyError:
            return 409, None, None  # README: raise HTTPException(409, ...) from None

        async def body() -> Any:
            try:
                async with contextlib.aclosing(agent.stream(text, session=handle)) as events:
                    async for event in events:
                        if isinstance(event, TextDelta):
                            yield event.text
            finally:
                await stack.aclose()  # save and release once streaming ends

        return 200, body(), stack.aclose

    return chat_stream


async def test_naive_streaming_response_exits_the_session_before_streaming() -> None:
    """Why the README pattern exists: returning from inside the block closes it."""
    agent = Agent(FakeProvider(ScriptedTurn.text("hi")), model="m")

    async def naive(sid: str) -> Any:
        async with agent.session(sid) as handle:

            async def body() -> Any:
                async for event in agent.stream("hello", session=handle):
                    yield event

            return body()

    response = await naive("s1")
    with pytest.raises(LogposeError, match="open 'async with agent.session"):
        async for _ in response:
            pass
    await agent.aclose()


async def test_readme_streaming_pattern_holds_lease_until_body_finishes(db: Path) -> None:
    provider = FakeProvider(
        [
            ScriptedTurn.text("hello there", chunks=["hello", " there"]),
            ScriptedTurn.text("again"),
        ]
    )
    agent = Agent(provider, model="m")
    chat_stream = _streaming_endpoint(agent)

    status, body, background = await chat_stream("s1", "hi")
    assert status == 200
    # Lease is held from before the response is built: a concurrent request is a 409
    # answered before any body is streamed.
    assert (await chat_stream("s1", "too soon"))[0] == 409
    assert [chunk async for chunk in body] == ["hello", " there"]
    await background()  # the server's background task: a no-op after the body closed it
    # Released and saved once the body finished.
    assert _rows(db, "SELECT lease_token FROM logpose_sessions") == [(None,)]
    assert _rows(db, "SELECT COUNT(*) FROM logpose_messages") == [(2,)]

    status, body, background = await chat_stream("s1", "more")
    assert status == 200
    assert [chunk async for chunk in body] == ["again"]
    await background()
    assert [m.text for m in provider.requests[1].messages] == ["hi", "hello there", "more"]
    await agent.aclose()


async def test_readme_streaming_pattern_releases_on_client_disconnect(db: Path) -> None:
    provider = FakeProvider(
        [ScriptedTurn.text("hello there", chunks=["hello", " there"]), ScriptedTurn.text("x")]
    )
    agent = Agent(provider, model="m")
    chat_stream = _streaming_endpoint(agent)

    _, body, background = await chat_stream("s1", "hi")
    assert await body.__anext__() == "hello"
    await body.aclose()  # the server closes the body iterator when the client leaves
    await background()
    assert _rows(db, "SELECT lease_token FROM logpose_sessions") == [(None,)]
    assert _rows(db, "SELECT outcome FROM logpose_runs") == [("cancelled",)]

    # A body that is never iterated (client gone before streaming began) is released
    # by the background task alone.
    status, body, background = await chat_stream("s1", "back")
    assert status == 200
    await body.aclose()  # closing an unstarted generator runs no finally
    assert _rows(db, "SELECT lease_token IS NOT NULL FROM logpose_sessions") == [(1,)]
    await background()
    assert _rows(db, "SELECT lease_token FROM logpose_sessions") == [(None,)]
    await agent.aclose()


# --- review fixes: an open stream on a closing block, idle leases ----------------------


def _assert_all_tool_uses_answered(messages: list[Any]) -> None:
    """Every tool_use is answered by the message right after it."""
    from logpose.messages import ToolResultBlock, ToolUseBlock

    for i, m in enumerate(messages):
        ids = [b.id for b in m.content if isinstance(b, ToolUseBlock)]
        if not ids:
            continue
        assert i + 1 < len(messages), f"unanswered tool_use {ids} at the end"
        following = messages[i + 1].content
        answered = [b.tool_use_id for b in following if isinstance(b, ToolResultBlock)]
        assert answered == ids


def _stored(db: Path, sid: str = "s1") -> list[Any]:
    import json

    from logpose.messages import Message

    return [
        Message.model_validate({"role": role, "content": json.loads(content)})
        for role, content in _rows(
            db,
            "SELECT role, content FROM logpose_messages WHERE session_id = ? ORDER BY seq",
            sid,
        )
    ]


async def test_block_exit_closes_stream_suspended_after_tool_call(db: Path) -> None:
    from logpose.events import ToolCall

    provider = FakeProvider(
        [ScriptedTurn.tool_use(tool_call("nap", {"seconds": 0}, id="t1")), ScriptedTurn.text("ok")]
    )
    agent = Agent(provider, model="m", tools=[nap])
    async with agent.session("s1") as handle:
        events = agent.stream("go", session=handle)
        async for event in events:
            if isinstance(event, ToolCall):
                break
        # The stream is left suspended right after the ToolCall; the block exits.
    stored = _stored(db)
    _assert_replayable(stored, ["t1"], _ENDED)
    assert _rows(db, "SELECT outcome FROM logpose_runs") == [("cancelled",)]
    assert _rows(db, "SELECT lease_token FROM logpose_sessions") == [(None,)]
    # The session stays usable.
    result = await agent.run("again", session_id="s1")
    assert result.text == "ok"
    _assert_all_tool_uses_answered(provider.requests[1].messages)
    await agent.aclose()


async def test_readme_pattern_with_tool_calls_and_held_stream_disconnect(db: Path) -> None:
    """Reviewer repro: chunks yields ToolCall events too and holds the stream in a
    variable (no aclosing); the client leaves after the ToolCall chunk."""
    from contextlib import AsyncExitStack

    from logpose.events import ToolCall

    provider = FakeProvider(
        [ScriptedTurn.tool_use(tool_call("nap", {"seconds": 0}, id="t1")), ScriptedTurn.text("ok")]
    )
    agent = Agent(provider, model="m", tools=[nap])
    stack = AsyncExitStack()
    handle = await stack.enter_async_context(agent.session("s1"))

    async def chunks() -> Any:
        events = agent.stream("go", session=handle)
        try:
            async for event in events:
                if isinstance(event, TextDelta):
                    yield event.text
                elif isinstance(event, ToolCall):
                    yield f"[calling {event.name}]"
        finally:
            await stack.aclose()

    body = chunks()
    assert await body.__anext__() == "[calling nap]"
    await body.aclose()  # client disconnected
    await stack.aclose()  # background task
    stored = _stored(db)
    _assert_all_tool_uses_answered(stored)
    _assert_replayable(stored, ["t1"], _ENDED)
    result = await agent.run("again", session_id="s1")
    assert result.text == "ok"
    await agent.aclose()


async def test_block_exit_repairs_trailing_unanswered_tool_use(db: Path) -> None:
    from logpose.messages import Message, TextBlock, ToolUseBlock

    agent = Agent(FakeProvider(ScriptedTurn.text("ok")), model="m")
    async with agent.session("s1") as handle:
        handle.conversation.append(Message(role="user", content=[TextBlock(text="go")]))
        handle.conversation.append(
            Message(role="assistant", content=[ToolUseBlock(id="t9", name="nap", input={})])
        )
    _assert_replayable(_stored(db), ["t9"], _ENDED)
    await agent.aclose()


def _short_lease(monkeypatch: pytest.MonkeyPatch, ttl: float) -> None:
    import logpose.store.sqlite as sqlite_mod

    real = sqlite_mod.SQLiteStore

    class ShortLease(real):  # type: ignore[misc,valid-type]
        def __init__(self, path: Any, **kw: Any) -> None:
            super().__init__(path, lease_ttl=ttl, **kw)

    monkeypatch.setattr(sqlite_mod, "SQLiteStore", ShortLease)


async def test_dropped_never_run_response_frees_lease(
    db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """README pattern whose response is dropped unrun; references stay alive (a cycle)."""
    _short_lease(monkeypatch, 0.2)
    agent = Agent(FakeProvider([ScriptedTurn.text("a"), ScriptedTurn.text("b")]), model="m")
    chat_stream = _streaming_endpoint(agent)
    status, body, background = await chat_stream("s1", "hi")
    assert status == 200
    kept = (body, background)  # never iterated, never closed
    assert (await chat_stream("s1", "x"))[0] == 409
    # Idle bound (3 x lease TTL) plus the TTL itself, plus slack.
    await asyncio.sleep(0.6 + 0.2 + 0.4)
    status, body2, background2 = await chat_stream("s1", "again")
    assert status == 200
    assert [c async for c in body2] == ["a"]
    await background2()
    del kept
    await agent.aclose()


async def test_lapsed_handle_reacquires_its_own_lease_for_a_new_run(
    db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A block held open across turns (WebSocket chat) resumes after idling."""
    _short_lease(monkeypatch, 0.2)
    provider = FakeProvider(
        [ScriptedTurn.tool_use(tool_call("nap", {"seconds": 0.6})), ScriptedTurn.text("done")]
    )
    agent = Agent(provider, model="m", tools=[nap])
    async with agent.session("s1") as handle:
        await asyncio.sleep(0.6 + 0.3)  # idle past the bound: renewal stopped
        assert handle.lapsed
        task = asyncio.create_task(agent.run("hi", session=handle))
        await asyncio.sleep(0.45)  # past the TTL: only a resumed renewal keeps the lease
        assert not handle.lapsed
        with pytest.raises(SessionBusyError):
            async with agent.session("s1"):
                pass
        result = await task
    assert result.text == "done"
    assert _rows(db, "SELECT COUNT(*) FROM logpose_messages") == [(4,)]
    assert _rows(db, "SELECT lease_token FROM logpose_sessions") == [(None,)]
    await agent.aclose()


async def test_lapsed_handle_refuses_run_when_lease_was_taken(
    db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _short_lease(monkeypatch, 0.2)
    provider = FakeProvider([ScriptedTurn.text("other"), ScriptedTurn.text("never")])
    agent = Agent(provider, model="m")
    with pytest.raises(SessionBusyError):
        async with agent.session("s1") as handle:
            await asyncio.sleep(0.6 + 0.2 + 0.3)  # lapsed and expired
            await agent.run("elsewhere", session_id="s1")  # another holder took it
            with pytest.raises(SessionBusyError):
                await agent.run("hi", session=handle)
            assert handle.lost
    assert provider.call_count == 1
    assert _rows(db, "SELECT COUNT(*) FROM logpose_messages") == [(2,)]
    await agent.aclose()


async def test_block_exit_during_run_in_another_task_keeps_live_history_valid(
    db: Path,
) -> None:
    """The block exits while another task is inside __anext__ (a slow tool): the
    stream cannot be closed, and the repair must not touch the live history."""
    from logpose.events import ToolCall
    from logpose.messages import ToolResultBlock

    provider = FakeProvider(
        [
            ScriptedTurn.tool_use(tool_call("nap", {"seconds": 0.3}, id="t1")),
            ScriptedTurn.text("done"),
        ]
    )
    agent = Agent(provider, model="m", tools=[nap])
    called = asyncio.Event()

    async def consume(handle: SessionHandle) -> list[Any]:
        seen = []
        async for event in agent.stream("go", session=handle):
            seen.append(event)
            if isinstance(event, ToolCall):
                called.set()
        return seen

    async with agent.session("s1") as handle:
        task = asyncio.create_task(consume(handle))
        await called.wait()
        await asyncio.sleep(0.05)  # the consumer is now awaiting __anext__ (tool running)
    stored = _stored(db)
    _assert_replayable(stored, ["t1"], _ENDED)
    await task
    second = provider.requests[1].messages
    results = [b.tool_use_id for m in second for b in m.content if isinstance(b, ToolResultBlock)]
    assert results == ["t1"]
    _assert_all_tool_uses_answered(second)
    await agent.aclose()


async def test_long_run_keeps_lease_past_idle_bound(
    db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _short_lease(monkeypatch, 0.2)
    provider = FakeProvider(
        [ScriptedTurn.tool_use(tool_call("nap", {"seconds": 1.2})), ScriptedTurn.text("done")]
    )
    agent = Agent(provider, model="m", tools=[nap])
    async with agent.session("s1") as handle:
        task = asyncio.create_task(agent.run("go", session=handle))
        await asyncio.sleep(1.0)  # past the idle bound, run still active
        with pytest.raises(SessionBusyError):
            async with agent.session("s1"):
                pass
        result = await task
    assert result.text == "done"
    assert _rows(db, "SELECT COUNT(*) FROM logpose_messages") == [(4,)]
    await agent.aclose()


def test_session_exit_timeout_exceeds_busy_timeout() -> None:
    from logpose.agent import SESSION_EXIT_TIMEOUT
    from logpose.store.sqlite import SQLiteStore

    assert SESSION_EXIT_TIMEOUT > SQLiteStore(":memory:")._busy_timeout_ms / 1000
