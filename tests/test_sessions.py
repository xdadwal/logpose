"""Automatic SQLite sessions: the store manager and Agent integration."""

from __future__ import annotations

import asyncio
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
        msgs = sync_agent.agent  # the sync facade drives Agent.run on its private loop
        from logpose.sync import _runner_for

        runner = _runner_for(msgs)
        result = runner.run(msgs.run("again", session_id="s1"))
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
