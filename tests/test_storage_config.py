"""Storage configuration: Agent(db_path=, session_ttl=) and export_traces(db_path=)."""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from logpose import Agent, LogposeError, export_traces
from logpose.store import manager
from logpose.tracing import semconv as sc
from logpose.tracing.spans import SpanRecord, Tracing
from tests.fake_provider import FakeProvider, ScriptedTurn

DAY_MS = 86_400_000


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv("LOGPOSE_DB_PATH", raising=False)
    manager.reset_stores_for_tests()
    yield
    manager.reset_stores_for_tests()


def _rows(path: Path, sql: str, *params: object) -> list[tuple[Any, ...]]:
    with sqlite3.connect(path) as conn:
        return list(conn.execute(sql, params))


def _agent(*texts: str, **kwargs: Any) -> Agent:
    turns = [ScriptedTurn.text(t) for t in texts] or [ScriptedTurn.text("ok")]
    return Agent(FakeProvider(turns, repeat_last=True), model="m", **kwargs)


def _text(raw: str) -> str:
    content = json.loads(raw)
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content)


def _contents(path: Path, session_id: str) -> list[str]:
    return [
        _text(r[0])
        for r in _rows(
            path,
            "SELECT content FROM logpose_messages WHERE session_id = ? ORDER BY seq",
            session_id,
        )
    ]


# --- db_path -------------------------------------------------------------------------------


async def test_db_path_argument_wins_over_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    a, b = tmp_path / "a.db", tmp_path / "b.db"
    monkeypatch.setenv("LOGPOSE_DB_PATH", str(a))
    agent = _agent(db_path=b)
    assert agent.db_path == b.resolve()
    await agent.run("hi", session_id="s1")
    await agent.aclose()
    assert b.exists()
    assert not a.exists()


async def test_db_path_env_fallback_and_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = tmp_path / "env.db"
    monkeypatch.setenv("LOGPOSE_DB_PATH", str(env))
    assert _agent().db_path == env
    assert _agent(db_path=None).db_path == env
    # A blank argument counts as unset.
    assert _agent(db_path="  ").db_path == env
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LOGPOSE_DB_PATH", "   ")
    assert _agent().db_path == Path(os.getcwd()) / "logpose.db"
    assert _agent(db_path="").db_path == Path(os.getcwd()) / "logpose.db"
    monkeypatch.delenv("LOGPOSE_DB_PATH")
    assert _agent().db_path == Path(os.getcwd()) / "logpose.db"


async def test_db_path_expands_user_and_is_absolute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    agent = _agent(db_path="~/data/x.db")
    assert agent.db_path == tmp_path / "data" / "x.db"
    assert agent.db_path.is_absolute()
    monkeypatch.chdir(tmp_path)
    rel = _agent(db_path=Path("rel") / "y.db")
    assert rel.db_path == tmp_path / "rel" / "y.db"
    assert rel.db_path.is_absolute()
    assert manager.db_path("~/z.db") == tmp_path / "z.db"


async def test_db_path_rejects_wrong_type() -> None:
    with pytest.raises(LogposeError):
        _agent(db_path=123)


async def test_two_agents_two_paths_isolated(tmp_path: Path) -> None:
    p1, p2 = tmp_path / "one.db", tmp_path / "two.db"
    one = _agent("from-one", db_path=p1, tracing=Tracing(), session_ttl=0.05)
    two = _agent("from-two", db_path=p2, tracing=Tracing(), session_ttl=0.05)
    await one.run("q1", session_id="chat", run_id="same-run")
    # The same run ID is free in a different database.
    await two.run("q2", session_id="chat", run_id="same-run")
    assert _contents(p1, "chat") == ["q1", "from-one"]
    assert _contents(p2, "chat") == ["q2", "from-two"]
    # Same loop, different paths: two stores.
    assert await manager.get_store(path=one.db_path) is not await manager.get_store(
        path=two.db_path
    )

    # delete_session on one leaves the other intact.
    await one.delete_session("chat")
    assert _contents(p1, "chat") == []
    assert _contents(p2, "chat") == ["q2", "from-two"]

    # purge_traces on one leaves the other's spans.
    await one.aclose()
    await two.aclose()
    spans_two = _rows(p2, "SELECT COUNT(*) FROM logpose_spans")[0][0]
    assert spans_two > 0
    # delete_session removed "chat"'s spans in one.db; the second run adds more.
    await one.run("q-traced")
    await one.aclose()
    spans_one = _rows(p1, "SELECT COUNT(*) FROM logpose_spans")[0][0]
    assert spans_one > 0
    assert await one.purge_traces(timedelta(0)) == spans_one
    assert _rows(p1, "SELECT COUNT(*) FROM logpose_spans")[0][0] == 0
    assert _rows(p2, "SELECT COUNT(*) FROM logpose_spans")[0][0] == spans_two

    # purge_expired_sessions on one leaves the other's expired session.
    await one.run("q3", session_id="other")
    await asyncio.sleep(0.1)
    assert await one.purge_expired_sessions() == 1
    assert _rows(p2, "SELECT COUNT(*) FROM logpose_sessions WHERE id = 'chat'")[0][0] == 1
    await one.aclose()
    await two.aclose()


async def test_tracing_uses_agent_db_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env, b = tmp_path / "env.db", tmp_path / "b.db"
    monkeypatch.setenv("LOGPOSE_DB_PATH", str(env))
    agent = _agent(db_path=b, tracing=Tracing())
    await agent.run("q")
    await agent.aclose()
    assert not env.exists()
    out = tmp_path / "o.jsonl"
    result = await export_traces(out, db_path=b)
    assert result.traces == 1
    assert not env.exists()


def _span(trace: str) -> SpanRecord:
    return SpanRecord(
        trace_id=trace,
        span_id="1" * 16,
        parent_span_id=None,
        run_id=f"run-{trace[:4]}",
        session_id="s",
        name="invoke_agent",
        kind=1,
        start_ns=1_700_000_000 * 1_000_000_000,
        end_ns=1_700_000_000 * 1_000_000_000 + 1000,
        status_code=0,
        status_message=None,
        attributes={sc.GEN_AI_OPERATION_NAME: "invoke_agent"},
        events=[],
    )


async def test_export_traces_db_path_overrides_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env, x = tmp_path / "env.db", tmp_path / "x.db"
    monkeypatch.setenv("LOGPOSE_DB_PATH", str(env))
    store = await manager.get_store(path=manager.db_path(x))
    await store.write_spans([_span("a" * 32)])  # type: ignore[attr-defined]
    out = tmp_path / "o.jsonl"
    assert (await export_traces(out, db_path=str(x))).traces == 1
    assert not env.exists()
    # Without db_path, the environment still decides.
    assert (await export_traces(out)).traces == 0
    assert env.exists()


# --- session_ttl ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad", [0, -1, float("inf"), float("nan"), timedelta(0), timedelta(seconds=-1), True, "7"]
)
def test_session_ttl_validation(bad: Any) -> None:
    with pytest.raises(LogposeError):
        _agent(session_ttl=bad)


def test_session_ttl_values() -> None:
    assert _agent(session_ttl=timedelta(days=7)).session_ttl == 604800.0
    assert _agent(session_ttl=90).session_ttl == 90.0
    assert _agent().session_ttl == 30 * 86400
    assert _agent(session_ttl=None).session_ttl == 30 * 86400


def _stamp(path: Path, session_id: str) -> int:
    (row,) = _rows(
        path, "SELECT expires_at - updated_at FROM logpose_sessions WHERE id = ?", session_id
    )
    return int(row[0])


async def test_session_ttl_stamped_per_write(tmp_path: Path) -> None:
    path = tmp_path / "s.db"
    week = _agent(db_path=path, session_ttl=timedelta(days=7))
    await week.run("a", session_id="s")
    assert _stamp(path, "s") == 7 * DAY_MS
    # Session creation is stamped with the agent's TTL too.
    async with week.session("t"):
        (row,) = _rows(path, "SELECT expires_at - created_at FROM logpose_sessions WHERE id='t'")
        assert row[0] == 7 * DAY_MS
    day = _agent(db_path=path, session_ttl=timedelta(days=1))
    await day.run("b", session_id="s")
    assert _stamp(path, "s") == DAY_MS
    await week.run("c", session_id="s")
    assert _stamp(path, "s") == 7 * DAY_MS
    assert _contents(path, "s") == ["a", "ok", "b", "ok", "c", "ok"]
    await week.aclose()
    await day.aclose()


async def test_short_ttl_expires(tmp_path: Path) -> None:
    path = tmp_path / "s.db"
    agent = _agent(db_path=path, session_ttl=0.05)
    await agent.run("first", session_id="gone")
    await asyncio.sleep(0.1)
    assert await agent.purge_expired_sessions() == 1

    await agent.run("first", session_id="lapsed")
    await asyncio.sleep(0.1)
    async with agent.session("lapsed") as handle:
        assert handle.created
        assert handle.conversation.messages == []
    await agent.aclose()


async def test_defaults_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = tmp_path / "env.db"
    monkeypatch.setenv("LOGPOSE_DB_PATH", str(env))
    agent = _agent()
    assert agent.db_path == manager.db_path()
    await agent.run("q", session_id="s")
    await agent.aclose()
    assert _stamp(env, "s") == 30 * DAY_MS


# --- fix round 1 ---------------------------------------------------------------------------


async def test_db_path_without_argument_is_resolved_at_each_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    early, late = tmp_path / "early.db", tmp_path / "late.db"
    monkeypatch.setenv("LOGPOSE_DB_PATH", str(early))
    agent = _agent(tracing=Tracing())
    assert agent.db_path == early
    monkeypatch.setenv("LOGPOSE_DB_PATH", str(late))
    assert agent.db_path == late
    await agent.run("q", session_id="s")
    assert late.exists()
    assert not early.exists()
    assert _contents(late, "s") == ["q", "ok"]
    # The default (no env) follows the working directory at use time too.
    monkeypatch.delenv("LOGPOSE_DB_PATH")
    monkeypatch.chdir(tmp_path)
    assert agent.db_path == tmp_path / "logpose.db"
    await agent.run("q2", session_id="d")
    await agent.aclose()
    assert _contents(tmp_path / "logpose.db", "d") == ["q2", "ok"]
    # Traces followed the path too.
    assert _rows(late, "SELECT COUNT(*) FROM logpose_spans")[0][0] > 0
    assert _rows(tmp_path / "logpose.db", "SELECT COUNT(*) FROM logpose_spans")[0][0] > 0


async def test_explicit_db_path_stays_fixed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixed = tmp_path / "fixed.db"
    monkeypatch.chdir(tmp_path)
    agent = _agent(db_path="fixed.db")
    monkeypatch.chdir(tmp_path.parent)
    monkeypatch.setenv("LOGPOSE_DB_PATH", str(tmp_path / "env.db"))
    assert agent.db_path == fixed
    with pytest.raises(AttributeError):
        agent.db_path = tmp_path / "other.db"  # type: ignore[misc]


def test_session_ttl_too_large() -> None:
    with pytest.raises(LogposeError):
        _agent(session_ttl=1e300)
    with pytest.raises(LogposeError):
        _agent(session_ttl=timedelta.max.total_seconds() * 2)
    assert _agent(session_ttl=timedelta.max).session_ttl == timedelta.max.total_seconds()


# --- load_session --------------------------------------------------------------------------


def _roles_texts(messages: list[Any]) -> list[tuple[str, str]]:
    return [(m.role, m.text) for m in messages]


async def test_load_session_returns_saved_history(tmp_path: Path) -> None:
    path = tmp_path / "s.db"
    agent = _agent("one", "two", db_path=path)
    await agent.run("q1", session_id="s")
    await agent.run("q2", session_id="s")
    loaded = await agent.load_session("s")
    assert _roles_texts(loaded) == [
        ("user", "q1"),
        ("assistant", "one"),
        ("user", "q2"),
        ("assistant", "two"),
    ]
    # A new list each call: mutating it changes nothing stored.
    loaded.clear()
    assert len(await agent.load_session("s")) == 4
    await agent.aclose()


async def test_load_session_returns_full_history_when_summarized(tmp_path: Path) -> None:
    from logpose.messages import Message, TextBlock

    path = tmp_path / "c.db"
    provider = FakeProvider([ScriptedTurn.text("SUMMARY-ONE"), ScriptedTurn.text("answer")])
    agent = Agent(provider, model="m", db_path=path, context_budget=2_000)
    history: list[Message] = []
    for i in range(10):
        history.append(Message(role="user", content=[TextBlock(text=f"<q{i}> " + "x" * 400)]))
        history.append(Message(role="assistant", content=[TextBlock(text=f"<a{i}> " + "y" * 400)]))
    async with agent.session("s") as handle:
        handle.conversation.extend(history)
        await agent.run("latest", session=handle)
    # Compaction ran and a summary was stored.
    assert _rows(path, "SELECT COUNT(*) FROM logpose_summaries WHERE session_id='s'")[0][0] == 1
    loaded = await agent.load_session("s")
    assert len(loaded) == 22
    assert loaded[:20] == history
    assert _roles_texts(loaded[20:]) == [("user", "latest"), ("assistant", "answer")]
    assert not any("SUMMARY-ONE" in m.text for m in loaded)
    await agent.aclose()


async def test_load_session_unknown_is_empty(tmp_path: Path) -> None:
    agent = _agent(db_path=tmp_path / "s.db")
    assert await agent.load_session("never-seen") == []
    await agent.aclose()


async def test_load_session_expired_is_empty_and_untouched(tmp_path: Path) -> None:
    path = tmp_path / "s.db"
    agent = _agent(db_path=path, session_ttl=0.05)
    await agent.run("q", session_id="s")
    before = _rows(path, "SELECT expires_at, updated_at, version FROM logpose_sessions")
    await asyncio.sleep(0.1)
    assert await agent.load_session("s") == []
    assert _rows(path, "SELECT COUNT(*) FROM logpose_messages WHERE session_id='s'")[0][0] > 0
    assert _rows(path, "SELECT expires_at, updated_at, version FROM logpose_sessions") == before
    await agent.aclose()


class _GatedProvider(FakeProvider):
    """Blocks inside ``stream`` while ``gated`` until ``release`` is set."""

    def __init__(self) -> None:
        super().__init__([ScriptedTurn.text("ok")], repeat_last=True)
        self.gated = False
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def stream(self, req: Any) -> Any:  # type: ignore[override]
        if self.gated:
            self.entered.set()
            await self.release.wait()
        async for event in super().stream(req):
            yield event


async def test_load_session_does_not_take_lease(tmp_path: Path) -> None:
    provider = _GatedProvider()
    agent = Agent(provider, model="m", db_path=tmp_path / "s.db")
    await agent.run("first", session_id="s")
    provider.gated = True

    async def hold() -> list[Any]:
        async with agent.session("s") as handle:
            return [event async for event in agent.stream("second", session=handle)]

    task = asyncio.create_task(hold())
    await asyncio.wait_for(provider.entered.wait(), 5)
    # The lease is held and the turn is mid-flight: reads see the saved history only.
    assert _roles_texts(await agent.load_session("s")) == [("user", "first"), ("assistant", "ok")]
    provider.release.set()
    events = await asyncio.wait_for(task, 5)
    assert events
    assert _roles_texts(await agent.load_session("s")) == [
        ("user", "first"),
        ("assistant", "ok"),
        ("user", "second"),
        ("assistant", "ok"),
    ]
    await agent.aclose()


async def test_load_session_invalid_id(tmp_path: Path) -> None:
    agent = _agent(db_path=tmp_path / "s.db")
    with pytest.raises(LogposeError):
        await agent.load_session("bad id!")
    await agent.aclose()


async def test_load_session_uses_agent_db_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env, b = tmp_path / "env.db", tmp_path / "b.db"
    monkeypatch.setenv("LOGPOSE_DB_PATH", str(env))
    writer = _agent(db_path=b)
    await writer.run("q", session_id="s")
    await writer.aclose()
    reader = _agent(db_path=b)
    assert _roles_texts(await reader.load_session("s")) == [("user", "q"), ("assistant", "ok")]
    # An agent on the environment's path sees nothing.
    assert await _agent().load_session("s") == []
    await reader.aclose()
    assert _contents(b, "s") == ["q", "ok"]


def test_sync_load_session(tmp_path: Path) -> None:
    from logpose import SyncAgent

    with SyncAgent(_agent(db_path=tmp_path / "s.db")) as sync_agent:
        sync_agent.run("q", session_id="s")
        loaded = sync_agent.load_session("s")
    assert _roles_texts(loaded) == [("user", "q"), ("assistant", "ok")]
