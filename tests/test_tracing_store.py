from __future__ import annotations

import json
import re
import sqlite3
import subprocess
from pathlib import Path

from logpose.store.base import RunRecord
from logpose.store.sqlite import _MIGRATIONS, SQLiteStore
from logpose.tracing.spans import SpanRecord, new_span_id, new_trace_id


def _span(
    trace: str = "t1",
    span: str = "s1",
    *,
    session: str | None = "sess",
    run: str = "r1",
    start: int = 100,
    parent: str | None = None,
) -> SpanRecord:
    return SpanRecord(
        trace_id=trace,
        span_id=span,
        parent_span_id=parent,
        run_id=run,
        session_id=session,
        name="chat",
        kind=3,
        start_ns=start,
        end_ns=start + 10,
        status_code=0,
        status_message=None,
        attributes={"gen_ai.request.model": "m", "n": 1},
        events=[{"name": "e", "time_unix_nano": start, "attributes": {"k": "v"}}],
    )


def _run(run_id: str = "r1", sid: str = "sess", metadata: dict | None = None) -> RunRecord:  # type: ignore[type-arg]
    return RunRecord(
        run_id=run_id,
        session_id=sid,
        model="m",
        provider="p",
        started_at_ms=1,
        ended_at_ms=2,
        outcome="completed",
        iterations=1,
        input_tokens=1,
        output_tokens=1,
        reasoning_tokens=0,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
        metadata=metadata,
    )


async def _read(s: SQLiteStore, **kw) -> list[SpanRecord]:  # type: ignore[no-untyped-def]
    return [x async for x in s.read_spans(**kw)]


def test_ids() -> None:
    assert re.fullmatch(r"[0-9a-f]{32}", new_trace_id())
    assert re.fullmatch(r"[0-9a-f]{16}", new_span_id())


async def test_migrates_v2_to_v3_preserving_data(tmp_path: Path) -> None:
    p = tmp_path / "t.db"
    conn = sqlite3.connect(p, isolation_level=None)
    conn.execute("CREATE TABLE logpose_schema (version INTEGER NOT NULL)")
    conn.execute("INSERT INTO logpose_schema (version) VALUES (2)")
    for script in _MIGRATIONS[:2]:
        for stmt in script:
            conn.execute(stmt)
    conn.execute(
        "INSERT INTO logpose_sessions (id, created_at, updated_at, expires_at, version, "
        "metadata) VALUES ('s1', 1, 1, 9999999999999, 4, '{}')"
    )
    conn.execute(
        "INSERT INTO logpose_runs (run_id, session_id, model, provider, started_at, ended_at, "
        "outcome, iterations, input_tokens, output_tokens, reasoning_tokens, "
        "cache_read_input_tokens, cache_creation_input_tokens) "
        "VALUES ('old', 's1', 'm', 'p', 1, 2, 'completed', 1, 1, 1, 0, 0, 0)"
    )
    conn.close()
    async with SQLiteStore(p) as s:
        assert await s.run_id_exists("old")
        await s.record_run(_run("new", "s1", {"a": 1}))
        await s.write_spans([_span()])
        assert len(await _read(s)) == 1
    conn = sqlite3.connect(p)
    assert conn.execute("SELECT version FROM logpose_schema").fetchone() == (3,)
    assert conn.execute("SELECT metadata FROM logpose_runs WHERE run_id='old'").fetchone() == (
        None,
    )
    row = conn.execute("SELECT metadata FROM logpose_runs WHERE run_id='new'").fetchone()
    assert json.loads(row[0]) == {"a": 1}
    assert conn.execute("SELECT version FROM logpose_sessions").fetchone() == (4,)
    conn.close()


async def test_write_and_read_spans_round_trip(tmp_path: Path) -> None:
    async with SQLiteStore(tmp_path / "t.db") as s:
        a = _span(parent="p1")
        b = _span("t2", "s2", session=None, start=50)
        await s.write_spans([a, b])
        await s.write_spans([])
        assert await _read(s) == [b, a]  # ordered by start_ns


async def test_read_spans_filters(tmp_path: Path) -> None:
    async with SQLiteStore(tmp_path / "t.db") as s:
        await s.write_spans(
            [
                _span("t1", "a", session="x", start=100),
                _span("t1", "b", session="x", start=200),
                _span("t2", "c", session="y", start=300),
                _span("t3", "d", session=None, start=400),
            ]
        )
        assert [x.span_id for x in await _read(s, session_id="x")] == ["a", "b"]
        assert [x.span_id for x in await _read(s, trace_id="t2")] == ["c"]
        assert [x.span_id for x in await _read(s, since_ns=200, until_ns=300)] == ["b", "c"]
        assert [x.span_id for x in await _read(s, since_ns=301)] == ["d"]
        assert await _read(s, session_id="x", trace_id="t2") == []


async def test_purge_spans_and_session_delete_remove_spans(tmp_path: Path) -> None:
    async with SQLiteStore(tmp_path / "t.db") as s:
        await s.write_spans(
            [
                _span("t1", "a", session="x", start=1_000_000),
                _span("t2", "b", session=None, start=2_000_000),
                _span("t3", "c", session=None, start=9_000_000_000_000),
            ]
        )
        # older_than_ms: spans whose start_ns is before that many ms.
        assert await s.purge_spans(3) == 2
        assert [x.span_id for x in await _read(s)] == ["c"]
        await s.write_spans([_span("t4", "d", session="x"), _span("t5", "e", session="y")])
        await s.delete_session("x")
        assert [x.span_id for x in await _read(s)] == ["e", "c"]


async def test_purge_expired_removes_spans(tmp_path: Path) -> None:
    async with SQLiteStore(tmp_path / "t.db", idle_ttl=0.0) as s:
        async with s.session("x"):
            pass
        await s.write_spans([_span(session="x"), _span("t2", "z", session=None)])
        assert await s.purge_expired(now_ms=2**60) == 1
        assert [x.span_id for x in await _read(s)] == ["z"]


async def test_expired_reset_on_access_removes_spans(tmp_path: Path) -> None:
    async with SQLiteStore(tmp_path / "t.db", idle_ttl=0.0) as s:
        async with s.session("x"):
            pass
        await s.write_spans([_span(session="x")])
        import asyncio

        await asyncio.sleep(0.01)
        async with s.session("x"):
            pass
        assert await _read(s) == []


async def test_run_id_exists(tmp_path: Path) -> None:
    async with SQLiteStore(tmp_path / "t.db") as s:
        assert not await s.run_id_exists("r1")
        await s.record_run(_run("r1"))
        assert await s.run_id_exists("r1")
        await s.write_spans([_span(run="r2")])
        assert await s.run_id_exists("r2")
        assert not await s.run_id_exists("r3")


def test_semconv_is_single_source() -> None:
    src = Path(__file__).resolve().parent.parent / "src"
    out = subprocess.run(
        ["grep", "-rlE", "[\"']gen_ai\\.", str(src), "--include=*.py"],
        capture_output=True,
        text=True,
    ).stdout.split()
    assert [Path(x).name for x in out] == ["semconv.py"]
    from logpose.tracing import semconv

    assert semconv.SCHEMA_URL == f"https://opentelemetry.io/schemas/{semconv.SEMCONV_VERSION}"
    assert semconv.GEN_AI_OPERATION_NAME == "gen_ai.operation.name"
