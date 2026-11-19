from __future__ import annotations

from pathlib import Path

import aiosqlite
import pytest

from logpose.errors import LogposeError
from logpose.messages import (
    Message,
    RawBlock,
    RedactedThinkingBlock,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from logpose.store.base import RunRecord, validate_session_id
from logpose.store.sqlite import SQLiteStore


def _run(sid: str = "s1", run_id: str = "r1") -> RunRecord:
    return RunRecord(
        run_id=run_id,
        session_id=sid,
        model="m",
        provider="p",
        started_at_ms=1000,
        ended_at_ms=2000,
        outcome="error",
        iterations=3,
        input_tokens=11,
        output_tokens=22,
        reasoning_tokens=33,
        cache_read_input_tokens=44,
        cache_creation_input_tokens=55,
        error_type="ProviderError",
    )


def _msg(text: str) -> Message:
    return Message(role="user", content=[TextBlock(text=text)])


async def _scalar(path: Path, sql: str, *params: object) -> object:
    async with aiosqlite.connect(path) as db, db.execute(sql, params) as cur:
        row = await cur.fetchone()
        return row[0] if row else None


async def test_open_creates_schema_version_1(tmp_path: Path) -> None:
    p = tmp_path / "t.db"
    async with SQLiteStore(p):
        pass
    assert await _scalar(p, "SELECT version FROM logpose_schema") == 1
    for t in ("sessions", "messages", "runs", "schema"):
        assert (
            await _scalar(p, "SELECT count(*) FROM sqlite_master WHERE name = ?", f"logpose_{t}")
            == 1
        )


async def test_open_is_idempotent(tmp_path: Path) -> None:
    p = tmp_path / "t.db"
    async with SQLiteStore(p):
        pass
    async with SQLiteStore(p) as s:
        await s.open()
    assert await _scalar(p, "SELECT version FROM logpose_schema") == 1
    assert await _scalar(p, "SELECT count(*) FROM logpose_schema") == 1


async def test_newer_schema_version_is_refused(tmp_path: Path) -> None:
    p = tmp_path / "t.db"
    async with SQLiteStore(p):
        pass
    async with aiosqlite.connect(p) as db:
        await db.execute("UPDATE logpose_schema SET version = 99")
        await db.commit()
    with pytest.raises(LogposeError, match=r"(?s)99.*1"):
        await SQLiteStore(p).open()


async def test_append_and_load_round_trip_all_blocks(tmp_path: Path) -> None:
    msgs = [
        Message(role="user", content=[TextBlock(text="héllo ☃")]),
        Message(
            role="assistant",
            content=[
                ThinkingBlock(thinking="hm", signature="sig"),
                RedactedThinkingBlock(data="xx"),
                TextBlock(text="ok"),
                ToolUseBlock(id="t1", name="f", input={"a": [1, {"b": None}]}),
                RawBlock(data={"k": "v"}),
            ],
        ),
        Message(
            role="user",
            content=[ToolResultBlock(tool_use_id="t1", content="r", is_error=True)],
        ),
    ]
    async with SQLiteStore(tmp_path / "t.db") as s:
        assert await s.append_messages("s1", msgs, start_seq=0) == 3
        assert await s.load_messages("s1") == msgs
        assert await s.load_messages("other") == []


async def test_append_continues_seq(tmp_path: Path) -> None:
    p = tmp_path / "t.db"
    async with SQLiteStore(p) as s:
        n = await s.append_messages("s1", [_msg("a"), _msg("b")], start_seq=0)
        assert n == 2
        m = await s.append_messages("s1", [_msg("c")], start_seq=n)
        assert m == 3
        loaded = await s.load_messages("s1")
    assert [b.text for x in loaded for b in x.content if isinstance(b, TextBlock)] == [
        "a",
        "b",
        "c",
    ]
    async with (
        aiosqlite.connect(p) as db,
        db.execute("SELECT seq FROM logpose_messages ORDER BY seq") as cur,
    ):
        assert [r[0] for r in await cur.fetchall()] == [0, 1, 2]


async def test_record_run_persists_all_fields(tmp_path: Path) -> None:
    p = tmp_path / "t.db"
    rec = _run()
    async with SQLiteStore(p) as s:
        await s.record_run(rec)
    async with aiosqlite.connect(p) as db, db.execute("SELECT * FROM logpose_runs") as cur:
        row = await cur.fetchone()
        cols = [d[0] for d in cur.description]
    assert row == (
        "r1",
        "s1",
        "m",
        "p",
        1000,
        2000,
        "error",
        3,
        11,
        22,
        33,
        44,
        55,
        "ProviderError",
    )
    assert cols[:4] == ["run_id", "session_id", "model", "provider"]


async def test_delete_session_removes_messages_and_runs(tmp_path: Path) -> None:
    p = tmp_path / "t.db"
    async with SQLiteStore(p) as s:
        await s.append_messages("s1", [_msg("a")], start_seq=0)
        await s.append_messages("s2", [_msg("b")], start_seq=0)
        await s.record_run(_run("s1", "r1"))
        await s.record_run(_run("s2", "r2"))
        await s.delete_session("s1")
        assert await s.load_messages("s1") == []
        assert len(await s.load_messages("s2")) == 1
    assert await _scalar(p, "SELECT count(*) FROM logpose_runs") == 1


async def test_purge_expired_counts_and_deletes(tmp_path: Path) -> None:
    p = tmp_path / "t.db"
    async with SQLiteStore(p) as s:
        for sid, exp in (("old", 100), ("new", 10_000)):
            await s._write(
                [
                    (
                        "INSERT INTO logpose_sessions (id, created_at, updated_at, expires_at, "
                        "version, metadata) VALUES (?, 0, 0, ?, 0, '{}')",
                        (sid, exp),
                    )
                ]
            )
            await s.append_messages(sid, [_msg(sid)], start_seq=0)
            await s.record_run(_run(sid, "r-" + sid))
        assert await s.purge_expired(now_ms=5000) == 1
        assert await s.load_messages("old") == []
        assert len(await s.load_messages("new")) == 1
    assert await _scalar(p, "SELECT count(*) FROM logpose_runs") == 1
    assert await _scalar(p, "SELECT id FROM logpose_sessions") == "new"


async def test_session_not_implemented(tmp_path: Path) -> None:
    async with SQLiteStore(tmp_path / "t.db") as s:
        with pytest.raises(NotImplementedError):
            s.session("s1")


@pytest.mark.parametrize("bad", ["", "a b", "x" * 129, "../x"])
def test_validate_session_id_rejects(bad: str) -> None:
    with pytest.raises(LogposeError):
        validate_session_id(bad)


def test_validate_session_id_accepts() -> None:
    assert validate_session_id("abc-1.2_x") == "abc-1.2_x"
