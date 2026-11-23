from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

import aiosqlite
import pytest

from logpose.errors import LogposeError, SessionBusyError
from logpose.messages import (
    Message,
    RawBlock,
    RedactedThinkingBlock,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from logpose.store.base import RunRecord, SessionHandle, validate_session_id
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


_FAR_FUTURE_MS = 1 << 62


async def _live(s: SQLiteStore, *session_ids: str) -> None:
    """Create unleased session rows that never expire, for the direct append tests.

    ``load_messages`` reads only a live session's messages, so a test that appends
    without leasing needs the row a lease would have created.
    """
    await s._write(
        [
            (
                "INSERT INTO logpose_sessions (id, created_at, updated_at, expires_at, "
                "version, metadata) VALUES (?, 0, 0, ?, 0, '{}')",
                (sid, _FAR_FUTURE_MS),
            )
            for sid in session_ids
        ]
    )


async def _scalar(path: Path, sql: str, *params: object) -> object:
    async with aiosqlite.connect(path) as db, db.execute(sql, params) as cur:
        row = await cur.fetchone()
        return row[0] if row else None


async def test_open_creates_schema_version_3(tmp_path: Path) -> None:
    p = tmp_path / "t.db"
    async with SQLiteStore(p):
        pass
    assert await _scalar(p, "SELECT version FROM logpose_schema") == 3
    for t in ("sessions", "messages", "runs", "schema", "summaries"):
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
    assert await _scalar(p, "SELECT version FROM logpose_schema") == 3
    assert await _scalar(p, "SELECT count(*) FROM logpose_schema") == 1


async def test_newer_schema_version_is_refused(tmp_path: Path) -> None:
    p = tmp_path / "t.db"
    async with SQLiteStore(p):
        pass
    async with aiosqlite.connect(p) as db:
        await db.execute("UPDATE logpose_schema SET version = 99")
        await db.commit()
    with pytest.raises(LogposeError, match=r"(?s)99.*3"):
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
        await _live(s, "s1")
        assert await s.append_messages("s1", msgs, start_seq=0) == 3
        assert await s.load_messages("s1") == msgs
        assert await s.load_messages("other") == []


async def test_append_continues_seq(tmp_path: Path) -> None:
    p = tmp_path / "t.db"
    async with SQLiteStore(p) as s:
        await _live(s, "s1")
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
        None,
    )
    assert cols[:4] == ["run_id", "session_id", "model", "provider"]


async def test_delete_session_removes_messages_and_runs(tmp_path: Path) -> None:
    p = tmp_path / "t.db"
    async with SQLiteStore(p) as s:
        await _live(s, "s1", "s2")
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
        for sid, exp in (("old", 100), ("new", _FAR_FUTURE_MS)):
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


@pytest.mark.parametrize("bad", ["", "a b", "x" * 129, "../x"])
def test_validate_session_id_rejects(bad: str) -> None:
    with pytest.raises(LogposeError):
        validate_session_id(bad)


def test_validate_session_id_accepts() -> None:
    assert validate_session_id("abc-1.2_x") == "abc-1.2_x"


def _texts(msgs: list[Message]) -> list[str]:
    return [b.text for m in msgs for b in m.content if isinstance(b, TextBlock)]


async def _no_renewal(handle: SessionHandle, token: str) -> None:
    return None


# --- session lease and lifecycle -------------------------------------------------------


async def test_new_session_is_created_with_metadata(tmp_path: Path) -> None:
    p = tmp_path / "t.db"
    async with SQLiteStore(p) as s:
        async with s.session("s1", metadata={"user": "u1"}) as h:
            assert h.id == "s1"
            assert h.created is True
            assert h.metadata == {"user": "u1"}
            assert h.conversation.messages == []
            assert h.loaded_count == 0
            assert h.next_seq == 0
            assert h.lost is False
            # The row exists from enter, before any message is saved.
            assert await _scalar(p, "SELECT count(*) FROM logpose_sessions WHERE id='s1'") == 1
    assert await _scalar(p, "SELECT metadata FROM logpose_sessions") == '{"user": "u1"}'
    assert await _scalar(p, "SELECT lease_token FROM logpose_sessions") is None


async def test_existing_session_loads_history_and_merges_metadata(tmp_path: Path) -> None:
    p = tmp_path / "t.db"
    async with SQLiteStore(p) as s:
        async with s.session("s1", metadata={"a": "1", "b": "2"}) as h:
            h.conversation.append(_msg("one"))
            h.conversation.append(_msg("two"))
        async with s.session("s1", metadata={"b": "3", "c": "4"}) as h:
            assert h.created is False
            assert _texts(h.conversation.messages) == ["one", "two"]
            assert h.loaded_count == 2
            assert h.next_seq == 2
            assert h.metadata == {"a": "1", "b": "3", "c": "4"}
        async with s.session("s1") as h:
            assert h.metadata == {"a": "1", "b": "3", "c": "4"}
    assert await _scalar(p, "SELECT version FROM logpose_sessions") == 3


async def test_messages_added_in_block_are_persisted_on_exit(tmp_path: Path) -> None:
    p = tmp_path / "t.db"
    async with SQLiteStore(p) as s:
        async with s.session("s1") as h:
            h.conversation.append(_msg("a"))
        async with s.session("s1") as h:
            h.conversation.append(_msg("b"))
            h.conversation.append(_msg("c"))
            assert h.new_messages() == h.conversation.messages[1:]
        assert _texts(await s.load_messages("s1")) == ["a", "b", "c"]
    async with (
        aiosqlite.connect(p) as db,
        db.execute("SELECT seq FROM logpose_messages ORDER BY seq") as cur,
    ):
        assert [r[0] for r in await cur.fetchall()] == [0, 1, 2]


async def test_messages_persist_when_block_raises(tmp_path: Path) -> None:
    async with SQLiteStore(tmp_path / "t.db") as s:
        with pytest.raises(ValueError, match="boom"):
            async with s.session("s1") as h:
                h.conversation.append(_msg("partial"))
                raise ValueError("boom")
        assert _texts(await s.load_messages("s1")) == ["partial"]
        # The lease was released: the session can be entered again at once.
        async with s.session("s1") as h:
            assert _texts(h.conversation.messages) == ["partial"]


async def test_second_holder_is_rejected_immediately(tmp_path: Path) -> None:
    p = tmp_path / "t.db"
    async with SQLiteStore(p) as s1, SQLiteStore(p) as s2:
        async with s1.session("s1") as h:
            h.conversation.append(_msg("mine"))
            with pytest.raises(SessionBusyError):
                async with s2.session("s1"):
                    pytest.fail("second holder entered")  # pragma: no cover
        assert _texts(await s2.load_messages("s1")) == ["mine"]


async def test_wait_succeeds_after_release(tmp_path: Path) -> None:
    p = tmp_path / "t.db"
    async with SQLiteStore(p) as s1, SQLiteStore(p) as s2:
        entered = asyncio.Event()

        async def first() -> None:
            async with s1.session("s1") as h:
                entered.set()
                h.conversation.append(_msg("first"))
                await asyncio.sleep(0.2)

        task = asyncio.create_task(first())
        await asyncio.wait_for(entered.wait(), 5)
        t0 = time.monotonic()
        async with s2.session("s1", wait=1.0) as h:
            waited = time.monotonic() - t0
            assert _texts(h.conversation.messages) == ["first"]
        await task
    assert 0.1 <= waited < 1.0


async def test_wait_times_out_with_busy(tmp_path: Path) -> None:
    p = tmp_path / "t.db"
    async with SQLiteStore(p) as s1, SQLiteStore(p) as s2:
        async with s1.session("s1"):
            t0 = time.monotonic()
            with pytest.raises(SessionBusyError):
                async with s2.session("s1", wait=0.2):
                    pass  # pragma: no cover
            assert time.monotonic() - t0 >= 0.15


async def test_expired_lease_can_be_taken(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = tmp_path / "t.db"
    async with SQLiteStore(p, lease_ttl=0.2) as s1, SQLiteStore(p) as s2:
        monkeypatch.setattr(s1, "_renewal_loop", _no_renewal)
        with pytest.raises(SessionBusyError):
            async with s1.session("s1"):
                await asyncio.sleep(0.3)
                async with s2.session("s1") as h2:
                    assert h2.created is False


async def test_stale_holder_save_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = tmp_path / "t.db"
    async with SQLiteStore(p, lease_ttl=0.2) as s1, SQLiteStore(p) as s2:
        monkeypatch.setattr(s1, "_renewal_loop", _no_renewal)
        with pytest.raises(SessionBusyError):
            async with s1.session("s1") as h1:
                h1.conversation.append(_msg("stale"))
                await asyncio.sleep(0.3)
                async with s2.session("s1") as h2:
                    h2.conversation.append(_msg("fresh"))
        assert _texts(await s2.load_messages("s1")) == ["fresh"]
    # The stale exit wrote nothing: version reflects the second holder's save only.
    assert await _scalar(p, "SELECT version FROM logpose_sessions") == 1


async def test_stale_holder_exception_is_chained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = tmp_path / "t.db"
    async with SQLiteStore(p, lease_ttl=0.2) as s1, SQLiteStore(p) as s2:
        monkeypatch.setattr(s1, "_renewal_loop", _no_renewal)
        with pytest.raises(SessionBusyError) as info:
            async with s1.session("s1") as h1:
                h1.conversation.append(_msg("stale"))
                await asyncio.sleep(0.3)
                async with s2.session("s1"):
                    pass
                raise ValueError("inner")
        assert isinstance(info.value.__cause__, ValueError)
        assert await s2.load_messages("s1") == []


async def test_renewal_marks_handle_lost_when_token_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = tmp_path / "t.db"
    async with SQLiteStore(p, lease_ttl=0.3) as s:
        with pytest.raises(SessionBusyError):
            async with s.session("s1") as h:
                h.conversation.append(_msg("x"))
                await s._write(
                    [("UPDATE logpose_sessions SET lease_token = 'other' WHERE id = 's1'", ())]
                )
                await asyncio.sleep(0.4)
                assert h.lost is True
        assert await s.load_messages("s1") == []


async def test_lost_handle_writes_no_messages(tmp_path: Path) -> None:
    p = tmp_path / "t.db"
    async with SQLiteStore(p) as s:
        with pytest.raises(SessionBusyError):
            async with s.session("s1") as h:
                h.conversation.append(_msg("x"))
                h.lost = True
        assert await s.load_messages("s1") == []
    assert await _scalar(p, "SELECT version FROM logpose_sessions") == 0


async def test_renewal_keeps_long_holder_alive(tmp_path: Path) -> None:
    p = tmp_path / "t.db"
    async with SQLiteStore(p, lease_ttl=0.3) as s1, SQLiteStore(p, lease_ttl=0.3) as s2:
        entered = asyncio.Event()

        async def holder() -> None:
            async with s1.session("s1") as h:
                entered.set()
                h.conversation.append(_msg("long"))
                await asyncio.sleep(1.0)
                assert h.lost is False

        task = asyncio.create_task(holder())
        await asyncio.wait_for(entered.wait(), 5)
        await asyncio.sleep(0.8)
        with pytest.raises(SessionBusyError):
            async with s2.session("s1"):
                pass  # pragma: no cover
        await task
        assert _texts(await s2.load_messages("s1")) == ["long"]


async def test_expired_session_starts_fresh_on_access(tmp_path: Path) -> None:
    p = tmp_path / "t.db"
    async with SQLiteStore(p) as s:
        async with s.session("s1", metadata={"k": "v"}) as h:
            h.conversation.append(_msg("old"))
        await s.record_run(_run("s1"))
        await s._write([("UPDATE logpose_sessions SET expires_at = 1 WHERE id = 's1'", ())])
        async with s.session("s1", metadata={"n": "1"}) as h:
            assert h.created is True
            assert h.conversation.messages == []
            assert h.metadata == {"n": "1"}
            assert h.next_seq == 0
    assert await _scalar(p, "SELECT count(*) FROM logpose_runs") == 0
    assert await _scalar(p, "SELECT count(*) FROM logpose_messages") == 0


async def test_session_exit_extends_expiry(tmp_path: Path) -> None:
    p = tmp_path / "t.db"
    async with SQLiteStore(p, idle_ttl=100.0) as s:
        before = int(time.time() * 1000)
        async with s.session("s1"):
            pass
    exp = await _scalar(p, "SELECT expires_at FROM logpose_sessions")
    assert isinstance(exp, int)
    assert before + 100_000 <= exp <= before + 110_000


async def test_session_rejects_bad_id(tmp_path: Path) -> None:
    async with SQLiteStore(tmp_path / "t.db") as s:
        with pytest.raises(LogposeError):
            async with s.session("../x"):
                pass  # pragma: no cover


# --- robustness ------------------------------------------------------------------------


async def test_failing_commit_rolls_back_and_next_write_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with SQLiteStore(tmp_path / "t.db") as s:
        await _live(s, "s1")
        db = s._db
        assert db is not None
        real = db.execute
        fired = False

        def fake(sql: str, *args: Any, **kwargs: Any) -> Any:
            nonlocal fired
            if sql == "COMMIT" and not fired:
                fired = True
                raise sqlite3.OperationalError("simulated commit failure")
            return real(sql, *args, **kwargs)

        monkeypatch.setattr(db, "execute", fake)
        with pytest.raises(sqlite3.OperationalError, match="simulated"):
            await s.append_messages("s1", [_msg("lost")], start_seq=0)
        assert not db.in_transaction
        await s.append_messages("s1", [_msg("kept")], start_seq=0)
        assert _texts(await s.load_messages("s1")) == ["kept"]


async def test_use_after_close_raises_not_open(tmp_path: Path) -> None:
    s = SQLiteStore(tmp_path / "t.db")
    await s.open()
    await s.close()
    with pytest.raises(LogposeError, match="not open"):
        await s.load_messages("s1")
    with pytest.raises(LogposeError, match="not open"):
        await s.append_messages("s1", [_msg("a")], start_seq=0)
    with pytest.raises(LogposeError, match="not open"):
        await s.record_run(_run())
    with pytest.raises(LogposeError, match="not open"):
        await s.delete_session("s1")
    with pytest.raises(LogposeError, match="not open"):
        await s.purge_expired()
    with pytest.raises(LogposeError, match="not open"):
        async with s.session("s1"):
            pass  # pragma: no cover


async def test_close_is_idempotent(tmp_path: Path) -> None:
    s = SQLiteStore(tmp_path / "t.db")
    await s.close()
    await s.open()
    await s.close()
    await s.close()


async def test_close_waits_for_in_flight_operation(tmp_path: Path) -> None:
    async with SQLiteStore(tmp_path / "t.db") as s:
        await asyncio.gather(s.append_messages("s1", [_msg("a")], start_seq=0), s.close())
    # Whichever order the lock gave, nothing raised mid-transaction and the store is closed.
    with pytest.raises(LogposeError, match="not open"):
        await s.load_messages("s1")


async def test_concurrent_open_does_not_leak(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0
    real = aiosqlite.connect

    def counting(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(aiosqlite, "connect", counting)
    s = SQLiteStore(tmp_path / "t.db")
    await asyncio.gather(s.open(), s.open(), s.open())
    try:
        assert calls == 1
    finally:
        await s.close()


async def test_cancel_during_begin_leaves_connection_usable(tmp_path: Path) -> None:
    async with SQLiteStore(tmp_path / "t.db") as s:
        await _live(s, "s1")
        db = s._db
        assert db is not None
        task = asyncio.create_task(s.append_messages("s1", [_msg("cancelled")], start_seq=0))
        await asyncio.sleep(0)  # let the task queue BEGIN IMMEDIATE, then cancel it
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not db.in_transaction
        await s.append_messages("s1", [_msg("kept")], start_seq=0)
        assert not db.in_transaction
        assert _texts(await s.load_messages("s1")) == ["kept"]


async def test_session_exit_survives_renewal_cancelled_mid_begin(tmp_path: Path) -> None:
    p = tmp_path / "t.db"
    async with SQLiteStore(p, lease_ttl=0.6) as s:
        async with aiosqlite.connect(p, isolation_level=None) as other:

            async def release_later() -> None:
                await asyncio.sleep(0.6)
                await other.execute("COMMIT")

            async with s.session("s1") as h:
                h.conversation.append(_msg("saved"))
                # Another connection takes the write lock, so the renewal at 0.2 s
                # blocks inside BEGIN IMMEDIATE (busy_timeout 5 s) when exit cancels it.
                await other.execute("BEGIN IMMEDIATE")
                releaser = asyncio.create_task(release_later())
                await asyncio.sleep(0.35)
            await releaser
        assert s._db is not None
        assert not s._db.in_transaction
        assert _texts(await s.load_messages("s1")) == ["saved"]
        async with s.session("s1") as h:
            assert _texts(h.conversation.messages) == ["saved"]


# --- opening a fresh or busy file -----------------------------------------------------


async def test_newer_schema_refusal_leaves_journal_mode_untouched(tmp_path: Path) -> None:
    p = tmp_path / "t.db"
    with sqlite3.connect(p) as conn:
        conn.execute("CREATE TABLE logpose_schema (version INTEGER NOT NULL)")
        conn.execute("INSERT INTO logpose_schema (version) VALUES (99)")
    with pytest.raises(LogposeError, match=r"(?s)99.*3"):
        await SQLiteStore(p).open()
    with sqlite3.connect(p) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "delete"


async def test_wal_switch_waits_for_a_writer_instead_of_failing(tmp_path: Path) -> None:
    """``PRAGMA journal_mode=WAL`` ignores busy_timeout; open() must retry it."""
    p = tmp_path / "t.db"
    writer = sqlite3.connect(p, isolation_level=None)
    writer.execute("CREATE TABLE other (x)")
    writer.execute("BEGIN IMMEDIATE")
    writer.execute("INSERT INTO other VALUES (1)")
    store = SQLiteStore(p)
    opening = asyncio.create_task(store.open())
    await asyncio.sleep(0.3)
    assert not opening.done()  # waiting, not failed
    writer.execute("COMMIT")
    writer.close()
    await opening
    await store.close()
    assert await _scalar(p, "PRAGMA journal_mode") == "wal"
    assert await _scalar(p, "SELECT version FROM logpose_schema") == 3


async def test_wal_switch_gives_up_with_logpose_error(tmp_path: Path) -> None:
    p = tmp_path / "t.db"
    writer = sqlite3.connect(p, isolation_level=None)
    writer.execute("CREATE TABLE other (x)")
    writer.execute("BEGIN IMMEDIATE")
    writer.execute("INSERT INTO other VALUES (1)")
    try:
        started = time.monotonic()
        with pytest.raises(LogposeError, match="locked"):
            await SQLiteStore(p, busy_timeout_ms=300).open()
        assert time.monotonic() - started >= 0.25
    finally:
        writer.execute("ROLLBACK")
        writer.close()


_RACE_CHILD = (
    "import asyncio, sys, time\n"
    "from logpose.store.sqlite import SQLiteStore\n"
    "async def main():\n"
    "    store = SQLiteStore(sys.argv[1])\n"
    "    while time.time() < float(sys.argv[2]):\n"
    "        pass\n"
    "    await store.open()\n"
    "    await store.close()\n"
    "asyncio.run(main())\n"
)


def test_processes_racing_on_a_fresh_file_all_open(tmp_path: Path) -> None:
    """Like ``uvicorn --workers N`` after a fresh deploy: every worker opens cleanly."""
    import subprocess
    import sys

    for round_ in range(2):
        p = tmp_path / f"race{round_}.db"
        start = time.time() + 1.5
        procs = [
            subprocess.Popen(
                [sys.executable, "-c", _RACE_CHILD, str(p), str(start)],
                stderr=subprocess.PIPE,
                text=True,
            )
            for _ in range(12)
        ]
        outputs = [(proc.communicate(timeout=60)[1], proc.returncode) for proc in procs]
        errors = [err for err, code in outputs if code]
        assert errors == []
        with sqlite3.connect(p) as conn:
            assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
            assert conn.execute("SELECT version FROM logpose_schema").fetchall() == [(3,)]


# --- schema version 2: persisted summaries ------------------------------------------------


def _make_v1_database(path: Path) -> None:
    """Build a database exactly as the version 1 release left it, with data in it."""
    from logpose.store.sqlite import _MIGRATIONS

    conn = sqlite3.connect(path, isolation_level=None)
    try:
        conn.execute("CREATE TABLE logpose_schema (version INTEGER NOT NULL)")
        conn.execute("INSERT INTO logpose_schema (version) VALUES (1)")
        for stmt in _MIGRATIONS[0]:
            conn.execute(stmt)
        conn.execute(
            "INSERT INTO logpose_sessions (id, created_at, updated_at, expires_at, version, "
            "metadata) VALUES ('s1', 1, 1, 9999999999999, 4, '{\"k\": \"v\"}')"
        )
        for seq, text in enumerate(["one", "two"]):
            conn.execute(
                "INSERT INTO logpose_messages (session_id, seq, role, content, created_at) "
                "VALUES ('s1', ?, 'user', ?, 1)",
                (seq, json.dumps([{"type": "text", "text": text}])),
            )
    finally:
        conn.close()


async def test_schema_migrates_v1_to_v3(tmp_path: Path) -> None:
    p = tmp_path / "t.db"
    _make_v1_database(p)
    async with SQLiteStore(p) as s:
        assert _texts(await s.load_messages("s1")) == ["one", "two"]
        assert await s.load_summary("s1") is None
        await s.save_summary("s1", 1, "S")
        assert await s.load_summary("s1") == (1, "S")
    assert await _scalar(p, "SELECT version FROM logpose_schema") == 3
    assert await _scalar(p, "SELECT count(*) FROM logpose_schema") == 1
    assert await _scalar(p, "SELECT metadata FROM logpose_sessions WHERE id = 's1'") == (
        '{"k": "v"}'
    )
    assert await _scalar(p, "SELECT version FROM logpose_sessions WHERE id = 's1'") == 4
    # Opening a version 2 file again changes nothing.
    async with SQLiteStore(p) as s:
        assert await s.load_summary("s1") == (1, "S")
        assert _texts(await s.load_messages("s1")) == ["one", "two"]
    assert await _scalar(p, "SELECT version FROM logpose_schema") == 3


async def test_load_summary_returns_latest(tmp_path: Path) -> None:
    async with SQLiteStore(tmp_path / "t.db") as s:
        assert await s.load_summary("s1") is None
        await s.save_summary("s1", 3, "first")
        await s.save_summary("s1", 9, "second")
        await s.save_summary("s2", 1, "other")
        assert await s.load_summary("s1") == (9, "second")
        # Saving again at an existing seq replaces that row and becomes the latest.
        await s.save_summary("s1", 3, "redone")
        assert await s.load_summary("s1") == (3, "redone")
        assert await s.load_summary("s2") == (1, "other")


async def test_delete_session_removes_summaries(tmp_path: Path) -> None:
    p = tmp_path / "t.db"
    async with SQLiteStore(p) as s:
        await s.save_summary("s1", 1, "a")
        await s.save_summary("s2", 1, "b")
        await s.delete_session("s1")
        assert await s.load_summary("s1") is None
        assert await s.load_summary("s2") == (1, "b")


async def test_purge_removes_summaries(tmp_path: Path) -> None:
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
            await s.save_summary(sid, 0, "summary " + sid)
        assert await s.purge_expired(now_ms=5000) == 1
        assert await s.load_summary("old") is None
        assert await s.load_summary("new") == (0, "summary new")
    assert await _scalar(p, "SELECT count(*) FROM logpose_summaries") == 1


async def test_expired_session_drops_its_summary_on_access(tmp_path: Path) -> None:
    p = tmp_path / "t.db"
    async with SQLiteStore(p) as s:
        async with s.session("s1") as h:
            h.conversation.append(_msg("old"))
        await s.save_summary("s1", 0, "stale")
        await s._write([("UPDATE logpose_sessions SET expires_at = 1 WHERE id = 's1'", ())])
        async with s.session("s1") as h:
            assert h.created is True
            assert h.conversation._compaction is None
        assert await s.load_summary("s1") is None


def _turns(n: int) -> list[Message]:
    out: list[Message] = []
    for i in range(n):
        out.append(Message(role="user", content=[TextBlock(text=f"q{i}")]))
        out.append(Message(role="assistant", content=[TextBlock(text=f"a{i}")]))
    return out


async def test_session_seeds_compaction_from_stored_summary(tmp_path: Path) -> None:
    from logpose.compaction import fingerprint

    async with SQLiteStore(tmp_path / "t.db") as s:
        async with s.session("s1") as h:
            h.conversation.extend(_turns(3))
        # upto_seq is the seq of the last summarized message: messages 1..3.
        await s.save_summary("s1", 3, "S")
        async with s.session("s1") as h:
            state = h.conversation._compaction
            assert state is not None
            assert state.summary == "S"
            assert state.cut_index == 4
            assert state.ratio == 1.0
            assert state.fingerprint == fingerprint(h.conversation.messages[:4])


async def test_session_ignores_summary_past_loaded_history(tmp_path: Path) -> None:
    async with SQLiteStore(tmp_path / "t.db") as s:
        async with s.session("s1") as h:
            h.conversation.extend(_turns(1))
        await s.save_summary("s1", 5, "from the future")
        async with s.session("s1") as h:
            assert h.conversation._compaction is None


async def test_session_saves_new_summary_on_exit(tmp_path: Path) -> None:
    from logpose.compaction import _CompactionState, fingerprint

    async with SQLiteStore(tmp_path / "t.db") as s:
        async with s.session("s1") as h:
            h.conversation.extend(_turns(3))
            h.conversation._compaction = _CompactionState(
                summary="S", cut_index=4, fingerprint=fingerprint(h.conversation.messages[:4])
            )
        assert await s.load_summary("s1") == (3, "S")
        # Unchanged on the next block: nothing new is written.
        async with s.session("s1") as h:
            h.conversation.extend(_turns(1))
        assert await s.load_summary("s1") == (3, "S")
        # A summary whose fingerprint no longer matches the history is not saved.
        async with s.session("s1") as h:
            h.conversation._compaction = _CompactionState(
                summary="stale", cut_index=6, fingerprint="0:nope"
            )
        assert await s.load_summary("s1") == (3, "S")


async def test_lost_handle_saves_no_summary(tmp_path: Path) -> None:
    from logpose.compaction import _CompactionState, fingerprint

    async with SQLiteStore(tmp_path / "t.db") as s:
        with pytest.raises(SessionBusyError):
            async with s.session("s1") as h:
                h.conversation.extend(_turns(2))
                h.conversation._compaction = _CompactionState(
                    summary="S", cut_index=2, fingerprint=fingerprint(h.conversation.messages[:2])
                )
                h.lost = True
        assert await s.load_summary("s1") is None


async def test_purge_expired_spares_a_leased_session(tmp_path: Path) -> None:
    p = tmp_path / "t.db"
    async with SQLiteStore(p) as s:
        async with s.session("s1") as h:
            h.conversation.append(_msg("hi"))
        async with s.session("s1") as h:
            # Past expires_at, but leased right now: purge must not touch it.
            await s._write([("UPDATE logpose_sessions SET expires_at = 1 WHERE id = 's1'", ())])
            assert await s.purge_expired() == 0
            assert len(await s.load_messages("s1")) == 1
            h.conversation.append(_msg("more"))
        assert len(await s.load_messages("s1")) == 2


async def test_purge_expired_spares_a_leased_sessions_summary(tmp_path: Path) -> None:
    async with SQLiteStore(tmp_path / "t.db") as s:
        async with s.session("s1") as h:
            h.conversation.append(_msg("hi"))
        await s.save_summary("s1", 0, "S")
        async with s.session("s1") as h:
            assert h.stored_summary == (0, "S")  # loaded after commit, under the lease
            await s._write([("UPDATE logpose_sessions SET expires_at = 1 WHERE id = 's1'", ())])
            assert await s.purge_expired() == 0
            assert await s.load_summary("s1") == (0, "S")


async def test_failed_history_load_releases_the_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import logpose.store.sqlite as sqlite_mod

    p = tmp_path / "t.db"
    async with SQLiteStore(p) as s:
        async with s.session("s1") as h:
            h.conversation.append(_msg("hi"))
        real = sqlite_mod._fetch_messages

        async def boom(db: Any, sid: str) -> Any:
            raise sqlite3.OperationalError("disk I/O error")

        monkeypatch.setattr(sqlite_mod, "_fetch_messages", boom)
        with pytest.raises(sqlite3.OperationalError):
            async with s.session("s1"):
                pass
        monkeypatch.setattr(sqlite_mod, "_fetch_messages", real)
        async with s.session("s1") as h:  # not left leased
            assert len(h.conversation.messages) == 1


async def test_renewal_stops_after_idle_bound(tmp_path: Path) -> None:
    p = tmp_path / "t.db"
    async with SQLiteStore(p, lease_ttl=0.2) as s:
        with pytest.raises(SessionBusyError):
            async with s.session("s1") as h:
                await asyncio.sleep(0.6 + 0.2 + 0.3)
                assert h.lapsed
                async with s.session("s1") as other:  # the idle lease lapsed
                    assert other.id == "s1"
        # The stale holder's exit is refused: the lease moved on meanwhile.


async def test_renewal_continues_while_a_run_is_active(tmp_path: Path) -> None:
    p = tmp_path / "t.db"
    async with SQLiteStore(p, lease_ttl=0.2) as s:
        async with s.session("s1") as h:
            h.idle_since = None  # a run is in progress
            await asyncio.sleep(0.6 + 0.2 + 0.3)
            assert not h.lapsed
            with pytest.raises(SessionBusyError):
                async with s.session("s1"):
                    pass
