"""OTLP/JSON export: shape, root-only, incremental, filters, purging."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

import logpose
from logpose import Agent, ExportResult, LogposeError, export_traces
from logpose.store import manager
from logpose.tracing import semconv as sc
from logpose.tracing.spans import SpanRecord
from tests.fake_provider import FakeProvider, ScriptedTurn

UTC = timezone.utc
T0 = 1_700_000_000 * 1_000_000_000  # ns


@pytest.fixture(autouse=True)
def _clean() -> Iterator[None]:
    manager.reset_stores_for_tests()
    yield
    manager.reset_stores_for_tests()


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "t.db"
    monkeypatch.setenv("LOGPOSE_DB_PATH", str(path))
    return path


def _span(
    trace: str,
    span: str,
    *,
    parent: str | None = None,
    start: int = T0,
    session: str | None = "s",
    attrs: dict[str, Any] | None = None,
    events: list[dict[str, Any]] | None = None,
    kind: int = 1,
    status: int = 0,
) -> SpanRecord:
    return SpanRecord(
        trace_id=trace,
        span_id=span,
        parent_span_id=parent,
        run_id=f"run-{trace}",
        session_id=session,
        name="invoke_agent" if parent is None else "chat",
        kind=kind,
        start_ns=start,
        end_ns=start + 5_000,
        status_code=status,
        status_message="boom" if status == 2 else None,
        attributes=attrs or {},
        events=events or [],
    )


def _tid(n: int) -> str:
    return f"{n:032x}"


def _sid(n: int) -> str:
    return f"{n:016x}"


async def _put(*spans: SpanRecord) -> None:
    store = await manager.get_store()
    await store.write_spans(list(spans))


def _lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text().splitlines()]


def _trace_ids(path: Path) -> list[str]:
    out = []
    for req in _lines(path):
        for rs in req["resourceSpans"]:
            for ss in rs["scopeSpans"]:
                out.extend({s["traceId"] for s in ss["spans"]})
    return out


async def test_export_otlp_json_shape(db: Path, tmp_path: Path) -> None:
    await _put(
        _span(
            _tid(1),
            _sid(2),
            parent=_sid(1),
            kind=3,
            status=2,
            start=T0 + 10,
            attrs={
                "s": "x",
                "i": 7,
                "f": 1.5,
                "b": True,
                "arr": ["a", "b"],
                "ints": [1, 2],
            },
            events=[{"name": "exception", "time_unix_nano": T0 + 12, "attributes": {"k": "v"}}],
        ),
        _span(_tid(1), _sid(1)),
    )
    out = tmp_path / "o.jsonl"
    result = await export_traces(out)
    assert isinstance(result, ExportResult)
    assert result.traces == 1
    assert result.through == datetime(2023, 11, 14, 22, 13, 20, tzinfo=UTC)
    (req,) = _lines(out)
    (rs,) = req["resourceSpans"]
    res = {a["key"]: a["value"] for a in rs["resource"]["attributes"]}
    assert res["service.name"] == {"stringValue": "logpose"}
    assert res["telemetry.sdk.name"] == {"stringValue": "logpose"}
    assert res["telemetry.sdk.version"] == {"stringValue": logpose.__version__}
    assert res["logpose.semconv.version"] == {"stringValue": sc.SEMCONV_VERSION}
    (ss,) = rs["scopeSpans"]
    assert ss["scope"] == {"name": "logpose", "version": logpose.__version__}
    assert ss["schemaUrl"] == sc.SCHEMA_URL
    root, child = ss["spans"]
    assert root["spanId"] == _sid(1) and "parentSpanId" not in root
    assert root["traceId"] == _tid(1)
    assert root["startTimeUnixNano"] == str(T0)
    assert root["endTimeUnixNano"] == str(T0 + 5_000)
    assert child["parentSpanId"] == _sid(1)
    assert child["kind"] == 3 and child["status"] == {"code": 2, "message": "boom"}
    assert root["status"] == {}
    attrs = {a["key"]: a["value"] for a in child["attributes"]}
    assert attrs["s"] == {"stringValue": "x"}
    assert attrs["i"] == {"intValue": "7"}
    assert attrs["f"] == {"doubleValue": 1.5}
    assert attrs["b"] == {"boolValue": True}
    assert attrs["arr"] == {"arrayValue": {"values": [{"stringValue": "a"}, {"stringValue": "b"}]}}
    assert attrs["ints"] == {"arrayValue": {"values": [{"intValue": "1"}, {"intValue": "2"}]}}
    (ev,) = child["events"]
    assert ev == {
        "name": "exception",
        "timeUnixNano": str(T0 + 12),
        "attributes": [{"key": "k", "value": {"stringValue": "v"}}],
    }


async def test_export_skips_live_traces_and_is_incremental(db: Path, tmp_path: Path) -> None:
    # Trace 1 ended; trace 2 is live (child written, root not yet).
    await _put(_span(_tid(1), _sid(1)))
    await _put(_span(_tid(2), _sid(20), parent=_sid(2), start=T0 + 1_000_000))
    out = tmp_path / "o.jsonl"
    first = await export_traces(out)
    assert first.traces == 1 and _trace_ids(out) == [_tid(1)]
    assert first.through is not None

    # Trace 2 ends. Trace 3 starts later still.
    await _put(_span(_tid(2), _sid(2), start=T0 + 1_000_000))
    await _put(_span(_tid(3), _sid(3), start=T0 + 2_000_000))
    second = await export_traces(out, since=first.through)
    assert sorted(_trace_ids(out)) == [_tid(2), _tid(3)]
    assert second.traces == 2

    again = await export_traces(out, since=second.through)
    assert again.traces == 0 and again.through is None
    assert out.read_text() == ""


async def test_incremental_never_repeats_equal_start_times(db: Path, tmp_path: Path) -> None:
    # Three roots in the same microsecond (distinct ns): one pass, then nothing.
    await _put(
        _span(_tid(1), _sid(1), start=T0 + 1),
        _span(_tid(2), _sid(2), start=T0 + 2),
        _span(_tid(3), _sid(3), start=T0 + 3),
    )
    out = tmp_path / "o.jsonl"
    r1 = await export_traces(out)
    assert r1.traces == 3
    r2 = await export_traces(out, since=r1.through)
    assert r2.traces == 0


async def test_export_filters_by_session_and_window(db: Path, tmp_path: Path) -> None:
    us = 1_000
    await _put(
        _span(_tid(1), _sid(1), start=T0, session="a"),
        _span(_tid(1), _sid(11), parent=_sid(1), start=T0 + 50 * us, session="a"),
        _span(_tid(2), _sid(2), start=T0 + 100 * us, session="b"),
        _span(_tid(3), _sid(3), start=T0 + 200 * us, session="a"),
    )
    out = tmp_path / "o.jsonl"
    r = await export_traces(out, session_id="a")
    assert sorted(_trace_ids(out)) == [_tid(1), _tid(3)] and r.traces == 2
    base = datetime.fromtimestamp(T0 / 1e9, UTC)
    # until is inclusive, window matches the ROOT start (child at +50 us is irrelevant).
    r = await export_traces(out, until=base + timedelta(microseconds=100))
    assert sorted(_trace_ids(out)) == [_tid(1), _tid(2)] and r.traces == 2
    # since is exclusive.
    r = await export_traces(
        out, since=base + timedelta(microseconds=100), until=base + timedelta(microseconds=200)
    )
    assert _trace_ids(out) == [_tid(3)]
    # A child inside the window of a root outside it does not export the trace.
    r = await export_traces(
        out, since=base + timedelta(microseconds=10), until=base + timedelta(microseconds=60)
    )
    assert r.traces == 0


async def test_export_rejects_naive_datetimes(db: Path, tmp_path: Path) -> None:
    with pytest.raises(LogposeError):
        await export_traces(tmp_path / "o", since=datetime(2024, 1, 1))
    with pytest.raises(LogposeError):
        await export_traces(tmp_path / "o", until=datetime(2024, 1, 1))


async def test_export_replaces_file_atomically(db: Path, tmp_path: Path) -> None:
    out = tmp_path / "o.jsonl"
    out.write_text("old\n")
    await _put(_span(_tid(1), _sid(1)))
    await export_traces(out)
    assert _trace_ids(out) == [_tid(1)]
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith("o")] == ["o.jsonl"]


async def test_purge_traces(db: Path) -> None:
    now_ns = int(datetime.now(UTC).timestamp() * 1e9)
    await _put(
        _span(_tid(1), _sid(1), start=now_ns - 10 * 86_400 * 10**9),
        _span(_tid(2), _sid(2), start=now_ns),
    )
    agent = Agent(FakeProvider([ScriptedTurn.text("x")]), model="m")
    assert await agent.purge_traces(timedelta(days=1)) == 1
    store = await manager.get_store()
    assert [s.trace_id async for s in store.read_spans()] == [_tid(2)]
    with pytest.raises(LogposeError):
        await agent.purge_traces(timedelta(days=-1))
    await agent.aclose()


def test_import_logpose_does_not_import_aiosqlite() -> None:
    code = "import sys, logpose; print('aiosqlite' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
