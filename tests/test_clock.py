"""Tests for the built-in ``current_time`` tool."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from logpose import Agent, clock, current_time
from logpose.messages import ToolResultBlock
from tests.fake_provider import FakeProvider, ScriptedTurn, tool_call

EXPECTED = "Thursday, 26 November 2026, 14:05 IST (Asia/Kolkata)"


@pytest.fixture(autouse=True)
def frozen(monkeypatch: pytest.MonkeyPatch) -> None:
    instant = datetime(2026, 11, 26, 8, 35, tzinfo=timezone.utc)
    monkeypatch.setattr(clock, "_now", lambda tz: instant.astimezone(tz))


def test_formats_time_in_zone() -> None:
    assert current_time(timezone="Asia/Kolkata") == EXPECTED


def test_default_is_utc() -> None:
    assert current_time().endswith("UTC (UTC)")


@pytest.mark.parametrize("bad", ["Mars/Base", "", "../etc/passwd"])
def test_invalid_zone_returns_message(bad: str) -> None:
    out = current_time(timezone=bad)
    assert repr(bad) in out
    assert "'Asia/Kolkata'" in out


def test_tool_schema() -> None:
    spec = current_time.spec()
    assert spec.name == "current_time"
    prop = spec.input_schema["properties"]["timezone"]
    assert prop["type"] == "string"
    assert prop["description"]
    assert "timezone" not in spec.input_schema.get("required", [])


async def test_agent_tool_round() -> None:
    provider = FakeProvider(
        [
            ScriptedTurn.tool_use(tool_call("current_time", {"timezone": "Asia/Kolkata"})),
            ScriptedTurn.text("done"),
        ]
    )
    result = await Agent(provider, model="m", tools=[current_time]).run("time?")
    block = result.messages[2].content[0]
    assert isinstance(block, ToolResultBlock)
    assert not block.is_error
    assert EXPECTED in block.content


def test_exported() -> None:
    import logpose

    assert "current_time" in logpose.__all__


@pytest.fixture
def no_tzdata(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(name: str) -> None:
        raise KeyError(name)

    monkeypatch.setattr(clock, "ZoneInfo", boom)
    monkeypatch.setattr(clock, "_tzdata_available", lambda: False)


def test_default_utc_without_tzdata(no_tzdata: None) -> None:
    assert current_time().endswith("UTC (UTC)")
    assert current_time(timezone="etc/utc").endswith("(etc/utc)")


def test_named_zone_without_tzdata_falls_back_to_utc(no_tzdata: None) -> None:
    out = current_time(timezone="Asia/Kolkata")
    assert "08:35 UTC" in out
    assert "not installed" in out


def test_long_bad_zone_is_truncated() -> None:
    out = current_time(timezone="x" * 5000)
    assert len(out) < 300
