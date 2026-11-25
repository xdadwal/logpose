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
    assert current_time.handler(timezone="Asia/Kolkata") == EXPECTED


def test_default_is_utc() -> None:
    assert current_time.handler().endswith("UTC (UTC)")


@pytest.mark.parametrize("bad", ["Mars/Base", "", "../etc/passwd"])
def test_invalid_zone_returns_message(bad: str) -> None:
    out = current_time.handler(timezone=bad)
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
