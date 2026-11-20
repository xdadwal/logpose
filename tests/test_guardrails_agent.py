"""Input guardrails wired into the agent loop (blocking mode)."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from logpose.agent import Agent, Conversation
from logpose.errors import GuardrailTripped, LogposeError, RunTimeoutError
from logpose.events import RunEnd, TextDelta
from logpose.guardrails import GuardrailInput, GuardrailVerdict, pattern_guard
from logpose.messages import Message, ToolResultBlock, Usage
from logpose.runtime import RuntimeEvent
from logpose.store import manager
from tests.fake_provider import FakeProvider, ScriptedTurn

PROMPT = "Tell me a SECRETPROMPT story"
REPLY = "I can only help with billing."
ERROR_REPLY = "Sorry, something went wrong. Please try again."


def allow_guard(inp: GuardrailInput) -> GuardrailVerdict:
    return GuardrailVerdict.allow(usage=Usage(input_tokens=3, output_tokens=1))


async def off_topic_guard(inp: GuardrailInput) -> GuardrailVerdict:
    return GuardrailVerdict.block("off_topic", REPLY, reason="r", scores={"on_topic": 0.1})


class Answer(BaseModel):
    value: str


async def _collect(agent: Agent, *args: Any, **kwargs: Any) -> list[Any]:
    return [event async for event in agent.stream(*args, **kwargs)]


async def test_allowed_run_unchanged() -> None:
    plain_provider = FakeProvider(ScriptedTurn.text("hello", usage=Usage(input_tokens=10)))
    guarded_provider = FakeProvider(ScriptedTurn.text("hello", usage=Usage(input_tokens=10)))
    plain = Agent(plain_provider, model="m")
    guarded = Agent(guarded_provider, model="m", input_guardrails=[allow_guard])
    a = await plain.run(PROMPT)
    b = await guarded.run(PROMPT)
    assert plain_provider.requests == guarded_provider.requests
    assert b.usage == a.usage + Usage(input_tokens=3, output_tokens=1)
    assert replace(b, usage=a.usage) == a
    assert b.guardrail is None


async def test_block_streams_placeholder_and_skips_model() -> None:
    provider = FakeProvider(ScriptedTurn.text("never"))
    agent = Agent(provider, model="m", input_guardrails=[off_topic_guard])
    events = await _collect(agent, PROMPT)
    assert provider.call_count == 0
    assert len(events) == 2
    assert events[0] == TextDelta(REPLY)
    assert isinstance(events[1], RunEnd)
    result = events[1].result
    assert result.text == REPLY
    assert result.stop_reason == "refusal"
    assert result.iterations == 0
    assert result.guardrail is not None
    assert result.guardrail.category == "off_topic"
    assert result.guardrail.guard == "off_topic_guard"
    assert result.guardrail.scores == {"on_topic": 0.1}


async def test_blocked_prompt_not_in_history() -> None:
    conv = Conversation([Message.user("earlier"), Message.assistant_text("ok")])
    before = list(conv.messages)
    calls = {"n": 0}

    def block_once(inp: GuardrailInput) -> GuardrailVerdict:
        calls["n"] += 1
        if "SECRETPROMPT" in inp.message.text:
            return GuardrailVerdict.block("off_topic", REPLY)
        return GuardrailVerdict.allow()

    provider = FakeProvider(ScriptedTurn.text("fine"))
    agent = Agent(provider, model="m", input_guardrails=[block_once])
    result = await agent.run(PROMPT, conversation=conv)
    assert result.text == REPLY
    assert conv.messages == before
    assert result.messages == before
    await agent.run("an allowed question", conversation=conv)
    sent = [m.text for m in provider.requests[0].messages]
    assert all("SECRETPROMPT" not in text for text in sent)
    assert sent == ["earlier", "ok", "an allowed question"]


async def test_sync_guard_short_circuits_async() -> None:
    called = {"async": False}

    async def recording_guard(inp: GuardrailInput) -> GuardrailVerdict:
        called["async"] = True
        return GuardrailVerdict.allow()

    provider = FakeProvider(ScriptedTurn.text("never"))
    agent = Agent(
        provider,
        model="m",
        # The async guard is listed first: sync guards still run before it.
        input_guardrails=[recording_guard, pattern_guard(max_chars=10, reply="too long")],
    )
    result = await agent.run("x" * 50)
    assert result.text == "too long"
    assert result.guardrail is not None and result.guardrail.category == "too_long"
    assert result.guardrail.guard == "pattern_guard"
    assert called["async"] is False
    assert provider.call_count == 0


async def test_first_async_block_wins_and_others_cancelled() -> None:
    cancelled = {"slow": False}

    async def slow_guard(inp: GuardrailInput) -> GuardrailVerdict:
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            cancelled["slow"] = True
            raise
        return GuardrailVerdict.block("slow", "slow reply")  # pragma: no cover

    async def fast_guard(inp: GuardrailInput) -> GuardrailVerdict:
        await asyncio.sleep(0.01)
        return GuardrailVerdict.block("unsafe", "fast reply")

    agent = Agent(FakeProvider(()), model="m", input_guardrails=[slow_guard, fast_guard])
    result = await asyncio.wait_for(agent.run(PROMPT), timeout=2)
    assert result.text == "fast reply"
    assert result.guardrail is not None and result.guardrail.guard == "fast_guard"
    assert cancelled["slow"] is True


async def test_guard_exception_blocks_with_error_reply() -> None:
    async def broken(inp: GuardrailInput) -> GuardrailVerdict:
        raise RuntimeError("boom SECRETPROMPT")

    def broken_sync(inp: GuardrailInput) -> GuardrailVerdict:
        raise ValueError("bad")

    for guard in (broken, broken_sync):
        provider = FakeProvider(ScriptedTurn.text("never"))
        agent = Agent(provider, model="m", input_guardrails=[guard])
        result = await agent.run(PROMPT)
        assert result.text == ERROR_REPLY
        assert result.stop_reason == "refusal"
        assert result.guardrail is not None and result.guardrail.category == "error"
        assert "SECRETPROMPT" not in (result.guardrail.reason or "")
        assert provider.call_count == 0


async def test_output_type_block_raises_guardrail_tripped() -> None:
    provider = FakeProvider(ScriptedTurn.text("never"))
    agent = Agent(provider, model="m", input_guardrails=[off_topic_guard])
    with pytest.raises(GuardrailTripped) as info:
        await agent.run(PROMPT, output_type=Answer)
    assert info.value.result.category == "off_topic"
    assert provider.call_count == 0


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    path = tmp_path / "s.db"
    monkeypatch.setenv("LOGPOSE_DB_PATH", str(path))
    manager.reset_stores_for_tests()
    yield path
    manager.reset_stores_for_tests()


def _rows(path: Path, sql: str) -> list[tuple[Any, ...]]:
    with sqlite3.connect(path) as conn:
        return list(conn.execute(sql))


async def test_session_records_blocked_outcome(db: Path) -> None:
    seen: list[GuardrailInput] = []

    async def guard(inp: GuardrailInput) -> GuardrailVerdict:
        seen.append(inp)
        if "SECRETPROMPT" in inp.message.text:
            return GuardrailVerdict.block(
                "off_topic", REPLY, usage=Usage(input_tokens=4, output_tokens=2)
            )
        return GuardrailVerdict.allow()

    provider = FakeProvider(ScriptedTurn.text("hi there"))
    agent = Agent(provider, model="m", input_guardrails=[guard])
    await agent.run("hello", session_id="s1", session_metadata={"tenant": "t1"})
    stored = _rows(db, "SELECT seq, role FROM logpose_messages ORDER BY seq")
    result = await agent.run(PROMPT, session_id="s1")
    assert result.text == REPLY
    assert _rows(db, "SELECT seq, role FROM logpose_messages ORDER BY seq") == stored
    rows = _rows(db, "SELECT outcome, iterations, input_tokens, output_tokens FROM logpose_runs")
    assert ("blocked", 0, 4, 2) in rows
    assert seen[-1].metadata == {"tenant": "t1"}
    assert [m.text for m in seen[-1].recent] == ["hello", "hi there"]

    agent2 = Agent(FakeProvider(()), model="m", input_guardrails=[off_topic_guard])
    with pytest.raises(GuardrailTripped):
        await agent2.run(PROMPT, session_id="s2", output_type=Answer)
    assert _rows(db, "SELECT outcome FROM logpose_runs WHERE session_id = 's2'") == [("blocked",)]
    await agent.aclose()
    await agent2.aclose()


async def test_guardrail_event_has_no_text() -> None:
    events: list[RuntimeEvent] = []
    agent = Agent(
        FakeProvider(()), model="m", input_guardrails=[off_topic_guard], observers=[events.append]
    )
    await agent.run(PROMPT)
    decided = [e for e in events if e.name == "guardrail.decided"]
    assert len(decided) == 1
    event = decided[0]
    assert event.guard_name == "off_topic_guard"
    assert event.guard_allowed is False
    assert event.guard_category == "off_topic"
    assert event.guard_scores is not None and json.loads(event.guard_scores) == {"on_topic": 0.1}
    assert event.duration_seconds is not None
    for e in events:
        dumped = json.dumps(e.log_fields(), default=str)
        assert "SECRETPROMPT" not in dumped
        assert REPLY not in dumped


async def test_deadline_during_guard_is_run_timeout() -> None:
    async def slow(inp: GuardrailInput) -> GuardrailVerdict:
        await asyncio.sleep(5)
        return GuardrailVerdict.allow()  # pragma: no cover

    agent = Agent(FakeProvider(()), model="m", input_guardrails=[slow], run_timeout=0.2)
    with pytest.raises(RunTimeoutError):
        await asyncio.wait_for(agent.run(PROMPT), timeout=2)


async def test_guard_usage_added_to_result() -> None:
    async def billed(inp: GuardrailInput) -> GuardrailVerdict:
        return GuardrailVerdict.allow(usage=Usage(input_tokens=5, reasoning_tokens=2))

    provider = FakeProvider(ScriptedTurn.text("ok", usage=Usage(input_tokens=10, output_tokens=1)))
    agent = Agent(provider, model="m", input_guardrails=[billed, allow_guard])
    result = await agent.run(PROMPT)
    assert result.usage == Usage(input_tokens=18, output_tokens=2, reasoning_tokens=2)

    blocked = Agent(
        FakeProvider(()),
        model="m",
        input_guardrails=[
            lambda inp: GuardrailVerdict.block("x", "no", usage=Usage(input_tokens=7))
        ],
    )
    assert (await blocked.run(PROMPT)).usage == Usage(input_tokens=7)


async def test_no_prompt_skips_guards() -> None:
    called = {"n": 0}

    def guard(inp: GuardrailInput) -> GuardrailVerdict:
        called["n"] += 1
        return GuardrailVerdict.block("x", "no")

    conv = Conversation([Message.user("hi")])
    agent = Agent(FakeProvider(ScriptedTurn.text("ok")), model="m", input_guardrails=[guard])
    result = await agent.run(None, conversation=conv)
    assert result.text == "ok"
    tool_only = Message(
        role="user", content=[ToolResultBlock(tool_use_id="t1", content="x", is_error=False)]
    )
    assert called["n"] == 0
    agent2 = Agent(FakeProvider(ScriptedTurn.text("ok")), model="m", input_guardrails=[guard])
    conv2 = Conversation([Message.user("q")])
    await agent2.run(tool_only, conversation=conv2)
    assert called["n"] == 0


def test_agent_validates_guardrail_options() -> None:
    provider = FakeProvider(())
    with pytest.raises(LogposeError):
        Agent(provider, model="m", input_guardrails=["nope"])  # type: ignore[list-item]
    with pytest.raises(LogposeError):
        Agent(provider, model="m", guardrail_mode="eventually")  # type: ignore[arg-type]
    agent = Agent(provider, model="m", input_guardrails=[allow_guard])
    assert agent.input_guardrails == (allow_guard,)
    assert agent.guardrail_mode == "blocking"
