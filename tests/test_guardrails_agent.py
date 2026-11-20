"""Input guardrails wired into the agent loop (blocking and parallel modes)."""

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

from logpose import tool
from logpose.agent import Agent, Conversation
from logpose.errors import GuardrailTripped, LogposeError, RunTimeoutError
from logpose.events import RunEnd, TextDelta
from logpose.guardrails import GuardrailInput, GuardrailVerdict, pattern_guard
from logpose.messages import Message, ToolResultBlock, Usage
from logpose.runtime import RuntimeEvent
from logpose.store import manager
from tests.fake_provider import FakeProvider, ScriptedTurn, tool_call

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


# --- Task 2 follow-ups: usage per decision, same-round decisions, no state touched


async def test_finished_guard_usage_survives_run_timeout(db: Path) -> None:
    async def fast_billed(inp: GuardrailInput) -> GuardrailVerdict:
        return GuardrailVerdict.allow(usage=Usage(input_tokens=6, output_tokens=2))

    async def slow(inp: GuardrailInput) -> GuardrailVerdict:
        await asyncio.sleep(5)
        return GuardrailVerdict.allow()  # pragma: no cover

    for mode in ("blocking", "parallel"):
        agent = Agent(
            FakeProvider(ScriptedTurn.text("never", delay=5)),
            model="m",
            input_guardrails=[fast_billed, slow],
            guardrail_mode=mode,  # type: ignore[arg-type]
            run_timeout=0.2,
        )
        with pytest.raises(RunTimeoutError):
            await asyncio.wait_for(agent.run(PROMPT, session_id=f"t-{mode}"), timeout=2)
        rows = _rows(
            db,
            "SELECT outcome, input_tokens, output_tokens FROM logpose_runs "
            f"WHERE session_id = 't-{mode}'",
        )
        assert rows == [("run_timeout", 6, 2)], mode
        await agent.aclose()


async def test_guards_finishing_together_are_all_decided() -> None:
    async def first(inp: GuardrailInput) -> GuardrailVerdict:
        return GuardrailVerdict.block("first", "first reply", usage=Usage(input_tokens=3))

    async def second(inp: GuardrailInput) -> GuardrailVerdict:
        return GuardrailVerdict.block("second", "second reply", usage=Usage(input_tokens=5))

    events: list[RuntimeEvent] = []
    agent = Agent(
        FakeProvider(()), model="m", input_guardrails=[first, second], observers=[events.append]
    )
    result = await agent.run(PROMPT)
    decided = [e.guard_name for e in events if e.name == "guardrail.decided"]
    assert sorted(decided) == ["first", "second"]
    assert result.usage == Usage(input_tokens=8)
    # Among guards decided in the same round, the first block in list order wins.
    assert result.text == "first reply"
    assert result.guardrail is not None and result.guardrail.guard == "first"


async def test_blocked_run_leaves_conversation_state_untouched() -> None:
    conv = Conversation([Message.user("earlier"), Message.assistant_text("ok")])
    assert conv._compaction is None
    agent = Agent(FakeProvider(()), model="m", input_guardrails=[off_topic_guard])
    await agent.run(PROMPT, conversation=conv)
    assert conv._compaction is None


# --- Parallel mode


async def test_parallel_allow_flushes_buffered_text_in_order() -> None:
    provider = FakeProvider(ScriptedTurn.text("abc", chunks=["a", "b", "c"]))
    state = {"resolved": False, "calls_seen": -1}

    async def slow_allow(inp: GuardrailInput) -> GuardrailVerdict:
        await asyncio.sleep(0.1)
        state["calls_seen"] = provider.call_count
        state["resolved"] = True
        return GuardrailVerdict.allow()

    agent = Agent(provider, model="m", input_guardrails=[slow_allow], guardrail_mode="parallel")
    received: list[tuple[Any, bool]] = []
    async for event in agent.stream(PROMPT):
        received.append((event, state["resolved"]))
    # The provider was called while the guard was still running.
    assert state["calls_seen"] == 1
    texts = [event for event, _ in received if isinstance(event, TextDelta)]
    assert texts == [TextDelta("a"), TextDelta("b"), TextDelta("c")]
    assert all(resolved for _, resolved in received)
    assert [type(event).__name__ for event, _ in received][:3] == ["TextDelta"] * 3
    final = received[-1][0]
    assert isinstance(final, RunEnd)
    assert final.result.text == "abc"
    assert final.result.guardrail is None


async def test_parallel_allowed_request_matches_blocking_mode() -> None:
    requests = []
    for mode in ("blocking", "parallel"):
        provider = FakeProvider(ScriptedTurn.text("hello"))
        conv = Conversation([Message.user("earlier"), Message.assistant_text("ok")])
        agent = Agent(
            provider,
            model="m",
            input_guardrails=[allow_guard],
            guardrail_mode=mode,  # type: ignore[arg-type]
        )
        result = await agent.run(PROMPT, conversation=conv)
        assert result.text == "hello"
        assert [m.text for m in conv.messages] == ["earlier", "ok", PROMPT, "hello"]
        assert conv._compaction is not None
        requests.append(provider.requests)
    assert requests[0] == requests[1]


async def test_parallel_block_cancels_provider_and_restores_history() -> None:
    conv = Conversation([Message.user("earlier"), Message.assistant_text("ok")])
    before = list(conv.messages)
    provider = FakeProvider(ScriptedTurn.text("never", delay=5))

    async def slow_block(inp: GuardrailInput) -> GuardrailVerdict:
        await asyncio.sleep(0.05)
        return GuardrailVerdict.block("off_topic", REPLY, usage=Usage(input_tokens=2))

    agent = Agent(provider, model="m", input_guardrails=[slow_block], guardrail_mode="parallel")
    started = asyncio.get_running_loop().time()
    result = await asyncio.wait_for(agent.run(PROMPT, conversation=conv), timeout=2)
    assert asyncio.get_running_loop().time() - started < 1  # not waiting out the turn
    assert provider.call_count == 1
    assert result.text == REPLY
    assert result.stop_reason == "refusal"
    assert result.iterations == 0
    assert result.usage == Usage(input_tokens=2)
    assert result.guardrail is not None and result.guardrail.category == "off_topic"
    assert conv.messages == before
    assert result.messages == before
    assert conv._compaction is None


async def test_parallel_block_rolls_back_summary_state() -> None:
    from tests.test_compaction import chat_history, small

    provider = FakeProvider(
        [
            ScriptedTurn.text("SUMMARY-ONE", usage=Usage(input_tokens=11)),
            ScriptedTurn.text("never", delay=5),
        ]
    )

    async def slow_block(inp: GuardrailInput) -> GuardrailVerdict:
        await asyncio.sleep(0.1)
        return GuardrailVerdict.block("off_topic", REPLY)

    agent = Agent(
        provider,
        model="test-model",
        tools=[small],
        context_budget=2_000,
        input_guardrails=[slow_block],
        guardrail_mode="parallel",
    )
    conv = Conversation(chat_history(10))
    before = list(conv.messages)
    result = await asyncio.wait_for(agent.run(PROMPT, conversation=conv), timeout=2)
    # The summary call ran (it never sees the new prompt) and its usage counts,
    # but the blocked run leaves no summary state on the conversation.
    assert provider.call_count == 2
    assert all(PROMPT not in m.text for m in provider.requests[0].messages)
    assert result.text == REPLY
    assert result.usage == Usage(input_tokens=11)
    assert conv.messages == before
    assert conv._compaction is None


async def test_parallel_tool_waits_for_guard() -> None:
    called = {"tool": False}

    @tool
    def probe() -> str:
        """Record that the tool ran."""
        called["tool"] = True
        return "ran"

    provider = FakeProvider([ScriptedTurn.tool_use(tool_call("probe")), ScriptedTurn.text("x")])

    async def slow_block(inp: GuardrailInput) -> GuardrailVerdict:
        await asyncio.sleep(0.1)
        return GuardrailVerdict.block("unsafe", REPLY)

    agent = Agent(
        provider, model="m", tools=[probe], input_guardrails=[slow_block], guardrail_mode="parallel"
    )
    events = await asyncio.wait_for(_collect(agent, PROMPT), timeout=2)
    assert called["tool"] is False
    assert provider.call_count == 1
    assert events[0] == TextDelta(REPLY)
    assert isinstance(events[-1], RunEnd) and events[-1].result.text == REPLY


async def test_parallel_block_emits_only_placeholder() -> None:
    provider = FakeProvider(
        ScriptedTurn.text("LEAKED", chunks=["LEA", "KED"], thinking="secret thoughts")
    )

    async def slow_block(inp: GuardrailInput) -> GuardrailVerdict:
        await asyncio.sleep(0.05)
        return GuardrailVerdict.block("off_topic", REPLY)

    agent = Agent(provider, model="m", input_guardrails=[slow_block], guardrail_mode="parallel")
    events = await asyncio.wait_for(_collect(agent, PROMPT), timeout=2)
    assert provider.call_count == 1
    assert len(events) == 2
    assert events[0] == TextDelta(REPLY)
    assert isinstance(events[1], RunEnd)
    assert events[1].result.text == REPLY
    assert events[1].result.messages == []


async def test_parallel_block_with_output_type_raises() -> None:
    provider = FakeProvider(ScriptedTurn.text('{"value": "x"}'))
    conv = Conversation()

    async def slow_block(inp: GuardrailInput) -> GuardrailVerdict:
        await asyncio.sleep(0.05)
        return GuardrailVerdict.block("off_topic", REPLY)

    agent = Agent(provider, model="m", input_guardrails=[slow_block], guardrail_mode="parallel")
    with pytest.raises(GuardrailTripped):
        await agent.run(PROMPT, conversation=conv, output_type=Answer)
    assert conv.messages == []


async def test_parallel_timeout_before_verdict_rolls_back_prompt() -> None:
    async def slow(inp: GuardrailInput) -> GuardrailVerdict:
        await asyncio.sleep(5)
        return GuardrailVerdict.allow()  # pragma: no cover

    conv = Conversation([Message.user("earlier"), Message.assistant_text("ok")])
    before = list(conv.messages)
    agent = Agent(
        FakeProvider(ScriptedTurn.text("fast answer")),
        model="m",
        input_guardrails=[slow],
        guardrail_mode="parallel",
        run_timeout=0.2,
    )
    with pytest.raises(RunTimeoutError) as info:
        await asyncio.wait_for(agent.run(PROMPT, conversation=conv), timeout=2)
    assert conv.messages == before
    assert info.value.messages == before
    assert conv._compaction is None


async def test_parallel_session_block_records_blocked(db: Path) -> None:
    provider = FakeProvider([ScriptedTurn.text("hi there"), ScriptedTurn.text("never", delay=5)])

    async def guard(inp: GuardrailInput) -> GuardrailVerdict:
        await asyncio.sleep(0.05)
        if "SECRETPROMPT" in inp.message.text:
            return GuardrailVerdict.block("off_topic", REPLY, usage=Usage(input_tokens=4))
        return GuardrailVerdict.allow()

    agent = Agent(provider, model="m", input_guardrails=[guard], guardrail_mode="parallel")
    first = await agent.run("hello", session_id="p1")
    assert first.text == "hi there"
    stored = _rows(db, "SELECT seq, role FROM logpose_messages ORDER BY seq")
    result = await asyncio.wait_for(agent.run(PROMPT, session_id="p1"), timeout=2)
    assert result.text == REPLY
    assert _rows(db, "SELECT seq, role FROM logpose_messages ORDER BY seq") == stored
    rows = _rows(db, "SELECT outcome, iterations, input_tokens FROM logpose_runs")
    assert ("blocked", 0, 4) in rows
    await agent.aclose()


async def test_parallel_provider_error_waits_for_block() -> None:
    from logpose.errors import ProviderError

    provider = FakeProvider(ScriptedTurn.failure(ProviderError("boom")))

    async def slow_block(inp: GuardrailInput) -> GuardrailVerdict:
        await asyncio.sleep(0.05)
        return GuardrailVerdict.block("off_topic", REPLY)

    agent = Agent(provider, model="m", input_guardrails=[slow_block], guardrail_mode="parallel")
    result = await asyncio.wait_for(agent.run(PROMPT), timeout=2)
    assert result.text == REPLY


def test_parallel_mode_through_sync_stream() -> None:
    from logpose.sync import close_sync, stream_sync

    async def slow_allow(inp: GuardrailInput) -> GuardrailVerdict:
        await asyncio.sleep(0.05)
        return GuardrailVerdict.allow()

    async def slow_block(inp: GuardrailInput) -> GuardrailVerdict:
        await asyncio.sleep(0.05)
        return GuardrailVerdict.block("off_topic", REPLY)

    provider = FakeProvider(ScriptedTurn.text("abc", chunks=["a", "b", "c"]))
    agent = Agent(provider, model="m", input_guardrails=[slow_allow], guardrail_mode="parallel")
    events = list(stream_sync(agent, PROMPT))
    assert [e for e in events if isinstance(e, TextDelta)] == [TextDelta(x) for x in "abc"]
    close_sync(agent)

    blocked = Agent(
        FakeProvider(ScriptedTurn.text("LEAK", delay=5)),
        model="m",
        input_guardrails=[slow_block],
        guardrail_mode="parallel",
    )
    events = list(stream_sync(blocked, PROMPT))
    assert events[0] == TextDelta(REPLY) and len(events) == 2
    close_sync(blocked)


# --- Fix round 1


async def test_parallel_sync_block_never_calls_model() -> None:
    results = []
    for mode in ("blocking", "parallel"):
        provider = FakeProvider(ScriptedTurn.text("never"))
        conv = Conversation([Message.user("earlier"), Message.assistant_text("ok")])
        agent = Agent(
            provider,
            model="m",
            input_guardrails=[off_topic_guard, pattern_guard(max_chars=10, reply="too long")],
            guardrail_mode=mode,  # type: ignore[arg-type]
        )
        events = await _collect(agent, "x" * 50, conversation=conv)
        assert provider.call_count == 0, mode
        assert conv._compaction is None
        assert events[0] == TextDelta("too long")
        results.append(events[-1].result)
    assert results[0] == results[1]


class _ProviderDied(BaseException):
    pass


async def test_parallel_provider_base_exception_does_not_hang() -> None:
    provider = FakeProvider(ScriptedTurn.failure(_ProviderDied()))

    async def slow_allow(inp: GuardrailInput) -> GuardrailVerdict:
        await asyncio.sleep(0.05)
        return GuardrailVerdict.allow()

    agent = Agent(provider, model="m", input_guardrails=[slow_allow], guardrail_mode="parallel")
    with pytest.raises(_ProviderDied):
        await asyncio.wait_for(agent.run(PROMPT), timeout=2)


async def test_parallel_exit_before_verdict_reports_zero_iterations(db: Path) -> None:
    async def slow(inp: GuardrailInput) -> GuardrailVerdict:
        await asyncio.sleep(5)
        return GuardrailVerdict.allow()  # pragma: no cover

    agent = Agent(
        FakeProvider(ScriptedTurn.text("fast answer")),
        model="m",
        input_guardrails=[slow],
        guardrail_mode="parallel",
        run_timeout=0.2,
    )
    with pytest.raises(RunTimeoutError):
        await asyncio.wait_for(agent.run(PROMPT, session_id="z1"), timeout=2)

    cancelled = Agent(
        FakeProvider(ScriptedTurn.text("fast answer")),
        model="m",
        input_guardrails=[slow],
        guardrail_mode="parallel",
    )
    task = asyncio.ensure_future(cancelled.run(PROMPT, session_id="z2"))
    await asyncio.sleep(0.1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    rows = _rows(db, "SELECT session_id, outcome, iterations FROM logpose_runs ORDER BY session_id")
    assert rows == [("z1", "run_timeout", 0), ("z2", "cancelled", 0)]
    await agent.aclose()
    await cancelled.aclose()


@pytest.mark.parametrize("bad", ["scores", "usage"])
async def test_malformed_verdict_blocks_as_error(db: Path, bad: str) -> None:
    async def guard(inp: GuardrailInput) -> GuardrailVerdict:
        if bad == "scores":
            return GuardrailVerdict(allowed=True, scores={"x": object()})  # type: ignore[dict-item]
        return GuardrailVerdict(allowed=True, usage="nope")  # type: ignore[arg-type]

    provider = FakeProvider(ScriptedTurn.text("never"))
    agent = Agent(provider, model="m", input_guardrails=[guard])
    result = await agent.run(PROMPT, session_id="m1")
    assert result.text == ERROR_REPLY
    assert result.guardrail is not None and result.guardrail.category == "error"
    assert result.guardrail.reason in ("guard raised TypeError", "guard raised ValueError")
    assert "SECRETPROMPT" not in (result.guardrail.reason or "")
    assert provider.call_count == 0
    assert _rows(db, "SELECT outcome FROM logpose_runs") == [("blocked",)]
    await agent.aclose()


async def test_fail_open_event_is_distinguishable() -> None:
    async def guard(inp: GuardrailInput) -> GuardrailVerdict:
        return GuardrailVerdict(allowed=True, category="fail_open", reason="timeout")

    events: list[RuntimeEvent] = []
    agent = Agent(
        FakeProvider(ScriptedTurn.text("ok")),
        model="m",
        input_guardrails=[guard],
        observers=[events.append],
    )
    result = await agent.run(PROMPT)
    assert result.text == "ok"
    (event,) = [e for e in events if e.name == "guardrail.decided"]
    assert event.guard_allowed is True and event.guard_category == "fail_open"


@pytest.mark.parametrize("mode", ["blocking", "parallel"])
async def test_cancel_mid_guard_releases_session(db: Path, mode: str) -> None:
    started = asyncio.Event()
    hang = True

    async def guard(inp: GuardrailInput) -> GuardrailVerdict:
        if hang:
            started.set()
            await asyncio.sleep(30)
        return GuardrailVerdict.allow()

    provider = FakeProvider([ScriptedTurn.text("first", delay=0.01), ScriptedTurn.text("again")])
    agent = Agent(provider, model="m", input_guardrails=[guard], guardrail_mode=mode)  # type: ignore[arg-type]
    task = asyncio.ensure_future(agent.run(PROMPT, session_id="c1"))
    await asyncio.wait_for(started.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert _rows(db, "SELECT outcome FROM logpose_runs") == [("cancelled",)]
    assert _rows(db, "SELECT seq FROM logpose_messages") == []
    hang = False
    result = await asyncio.wait_for(agent.run("hello", session_id="c1"), 2)
    assert result.text in ("first", "again")
    await agent.aclose()
