"""Persistence callbacks fail closed and never repeat tool effects."""

import pytest

from logpose import Agent, ToolUseBlock, tool
from logpose.runtime import RuntimeContext
from tests.fake_provider import FakeProvider


@pytest.mark.asyncio
@pytest.mark.parametrize("phase,expected_calls", [("intent", 0), ("outcome", 1)])
async def test_persistence_failure_aborts_without_repeating_handler(phase, expected_calls):
    from logpose.lifecycle import ToolObservationError

    calls = []

    @tool
    def effect() -> str:
        """Record one observed effect."""
        calls.append(1)
        return "done"

    async def persist(event):
        if event.phase == phase:
            raise OSError("storage unavailable")

    agent = Agent(FakeProvider(), tools=[effect], tool_observer=persist)
    context = RuntimeContext(run_id="run", provider="fake", model="fake")
    with pytest.raises(ToolObservationError) as error:
        await agent._execute([ToolUseBlock(id="call", name="effect", input={})], context)
    assert error.value.phase == phase
    assert len(calls) == expected_calls


@pytest.mark.asyncio
async def test_live_loop_stops_before_next_model_request_on_outcome_failure():
    from logpose import ToolObservationError
    from tests.fake_provider import ScriptedTurn, tool_call

    @tool
    def effect() -> str:
        """Produce one result."""
        return "done"

    async def persist(event):
        if event.phase == "outcome":
            raise OSError("unavailable")

    provider = FakeProvider(
        [ScriptedTurn.tool_use(tool_call("effect", {})), ScriptedTurn.text("should not run")]
    )
    with pytest.raises(ToolObservationError):
        await Agent(provider, tools=[effect], tool_observer=persist).run("execute")
    assert provider.call_count == 1


@pytest.mark.asyncio
async def test_denied_tool_is_not_journaled():
    from logpose import ToolGateResult

    observations = []

    @tool
    def effect() -> str:
        """Must not execute."""
        raise AssertionError("denied tool ran")

    async def persist(event):
        observations.append(event)

    async def deny(call):
        return ToolGateResult("denied")

    agent = Agent(FakeProvider(), tools=[effect], on_tool_call=deny, tool_observer=persist)
    await agent._execute(
        [ToolUseBlock(id="call", name="effect", input={})],
        RuntimeContext(run_id="run", provider="fake", model="fake"),
    )
    assert observations == []


@pytest.mark.asyncio
async def test_observer_mutation_cannot_change_execution_or_result():
    @tool
    def echo(value: str) -> str:
        """Return its input.

        Args:
            value: Text to return.
        """
        return value

    async def persist(event):
        event.call.input["value"] = "changed"
        if event.result is not None:
            event.result.content = "changed"

    agent = Agent(FakeProvider(), tools=[echo], tool_observer=persist)
    call = ToolUseBlock(id="call", name="echo", input={"value": "original"})
    result = await agent._execute(
        [call], RuntimeContext(run_id="run", provider="fake", model="fake")
    )
    assert result[0].content == "original"
    assert call.input == {"value": "original"}


@pytest.mark.asyncio
async def test_batch_persistence_failure_drains_synchronous_sibling():
    import asyncio
    import threading

    from logpose import ToolObservationError

    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    observed = []

    @tool
    def slow() -> str:
        """Finish an already approved synchronous effect."""
        started.set()
        assert release.wait(5)
        finished.set()
        return "finished"

    @tool
    def other() -> str:
        """An independent tool."""
        return "other"

    async def persist(event):
        if event.call.id == "bad" and event.phase == "intent":
            assert await asyncio.to_thread(started.wait, 5)
            asyncio.get_running_loop().call_later(0.05, release.set)
            raise OSError("unavailable")
        observed.append((event.call.id, event.phase))

    agent = Agent(FakeProvider(), tools=[slow, other], tool_observer=persist)
    with pytest.raises(ToolObservationError):
        await agent._execute(
            [
                ToolUseBlock(id="slow", name="slow", input={}),
                ToolUseBlock(id="bad", name="other", input={}),
            ],
            RuntimeContext(run_id="run", provider="fake", model="fake"),
        )
    assert finished.is_set()
    assert ("slow", "outcome") in observed
