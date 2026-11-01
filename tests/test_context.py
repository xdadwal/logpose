"""Prepared model context never replaces the retained transcript."""

import asyncio

import pytest

from logpose import Agent, Conversation, Message, TextBlock
from tests.fake_provider import FakeProvider, ScriptedTurn


@pytest.mark.asyncio
async def test_async_policy_changes_request_without_mutating_canonical_history():
    from logpose.context import PreparedContext

    async def policy(context):
        context.messages[0].content[0].text = "selected"
        return PreparedContext(context.messages, context.system, {"policy": "test"})

    history = Conversation([Message(role="user", content=[TextBlock(text="original")])])
    provider = FakeProvider([ScriptedTurn.text("answer")])
    result = await Agent(provider, context_policy=policy).run(conversation=history)
    assert provider.requests[0].messages[0].content[0].text == "selected"
    assert history.messages[0].content[0].text == "original"
    assert result.messages[0].content[0].text == "original"


@pytest.mark.asyncio
async def test_preparation_failure_never_requests_provider():
    from logpose.context import ContextPreparationError

    def broken(context):
        raise ValueError("cannot prepare")

    provider = FakeProvider([ScriptedTurn.text("must not run")])
    with pytest.raises(ContextPreparationError):
        await Agent(provider, context_policy=broken).run("request")
    assert provider.call_count == 0


@pytest.mark.asyncio
async def test_sync_policy_runs_each_iteration_and_preserves_opaque_blocks():
    from logpose import RawBlock, ThinkingBlock, tool
    from logpose.context import PreparedContext
    from tests.fake_provider import tool_call

    @tool
    def inspect() -> str:
        """Return evidence."""
        return "evidence"

    seen = []

    def policy(context):
        seen.append(context)
        return PreparedContext(
            context.messages, context.system, {"messages": len(context.messages)}
        )

    raw = RawBlock(data={"type": "reasoning", "encrypted_content": "opaque"})
    thinking = ThinkingBlock(thinking="reason", signature="signature")
    history = Conversation([Message(role="assistant", content=[raw, thinking])])
    provider = FakeProvider(
        [ScriptedTurn.tool_use(tool_call("inspect", {})), ScriptedTurn.text("done")]
    )
    agent = Agent(
        provider, tools=[inspect], context_policy=policy, system="instructions", max_tokens=42
    )
    await agent.run("request", conversation=history)
    assert len(seen) == 2
    assert provider.requests[1].messages[0].content == [raw, thinking]
    assert provider.requests[1].max_tokens == 42
    assert provider.requests[1].system == "instructions"
    assert provider.requests[1].tools[0].name == "inspect"
    assert agent.last_context_diagnostics["messages"] == len(seen[1].messages)


@pytest.mark.asyncio
async def test_invalid_policy_return_stops_before_provider():
    from logpose.context import ContextPreparationError

    provider = FakeProvider([ScriptedTurn.text("must not run")])
    with pytest.raises(ContextPreparationError):
        await Agent(provider, context_policy=lambda context: None).run("request")
    assert provider.call_count == 0


@pytest.mark.asyncio
async def test_cancellation_during_preparation_propagates_without_inference():
    entered = asyncio.Event()

    async def policy(context):
        entered.set()
        await asyncio.Event().wait()

    provider = FakeProvider([ScriptedTurn.text("must not run")])
    task = asyncio.create_task(Agent(provider, context_policy=policy).run("request"))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert provider.call_count == 0
