"""Tests for the agentic loop.

Everything runs against :class:`tests.fake_provider.FakeProvider`: no network, no
credentials, and full control over what the "model" says on every turn. The ten
loop semantics the agent must guarantee are grouped into the sections below.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator
from typing import Any

import pytest

from logpose.agent import (
    DEFAULT_MAX_TOKENS,
    EMPTY_TOOL_RESULT,
    Agent,
    Conversation,
    ToolGateResult,
)
from logpose.errors import LogposeError, MaxIterationsError, ProviderError
from logpose.events import (
    Event,
    RunEnd,
    TextDelta,
    ThinkingDelta,
    ToolCall,
    ToolResult,
    TurnEnd,
)
from logpose.messages import (
    Message,
    RedactedThinkingBlock,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
)
from logpose.providers.base import CompletionDone, Provider, ProviderTextDelta
from logpose.retry import RetryPolicy
from logpose.tools import ToolDef, tool
from tests.fake_provider import FakeProvider, ScriptedTurn, tool_call

SIGNATURE = "ErUBCkYIBRgCIkDdQ7/vNdkKgFQ0oX+signature+bytes+must+survive=="
REDACTED_DATA = "EroBCkYIBRgCKkB0aGlzLWlzLWVuY3J5cHRlZA=="


# ---------------------------------------------------------------------------
# tools used by the tests
# ---------------------------------------------------------------------------


@tool
def add(a: int, b: int) -> int:
    """Add two numbers.

    Args:
        a: Left operand.
        b: Right operand.
    """
    return a + b


@tool
def shout(text: str) -> str:
    """Uppercase some text.

    Args:
        text: The text to shout.
    """
    return text.upper()


@tool
def boom() -> str:
    """Always fails."""
    raise RuntimeError("kaboom")


@tool
def blank() -> str:
    """Return an empty string."""
    return ""


class Detonation(BaseException):
    """A failure that is not an ``Exception``, so no tool machinery catches it."""


class Tracker:
    """Records how tool executions overlapped in time."""

    def __init__(self) -> None:
        self.active = 0
        self.peak = 0
        self.started: list[str] = []
        self.finished: list[str] = []
        self.cancelled: list[str] = []

    def enter(self, name: str) -> None:
        self.active += 1
        self.peak = max(self.peak, self.active)
        self.started.append(name)

    def exit(self, name: str) -> None:
        self.active -= 1


def sleeping_tool(name: str, tracker: Tracker, delay: float = 0.05) -> ToolDef:
    """Build an async tool that sleeps, recording overlap and cancellation.

    Args:
        name: Tool name.
        tracker: Shared overlap recorder.
        delay: Seconds to sleep.

    Returns:
        The tool definition.
    """

    async def handler() -> str:
        """Sleep for a while."""
        tracker.enter(name)
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            tracker.cancelled.append(name)
            raise
        finally:
            tracker.exit(name)
        tracker.finished.append(name)
        return f"{name} done"

    handler.__name__ = name
    return tool(handler)


async def collect(agent: Agent, *args: Any, **kwargs: Any) -> list[Event]:
    """Drain ``agent.stream`` into a list.

    Args:
        agent: The agent to run.
        *args: Positional arguments for :meth:`Agent.stream`.
        **kwargs: Keyword arguments for :meth:`Agent.stream`.

    Returns:
        Every event the run emitted, in order.
    """
    return [event async for event in agent.stream(*args, **kwargs)]


# ---------------------------------------------------------------------------
# 1. request construction + delta re-emission
# ---------------------------------------------------------------------------


def test_fake_provider_satisfies_the_provider_protocol() -> None:
    assert isinstance(FakeProvider(), Provider)


async def test_request_carries_history_system_and_tool_specs() -> None:
    provider = FakeProvider([ScriptedTurn.text("hi")])
    agent = Agent(provider, system="Be terse.", tools=[add, shout], max_tokens=512)

    await agent.run("hello")

    request = provider.last_request
    assert request.system == "Be terse."
    assert request.max_tokens == 512
    assert [message.role for message in request.messages] == ["user"]
    assert request.messages[0].text == "hello"
    assert [spec.name for spec in request.tools] == ["add", "shout"]
    assert request.tools[0].input_schema == add.input_schema
    # ToolSpec deliberately carries no handler: providers advertise, never execute.
    assert not hasattr(request.tools[0], "handler")


async def test_a_message_prompt_is_sent_as_is() -> None:
    provider = FakeProvider([ScriptedTurn.text("hi")])
    agent = Agent(provider)
    prompt = Message(role="user", content=[TextBlock(text="already a message")])

    await agent.run(prompt)

    assert provider.last_request.messages[0] is prompt


async def test_model_override_is_sent_and_defaults_to_the_provider() -> None:
    provider = FakeProvider([ScriptedTurn.text("hi")])
    await Agent(provider, model="claude-opus-5").run("hello")
    assert provider.last_request.model == "claude-opus-5"

    defaulting = FakeProvider([ScriptedTurn.text("hi")], model_default="fake-1")
    await Agent(defaulting).run("hello")
    assert defaulting.last_request.model == "fake-1"


def test_model_default_is_part_of_the_provider_protocol() -> None:
    # Regression: `model_default` was read off the provider by getattr but was
    # not declared on the protocol, so a provider written to the documented
    # contract passed isinstance() and then silently received model="".
    class NoDefault:
        name = "nodefault"

        async def stream(self, req: object) -> AsyncIterator[object]:  # pragma: no cover
            yield object()

    assert not isinstance(NoDefault(), Provider)
    with pytest.raises(LogposeError, match="model_default"):
        Agent(NoDefault())  # type: ignore[arg-type]


async def test_an_empty_model_default_fails_loudly_before_the_request() -> None:
    # A provider registered by name skips the isinstance check, so the empty
    # model has to be caught when the request is built rather than sent as ""
    # and returned as an opaque provider 400.
    provider = FakeProvider([ScriptedTurn.text("hi")], model_default="")

    with pytest.raises(LogposeError, match="model_default"):
        await Agent(provider).run("hello")

    assert provider.call_count == 0


async def test_extra_is_forwarded_to_every_request() -> None:
    # Regression: CompletionRequest.extra is the documented provider escape
    # hatch and is consumed by AnthropicProvider, but Agent offered no way in.
    provider = FakeProvider([ScriptedTurn.tool_use(tool_call("add", {"a": 1, "b": 2}))
                             , ScriptedTurn.text("3")])
    extra = {"tool_choice": {"type": "any"}, "stop_sequences": ["</done>"]}
    agent = Agent(provider, tools=[add], extra=extra)

    await agent.run("add them")

    assert provider.call_count == 2
    assert all(request.extra == extra for request in provider.requests)
    # Copied on the way in and on the way out: mutating either side is inert.
    extra["tool_choice"] = {"type": "none"}
    assert provider.requests[0].extra["tool_choice"] == {"type": "any"}
    assert Agent(FakeProvider()).extra == {}


async def test_max_tokens_falls_back_to_the_providers_own_ceiling() -> None:
    # Regression: Agent.max_tokens always had a value, so `req.max_tokens or
    # self.max_tokens` in the provider never fell back and a provider
    # configured with max_tokens=2048 was silently billed 16000-token turns.
    capped = FakeProvider([ScriptedTurn.text("hi")])
    capped.max_tokens = 2048  # type: ignore[attr-defined]
    await Agent(capped).run("hello")
    assert capped.last_request.max_tokens == 2048

    # An explicit override still wins.
    overridden = FakeProvider([ScriptedTurn.text("hi")])
    overridden.max_tokens = 2048  # type: ignore[attr-defined]
    await Agent(overridden, max_tokens=99).run("hello")
    assert overridden.last_request.max_tokens == 99

    # A provider with no opinion gets the loop's documented default.
    bare = FakeProvider([ScriptedTurn.text("hi")])
    await Agent(bare).run("hello")
    assert bare.last_request.max_tokens == DEFAULT_MAX_TOKENS


async def test_deltas_are_re_emitted_as_public_events() -> None:
    provider = FakeProvider(
        [ScriptedTurn.text("hello world", chunks=["hello ", "world"], thinking="pondering")]
    )
    events = await collect(Agent(provider), "hi")

    assert isinstance(events[0], ThinkingDelta)
    assert events[0].text == "pondering"
    assert [event.text for event in events if isinstance(event, TextDelta)] == [
        "hello ",
        "world",
    ]


def test_stream_returns_an_async_iterator_not_a_coroutine() -> None:
    stream = Agent(FakeProvider([ScriptedTurn.text("hi")])).stream("hello")
    assert hasattr(stream, "__anext__")
    assert not asyncio.iscoroutine(stream)


async def test_request_messages_are_snapshots_not_the_live_history() -> None:
    provider = FakeProvider(
        [
            ScriptedTurn.tool_use(tool_call("add", {"a": 1, "b": 1}, id="t1")),
            ScriptedTurn.text("2"),
        ]
    )
    conversation = Conversation()
    await Agent(provider, tools=[add]).run("1+1?", conversation=conversation)

    # The first request must still show the history as it was on turn one.
    assert len(provider.requests[0].messages) == 1
    assert len(provider.requests[1].messages) == 3
    assert provider.requests[0].messages is not conversation.messages


# ---------------------------------------------------------------------------
# 2. verbatim assistant messages, TurnEnd, usage
# ---------------------------------------------------------------------------


async def test_assistant_message_is_appended_verbatim() -> None:
    scripted = ScriptedTurn(
        message=Message(
            role="assistant",
            content=[
                ThinkingBlock(thinking="hmm", signature=SIGNATURE),
                RedactedThinkingBlock(data=REDACTED_DATA),
                TextBlock(text="the answer"),
            ],
        ),
        usage=Usage(input_tokens=3, output_tokens=4),
    )
    provider = FakeProvider([scripted])
    result = await Agent(provider).run("go")

    appended = result.messages[-1]
    assert appended is scripted.message
    assert appended.model_dump() == scripted.message.model_dump()
    assert appended.content[0].signature == SIGNATURE
    assert appended.content[1].data == REDACTED_DATA


async def test_thinking_blocks_are_echoed_back_byte_identically() -> None:
    first = ScriptedTurn.tool_use(
        tool_call("add", {"a": 1, "b": 1}, id="t1"),
        thinking="I should add these",
        signature=SIGNATURE,
    )
    provider = FakeProvider([first, ScriptedTurn.text("2")])

    await Agent(provider, tools=[add]).run("1+1?")

    assert provider.call_count == 2
    echoed = provider.requests[1].messages[1]
    assert echoed is first.message  # not rebuilt, not filtered
    thinking = echoed.content[0]
    assert isinstance(thinking, ThinkingBlock)
    assert thinking.thinking == "I should add these"
    assert thinking.signature == SIGNATURE
    assert echoed.model_dump() == first.message.model_dump()


async def test_turn_end_carries_the_stop_reason_and_that_turns_usage() -> None:
    provider = FakeProvider(
        [
            ScriptedTurn.tool_use(
                tool_call("add", {"a": 1, "b": 1}, id="t1"),
                usage=Usage(input_tokens=10, output_tokens=2),
            ),
            ScriptedTurn.text("2", usage=Usage(input_tokens=20, output_tokens=3)),
        ]
    )
    events = await collect(Agent(provider, tools=[add]), "1+1?")
    turn_ends = [event for event in events if isinstance(event, TurnEnd)]

    assert [turn.stop_reason for turn in turn_ends] == ["tool_use", "end_turn"]
    assert turn_ends[0].usage == Usage(input_tokens=10, output_tokens=2)
    assert turn_ends[1].usage == Usage(input_tokens=20, output_tokens=3)


# ---------------------------------------------------------------------------
# 3. parallel tool execution, one user message
# ---------------------------------------------------------------------------


async def test_two_tool_calls_land_in_exactly_one_user_message_in_order() -> None:
    provider = FakeProvider(
        [
            ScriptedTurn.tool_use(
                tool_call("add", {"a": 2, "b": 3}, id="t1"),
                tool_call("shout", {"text": "hey"}, id="t2"),
                text="calling both",
            ),
            ScriptedTurn.text("5 and HEY"),
        ]
    )
    result = await Agent(provider, tools=[add, shout]).run("do both")

    assert [message.role for message in result.messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    results_message = result.messages[2]
    assert all(isinstance(block, ToolResultBlock) for block in results_message.content)
    assert [block.tool_use_id for block in results_message.content] == ["t1", "t2"]
    assert [block.content for block in results_message.content] == ["5", "HEY"]

    # The single-message invariant, asserted against what the provider was sent.
    sent = provider.requests[1].messages
    carriers = [
        message
        for message in sent
        if any(isinstance(block, ToolResultBlock) for block in message.content)
    ]
    assert len(carriers) == 1
    assert len(carriers[0].content) == 2


async def test_results_are_ordered_by_request_not_by_completion() -> None:
    tracker = Tracker()
    slow = sleeping_tool("slow", tracker, delay=0.04)
    fast = sleeping_tool("fast", tracker, delay=0.0)
    provider = FakeProvider(
        [
            ScriptedTurn.tool_use(
                tool_call("slow", id="t1"),
                tool_call("fast", id="t2"),
            ),
            ScriptedTurn.text("done"),
        ]
    )
    result = await Agent(provider, tools=[slow, fast]).run("go")

    assert tracker.finished == ["fast", "slow"]  # completion order really did differ
    results_message = result.messages[2]
    assert [block.tool_use_id for block in results_message.content] == ["t1", "t2"]
    assert [block.content for block in results_message.content] == ["slow done", "fast done"]


async def test_tools_actually_run_concurrently() -> None:
    tracker = Tracker()
    tools = [sleeping_tool("alpha", tracker), sleeping_tool("beta", tracker)]
    provider = FakeProvider(
        [
            ScriptedTurn.tool_use(tool_call("alpha", id="t1"), tool_call("beta", id="t2")),
            ScriptedTurn.text("done"),
        ]
    )

    start = time.perf_counter()
    await Agent(provider, tools=tools).run("go")
    elapsed = time.perf_counter() - start

    assert tracker.peak == 2  # both were in flight at the same moment
    assert elapsed < 0.09  # 2 x 50ms run serially would take at least 100ms


async def test_events_are_emitted_in_loop_order() -> None:
    provider = FakeProvider(
        [
            ScriptedTurn.tool_use(
                tool_call("add", {"a": 1, "b": 1}, id="t1"),
                tool_call("shout", {"text": "x"}, id="t2"),
                thinking="think",
                text="calling",
            ),
            ScriptedTurn.text("done"),
        ]
    )
    events = await collect(Agent(provider, tools=[add, shout]), "go")

    assert [type(event) for event in events] == [
        ThinkingDelta,
        TextDelta,
        TurnEnd,
        ToolCall,
        ToolCall,
        ToolResult,
        ToolResult,
        TextDelta,
        TurnEnd,
        RunEnd,
    ]
    calls = [event for event in events if isinstance(event, ToolCall)]
    assert [(call.id, call.name, call.input) for call in calls] == [
        ("t1", "add", {"a": 1, "b": 1}),
        ("t2", "shout", {"text": "x"}),
    ]
    results = [event for event in events if isinstance(event, ToolResult)]
    assert [(result.id, result.name, result.content) for result in results] == [
        ("t1", "add", "2"),
        ("t2", "shout", "X"),
    ]
    assert all(result.is_error is False for result in results)


# ---------------------------------------------------------------------------
# 4. tool failures never crash the loop
# ---------------------------------------------------------------------------


async def test_failing_tool_becomes_an_error_result_and_the_run_continues() -> None:
    provider = FakeProvider(
        [
            ScriptedTurn.tool_use(tool_call("boom", id="t1")),
            ScriptedTurn.text("I could not do that, sorry."),
        ]
    )
    events = await collect(Agent(provider, tools=[boom]), "explode")
    result = events[-1].result

    error_events = [event for event in events if isinstance(event, ToolResult)]
    assert len(error_events) == 1
    assert error_events[0].is_error is True
    assert "kaboom" in error_events[0].content

    block = result.messages[2].content[0]
    assert isinstance(block, ToolResultBlock)
    assert block.is_error is True
    assert block.tool_use_id == "t1"
    assert "RuntimeError" in block.content
    assert result.text == "I could not do that, sorry."
    assert result.stop_reason == "end_turn"
    assert result.iterations == 2


async def test_invalid_tool_arguments_become_an_error_result() -> None:
    provider = FakeProvider(
        [
            ScriptedTurn.tool_use(tool_call("add", {"a": "not a number", "b": 1}, id="t1")),
            ScriptedTurn.text("recovered"),
        ]
    )
    result = await Agent(provider, tools=[add]).run("go")

    block = result.messages[2].content[0]
    assert block.is_error is True
    assert "a:" in block.content
    assert result.text == "recovered"


async def test_unknown_tool_name_reports_the_available_tools() -> None:
    provider = FakeProvider(
        [
            ScriptedTurn.tool_use(tool_call("teleport", {"to": "mars"}, id="t1")),
            ScriptedTurn.text("ok, no teleporting"),
        ]
    )
    events = await collect(Agent(provider, tools=[add, shout]), "teleport me")
    result = events[-1].result

    block = result.messages[2].content[0]
    assert block.is_error is True
    assert block.tool_use_id == "t1"
    assert "teleport" in block.content
    assert "add, shout" in block.content
    assert any(isinstance(event, ToolResult) and event.is_error for event in events)
    assert result.text == "ok, no teleporting"


async def test_unknown_tool_when_no_tools_are_configured() -> None:
    provider = FakeProvider(
        [
            ScriptedTurn.tool_use(tool_call("anything", id="t1")),
            ScriptedTurn.text("nothing to call"),
        ]
    )
    result = await Agent(provider).run("go")

    block = result.messages[2].content[0]
    assert block.is_error is True
    assert "no tools" in block.content


async def test_empty_tool_output_is_replaced_with_a_placeholder() -> None:
    provider = FakeProvider(
        [ScriptedTurn.tool_use(tool_call("blank", id="t1")), ScriptedTurn.text("ok")]
    )
    result = await Agent(provider, tools=[blank]).run("go")

    block = result.messages[2].content[0]
    assert block.content == EMPTY_TOOL_RESULT
    assert block.is_error is False


async def test_one_failing_tool_does_not_disturb_its_siblings() -> None:
    provider = FakeProvider(
        [
            ScriptedTurn.tool_use(
                tool_call("boom", id="t1"),
                tool_call("add", {"a": 20, "b": 2}, id="t2"),
            ),
            ScriptedTurn.text("22"),
        ]
    )
    result = await Agent(provider, tools=[boom, add]).run("go")

    blocks = result.messages[2].content
    assert [block.is_error for block in blocks] == [True, False]
    assert blocks[1].content == "22"


# ---------------------------------------------------------------------------
# 5. pause_turn
# ---------------------------------------------------------------------------


async def test_pause_turn_resumes_transparently() -> None:
    provider = FakeProvider(
        [
            ScriptedTurn.text("thinking out lou", stop_reason="pause_turn"),
            ScriptedTurn.text("d... done!"),
        ]
    )
    events = await collect(Agent(provider), "go")
    result = events[-1].result

    assert provider.call_count == 2
    # The paused assistant turn is part of the next request, unmodified.
    assert provider.requests[1].messages[1].text == "thinking out lou"
    assert [event.stop_reason for event in events if isinstance(event, TurnEnd)] == [
        "pause_turn",
        "end_turn",
    ]
    assert result.stop_reason == "end_turn"
    assert result.text == "d... done!"
    assert result.iterations == 2


async def test_pause_turn_can_be_followed_by_tool_use() -> None:
    provider = FakeProvider(
        [
            ScriptedTurn.text("hold on", stop_reason="pause_turn"),
            ScriptedTurn.tool_use(tool_call("add", {"a": 1, "b": 2}, id="t1")),
            ScriptedTurn.text("3"),
        ]
    )
    result = await Agent(provider, tools=[add]).run("go")

    assert result.iterations == 3
    assert [message.role for message in result.messages] == [
        "user",
        "assistant",
        "assistant",
        "user",
        "assistant",
    ]


# ---------------------------------------------------------------------------
# 6. terminal stop reasons
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stop_reason", ["end_turn", "max_tokens", "stop_sequence", "refusal"])
async def test_terminal_stop_reasons_end_the_run(stop_reason: str) -> None:
    provider = FakeProvider([ScriptedTurn.text("final", stop_reason=stop_reason)])  # type: ignore[arg-type]
    result = await Agent(provider).run("go")

    assert provider.call_count == 1
    assert result.stop_reason == stop_reason
    assert result.iterations == 1


async def test_refusal_is_surfaced_not_raised() -> None:
    provider = FakeProvider([ScriptedTurn.text("I can't help with that.", stop_reason="refusal")])
    result = await Agent(provider).run("do something disallowed")

    assert result.stop_reason == "refusal"
    assert result.text == "I can't help with that."


async def test_a_pre_output_refusal_never_enters_the_conversation() -> None:
    # Regression: a refusal that fires before any output is returned as HTTP 200
    # with an empty content array. Appending it verbatim left the Conversation
    # holding {"role": "assistant", "content": []}, which every provider rejects
    # — so the *next* run failed with an error naming a message the consumer
    # never wrote, and the conversation was permanently unusable.
    empty = ScriptedTurn(
        message=Message(role="assistant", content=[]),
        stop_reason="refusal",
    )
    provider = FakeProvider([empty, ScriptedTurn.text("sure, here you go")])
    agent = Agent(provider)
    conversation = Conversation()

    refused = await agent.run("something disallowed", conversation=conversation)

    assert refused.stop_reason == "refusal"
    assert refused.text == ""
    # The user turn is there; the unsendable assistant turn is not.
    assert [message.role for message in conversation] == ["user"]
    assert all(message.content for message in conversation)

    # The conversation still works.
    followup = await agent.run("ok, something else", conversation=conversation)

    assert followup.stop_reason == "end_turn"
    assert all(message.content for message in provider.last_request.messages)


async def test_an_empty_paused_turn_ends_the_run_instead_of_spinning() -> None:
    # With nothing appended there is nothing to resume from, so re-issuing would
    # be an identical request in a tight loop until the iteration cap.
    empty = ScriptedTurn(
        message=Message(role="assistant", content=[]),
        stop_reason="pause_turn",
    )
    provider = FakeProvider([empty], repeat_last=True)

    result = await Agent(provider).run("go")

    assert provider.call_count == 1
    assert result.stop_reason == "pause_turn"


async def test_tool_use_without_tool_blocks_terminates_instead_of_spinning() -> None:
    provider = FakeProvider(
        [ScriptedTurn.text("I meant to call a tool", stop_reason="tool_use")],
        repeat_last=True,
    )
    result = await Agent(provider, tools=[add]).run("go")

    assert provider.call_count == 1
    assert result.stop_reason == "tool_use"
    assert result.iterations == 1


# ---------------------------------------------------------------------------
# 7. max_iterations
# ---------------------------------------------------------------------------


async def test_max_iterations_raises_with_the_partial_conversation_attached() -> None:
    provider = FakeProvider(
        [ScriptedTurn.tool_use(tool_call("add", {"a": 1, "b": 1}, id="t1"))],
        repeat_last=True,
    )
    agent = Agent(provider, tools=[add], max_iterations=3)

    with pytest.raises(MaxIterationsError) as excinfo:
        await agent.run("loop forever")

    error = excinfo.value
    assert error.max_iterations == 3
    assert provider.call_count == 3
    # user prompt + 3 x (assistant tool_use + user tool results)
    assert len(error.messages) == 7
    assert [message.role for message in error.messages[:3]] == ["user", "assistant", "user"]
    assert error.messages[-1].content[0].content == "2"


async def test_max_iterations_of_one_still_allows_a_single_turn() -> None:
    provider = FakeProvider([ScriptedTurn.text("done")])
    result = await Agent(provider, max_iterations=1).run("go")

    assert result.iterations == 1


async def test_max_iterations_partial_history_lands_in_the_conversation() -> None:
    provider = FakeProvider(
        [ScriptedTurn.tool_use(tool_call("add", {"a": 1, "b": 1}, id="t1"))],
        repeat_last=True,
    )
    conversation = Conversation()

    with pytest.raises(MaxIterationsError) as excinfo:
        await Agent(provider, tools=[add], max_iterations=2).run("go", conversation=conversation)

    assert conversation.messages == excinfo.value.messages


@pytest.mark.parametrize("bad", [0, -1])
def test_max_iterations_must_be_positive(bad: int) -> None:
    with pytest.raises(LogposeError):
        Agent(FakeProvider(), max_iterations=bad)


def test_max_tokens_must_be_positive() -> None:
    with pytest.raises(LogposeError):
        Agent(FakeProvider(), max_tokens=0)


# ---------------------------------------------------------------------------
# 8. RunEnd / RunResult
# ---------------------------------------------------------------------------


async def test_run_end_is_the_last_event_and_carries_the_whole_result() -> None:
    provider = FakeProvider(
        [
            ScriptedTurn.tool_use(
                tool_call("add", {"a": 1, "b": 1}, id="t1"),
                usage=Usage(input_tokens=1, output_tokens=1),
            ),
            ScriptedTurn.text("2", usage=Usage(input_tokens=2, output_tokens=2)),
        ]
    )
    events = await collect(Agent(provider, tools=[add]), "1+1?")

    assert isinstance(events[-1], RunEnd)
    assert not any(isinstance(event, RunEnd) for event in events[:-1])
    result = events[-1].result
    assert result.text == "2"
    assert result.iterations == 2
    assert result.stop_reason == "end_turn"
    assert result.usage == Usage(input_tokens=3, output_tokens=3)
    assert len(result.messages) == 4


async def test_usage_is_aggregated_across_three_turns() -> None:
    provider = FakeProvider(
        [
            ScriptedTurn.tool_use(
                tool_call("add", {"a": 1, "b": 1}, id="t1"),
                usage=Usage(input_tokens=10, output_tokens=1, cache_read_input_tokens=100),
            ),
            ScriptedTurn.tool_use(
                tool_call("add", {"a": 2, "b": 2}, id="t2"),
                usage=Usage(input_tokens=20, output_tokens=2, cache_creation_input_tokens=7),
            ),
            ScriptedTurn.text("done", usage=Usage(input_tokens=30, output_tokens=3)),
        ]
    )
    result = await Agent(provider, tools=[add]).run("go")

    assert result.iterations == 3
    assert result.usage == Usage(
        input_tokens=60,
        output_tokens=6,
        cache_read_input_tokens=100,
        cache_creation_input_tokens=7,
    )


async def test_run_returns_the_same_result_the_stream_ends_with() -> None:
    script = [
        ScriptedTurn.tool_use(tool_call("add", {"a": 1, "b": 1}, id="t1")),
        ScriptedTurn.text("2", usage=Usage(output_tokens=9)),
    ]
    streamed = await collect(Agent(FakeProvider(list(script)), tools=[add]), "1+1?")
    ran = await Agent(FakeProvider(list(script)), tools=[add]).run("1+1?")

    assert streamed[-1].result == ran


async def test_result_messages_are_a_snapshot_of_the_conversation() -> None:
    provider = FakeProvider([ScriptedTurn.text("one"), ScriptedTurn.text("two")])
    conversation = Conversation()
    agent = Agent(provider)

    first = await agent.run("a", conversation=conversation)
    assert first.messages is not conversation.messages
    assert len(first.messages) == 2

    await agent.run("b", conversation=conversation)
    assert len(first.messages) == 2  # not retroactively grown
    assert len(conversation.messages) == 4


# ---------------------------------------------------------------------------
# 9. cancellation safety
# ---------------------------------------------------------------------------


async def test_breaking_out_of_the_stream_closes_the_provider_stream() -> None:
    provider = FakeProvider([ScriptedTurn.text("hello", chunks=["he", "llo"])])
    stream = Agent(provider).stream("hi")

    async for event in stream:
        if isinstance(event, TextDelta):
            break
    await stream.aclose()  # type: ignore[attr-defined]

    assert provider.closed == 1
    assert provider.call_count == 1


async def test_cancelling_a_run_cancels_in_flight_tools() -> None:
    tracker = Tracker()
    tools = [sleeping_tool("alpha", tracker, delay=5), sleeping_tool("beta", tracker, delay=5)]
    provider = FakeProvider(
        [ScriptedTurn.tool_use(tool_call("alpha", id="t1"), tool_call("beta", id="t2"))],
        repeat_last=True,
    )
    task = asyncio.create_task(Agent(provider, tools=tools).run("go"))

    for _ in range(200):
        if tracker.active == 2:
            break
        await asyncio.sleep(0)
    assert tracker.active == 2, "tools never started"

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert sorted(tracker.cancelled) == ["alpha", "beta"]
    assert tracker.active == 0
    assert tracker.finished == []


async def test_a_timeout_around_run_cancels_in_flight_tools() -> None:
    tracker = Tracker()
    tools = [sleeping_tool("alpha", tracker, delay=5), sleeping_tool("beta", tracker, delay=5)]
    provider = FakeProvider(
        [ScriptedTurn.tool_use(tool_call("alpha", id="t1"), tool_call("beta", id="t2"))],
        repeat_last=True,
    )
    agent = Agent(provider, tools=tools)

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(agent.run("go"), timeout=0.05)

    assert sorted(tracker.cancelled) == ["alpha", "beta"]
    assert tracker.active == 0


async def test_a_tool_escaping_with_a_base_exception_cancels_its_siblings() -> None:
    tracker = Tracker()
    slow = sleeping_tool("slow", tracker, delay=5)

    async def detonate() -> str:
        """Fail in a way ToolDef.invoke does not catch."""
        raise Detonation("not an Exception")

    detonate.__name__ = "detonate"
    provider = FakeProvider(
        [ScriptedTurn.tool_use(tool_call("slow", id="t1"), tool_call("detonate", id="t2"))],
        repeat_last=True,
    )
    agent = Agent(provider, tools=[slow, tool(detonate)])

    with pytest.raises(Detonation):
        await agent.run("go")

    # gather() propagates the failure immediately; without the loop's finally the
    # sibling would keep running as an orphaned task.
    assert tracker.cancelled == ["slow"]
    assert tracker.active == 0
    assert tracker.finished == []


async def test_abandoning_a_stream_mid_tool_leaves_no_stray_tasks() -> None:
    tracker = Tracker()
    tools = [sleeping_tool("alpha", tracker, delay=5), sleeping_tool("beta", tracker, delay=5)]
    provider = FakeProvider(
        [ScriptedTurn.tool_use(tool_call("alpha", id="t1"), tool_call("beta", id="t2"))],
        repeat_last=True,
    )
    stream = Agent(provider, tools=tools).stream("go")

    async for event in stream:
        if isinstance(event, ToolCall) and event.id == "t2":
            break

    before = len(asyncio.all_tasks())
    await stream.aclose()  # type: ignore[attr-defined]

    assert len(asyncio.all_tasks()) <= before
    assert tracker.active == 0
    assert tracker.finished == []


# ---------------------------------------------------------------------------
# 10. Conversation
# ---------------------------------------------------------------------------


async def test_conversation_persists_across_runs_on_the_same_object() -> None:
    provider = FakeProvider(
        [ScriptedTurn.text("Hello Ada."), ScriptedTurn.text("Your name is Ada.")]
    )
    agent = Agent(provider)
    conversation = Conversation()

    await agent.run("My name is Ada.", conversation=conversation)
    result = await agent.run("What is my name?", conversation=conversation)

    assert [message.role for message in conversation.messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert [message.text for message in provider.requests[1].messages] == [
        "My name is Ada.",
        "Hello Ada.",
        "What is my name?",
    ]
    assert result.messages == conversation.messages
    assert conversation.text == "Your name is Ada."
    assert len(conversation) == 4


async def test_runs_without_a_conversation_are_independent() -> None:
    provider = FakeProvider([ScriptedTurn.text("one"), ScriptedTurn.text("two")])
    agent = Agent(provider)

    await agent.run("first")
    await agent.run("second")

    assert len(provider.requests[0].messages) == 1
    assert len(provider.requests[1].messages) == 1
    assert provider.requests[1].messages[0].text == "second"


async def test_a_conversation_can_be_seeded_and_continued_without_a_prompt() -> None:
    provider = FakeProvider([ScriptedTurn.text("continued")])
    conversation = Conversation([Message.user("seeded")])

    result = await Agent(provider).run(conversation=conversation)

    assert provider.last_request.messages[0].text == "seeded"
    assert result.text == "continued"
    assert len(conversation) == 2


async def test_seed_messages_are_copied_into_the_conversation() -> None:
    seed = [Message.user("seeded")]
    conversation = Conversation(seed)
    conversation.append(Message.assistant_text("later"))

    assert len(seed) == 1
    assert len(conversation) == 2
    assert list(conversation) == conversation.messages
    assert repr(conversation) == "Conversation(messages=2)"

    conversation.clear()
    assert len(conversation) == 0
    assert conversation.text == ""


def test_nothing_to_send_raises() -> None:
    agent = Agent(FakeProvider())
    with pytest.raises(LogposeError, match="Nothing to send"):
        agent.stream()
    with pytest.raises(LogposeError, match="Nothing to send"):
        agent.stream(conversation=Conversation())


def test_a_bad_prompt_type_raises_before_any_request() -> None:
    provider = FakeProvider([ScriptedTurn.text("hi")])
    with pytest.raises(LogposeError, match="prompt must be"):
        Agent(provider).stream(42)  # type: ignore[arg-type]
    assert provider.call_count == 0


@pytest.mark.parametrize("blank_prompt", ["", "   \n"])
def test_a_blank_prompt_raises_rather_than_sending_an_empty_block(blank_prompt: str) -> None:
    with pytest.raises(LogposeError, match="must not be blank"):
        Agent(FakeProvider()).stream(blank_prompt)


# ---------------------------------------------------------------------------
# 11. the on_tool_call gate
# ---------------------------------------------------------------------------


def logging_tool(name: str, log: list[str]) -> ToolDef:
    """Build a tool that records the fact that it ran.

    Args:
        name: Tool name.
        log: Shared ordering log, shared with the gate under test.

    Returns:
        The tool definition.
    """

    async def handler() -> str:
        """Record that it ran."""
        log.append(f"ran:{name}")
        return f"{name} ok"

    handler.__name__ = name
    return tool(handler)


def tool_use_provider(*names: str) -> FakeProvider:
    """Script one tool-use turn asking for ``names``, then a closing text turn.

    Args:
        *names: Tool names to request, in wire order. Ids are ``t1``, ``t2``, ...

    Returns:
        The scripted provider.
    """
    calls = [tool_call(name, id=f"t{index}") for index, name in enumerate(names, start=1)]
    return FakeProvider([ScriptedTurn.tool_use(*calls), ScriptedTurn.text("done")])


async def test_the_gate_sees_every_call_in_wire_order_before_any_handler_runs() -> None:
    log: list[str] = []
    tools = [logging_tool("alpha", log), logging_tool("beta", log)]

    async def gate(call: ToolUseBlock) -> None:
        log.append(f"gate:{call.name}")
        await asyncio.sleep(0)  # a real gate awaits; ordering must survive it
        return None

    await Agent(tool_use_provider("alpha", "beta"), tools=tools, on_tool_call=gate).run("go")

    assert log[:2] == ["gate:alpha", "gate:beta"]  # in wire order, and both first
    assert sorted(log[2:]) == ["ran:alpha", "ran:beta"]


async def test_a_blocked_call_never_reaches_its_handler() -> None:
    log: list[str] = []

    def gate(call: ToolUseBlock) -> str:
        return "Denied by the user."

    result = await Agent(
        tool_use_provider("alpha"), tools=[logging_tool("alpha", log)], on_tool_call=gate
    ).run("go")

    assert log == []  # the handler never ran
    results_message = result.messages[2]
    assert [block.content for block in results_message.content] == ["Denied by the user."]
    assert [block.is_error for block in results_message.content] == [True]


async def test_a_gate_can_block_without_calling_it_an_error() -> None:
    """A permission layer that declines *and steers* is not reporting a failure."""

    def gate(call: ToolUseBlock) -> ToolGateResult:
        return ToolGateResult(content="Do it the other way instead.", is_error=False)

    events = await collect(
        Agent(tool_use_provider("alpha"), tools=[logging_tool("alpha", [])], on_tool_call=gate),
        "go",
    )

    results = [event for event in events if isinstance(event, ToolResult)]
    assert [(result.content, result.is_error) for result in results] == [
        ("Do it the other way instead.", False)
    ]


async def test_blocked_and_allowed_calls_keep_request_order() -> None:
    log: list[str] = []
    tools = [logging_tool(name, log) for name in ("alpha", "beta", "gamma")]

    def gate(call: ToolUseBlock) -> str | None:
        return "blocked" if call.name == "beta" else None

    result = await Agent(
        tool_use_provider("alpha", "beta", "gamma"), tools=tools, on_tool_call=gate
    ).run("go")

    assert sorted(log) == ["ran:alpha", "ran:gamma"]
    results_message = result.messages[2]
    assert [block.tool_use_id for block in results_message.content] == ["t1", "t2", "t3"]
    assert [block.content for block in results_message.content] == [
        "alpha ok",
        "blocked",
        "gamma ok",
    ]


async def test_a_blocked_call_still_emits_its_call_and_result_events() -> None:
    def gate(call: ToolUseBlock) -> str:
        return "blocked"

    events = await collect(
        Agent(tool_use_provider("alpha"), tools=[logging_tool("alpha", [])], on_tool_call=gate),
        "go",
    )

    assert [type(event) for event in events] == [
        TurnEnd,
        ToolCall,
        ToolResult,
        TextDelta,
        TurnEnd,
        RunEnd,
    ]


async def test_the_gate_sees_unknown_tool_names_too() -> None:
    seen: list[str] = []

    def gate(call: ToolUseBlock) -> None:
        seen.append(call.name)
        return None

    result = await Agent(tool_use_provider("nope"), tools=[add], on_tool_call=gate).run("go")

    assert seen == ["nope"]
    # Letting it through leaves the loop's own unknown-tool report intact.
    assert "Unknown tool 'nope'" in result.messages[2].content[0].content


async def test_an_empty_gate_result_is_replaced_with_the_placeholder() -> None:
    def gate(call: ToolUseBlock) -> str:
        return "   "

    result = await Agent(
        tool_use_provider("alpha"), tools=[logging_tool("alpha", [])], on_tool_call=gate
    ).run("go")

    assert result.messages[2].content[0].content == EMPTY_TOOL_RESULT


async def test_a_raising_gate_ends_the_run_rather_than_failing_open() -> None:
    log: list[str] = []

    def gate(call: ToolUseBlock) -> None:
        raise PermissionError("the permission service is down")

    with pytest.raises(PermissionError, match="permission service is down"):
        await Agent(
            tool_use_provider("alpha"), tools=[logging_tool("alpha", log)], on_tool_call=gate
        ).run("go")

    assert log == []


async def test_a_gate_that_raises_late_still_stops_its_already_allowed_siblings() -> None:
    """The whole point of gating *entirely* before execution.

    An implementation that created each task as its call cleared the gate would
    pass every other test here while letting alpha run before the gate refused
    beta.
    """
    log: list[str] = []
    tools = [logging_tool("alpha", log), logging_tool("beta", log)]

    def gate(call: ToolUseBlock) -> None:
        if call.name == "beta":
            raise PermissionError("no")
        return None

    with pytest.raises(PermissionError):
        await Agent(tool_use_provider("alpha", "beta"), tools=tools, on_tool_call=gate).run("go")

    assert log == []


async def test_abandoning_a_stream_mid_gate_leaves_no_stray_tasks() -> None:
    """Section 9's invariant, extended to the loop's new suspension point."""
    log: list[str] = []
    entered = asyncio.Event()

    async def gate(call: ToolUseBlock) -> None:
        entered.set()
        await asyncio.Event().wait()  # never fires; the consumer walks away
        return None

    stream = Agent(
        tool_use_provider("alpha"), tools=[logging_tool("alpha", log)], on_tool_call=gate
    ).stream("go")

    async def drain() -> None:
        async for _event in stream:
            pass

    consumer = asyncio.create_task(drain())
    await entered.wait()
    before = len(asyncio.all_tasks())
    consumer.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await consumer
    await stream.aclose()  # type: ignore[attr-defined]

    assert log == []  # nothing started while the gate was still deciding
    assert len(asyncio.all_tasks()) < before


async def test_a_gate_returning_an_unsupported_type_raises() -> None:
    def gate(call: ToolUseBlock) -> Any:
        return 42

    with pytest.raises(LogposeError, match="must return None, a str, or a ToolGateResult"):
        await Agent(
            tool_use_provider("alpha"), tools=[logging_tool("alpha", [])], on_tool_call=gate
        ).run("go")


def test_a_gate_result_rejects_non_string_content_where_it_is_built() -> None:
    # Otherwise it surfaces as an AttributeError from inside the loop, naming
    # nothing the caller wrote.
    with pytest.raises(LogposeError, match="ToolGateResult.content must be a str"):
        ToolGateResult(content=None)  # type: ignore[arg-type]


async def test_without_a_gate_every_call_runs() -> None:
    log: list[str] = []
    tools = [logging_tool("alpha", log), logging_tool("beta", log)]

    await Agent(tool_use_provider("alpha", "beta"), tools=tools).run("go")

    assert sorted(log) == ["ran:alpha", "ran:beta"]


def test_on_tool_call_must_be_callable() -> None:
    with pytest.raises(LogposeError, match="on_tool_call must be callable"):
        Agent(FakeProvider(), on_tool_call="nope")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# construction and provider-contract enforcement
# ---------------------------------------------------------------------------


def test_unknown_provider_name_raises() -> None:
    with pytest.raises(LogposeError, match="Unknown provider"):
        Agent("does-not-exist")


def test_named_provider_is_resolved_with_provider_kwargs() -> None:
    agent = Agent("claude-code", model="claude-opus-5", compat_claude_code=False)

    assert agent.provider_name == "claude-code"
    assert agent.provider.compat_claude_code is False  # type: ignore[attr-defined]
    assert agent.model == "claude-opus-5"


def test_provider_kwargs_alongside_a_built_provider_raise() -> None:
    with pytest.raises(LogposeError, match="already-built provider"):
        Agent(FakeProvider(), api_key="sk-should-not-be-here")


def test_an_object_that_is_not_a_provider_raises() -> None:
    with pytest.raises(LogposeError, match="Provider protocol"):
        Agent(object())  # type: ignore[arg-type]


def test_duplicate_tool_names_raise() -> None:
    with pytest.raises(LogposeError, match="Duplicate tool name"):
        Agent(FakeProvider(), tools=[add, add])


def test_a_plain_function_is_rejected_with_a_hint() -> None:
    def not_a_tool() -> str:
        return "nope"

    with pytest.raises(LogposeError, match="@logpose.tool"):
        Agent(FakeProvider(), tools=[not_a_tool])  # type: ignore[list-item]


def test_repr_describes_the_configuration() -> None:
    agent = Agent(FakeProvider(), model="m", tools=[add])
    assert repr(agent) == "Agent(provider='fake', model='m', tools=1, max_iterations=25)"


async def test_provider_errors_propagate() -> None:
    provider = FakeProvider([ScriptedTurn.failure(ProviderError("upstream is down"))])
    with pytest.raises(ProviderError, match="upstream is down"):
        await Agent(provider).run("go")


async def test_retryable_pre_delta_provider_failure_is_retried() -> None:
    provider = FakeProvider(
        [
            ScriptedTurn.failure(ProviderError("temporary", retryable=True)),
            ScriptedTurn.text("recovered"),
        ]
    )
    agent = Agent(provider, retry_policy=RetryPolicy(initial_delay=0, jitter=0))

    result = await agent.run("go")

    assert result.text == "recovered"
    assert provider.call_count == 2


async def test_retry_after_is_honored_for_a_pre_delta_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delays: list[float] = []

    async def record_delay(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr("logpose.agent.asyncio.sleep", record_delay)
    provider = FakeProvider(
        [
            ScriptedTurn.failure(ProviderError("temporary", retryable=True, retry_after=2)),
            ScriptedTurn.text("recovered"),
        ]
    )
    agent = Agent(provider, retry_policy=RetryPolicy(initial_delay=0.5, jitter=0))

    await agent.run("go")

    assert delays == [2]


async def test_post_delta_provider_failure_is_not_retried() -> None:
    class PartialProvider:
        name = "partial"
        model_default = "partial-1"

        def __init__(self) -> None:
            self.calls = 0

        async def stream(self, req: Any) -> Any:
            self.calls += 1
            yield ProviderTextDelta(text="half")
            raise ProviderError("connection lost", retryable=True)

    provider = PartialProvider()
    agent = Agent(provider, retry_policy=RetryPolicy(initial_delay=0, jitter=0))

    events: list[Any] = []
    with pytest.raises(ProviderError) as excinfo:
        async for event in agent.stream("go"):
            events.append(event)

    assert [event.text for event in events if isinstance(event, TextDelta)] == ["half"]
    assert provider.calls == 1
    assert excinfo.value.partial is True
    assert excinfo.value.attempts == 1


async def test_retryable_failure_reports_exhausted_attempt_count() -> None:
    provider = FakeProvider(
        [
            ScriptedTurn.failure(ProviderError("temporary", retryable=True)),
            ScriptedTurn.failure(ProviderError("temporary", retryable=True)),
        ]
    )
    agent = Agent(
        provider,
        retry_policy=RetryPolicy(max_attempts=2, initial_delay=0, jitter=0),
    )

    with pytest.raises(ProviderError) as excinfo:
        await agent.run("go")

    assert provider.call_count == 2
    assert excinfo.value.attempts == 2


async def test_pre_delta_provider_turn_timeout_is_retried() -> None:
    provider = FakeProvider(
        [
            ScriptedTurn.text("late", delay=0.05),
            ScriptedTurn.text("recovered"),
        ]
    )
    agent = Agent(
        provider,
        provider_turn_timeout=0.01,
        retry_policy=RetryPolicy(initial_delay=0, jitter=0),
    )

    result = await agent.run("go")

    assert result.text == "recovered"
    assert provider.call_count == 2


async def test_post_delta_provider_turn_timeout_is_partial_and_not_retried() -> None:
    class SlowAfterDelta:
        name = "slow"
        model_default = "slow-1"

        def __init__(self) -> None:
            self.calls = 0

        async def stream(self, req: Any) -> Any:
            self.calls += 1
            yield ProviderTextDelta(text="half")
            await asyncio.sleep(0.05)
            yield CompletionDone(
                message=Message.assistant_text("half"),
                stop_reason="end_turn",
                usage=Usage(),
            )

    provider = SlowAfterDelta()
    agent = Agent(
        provider,
        provider_turn_timeout=0.01,
        retry_policy=RetryPolicy(initial_delay=0, jitter=0),
    )

    with pytest.raises(ProviderError) as excinfo:
        await agent.run("go")

    assert provider.calls == 1
    assert excinfo.value.error_code == "turn_timeout"
    assert excinfo.value.partial is True


async def test_none_disables_the_provider_turn_deadline() -> None:
    provider = FakeProvider([ScriptedTurn.text("slow but complete", delay=0.02)])
    result = await Agent(provider, provider_turn_timeout=None).run("go")
    assert result.text == "slow but complete"


async def test_a_stream_without_completion_done_is_a_provider_error() -> None:
    provider = FakeProvider([ScriptedTurn(emit_done=False)])
    with pytest.raises(ProviderError, match="without a CompletionDone"):
        await Agent(provider).run("go")


async def test_events_after_completion_done_are_a_provider_error() -> None:
    class Chatterbox:
        name = "chatty"
        model_default = "chatty-1"

        async def stream(self, req: Any) -> Any:
            yield CompletionDone(
                message=Message.assistant_text("hi"),
                stop_reason="end_turn",
                usage=Usage(),
            )
            yield CompletionDone(
                message=Message.assistant_text("again"),
                stop_reason="end_turn",
                usage=Usage(),
            )

    with pytest.raises(ProviderError, match="after CompletionDone"):
        await Agent(Chatterbox()).run("go")


async def test_unsupported_provider_events_are_rejected() -> None:
    class Weird:
        name = "weird"
        model_default = "weird-1"

        async def stream(self, req: Any) -> Any:
            yield "not an event"

    with pytest.raises(ProviderError, match="unsupported event"):
        await Agent(Weird()).run("go")


async def test_aclose_closes_only_a_provider_the_agent_built() -> None:
    class Closable:
        name = "closable"
        model_default = "closable-1"
        closed = False

        async def stream(self, req: Any) -> Any:  # pragma: no cover - never run
            yield CompletionDone(
                message=Message.assistant_text("hi"),
                stop_reason="end_turn",
                usage=Usage(),
            )

        async def aclose(self) -> None:
            self.closed = True

    provider = Closable()
    await Agent(provider).aclose()
    assert provider.closed is False  # not ours to close


async def test_tool_use_block_input_reaches_the_handler_unparsed() -> None:
    seen: list[dict[str, Any]] = []

    @tool
    def record(payload: dict[str, Any]) -> str:
        """Record a payload.

        Args:
            payload: Arbitrary JSON object.
        """
        seen.append(payload)
        return "recorded"

    block = ToolUseBlock(id="t1", name="record", input={"payload": {"nested": [1, 2]}})
    provider = FakeProvider([ScriptedTurn.tool_use(block), ScriptedTurn.text("ok")])

    await Agent(provider, tools=[record]).run("go")

    assert seen == [{"nested": [1, 2]}]
