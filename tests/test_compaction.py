"""Built-in context compaction: tool output pointers, the recall tool, and settings."""

from __future__ import annotations

import pytest

from logpose import Agent, Conversation, Message, TextBlock, tool
from logpose.compaction import apply_pointers, estimate_tokens, make_recall_tool
from logpose.context import PreparedContext
from logpose.errors import LogposeError
from logpose.messages import ToolResultBlock, ToolUseBlock
from logpose.providers.base import CompletionRequest, ToolSpec
from tests.fake_provider import FakeProvider, ScriptedTurn, tool_call

RECALL = "recall_tool_output"


@tool
def big(n: int) -> str:
    """Return a 5,000-character output made of the digit ``n``."""
    return str(n) * 5000


@tool
def small() -> str:
    """Return a 500-character output."""
    return "s" * 500


@tool
def failing() -> str:
    """Always fail with a long message."""
    raise RuntimeError("boom " * 400)


def pointer(call_id: str, chars: int) -> str:
    return (
        f"[tool output stored: {call_id}, {chars} chars. "
        f'Call recall_tool_output("{call_id}") to read it.]'
    )


def results_by_id(request: CompletionRequest) -> dict[str, ToolResultBlock]:
    return {
        block.tool_use_id: block
        for message in request.messages
        for block in message.content
        if isinstance(block, ToolResultBlock)
    }


def two_round_script() -> list[ScriptedTurn]:
    return [
        ScriptedTurn.tool_use(tool_call("big", {"n": 1}, id="t1")),
        ScriptedTurn.tool_use(tool_call("big", {"n": 2}, id="t2")),
        ScriptedTurn.text("run one done"),
        ScriptedTurn.text("run two done"),
    ]


async def test_pointer_replaces_long_outputs_from_earlier_runs_only() -> None:
    provider = FakeProvider(two_round_script())
    agent = Agent(provider, model="test-model", tools=[big])
    conv = Conversation()
    await agent.run("first", conversation=conv)
    await agent.run("second", conversation=conv)

    # Run 1's own outputs stay full on every turn of run 1.
    assert results_by_id(provider.requests[1])["t1"].content == "1" * 5000
    run_one_last = results_by_id(provider.requests[2])
    assert run_one_last["t1"].content == "1" * 5000
    assert run_one_last["t2"].content == "2" * 5000

    # Run 2 sees run 1's outputs as pointers.
    run_two = results_by_id(provider.requests[3])
    assert run_two["t1"].content == pointer("t1", 5000)
    assert run_two["t2"].content == pointer("t2", 5000)


async def test_short_outputs_stay_inline() -> None:
    provider = FakeProvider(
        [
            ScriptedTurn.tool_use(tool_call("small", id="s1")),
            ScriptedTurn.text("one"),
            ScriptedTurn.text("two"),
        ]
    )
    agent = Agent(provider, model="test-model", tools=[small])
    conv = Conversation()
    await agent.run("first", conversation=conv)
    await agent.run("second", conversation=conv)
    assert results_by_id(provider.requests[2])["s1"].content == "s" * 500


async def test_pointer_keeps_pairing() -> None:
    provider = FakeProvider(
        [
            ScriptedTurn.tool_use(tool_call("failing", id="f1")),
            ScriptedTurn.text("one"),
            ScriptedTurn.text("two"),
        ]
    )
    agent = Agent(provider, model="test-model", tools=[failing], tool_error_mode="message")
    conv = Conversation()
    await agent.run("first", conversation=conv)
    full = results_by_id(provider.requests[1])["f1"]
    assert full.is_error and len(full.content) > 1000
    await agent.run("second", conversation=conv)
    pointed = results_by_id(provider.requests[2])["f1"]
    assert pointed.tool_use_id == "f1"
    assert pointed.is_error is True
    assert pointed.content == pointer("f1", len(full.content))
    # The tool_use it answers is still in the request, immediately before it.
    messages = provider.requests[2].messages
    index = next(i for i, m in enumerate(messages) if pointed in m.content)
    uses = [b for b in messages[index - 1].content if isinstance(b, ToolUseBlock)]
    assert [u.id for u in uses] == ["f1"]


async def test_stored_history_is_unchanged() -> None:
    provider = FakeProvider(two_round_script())
    agent = Agent(provider, model="test-model", tools=[big])
    conv = Conversation()
    await agent.run("first", conversation=conv)
    result = await agent.run("second", conversation=conv)
    for messages in (conv.messages, result.messages):
        stored = {
            block.tool_use_id: block.content
            for message in messages
            for block in message.content
            if isinstance(block, ToolResultBlock)
        }
        assert stored == {"t1": "1" * 5000, "t2": "2" * 5000}


def test_apply_pointers_never_mutates_input() -> None:
    messages = [
        Message(role="user", content=[TextBlock(text="q")]),
        Message(role="assistant", content=[ToolUseBlock(id="x", name="big", input={})]),
        Message(role="user", content=[ToolResultBlock(tool_use_id="x", content="z" * 2000)]),
        Message(role="user", content=[TextBlock(text="next")]),
    ]
    before = [m.model_dump() for m in messages]
    out, applied = apply_pointers(messages, current_start=3, min_chars=1000)
    assert applied == 1
    assert out[2].content[0].content == pointer("x", 2000)
    assert [m.model_dump() for m in messages] == before
    # Boundary: exactly min_chars stays inline; current-run outputs stay inline.
    _, none_applied = apply_pointers(messages, current_start=3, min_chars=2000)
    assert none_applied == 0
    _, current = apply_pointers(messages, current_start=2, min_chars=1000)
    assert current == 0


def test_estimate_tokens_counts_all_pieces() -> None:
    messages = [Message(role="user", content=[TextBlock(text="a" * 400)])]
    spec = ToolSpec(name="t", description="d" * 400, input_schema={"type": "object"})
    bare = estimate_tokens(messages, None, [])
    assert bare >= 100
    assert estimate_tokens(messages, "s" * 400, []) >= bare + 100
    assert estimate_tokens(messages, None, [spec]) >= bare + 100


async def test_recall_tool_reads_full_output_and_pages() -> None:
    history = [
        Message(role="assistant", content=[ToolUseBlock(id="r1", name="big", input={})]),
        Message(
            role="user",
            content=[ToolResultBlock(tool_use_id="r1", content="a" * 8000 + "b" * 2000)],
        ),
    ]
    recall = make_recall_tool(lambda: history)
    assert recall.name == RECALL
    assert set(recall.input_schema["properties"]) == {"call_id", "offset"}
    assert recall.input_schema["required"] == ["call_id"]

    first = await recall.invoke({"call_id": "r1"})
    assert first.startswith("a" * 8000)
    hint = first[8000:]
    assert "b" not in hint.replace("offset", "")
    assert "offset=8000" in hint

    rest = await recall.invoke({"call_id": "r1", "offset": 8000})
    assert rest == "b" * 2000

    assert await recall.invoke({"call_id": "nope"}) == "No stored tool output with id 'nope'."


async def test_recall_tool_reads_run_history_end_to_end() -> None:
    provider = FakeProvider(
        [
            ScriptedTurn.tool_use(tool_call("big", {"n": 7}, id="t1")),
            ScriptedTurn.text("one"),
            ScriptedTurn.tool_use(tool_call(RECALL, {"call_id": "t1"}, id="r1")),
            ScriptedTurn.text("two"),
        ]
    )
    agent = Agent(provider, model="test-model", tools=[big])
    conv = Conversation()
    await agent.run("first", conversation=conv)
    await agent.run("second", conversation=conv)
    recalled = results_by_id(provider.requests[3])["r1"]
    assert recalled.is_error is False
    assert recalled.content == "7" * 5000


async def test_recall_tool_present_only_with_tools() -> None:
    provider = FakeProvider([ScriptedTurn.text("hi")])
    await Agent(provider, model="test-model", tools=[big]).run("x")
    names = [spec.name for spec in provider.requests[0].tools]
    assert names == ["big", RECALL]

    bare = FakeProvider([ScriptedTurn.text("hi")])
    await Agent(bare, model="test-model").run("x")
    assert bare.requests[0].tools == []


def test_recall_name_collision_rejected() -> None:
    @tool(name=RECALL)
    def mine(call_id: str) -> str:
        """App tool with the reserved name."""
        return call_id

    with pytest.raises(LogposeError, match=RECALL):
        Agent(FakeProvider([]), model="test-model", tools=[mine])
    # With the built-in path disabled the name is the app's to use.
    Agent(FakeProvider([]), model="test-model", tools=[mine], context_policy=None)


async def test_custom_policy_disables_builtin() -> None:
    def passthrough(context):
        return PreparedContext(context.messages, context.system)

    provider = FakeProvider(two_round_script())
    agent = Agent(provider, model="test-model", tools=[big], context_policy=passthrough)
    conv = Conversation()
    await agent.run("first", conversation=conv)
    await agent.run("second", conversation=conv)
    assert results_by_id(provider.requests[3])["t1"].content == "1" * 5000
    assert [spec.name for spec in provider.requests[3].tools] == ["big"]


async def test_none_policy_sends_full_history() -> None:
    provider = FakeProvider(two_round_script())
    agent = Agent(provider, model="test-model", tools=[big], context_policy=None)
    conv = Conversation()
    await agent.run("first", conversation=conv)
    await agent.run("second", conversation=conv)
    run_two = results_by_id(provider.requests[3])
    assert run_two["t1"].content == "1" * 5000
    assert run_two["t2"].content == "2" * 5000
    assert [spec.name for spec in provider.requests[3].tools] == ["big"]


def test_input_budget_validation() -> None:
    with pytest.raises(LogposeError, match="input"):
        Agent(FakeProvider([]), model="test-model", context_window=10_000, max_tokens=16_384)

    provider = FakeProvider([])
    provider.max_tokens = 9_000  # type: ignore[attr-defined]
    with pytest.raises(LogposeError, match="input"):
        Agent(provider, model="test-model", context_window=9_000)

    assert Agent(FakeProvider([]), model="test-model").input_budget == 32_000
    unbounded = Agent(FakeProvider([]), model="test-model", context_budget=None)
    assert unbounded.input_budget == 128_000 - 16_384
    assert Agent(FakeProvider([]), model="test-model")._context_policy_is_default is True
    assert (
        Agent(FakeProvider([]), model="test-model", context_policy=None)._context_policy_is_default
        is False
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"context_window": 0},
        {"context_window": True},
        {"context_window": 1.5},
        {"context_budget": 0},
        {"context_budget": -5},
        {"pointer_min_chars": 0},
        {"compaction_threshold": 1.1},
        {"compaction_threshold": 0.5, "compaction_target": 0.5},
        {"compaction_target": 0.0},
        {"compaction_target": 0.9},
        {"compaction_threshold": "high"},
    ],
)
def test_invalid_settings_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(LogposeError):
        Agent(FakeProvider([]), model="test-model", **kwargs)  # type: ignore[arg-type]
