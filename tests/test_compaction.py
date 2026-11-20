"""Built-in context compaction: tool output pointers, the recall tool, and settings."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from logpose import (
    Agent,
    ContextOverflowError,
    Conversation,
    Message,
    RuntimeEvent,
    TextBlock,
    Usage,
    tool,
)
from logpose.compaction import apply_pointers, estimate_tokens, make_recall_tool, turn_starts
from logpose.context import PreparedContext
from logpose.errors import LogposeError, ProviderError
from logpose.messages import ToolResultBlock, ToolUseBlock
from logpose.providers.base import CompletionRequest, ToolSpec
from logpose.retry import RetryPolicy
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
        Message(role="assistant", content=[TextBlock(text="answer")]),
        Message(role="user", content=[TextBlock(text="next")]),
    ]
    before = [m.model_dump() for m in messages]
    out, applied = apply_pointers(messages, current_start=4, min_chars=1000)
    assert applied == 1
    assert out[2].content[0].content == pointer("x", 2000)
    assert [m.model_dump() for m in messages] == before
    # Boundary: exactly min_chars stays inline; current-run outputs stay inline.
    _, none_applied = apply_pointers(messages, current_start=4, min_chars=2000)
    assert none_applied == 0
    _, current = apply_pointers(messages, current_start=2, min_chars=1000)
    assert current == 0
    # Before the model answered it (no assistant message after it), it stays full.
    _, unanswered = apply_pointers(messages, current_start=3, min_chars=1000)
    assert unanswered == 0


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


async def test_unanswered_results_stay_full_until_the_model_answers() -> None:
    conv = Conversation(
        [
            Message(role="user", content=[TextBlock(text="go")]),
            Message(role="assistant", content=[ToolUseBlock(id="t1", name="big", input={"n": 1})]),
            Message(role="user", content=[ToolResultBlock(tool_use_id="t1", content="1" * 5000)]),
        ]
    )
    provider = FakeProvider([ScriptedTurn.text("seen it"), ScriptedTurn.text("next")])
    agent = Agent(provider, model="test-model", tools=[big])
    await agent.run(conversation=conv)
    assert results_by_id(provider.requests[0])["t1"].content == "1" * 5000
    await agent.run("again", conversation=conv)
    assert results_by_id(provider.requests[1])["t1"].content == pointer("t1", 5000)


async def test_recall_results_are_never_pointed() -> None:
    provider = FakeProvider(
        [
            ScriptedTurn.tool_use(tool_call("big", {"n": 7}, id="t1")),
            ScriptedTurn.text("one"),
            ScriptedTurn.tool_use(tool_call(RECALL, {"call_id": "t1"}, id="r1")),
            ScriptedTurn.text("two"),
            ScriptedTurn.text("three"),
        ]
    )
    agent = Agent(provider, model="test-model", tools=[big])
    conv = Conversation()
    for prompt in ("first", "second", "third"):
        await agent.run(prompt, conversation=conv)
    third = results_by_id(provider.requests[4])
    assert third["t1"].content == pointer("t1", 5000)
    assert third["r1"].content == "7" * 5000


# -- Task 2: calibration and the summarization fallback --------------------

SUMMARY_PREFIX = "[Summary of earlier conversation]\n"


@pytest.mark.parametrize("char", ["\u4e2d", "\U0001f600", "\u00e9"])
def test_estimate_tokens_counts_non_ascii_as_characters(char: str) -> None:
    text = char * 100_000
    messages = [Message(role="user", content=[TextBlock(text=text)])]
    estimate = estimate_tokens(messages, None, [])
    # About len/4, not the 6-12x a \\uXXXX escape per character would give.
    assert 25_000 <= estimate <= 25_100


def test_message_chars_matches_estimate_for_non_ascii() -> None:
    from logpose.compaction import message_chars

    message = Message(role="user", content=[TextBlock(text="\u4e2d" * 1_000)])
    assert message_chars(message) < 1_100


async def test_large_cjk_prompt_is_not_refused_preflight() -> None:
    provider = FakeProvider([ScriptedTurn.text("ok")])
    agent = Agent(provider, model="test-model")
    result = await agent.run("\u4e2d" * 80_000)
    assert result.text == "ok"
    assert provider.call_count == 1


def chat_history(turns: int, chars: int = 400, tag: str = "") -> list[Message]:
    """``turns`` finished user/assistant exchanges, each message ~``chars`` long."""
    history: list[Message] = []
    for i in range(turns):
        history.append(
            Message(role="user", content=[TextBlock(text=f"<q{tag}{i}> " + "x" * chars)])
        )
        history.append(
            Message(role="assistant", content=[TextBlock(text=f"<a{tag}{i}> " + "y" * chars)])
        )
    return history


def all_text(messages: list[Message]) -> str:
    return "\n".join(
        block.text if isinstance(block, TextBlock) else getattr(block, "content", "")
        for message in messages
        for block in message.content
    )


def is_summary_call(request: CompletionRequest) -> bool:
    return request.tools == [] and request.max_tokens == 2048


def summary_messages(request: CompletionRequest) -> list[Message]:
    return [m for m in request.messages if m.text.startswith(SUMMARY_PREFIX)]


def assert_request_valid(request: CompletionRequest) -> None:
    """Every tool_use is answered by the next message, and no result is orphaned."""
    messages = request.messages
    assert messages and messages[0].role == "user"
    for index, message in enumerate(messages):
        uses = [b.id for b in message.content if isinstance(b, ToolUseBlock)]
        results = [b.tool_use_id for b in message.content if isinstance(b, ToolResultBlock)]
        if uses:
            assert message.role == "assistant"
            following = messages[index + 1]
            answered = [b.tool_use_id for b in following.content if isinstance(b, ToolResultBlock)]
            assert answered == uses
        if results:
            previous = messages[index - 1]
            assert previous.role == "assistant"
            assert results == [b.id for b in previous.content if isinstance(b, ToolUseBlock)]
        if index:
            # Strict alternation: chat templates on vLLM, llama.cpp and Ollama
            # reject two user or two assistant messages in a row.
            assert message.role != messages[index - 1].role, f"roles repeat at {index}"
    assert messages[-1].role == "user"  # never a trailing assistant message (prefill)
    # The Chat Completions wire form pairs every tool message with a preceding call.
    from logpose.providers.openai_compat import _messages_to_wire

    wire = _messages_to_wire(list(messages), request.system)
    open_calls: set[str] = set()
    for entry in wire:
        if entry["role"] == "assistant":
            assert not open_calls
            open_calls = {call["id"] for call in entry.get("tool_calls", [])}
        elif entry["role"] == "tool":
            assert entry["tool_call_id"] in open_calls
            open_calls.discard(entry["tool_call_id"])
        else:
            assert not open_calls
    assert not open_calls


def test_turn_starts() -> None:
    messages = [
        Message(role="user", content=[TextBlock(text="q")]),
        Message(role="assistant", content=[ToolUseBlock(id="x", name="big", input={})]),
        Message(role="user", content=[ToolResultBlock(tool_use_id="x", content="r")]),
        Message(role="assistant", content=[TextBlock(text="answer")]),
        Message(role="user", content=[TextBlock(text="next")]),
    ]
    assert turn_starts(messages) == [0, 4]
    assert turn_starts([]) == []


async def test_no_summary_below_threshold() -> None:
    provider = FakeProvider([ScriptedTurn.text("one"), ScriptedTurn.text("two")])
    agent = Agent(provider, model="test-model", context_budget=2_000)
    conv = Conversation()
    await agent.run("hello", conversation=conv)
    await agent.run("again", conversation=conv)
    assert provider.call_count == 2
    assert not any(summary_messages(request) for request in provider.requests)
    assert conv._compaction is not None and conv._compaction.summary is None


async def test_summary_at_threshold() -> None:
    provider = FakeProvider([ScriptedTurn.text("SUMMARY-ONE"), ScriptedTurn.text("answer")])
    agent = Agent(provider, model="test-model", tools=[small], context_budget=2_000)
    conv = Conversation(chat_history(10))
    full = estimate_tokens(conv.messages, None, [])
    assert full >= 0.85 * 2_000
    await agent.run("latest question", conversation=conv)

    assert provider.call_count == 2
    summary_call, turn = provider.requests
    assert is_summary_call(summary_call)
    assert summary_call.model == "test-model"
    transcript = all_text(summary_call.messages)
    assert "<a0>" in transcript and "<q1>" in transcript
    assert "<q0>" not in transcript

    assert turn.messages[0] == conv.messages[0]
    assert turn.messages[1].role == "assistant"
    assert turn.messages[1].text == SUMMARY_PREFIX + "SUMMARY-ONE"
    assert_request_valid(turn)
    sent = all_text(turn.messages)
    assert "<q1>" not in sent and "<a1>" not in sent
    assert "<q9>" in sent and "latest question" in sent
    assert estimate_tokens(turn.messages, turn.system, turn.tools) <= 0.60 * 2_000
    # Stored history keeps everything.
    assert len(conv.messages) == 20 + 2
    assert "<q1>" in all_text(conv.messages)


async def test_summary_is_reused_until_threshold_again() -> None:
    provider = FakeProvider(
        [ScriptedTurn.text("SUMMARY-ONE"), ScriptedTurn.text("answer"), ScriptedTurn.text("more")]
    )
    agent = Agent(provider, model="test-model", context_budget=2_000)
    conv = Conversation(chat_history(10))
    await agent.run("latest question", conversation=conv)
    await agent.run("follow up", conversation=conv)

    assert provider.call_count == 3
    first_turn, second_turn = provider.requests[1], provider.requests[2]
    assert not is_summary_call(second_turn) or second_turn.tools != []
    assert summary_messages(second_turn)[0].text == SUMMARY_PREFIX + "SUMMARY-ONE"
    # The request prefix is unchanged, so provider prompt caching keeps working.
    assert second_turn.messages[: len(first_turn.messages)] == first_turn.messages
    assert second_turn.messages[-1].text == "follow up"
    assert_request_valid(first_turn)
    assert_request_valid(second_turn)


async def test_summary_extends_when_threshold_reached_again() -> None:
    provider = FakeProvider(
        [
            ScriptedTurn.text("SUMMARY-ONE"),
            ScriptedTurn.text("answer"),
            ScriptedTurn.text("SUMMARY-TWO"),
            ScriptedTurn.text("more"),
        ]
    )
    agent = Agent(provider, model="test-model", context_budget=2_000)
    conv = Conversation(chat_history(10))
    await agent.run("latest question", conversation=conv)
    conv.extend(chat_history(6, tag="b"))
    await agent.run("follow up", conversation=conv)

    assert provider.call_count == 4
    extend_call = provider.requests[2]
    assert is_summary_call(extend_call)
    # The earlier summary is carried forward, not the turns it already covers.
    transcript = all_text(extend_call.messages)
    assert "SUMMARY-ONE" in transcript
    assert "<q1>" not in transcript
    assert "<qb0>" in transcript
    turn = provider.requests[3]
    assert [m.text for m in summary_messages(turn)] == [SUMMARY_PREFIX + "SUMMARY-TWO"]
    assert_request_valid(turn)
    assert estimate_tokens(turn.messages, turn.system, turn.tools) <= 0.60 * 2_000


async def test_first_user_turn_never_summarized() -> None:
    provider = FakeProvider([ScriptedTurn.text("S"), ScriptedTurn.text("answer")])
    agent = Agent(provider, model="test-model", context_budget=2_000)
    history = chat_history(10)
    history[0] = Message(role="user", content=[TextBlock(text="FIRST-TURN " + "f" * 400)])
    conv = Conversation(history)
    await agent.run("latest", conversation=conv)
    summary_call, turn = provider.requests
    assert "FIRST-TURN" not in all_text(summary_call.messages)
    assert turn.messages[0] == history[0]
    assert turn.messages[1].text == SUMMARY_PREFIX + "S"
    assert turn.messages[1].role == "assistant"
    assert_request_valid(turn)


async def test_cut_never_splits_tool_pair() -> None:
    history: list[Message] = []
    for i in range(8):
        history.append(Message(role="user", content=[TextBlock(text=f"<q{i}> " + "x" * 100)]))
        history.append(
            Message(
                role="assistant",
                content=[ToolUseBlock(id=f"u{i}", name="small", input={})],
            )
        )
        history.append(
            Message(role="user", content=[ToolResultBlock(tool_use_id=f"u{i}", content="r" * 700)])
        )
        history.append(Message(role="assistant", content=[TextBlock(text=f"<a{i}> done")]))
    provider = FakeProvider(
        [
            ScriptedTurn.text("S"),
            ScriptedTurn.tool_use(tool_call("small", id="now")),
            ScriptedTurn.text("answer"),
        ]
    )
    agent = Agent(provider, model="test-model", tools=[small], context_budget=2_000)
    conv = Conversation(history)
    await agent.run("latest", conversation=conv)
    assert is_summary_call(provider.requests[0])
    for request in provider.requests[1:]:
        assert summary_messages(request)
        assert_request_valid(request)
        # The message after the summary starts a caller turn.
        after = request.messages[2]
        assert after.role == "user"
        assert not any(isinstance(b, ToolResultBlock) for b in after.content)


async def test_edited_history_invalidates_cache() -> None:
    provider = FakeProvider(
        [
            ScriptedTurn.text("SUMMARY-OLD"),
            ScriptedTurn.text("answer"),
            ScriptedTurn.text("SUMMARY-NEW"),
            ScriptedTurn.text("more"),
        ]
    )
    agent = Agent(provider, model="test-model", context_budget=2_000)
    conv = Conversation(chat_history(10))
    await agent.run("latest question", conversation=conv)
    old_cut = conv._compaction.cut_index if conv._compaction else 0
    assert old_cut > 3
    del conv.messages[2:4]  # the caller drops <q1>/<a1>, inside the summarized span
    await agent.run("follow up", conversation=conv)

    assert provider.call_count == 4
    resummarize = provider.requests[2]
    assert is_summary_call(resummarize)
    transcript = all_text(resummarize.messages)
    assert "SUMMARY-OLD" not in transcript
    assert "<q1>" not in transcript and "<a0>" in transcript
    turn = provider.requests[3]
    assert [m.text for m in summary_messages(turn)] == [SUMMARY_PREFIX + "SUMMARY-NEW"]
    assert_request_valid(turn)
    # The recent turns are the real tail of the edited history.
    assert (
        turn.messages[2:] == conv.messages[len(conv.messages) - (len(turn.messages) - 2) - 1 : -1]
    )


async def test_edited_history_that_now_fits_drops_summary() -> None:
    provider = FakeProvider(
        [ScriptedTurn.text("SUMMARY-OLD"), ScriptedTurn.text("answer"), ScriptedTurn.text("more")]
    )
    agent = Agent(provider, model="test-model", context_budget=2_000)
    conv = Conversation(chat_history(10))
    await agent.run("latest question", conversation=conv)
    del conv.messages[1:16]
    await agent.run("follow up", conversation=conv)
    assert provider.call_count == 3
    assert provider.requests[2].messages == conv.messages[:-1]


async def test_edit_after_cut_that_moves_turn_start_invalidates_cache() -> None:
    provider = FakeProvider(
        [
            ScriptedTurn.text("SUMMARY-OLD"),
            ScriptedTurn.text("answer"),
            ScriptedTurn.text("SUMMARY-NEW"),
            ScriptedTurn.text("more"),
        ]
    )
    agent = Agent(provider, model="test-model", context_budget=2_000)
    conv = Conversation(chat_history(10))
    await agent.run("latest question", conversation=conv)
    assert conv._compaction is not None
    cut = conv._compaction.cut_index
    del conv.messages[cut]  # the user turn at the cut; an assistant message moves there
    await agent.run("follow up", conversation=conv)
    assert is_summary_call(provider.requests[2])
    turn = provider.requests[3]
    assert [m.text for m in summary_messages(turn)] == [SUMMARY_PREFIX + "SUMMARY-NEW"]
    assert turn.messages[2].role == "user"


async def test_summary_failure_sends_uncompacted_when_it_fits() -> None:
    events: list[RuntimeEvent] = []
    provider = FakeProvider(
        [ScriptedTurn.failure(ProviderError("down", status_code=400)), ScriptedTurn.text("ok")]
    )
    agent = Agent(provider, model="test-model", context_budget=2_000, observers=[events.append])
    conv = Conversation(chat_history(10))
    result = await agent.run("latest", conversation=conv)
    assert result.text == "ok"
    assert provider.call_count == 2
    assert provider.requests[1].messages == conv.messages[:-1]
    compacted = [e for e in events if e.name == "context.compacted"]
    assert len(compacted) == 1 and compacted[0].summarized is False
    assert conv._compaction is not None and conv._compaction.summary is None


async def test_summary_failure_overflows_when_it_does_not() -> None:
    cause = ProviderError("down", status_code=400)
    provider = FakeProvider([ScriptedTurn.failure(cause)])
    agent = Agent(
        provider, model="test-model", context_window=3_000, max_tokens=1_000, context_budget=None
    )
    assert agent.input_budget == 2_000
    conv = Conversation(chat_history(10))
    with pytest.raises(ContextOverflowError) as info:
        await agent.run("latest", conversation=conv)
    assert info.value.__cause__ is cause
    assert info.value.input_budget == 2_000
    assert info.value.estimated_tokens > 2_000
    assert provider.call_count == 1


async def test_summary_failure_retries_per_policy() -> None:
    provider = FakeProvider(
        [
            ScriptedTurn.failure(ProviderError("busy", status_code=503, retryable=True)),
            ScriptedTurn.text("S"),
            ScriptedTurn.text("ok"),
        ]
    )
    agent = Agent(
        provider,
        model="test-model",
        context_budget=2_000,
        retry_policy=RetryPolicy(max_attempts=2, initial_delay=0.0, max_delay=0.0, jitter=0),
    )
    conv = Conversation(chat_history(10))
    result = await agent.run("latest", conversation=conv)
    assert result.text == "ok"
    assert result.iterations == 1
    assert provider.call_count == 3
    assert summary_messages(provider.requests[2])


async def test_overflow_after_summary_raises() -> None:
    # The summary itself is too long to fit the window.
    provider = FakeProvider([ScriptedTurn.text("s" * 9_000)])
    agent = Agent(
        provider, model="test-model", context_window=3_000, max_tokens=1_000, context_budget=None
    )
    conv = Conversation(chat_history(10))
    with pytest.raises(ContextOverflowError) as info:
        await agent.run("latest", conversation=conv)
    assert provider.call_count == 1
    assert info.value.estimated_tokens > 2_000

    # A current turn that alone exceeds the window cannot be helped by a summary:
    # it raises without paying for the summary call.
    provider = FakeProvider([ScriptedTurn.text("S")])
    agent = Agent(
        provider, model="test-model", context_window=3_000, max_tokens=1_000, context_budget=None
    )
    with pytest.raises(ContextOverflowError):
        await agent.run("z" * 10_000, conversation=Conversation(chat_history(10)))
    assert provider.call_count == 0


def clamp(ratio: float) -> float:
    return min(2.0, max(0.5, ratio))


async def test_calibration_ratio_updates_from_reported_usage() -> None:
    provider = FakeProvider(
        [
            ScriptedTurn.text("one", usage=Usage(input_tokens=12, cache_read_input_tokens=8)),
            ScriptedTurn.text("two", usage=Usage(input_tokens=30)),
            ScriptedTurn.text("three"),
        ]
    )
    agent = Agent(provider, model="test-model")
    conv = Conversation()
    await agent.run("hello", conversation=conv)
    state = conv._compaction
    assert state is not None
    first = provider.requests[0]
    est1 = estimate_tokens(first.messages, first.system, first.tools)
    expected = 0.7 * 1.0 + 0.3 * (20 / est1)
    assert 0.5 < expected < 2.0  # inside the clamp: the formula itself is checked
    assert state.ratio == pytest.approx(expected)

    await agent.run("again", conversation=conv)
    second = provider.requests[1]
    est2 = estimate_tokens(second.messages, second.system, second.tools)
    expected = clamp(0.7 * expected + 0.3 * (30 / est2))
    assert state.ratio == pytest.approx(expected)

    # No reported usage leaves the ratio alone.
    await agent.run("third", conversation=conv)
    assert state.ratio == pytest.approx(expected)


async def test_calibration_ratio_scales_the_threshold() -> None:
    # A small history the raw estimate keeps under 85%, but the server reports
    # it as three times larger: the next request is summarized.
    provider = FakeProvider(
        [
            ScriptedTurn.text("one", usage=Usage(input_tokens=6_000)),
            ScriptedTurn.text("S"),
            ScriptedTurn.text("two"),
        ]
    )
    agent = Agent(provider, model="test-model", context_budget=2_000)
    conv = Conversation(chat_history(5))
    await agent.run("hello", conversation=conv)
    assert not summary_messages(provider.requests[0])
    await agent.run("again", conversation=conv)
    assert is_summary_call(provider.requests[1])
    assert summary_messages(provider.requests[2])


async def test_compacted_event_emitted() -> None:
    events: list[RuntimeEvent] = []
    provider = FakeProvider([ScriptedTurn.text("S"), ScriptedTurn.text("answer")])
    agent = Agent(provider, model="test-model", context_budget=2_000, observers=[events.append])
    await agent.run("latest", conversation=Conversation(chat_history(10)))
    compacted = [e for e in events if e.name == "context.compacted"]
    assert len(compacted) == 1
    event = compacted[0]
    assert event.summarized is True
    assert event.pointers_applied == 0
    assert event.input_budget == 2_000
    assert event.estimated_tokens_before is not None and event.estimated_tokens_before >= 1_700
    assert event.estimated_tokens_after is not None and event.estimated_tokens_after <= 1_200
    # Metadata only: no strings beyond identifiers.
    for name, value in vars(event).items():
        if isinstance(value, str):
            assert name in {"name", "run_id", "provider", "model", "turn_id", "request_id"}
    # The summary call is not a provider attempt.
    assert [e.name for e in events].count("provider.attempt.started") == 1
    assert agent.last_context_diagnostics["summarized"] is True


async def test_pointer_only_turn_emits_event() -> None:
    events: list[RuntimeEvent] = []
    provider = FakeProvider(two_round_script())
    agent = Agent(provider, model="test-model", tools=[big], observers=[events.append])
    conv = Conversation()
    await agent.run("first", conversation=conv)
    assert not [e for e in events if e.name == "context.compacted"]
    await agent.run("second", conversation=conv)
    compacted = [e for e in events if e.name == "context.compacted"]
    assert len(compacted) == 1
    assert compacted[0].pointers_applied == 2
    assert compacted[0].summarized is False


async def test_summary_usage_counted_in_run_result() -> None:
    provider = FakeProvider(
        [
            ScriptedTurn.text("S", usage=Usage(input_tokens=500, output_tokens=50)),
            ScriptedTurn.text("answer", usage=Usage(input_tokens=100, output_tokens=10)),
        ]
    )
    agent = Agent(provider, model="test-model", context_budget=2_000, max_iterations=1)
    result = await agent.run("latest", conversation=Conversation(chat_history(10)))
    assert result.usage.input_tokens == 600
    assert result.usage.output_tokens == 60
    assert result.iterations == 1


async def test_summary_usage_counted_in_session_run_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from logpose.store import manager

    db = tmp_path / "s.db"
    monkeypatch.setenv("LOGPOSE_DB_PATH", str(db))
    manager.reset_stores_for_tests()
    try:
        provider = FakeProvider(
            [
                ScriptedTurn.text("S", usage=Usage(input_tokens=500, output_tokens=50)),
                ScriptedTurn.text("answer", usage=Usage(input_tokens=100, output_tokens=10)),
            ]
        )
        agent = Agent(provider, model="test-model", context_budget=2_000)
        async with agent.session("chat-1") as handle:
            handle.conversation.extend(chat_history(10))
            await agent.run("latest", session=handle)
        await agent.aclose()
        with sqlite3.connect(db) as conn:
            rows = list(
                conn.execute("SELECT iterations, input_tokens, output_tokens FROM logpose_runs")
            )
        assert rows == [(1, 600, 60)]
    finally:
        manager.reset_stores_for_tests()


async def test_structured_output_correction_with_summary() -> None:
    from pydantic import BaseModel

    class Answer(BaseModel):
        value: int

    provider = FakeProvider(
        [
            ScriptedTurn.text("S"),
            ScriptedTurn.text("not json"),
            ScriptedTurn.text('{"value": 3}'),
        ]
    )
    agent = Agent(provider, model="test-model", context_budget=2_000)
    conv = Conversation(chat_history(10))
    result = await agent.run("latest", conversation=conv, output_type=Answer, output_retries=1)
    assert result.output == Answer(value=3)
    assert provider.call_count == 3
    correction = provider.requests[2]
    assert summary_messages(correction)
    assert correction.messages[-1].role == "user"
    assert correction.messages[-2].text == "not json"
    assert_request_valid(correction)


@pytest.mark.parametrize("turns", [4, 5, 6, 8, 10, 14])
@pytest.mark.parametrize("chars", [150, 400, 700])
async def test_every_cut_configuration_alternates(turns: int, chars: int) -> None:
    provider = FakeProvider([ScriptedTurn.text("S"), ScriptedTurn.text("answer")], repeat_last=True)
    agent = Agent(provider, model="test-model", context_budget=2_000)
    conv = Conversation(chat_history(turns, chars))
    await agent.run("latest", conversation=conv)
    for request in provider.requests:
        if not is_summary_call(request):
            assert_request_valid(request)
    # And again with tool rounds in every earlier turn and in the current run.
    history: list[Message] = []
    for i in range(turns):
        history.append(Message(role="user", content=[TextBlock(text=f"<q{i}> " + "x" * chars)]))
        history.append(
            Message(role="assistant", content=[ToolUseBlock(id=f"u{i}", name="small", input={})])
        )
        history.append(
            Message(role="user", content=[ToolResultBlock(tool_use_id=f"u{i}", content="r" * 500)])
        )
        history.append(Message(role="assistant", content=[TextBlock(text=f"<a{i}> done")]))
    provider = FakeProvider(
        [
            ScriptedTurn.text("S"),
            ScriptedTurn.tool_use(tool_call("small", id="now")),
            ScriptedTurn.text("answer"),
        ]
    )
    agent = Agent(provider, model="test-model", tools=[small], context_budget=2_000)
    await agent.run("latest", conversation=Conversation(history))
    for request in provider.requests:
        if not is_summary_call(request):
            assert_request_valid(request)


async def test_calibration_ratio_is_clamped_low() -> None:
    # A server that under-reports prompt tokens (2% of the estimate) cannot
    # collapse the estimate below half: the threshold still fires.
    history = chat_history(10)
    probe = estimate_tokens(history, None, [])
    under = Usage(input_tokens=max(1, probe // 50))
    provider = FakeProvider(
        [ScriptedTurn.text(f"r{i}", usage=under) for i in range(12)]
        + [ScriptedTurn.text("S"), ScriptedTurn.text("final")]
    )
    agent = Agent(provider, model="test-model", context_budget=4_000)
    conv = Conversation(chat_history(2, chars=50))
    for i in range(12):
        await agent.run(f"p{i}", conversation=conv)
        assert conv._compaction is not None and conv._compaction.ratio >= 0.5
    assert conv._compaction.ratio == pytest.approx(0.5)
    # Raw history now ~1.8x the 85% threshold of 4,000: halved, it still fires.
    conv.extend(chat_history(30, tag="z"))
    raw = estimate_tokens(conv.messages, None, [])
    assert raw * 0.5 >= 0.85 * 4_000
    await agent.run("go", conversation=conv)
    assert is_summary_call(provider.requests[12])


async def test_calibration_ratio_is_clamped_high() -> None:
    provider = FakeProvider(
        [ScriptedTurn.text(f"r{i}", usage=Usage(input_tokens=1_000_000)) for i in range(8)]
    )
    agent = Agent(provider, model="test-model")
    conv = Conversation()
    for i in range(8):
        await agent.run(f"p{i}", conversation=conv)
        assert conv._compaction is not None and conv._compaction.ratio <= 2.0
    assert conv._compaction.ratio == pytest.approx(2.0)


async def test_failed_summary_not_retried_within_the_run() -> None:
    events: list[RuntimeEvent] = []
    provider = FakeProvider(
        [
            ScriptedTurn.failure(ProviderError("down", status_code=400)),
            ScriptedTurn.tool_use(tool_call("small", id="now")),
            ScriptedTurn.text("done"),
            ScriptedTurn.text("S"),
            ScriptedTurn.text("next run"),
        ]
    )
    agent = Agent(
        provider, model="test-model", tools=[small], context_budget=2_000, observers=[events.append]
    )
    conv = Conversation(chat_history(10))
    result = await agent.run("latest", conversation=conv)
    assert result.text == "done"
    # One failed summary call, then two uncompacted turns: no second attempt.
    assert provider.call_count == 3
    assert not any(summary_messages(r) for r in provider.requests[1:3])
    compacted = [e for e in events if e.name == "context.compacted"]
    # Only the turn that attempted the summary reports; the next one has
    # nothing to report (no pointers, no summary, no attempt).
    assert [e.summarized for e in compacted] == [False]
    # A new run tries again.
    await agent.run("again", conversation=conv)
    assert is_summary_call(provider.requests[3])
    assert summary_messages(provider.requests[4])


# -- Task 3: persisted summaries and the reactive context-length retry ----------


@pytest.fixture
def session_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from logpose.store import manager

    db = tmp_path / "s.db"
    monkeypatch.setenv("LOGPOSE_DB_PATH", str(db))
    manager.reset_stores_for_tests()
    yield db
    manager.reset_stores_for_tests()


async def test_summary_persists_across_requests(session_db: Path) -> None:
    from logpose.store import manager

    provider = FakeProvider([ScriptedTurn.text("SUMMARY-ONE"), ScriptedTurn.text("answer")])
    agent = Agent(provider, model="test-model", context_budget=2_000)
    async with agent.session("chat-1") as handle:
        handle.conversation.extend(chat_history(10))
        await agent.run("latest", session=handle)
    await agent.aclose()
    assert is_summary_call(provider.requests[0])
    first_turn = provider.requests[1]
    with sqlite3.connect(session_db) as conn:
        rows = list(conn.execute("SELECT session_id, upto_seq, text FROM logpose_summaries"))
    # upto_seq is the stored seq of the last summarized message (cut index - 1).
    # The request was [first turn, summary, *history[cut:]] over 21 messages.
    cut = 21 - (len(first_turn.messages) - 2)
    assert first_turn.messages[2] == Message(
        role="user", content=[TextBlock(text=f"<q{cut // 2}> " + "x" * 400)]
    )
    assert rows == [("chat-1", cut - 1, "SUMMARY-ONE")]

    # A new process: fresh stores, agent and provider.
    manager.reset_stores_for_tests()
    provider_b = FakeProvider([ScriptedTurn.text("answer B")])
    agent_b = Agent(provider_b, model="test-model", context_budget=2_000)
    result = await agent_b.run("follow up", session_id="chat-1")
    await agent_b.aclose()
    assert result.text == "answer B"
    assert provider_b.call_count == 1
    request = provider_b.requests[0]
    assert not is_summary_call(request)
    assert [m.text for m in summary_messages(request)] == [SUMMARY_PREFIX + "SUMMARY-ONE"]
    # Same prefix as run A's turn, so prompt caching survives the new process.
    assert request.messages[: len(first_turn.messages)] == first_turn.messages
    assert request.messages[-1].text == "follow up"
    assert_request_valid(request)


async def test_persisted_summary_survives_structured_output_rollback(session_db: Path) -> None:
    from pydantic import BaseModel

    from logpose.store import manager

    class Answer(BaseModel):
        value: int

    provider = FakeProvider(
        [ScriptedTurn.text("S"), ScriptedTurn.text("not json"), ScriptedTurn.text('{"value": 3}')]
    )
    agent = Agent(provider, model="test-model", context_budget=2_000)
    async with agent.session("chat-2") as handle:
        handle.conversation.extend(chat_history(10))
        result = await agent.run("latest", session=handle, output_type=Answer, output_retries=1)
        assert result.output == Answer(value=3)
    await agent.aclose()

    manager.reset_stores_for_tests()
    provider_b = FakeProvider([ScriptedTurn.text("B")])
    agent_b = Agent(provider_b, model="test-model", context_budget=2_000)
    async with agent_b.session("chat-2") as handle:
        # The invalid answer and its correction were dropped before saving.
        assert len(handle.conversation.messages) == 22
        state = handle.conversation._compaction
        assert state is not None and state.summary == "S"
        await agent_b.run("next", session=handle)
    await agent_b.aclose()
    assert provider_b.call_count == 1
    assert [m.text for m in summary_messages(provider_b.requests[0])] == [SUMMARY_PREFIX + "S"]
    assert_request_valid(provider_b.requests[0])


async def test_extended_summary_is_persisted(session_db: Path) -> None:
    from logpose.store import manager

    provider = FakeProvider([ScriptedTurn.text("SUMMARY-ONE"), ScriptedTurn.text("answer")])
    agent = Agent(provider, model="test-model", context_budget=2_000)
    async with agent.session("chat-3") as handle:
        handle.conversation.extend(chat_history(10))
        await agent.run("latest", session=handle)
    await agent.aclose()

    manager.reset_stores_for_tests()
    provider_b = FakeProvider([ScriptedTurn.text("SUMMARY-TWO"), ScriptedTurn.text("more")])
    agent_b = Agent(provider_b, model="test-model", context_budget=2_000)
    async with agent_b.session("chat-3") as handle:
        handle.conversation.extend(chat_history(6, tag="b"))
        await agent_b.run("follow up", session=handle)
    await agent_b.aclose()
    extend_call = provider_b.requests[0]
    assert is_summary_call(extend_call)
    assert "SUMMARY-ONE" in all_text(extend_call.messages)
    with sqlite3.connect(session_db) as conn:
        latest = conn.execute(
            "SELECT text FROM logpose_summaries ORDER BY upto_seq DESC LIMIT 1"
        ).fetchone()
    assert latest == ("SUMMARY-TWO",)


CONTEXT_ERROR = (
    "This model's maximum context length is 8192 tokens. However, your messages "
    "resulted in 9000 tokens. Please reduce the length of the messages."
)


async def test_reactive_context_length_retry() -> None:
    events: list[RuntimeEvent] = []
    rejection = ProviderError(CONTEXT_ERROR, status_code=400)
    provider = FakeProvider(
        [ScriptedTurn.failure(rejection), ScriptedTurn.text("S"), ScriptedTurn.text("ok")]
    )
    agent = Agent(provider, model="test-model", observers=[events.append])
    conv = Conversation(chat_history(4))
    result = await agent.run("latest", conversation=conv)

    assert result.text == "ok"
    assert result.iterations == 1
    assert provider.call_count == 3
    # The first request was below the threshold: sent whole.
    assert provider.requests[0].messages == conv.messages[:-1]
    assert sum(is_summary_call(r) for r in provider.requests) == 1
    assert is_summary_call(provider.requests[1])
    retried = provider.requests[2]
    assert [m.text for m in summary_messages(retried)] == [SUMMARY_PREFIX + "S"]
    assert retried.messages[-1].text == "latest"
    assert_request_valid(retried)
    assert conv._compaction is not None and conv._compaction.summary == "S"
    assert any(e.name == "context.compacted" and e.summarized for e in events)


async def test_second_context_length_rejection_propagates() -> None:
    first = ProviderError(CONTEXT_ERROR, status_code=400)
    second = ProviderError(CONTEXT_ERROR, status_code=400)
    provider = FakeProvider(
        [ScriptedTurn.failure(first), ScriptedTurn.text("S"), ScriptedTurn.failure(second)]
    )
    agent = Agent(provider, model="test-model")
    with pytest.raises(ProviderError) as info:
        await agent.run("latest", conversation=Conversation(chat_history(4)))
    assert not isinstance(info.value, ContextOverflowError)
    assert info.value is second
    assert provider.call_count == 3


@pytest.mark.parametrize(
    "message",
    [
        "Error: context_length_exceeded",
        "prompt exceeds the model's CONTEXT WINDOW",
        "Maximum Context Length reached",
        # llama.cpp server
        "the request exceeds the available context size, try increasing it",
        "exceed_context_size_error",
        # LM Studio
        "Trying to keep the first 9000 tokens when Context Length is 4096",
        "EXCEED_CONTEXT",
    ],
)
async def test_context_length_phrasings_detected(message: str) -> None:
    provider = FakeProvider(
        [
            ScriptedTurn.failure(ProviderError(message, status_code=400)),
            ScriptedTurn.text("S"),
            ScriptedTurn.text("ok"),
        ]
    )
    agent = Agent(provider, model="test-model")
    result = await agent.run("latest", conversation=Conversation(chat_history(4)))
    assert result.text == "ok"
    assert provider.call_count == 3


async def test_llama_cpp_error_type_detected_from_error_code() -> None:
    rejection = ProviderError(
        "bad request", status_code=400, error_code="exceed_context_size_error"
    )
    provider = FakeProvider(
        [ScriptedTurn.failure(rejection), ScriptedTurn.text("S"), ScriptedTurn.text("ok")]
    )
    agent = Agent(provider, model="test-model")
    result = await agent.run("latest", conversation=Conversation(chat_history(4)))
    assert result.text == "ok"
    assert provider.call_count == 3


@pytest.mark.parametrize(
    "error",
    [
        ProviderError("invalid request", status_code=400),
        ProviderError(CONTEXT_ERROR, status_code=500),
        ProviderError(CONTEXT_ERROR),
    ],
)
async def test_other_errors_are_not_reactive(error: ProviderError) -> None:
    provider = FakeProvider([ScriptedTurn.failure(error)])
    agent = Agent(provider, model="test-model")
    with pytest.raises(ProviderError) as info:
        await agent.run("latest", conversation=Conversation(chat_history(4)))
    assert info.value is error
    assert provider.call_count == 1


async def test_structured_output_unsupported_is_not_reactive() -> None:
    from pydantic import BaseModel

    from logpose.errors import StructuredOutputUnsupportedError

    class Answer(BaseModel):
        value: int

    error = StructuredOutputUnsupportedError(
        "response_format json_schema unsupported (context window)", status_code=400
    )
    provider = FakeProvider([ScriptedTurn.failure(error)])
    agent = Agent(provider, model="test-model")
    with pytest.raises(StructuredOutputUnsupportedError) as info:
        await agent.run("latest", conversation=Conversation(chat_history(4)), output_type=Answer)
    assert info.value is error
    assert provider.call_count == 1


async def test_context_length_without_anything_to_summarize_propagates() -> None:
    rejection = ProviderError(CONTEXT_ERROR, status_code=400)
    provider = FakeProvider([ScriptedTurn.failure(rejection)])
    agent = Agent(provider, model="test-model")
    with pytest.raises(ProviderError) as info:
        await agent.run("x" * 400)
    assert info.value is rejection
    assert provider.call_count == 1


async def test_context_length_with_custom_policy_propagates() -> None:
    def passthrough(context):
        return PreparedContext(context.messages, context.system)

    rejection = ProviderError(CONTEXT_ERROR, status_code=400)
    provider = FakeProvider([ScriptedTurn.failure(rejection)])
    agent = Agent(provider, model="test-model", context_policy=passthrough)
    with pytest.raises(ProviderError) as info:
        await agent.run("latest", conversation=Conversation(chat_history(4)))
    assert info.value is rejection
    assert provider.call_count == 1


async def test_context_length_when_summary_fails_propagates_rejection() -> None:
    rejection = ProviderError(CONTEXT_ERROR, status_code=400)
    provider = FakeProvider(
        [ScriptedTurn.failure(rejection), ScriptedTurn.failure(ProviderError("down"))]
    )
    agent = Agent(provider, model="test-model")
    with pytest.raises(ProviderError) as info:
        await agent.run("latest", conversation=Conversation(chat_history(4)))
    assert info.value is rejection
    assert provider.call_count == 2


async def test_reactive_retry_respects_run_deadline() -> None:
    from logpose import RunTimeoutError

    provider = FakeProvider(
        [
            ScriptedTurn.failure(ProviderError(CONTEXT_ERROR, status_code=400)),
            ScriptedTurn.text("S", delay=5.0),
        ]
    )
    agent = Agent(provider, model="test-model", run_timeout=0.2)
    with pytest.raises(RunTimeoutError):
        await agent.run("latest", conversation=Conversation(chat_history(4)))
    assert provider.call_count == 2


async def test_reactive_retry_once_per_turn_not_per_run() -> None:
    provider = FakeProvider(
        [
            ScriptedTurn.failure(ProviderError(CONTEXT_ERROR, status_code=400)),
            ScriptedTurn.text("S1"),
            ScriptedTurn.tool_use(tool_call("small", id="c1")),
            ScriptedTurn.failure(ProviderError(CONTEXT_ERROR, status_code=400)),
            ScriptedTurn.text("S2"),
            ScriptedTurn.text("done"),
        ]
    )
    agent = Agent(provider, model="test-model", tools=[small], max_iterations=2)
    conv = Conversation(chat_history(6))
    result = await agent.run("latest", conversation=conv)
    assert result.text == "done"
    assert result.iterations == 2
    assert provider.call_count == 6
    assert is_summary_call(provider.requests[1]) and is_summary_call(provider.requests[4])
    # The second forced summary extends the first.
    assert "S1" in all_text(provider.requests[4].messages)
    assert [m.text for m in summary_messages(provider.requests[5])] == [SUMMARY_PREFIX + "S2"]
    assert_request_valid(provider.requests[5])
