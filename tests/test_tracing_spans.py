"""Task 2 of Plan 4: GenAI spans built by the agent loop (in-memory sink)."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from logpose import Agent, Conversation, tool
from logpose.errors import LogposeError, ProviderError
from logpose.guardrails import GuardrailInput, GuardrailVerdict
from logpose.messages import Message, TextBlock, Usage
from logpose.providers._redact import redact
from logpose.providers.base import ProviderTextDelta
from logpose.retry import RetryPolicy
from logpose.runtime import RuntimeEvent, current_runtime_context
from logpose.tracing import semconv as sc
from logpose.tracing.spans import SpanRecord, Tracing
from tests.fake_provider import FakeProvider, ScriptedTurn, tool_call

INTERNAL, CLIENT = 1, 3
UNSET, ERROR = 0, 2

KEY = "sk-test-SECRETKEY-0123456789abcdef"


@tool
def add(a: int, b: int) -> str:
    """Add two numbers.

    Args:
        a: First.
        b: Second.
    """
    return str(a + b)


class _MemorySink:
    """Test-only span sink: keeps spans in memory instead of the store's writer."""

    def __init__(self) -> None:
        self.spans: list[SpanRecord] = []

    def submit(self, span: SpanRecord) -> None:
        self.spans.append(span)

    async def run_id_exists(self, run_id: str) -> bool:
        return any(span.run_id == run_id for span in self.spans)

    async def flush(self, run_id: str | None, timeout: float) -> bool:
        return True

    def pending_for(self, run_id: str) -> int:
        return 0


@pytest.fixture(autouse=True)
def _memory_sink(monkeypatch: pytest.MonkeyPatch) -> None:
    """Route each traced agent's spans to its own in-memory sink (no store writes)."""

    async def open_sink(self: Agent) -> _MemorySink | None:
        if self.tracing is None:
            return None
        sink = self.__dict__.get("_test_sink")
        if sink is None:
            sink = self.__dict__["_test_sink"] = _MemorySink()
        return sink

    monkeypatch.setattr(Agent, "_open_trace_sink", open_sink)


def spans_of(agent: Agent) -> list[SpanRecord]:
    sink = agent.__dict__.get("_test_sink")
    assert sink is not None
    return list(sink.spans)


def by_op(spans: list[SpanRecord], op: str) -> list[SpanRecord]:
    return [s for s in spans if s.attributes.get(sc.GEN_AI_OPERATION_NAME) == op]


def named(spans: list[SpanRecord], name: str) -> list[SpanRecord]:
    return [s for s in spans if s.name == name]


def root(spans: list[SpanRecord]) -> SpanRecord:
    (span,) = by_op(spans, sc.OPERATION_INVOKE_AGENT)
    return span


def in_order(spans: list[SpanRecord]) -> list[SpanRecord]:
    return sorted(spans, key=lambda s: (s.start_ns, s.end_ns))


@pytest.fixture(autouse=True)
def _no_trace_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LOGPOSE_TRACE", raising=False)
    monkeypatch.delenv("LOGPOSE_TRACE_CONTENT", raising=False)


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    path = tmp_path / "t.db"
    monkeypatch.setenv("LOGPOSE_DB_PATH", str(path))
    yield path


def _rows(path: Path, sql: str) -> list[tuple[Any, ...]]:
    with sqlite3.connect(path) as conn:
        return conn.execute(sql).fetchall()


async def test_span_tree_shape() -> None:
    provider = FakeProvider(
        [ScriptedTurn.tool_use(tool_call("add", {"a": 1, "b": 2}, id="c1")), ScriptedTurn.text("3")]
    )
    agent = Agent(provider, model="m", tools=[add], tracing=Tracing())
    await agent.run("1+2?")
    spans = spans_of(agent)
    assert len({s.trace_id for s in spans}) == 1
    top = root(spans)
    assert top.parent_span_id is None
    assert top.name == "invoke_agent logpose"
    assert top.kind == INTERNAL
    children = in_order([s for s in spans if s is not top])
    assert [s.attributes[sc.GEN_AI_OPERATION_NAME] for s in children] == [
        "chat",
        "execute_tool",
        "chat",
    ]
    assert all(s.parent_span_id == top.span_id for s in children)
    assert [s.name for s in children] == ["chat m", "execute_tool add", "chat m"]
    assert [s.kind for s in children] == [CLIENT, INTERNAL, CLIENT]
    tool_span = children[1]
    assert tool_span.attributes[sc.GEN_AI_TOOL_NAME] == "add"
    assert tool_span.attributes[sc.GEN_AI_TOOL_CALL_ID] == "c1"
    assert tool_span.attributes[sc.LOGPOSE_TOOL_IS_ERROR] is False
    assert tool_span.attributes[sc.LOGPOSE_TOOL_TIMED_OUT] is False
    assert isinstance(tool_span.attributes[sc.LOGPOSE_TOOL_QUEUE_SECONDS], float)
    first_chat = children[0]
    assert first_chat.attributes[sc.GEN_AI_REQUEST_MODEL] == "m"
    assert first_chat.attributes[sc.GEN_AI_PROVIDER_NAME] == "fake"
    assert first_chat.attributes[sc.LOGPOSE_ATTEMPT_NUMBER] == 1
    assert first_chat.attributes[sc.GEN_AI_RESPONSE_FINISH_REASONS] == ["tool_call"]
    assert children[2].attributes[sc.GEN_AI_RESPONSE_FINISH_REASONS] == ["stop"]
    for span in spans:
        assert span.start_ns <= span.end_ns
        assert span.run_id == top.attributes[sc.LOGPOSE_RUN_ID]
        assert span.session_id is None


async def test_invoke_agent_attributes(db: Path) -> None:
    usage = Usage(input_tokens=10, cache_read_input_tokens=5, output_tokens=3, reasoning_tokens=2)
    provider = FakeProvider(ScriptedTurn.text("hi", usage=usage))
    agent = Agent(
        provider, model="m", system="Be terse.", tools=[add], tracing=Tracing(capture_content=True)
    )
    await agent.run("hello", session_id="s1", metadata={"exp": "baseline", "n": 3, "ok": True})
    spans = spans_of(agent)
    top = root(spans)
    attrs = top.attributes
    assert attrs[sc.LOGPOSE_RUN_OUTCOME] == "completed"
    assert attrs[sc.LOGPOSE_RUN_ITERATIONS] == 1
    assert attrs[sc.GEN_AI_USAGE_INPUT_TOKENS] == 15
    assert attrs[sc.GEN_AI_USAGE_OUTPUT_TOKENS] == 3
    assert attrs[sc.LOGPOSE_USAGE_INPUT_TOKENS] == 10
    assert attrs[sc.LOGPOSE_USAGE_CACHE_READ_INPUT_TOKENS] == 5
    assert attrs[sc.LOGPOSE_USAGE_CACHE_CREATION_INPUT_TOKENS] == 0
    assert attrs[sc.LOGPOSE_USAGE_REASONING_TOKENS] == 2
    assert attrs[sc.GEN_AI_CONVERSATION_ID] == "s1"
    assert attrs[sc.GEN_AI_AGENT_NAME] == "logpose"
    assert attrs[sc.GEN_AI_PROVIDER_NAME] == "fake"
    assert attrs[sc.GEN_AI_REQUEST_MODEL] == "m"
    assert attrs[sc.LOGPOSE_METADATA_PREFIX + "exp"] == "baseline"
    assert attrs[sc.LOGPOSE_METADATA_PREFIX + "n"] == "3"
    assert attrs[sc.LOGPOSE_METADATA_PREFIX + "ok"] == "True"
    cfg = sc.LOGPOSE_CONFIG_PREFIX
    assert attrs[cfg + "max_iterations"] == 10
    assert attrs[cfg + "max_tokens"] == agent._max_tokens()
    assert attrs[cfg + "run_timeout"] == agent.run_timeout
    assert attrs[cfg + "provider_turn_timeout"] == agent.provider_turn_timeout
    assert attrs[cfg + "tool_timeout"] == agent.tool_timeout
    assert attrs[cfg + "max_concurrent_tools"] == agent.max_concurrent_tools
    assert attrs[cfg + "retry.max_attempts"] == agent.retry_policy.max_attempts
    assert attrs[cfg + "tools"] == ["add", "recall_tool_output"]
    assert len(attrs[cfg + "system_prompt_sha256"]) == 64
    assert "Be terse." in attrs[sc.GEN_AI_SYSTEM_INSTRUCTIONS]
    assert "add" in attrs[sc.GEN_AI_TOOL_DEFINITIONS]
    assert all(s.session_id == "s1" for s in spans)
    (chat,) = by_op(spans, "chat")
    assert chat.attributes[sc.GEN_AI_USAGE_INPUT_TOKENS] == 15
    assert chat.attributes[sc.LOGPOSE_USAGE_REASONING_TOKENS] == 2
    # Metadata is also recorded on the run row.
    (row,) = _rows(db, "SELECT run_id, metadata FROM logpose_runs")
    assert row[0] == top.run_id
    assert json.loads(row[1]) == {"exp": "baseline", "n": 3, "ok": True}
    await agent.aclose()


async def test_retry_creates_sibling_chat_spans() -> None:
    provider = FakeProvider(
        [
            ScriptedTurn.failure(ProviderError("busy", retryable=True, status_code=503)),
            ScriptedTurn.text("ok"),
        ]
    )
    agent = Agent(
        provider,
        model="m",
        retry_policy=RetryPolicy(max_attempts=2, initial_delay=0.0, max_delay=0.0, jitter=0.0),
        tracing=Tracing(capture_content=True),
    )
    await agent.run("PROMPT-TEXT")
    spans = spans_of(agent)
    top = root(spans)
    first, second = in_order(by_op(spans, "chat"))
    assert first.parent_span_id == second.parent_span_id == top.span_id
    assert first.attributes[sc.LOGPOSE_ATTEMPT_NUMBER] == 1
    assert second.attributes[sc.LOGPOSE_ATTEMPT_NUMBER] == 2
    assert "PROMPT-TEXT" in first.attributes[sc.GEN_AI_INPUT_MESSAGES]
    assert sc.GEN_AI_INPUT_MESSAGES not in second.attributes
    assert second.attributes[sc.LOGPOSE_INPUT_SAME_AS_SPAN] == first.span_id
    assert first.status_code == ERROR
    assert first.attributes[sc.ERROR_TYPE] == "ProviderError"
    assert first.attributes[sc.LOGPOSE_ATTEMPT_RETRYABLE] is True
    assert [e["name"] for e in first.events] == [sc.EVENT_RETRY_SCHEDULED]
    assert second.status_code == UNSET
    assert "ok" in second.attributes[sc.GEN_AI_OUTPUT_MESSAGES]


class KeyedProvider(FakeProvider):
    def _secrets(self) -> tuple[str, ...]:
        return (KEY,)


@tool
def leak(q: str) -> str:
    """Return something sensitive.

    Args:
        q: Query.
    """
    return f"TOOL-OUTPUT the key is {KEY}"


def _leaky_run_script() -> list[ScriptedTurn]:
    return [
        ScriptedTurn.tool_use(tool_call("leak", {"q": "TOOL-ARG"}, id="c1")),
        ScriptedTurn.text("ANSWER-TEXT", thinking="THINKING-TEXT"),
    ]


def _dump(spans: list[SpanRecord]) -> str:
    return json.dumps(
        [[s.name, s.attributes, s.events, s.status_message] for s in spans],
        default=str,
        ensure_ascii=False,
    )


async def test_capture_off_stores_no_content() -> None:
    agent = Agent(
        KeyedProvider(_leaky_run_script()),
        model="m",
        system="SYSTEM-TEXT",
        tools=[leak],
        tracing=Tracing(),
    )
    result = await agent.run("PROMPT-TEXT", metadata={"label": "x"})
    assert result.text == "ANSWER-TEXT"
    dumped = _dump(spans_of(agent))
    for needle in (KEY, "PROMPT-TEXT", "TOOL-OUTPUT", "TOOL-ARG", "ANSWER-TEXT", "THINKING-TEXT"):
        assert needle not in dumped
    assert "SYSTEM-TEXT" not in dumped
    for span in spans_of(agent):
        for name in (
            sc.GEN_AI_INPUT_MESSAGES,
            sc.GEN_AI_OUTPUT_MESSAGES,
            sc.GEN_AI_SYSTEM_INSTRUCTIONS,
            sc.GEN_AI_TOOL_DEFINITIONS,
            sc.GEN_AI_TOOL_CALL_ARGUMENTS,
            sc.GEN_AI_TOOL_CALL_RESULT,
        ):
            assert name not in span.attributes


async def test_capture_on_scrubs_configured_keys() -> None:
    agent = Agent(
        KeyedProvider(_leaky_run_script()),
        model="m",
        tools=[leak],
        tracing=Tracing(capture_content=True),
    )
    await agent.run("PROMPT-TEXT")
    spans = spans_of(agent)
    dumped = _dump(spans)
    assert KEY not in dumped
    assert redact(KEY) in dumped
    (tool_span,) = by_op(spans, "execute_tool")
    assert "TOOL-ARG" in tool_span.attributes[sc.GEN_AI_TOOL_CALL_ARGUMENTS]
    assert "TOOL-OUTPUT" in tool_span.attributes[sc.GEN_AI_TOOL_CALL_RESULT]
    last_chat = in_order(by_op(spans, "chat"))[-1]
    output = json.loads(last_chat.attributes[sc.GEN_AI_OUTPUT_MESSAGES])
    assert output[0]["role"] == "assistant"
    assert {"type": "reasoning", "content": "THINKING-TEXT"} in output[0]["parts"]
    assert {"type": "text", "content": "ANSWER-TEXT"} in output[0]["parts"]


async def test_capture_on_stores_messages_with_tail_truncation() -> None:
    big = "".join(f"{i:09d}," for i in range(10_000))
    assert len(big) == 100_000

    @tool
    def huge() -> str:
        """Return a lot."""
        return big

    provider = FakeProvider(
        [ScriptedTurn.tool_use(tool_call("huge", {}, id="c1")), ScriptedTurn.text("done")]
    )
    agent = Agent(provider, model="m", tools=[huge], tracing=Tracing(capture_content=True))
    await agent.run("go")
    spans = spans_of(agent)
    (tool_span,) = by_op(spans, "execute_tool")
    stored = tool_span.attributes[sc.GEN_AI_TOOL_CALL_RESULT]
    assert len(stored) == 64_000
    assert stored.startswith("…[truncated ")
    marker_end = stored.index(" chars]") + len(" chars]")
    dropped = int(stored[len("…[truncated ") : stored.index(" chars]")])
    kept = stored[marker_end:]
    assert dropped == 100_000 - len(kept)
    assert big.endswith(kept)
    assert tool_span.attributes[sc.LOGPOSE_CONTENT_TRUNCATED] is True
    first_chat = in_order(by_op(spans, "chat"))[0]
    assert sc.LOGPOSE_CONTENT_TRUNCATED not in first_chat.attributes
    messages = json.loads(first_chat.attributes[sc.GEN_AI_INPUT_MESSAGES])
    assert messages == [{"role": "user", "parts": [{"type": "text", "content": "go"}]}]


async def test_content_cap_is_configurable() -> None:
    agent = Agent(
        FakeProvider(ScriptedTurn.text("x" * 500)),
        model="m",
        tracing=Tracing(capture_content=True, content_max_chars=100),
    )
    await agent.run("go")
    (chat,) = by_op(spans_of(agent), "chat")
    assert len(chat.attributes[sc.GEN_AI_OUTPUT_MESSAGES]) == 100
    assert chat.attributes[sc.LOGPOSE_CONTENT_TRUNCATED] is True


async def test_guardrail_and_compaction_spans() -> None:
    def block_guard(inp: GuardrailInput) -> GuardrailVerdict:
        return GuardrailVerdict.block("unsafe", "No.", scores={"unsafe": 0.9})

    provider = FakeProvider(ScriptedTurn.text("never"))
    agent = Agent(provider, model="m", input_guardrails=[block_guard], tracing=Tracing())
    result = await agent.run("bad")
    assert result.guardrail is not None
    assert provider.call_count == 0
    spans = spans_of(agent)
    top = root(spans)
    assert top.attributes[sc.LOGPOSE_RUN_OUTCOME] == "blocked"
    assert top.attributes[sc.LOGPOSE_RUN_ITERATIONS] == 0
    (guard,) = named(spans, sc.SPAN_GUARDRAIL)
    assert guard.parent_span_id == top.span_id
    assert guard.kind == INTERNAL
    assert guard.attributes[sc.LOGPOSE_GUARDRAIL_NAME] == "block_guard"
    assert guard.attributes[sc.LOGPOSE_GUARDRAIL_ALLOWED] is False
    assert guard.attributes[sc.LOGPOSE_GUARDRAIL_CATEGORY] == "unsafe"
    assert json.loads(guard.attributes[sc.LOGPOSE_GUARDRAIL_SCORES]) == {"unsafe": 0.9}
    assert sc.LOGPOSE_GUARDRAIL_INPUT not in guard.attributes
    assert not by_op(spans, "chat")

    # Pointers on a later run produce a compaction span.
    @tool
    def long_output() -> str:
        """Return a long string."""
        return "z" * 5_000

    provider = FakeProvider(
        [
            ScriptedTurn.tool_use(tool_call("long_output", {}, id="c1")),
            ScriptedTurn.text("first"),
            ScriptedTurn.text("second"),
        ]
    )
    agent = Agent(provider, model="m", tools=[long_output], tracing=Tracing())
    conv = Conversation()
    await agent.run("one", conversation=conv)
    assert not named(spans_of(agent), sc.SPAN_COMPACTION)
    await agent.run("two", conversation=conv)
    second_trace = [s for s in spans_of(agent) if s.trace_id != spans_of(agent)[0].trace_id]
    (compaction,) = named(second_trace, sc.SPAN_COMPACTION)
    assert compaction.parent_span_id == root(second_trace).span_id
    assert compaction.attributes[sc.LOGPOSE_COMPACTION_POINTERS] == 1
    assert compaction.attributes[sc.LOGPOSE_COMPACTION_SUMMARIZED] is False
    assert compaction.attributes[sc.LOGPOSE_COMPACTION_BUDGET] == agent.input_budget
    before = compaction.attributes[sc.LOGPOSE_COMPACTION_ESTIMATE_BEFORE]
    after = compaction.attributes[sc.LOGPOSE_COMPACTION_ESTIMATE_AFTER]
    assert isinstance(before, int) and isinstance(after, int)
    (chat,) = by_op(second_trace, "chat")
    assert chat.attributes[sc.LOGPOSE_CONTEXT_PREFIX + "pointers_applied"] == 1


async def test_guardrail_span_capture_and_usage() -> None:
    async def judge(inp: GuardrailInput) -> GuardrailVerdict:
        return GuardrailVerdict(
            allowed=True,
            category="fail_open",
            reason="missing_api_key",
            usage=Usage(input_tokens=4, output_tokens=1),
        )

    agent = Agent(
        FakeProvider(ScriptedTurn.text("hi")),
        model="m",
        input_guardrails=[judge],
        tracing=Tracing(capture_content=True),
    )
    await agent.run("GUARDED-TEXT")
    (guard,) = named(spans_of(agent), sc.SPAN_GUARDRAIL)
    assert guard.attributes[sc.LOGPOSE_GUARDRAIL_ALLOWED] is True
    assert guard.attributes[sc.LOGPOSE_GUARDRAIL_CATEGORY] == "fail_open"
    assert guard.attributes[sc.LOGPOSE_GUARDRAIL_INPUT] == "GUARDED-TEXT"
    assert guard.attributes[sc.GEN_AI_USAGE_INPUT_TOKENS] == 4
    assert guard.attributes[sc.GEN_AI_USAGE_OUTPUT_TOKENS] == 1


async def test_summary_call_is_a_chat_span() -> None:
    from tests.test_compaction import chat_history

    provider = FakeProvider([ScriptedTurn.text("SUMMARY-ONE"), ScriptedTurn.text("answer")])
    agent = Agent(
        provider,
        model="m",
        tools=[add],
        context_budget=2_000,
        tracing=Tracing(capture_content=True),
    )
    conv = Conversation(chat_history(10))
    await agent.run("latest question", conversation=conv)
    spans = spans_of(agent)
    top = root(spans)
    summary_chat, turn_chat = in_order(by_op(spans, "chat"))
    assert summary_chat.attributes[sc.LOGPOSE_CHAT_PURPOSE] == "summary"
    assert sc.LOGPOSE_CHAT_PURPOSE not in turn_chat.attributes
    assert summary_chat.parent_span_id == top.span_id
    (compaction,) = named(spans, sc.SPAN_COMPACTION)
    assert compaction.attributes[sc.LOGPOSE_COMPACTION_SUMMARIZED] is True
    assert compaction.attributes[sc.LOGPOSE_COMPACTION_SUMMARY] == "SUMMARY-ONE"


async def test_error_status_on_failures() -> None:
    provider = FakeProvider(ScriptedTurn.failure(ProviderError("boom", retryable=False)))
    agent = Agent(provider, model="m", tracing=Tracing())
    with pytest.raises(ProviderError):
        await agent.run("hi")
    spans = spans_of(agent)
    (chat,) = by_op(spans, "chat")
    assert chat.status_code == ERROR
    assert chat.attributes[sc.ERROR_TYPE] == "ProviderError"
    assert chat.attributes[sc.LOGPOSE_ATTEMPT_RETRYABLE] is False
    assert chat.status_message is None  # messages only with capture
    top = root(spans)
    assert top.status_code == ERROR
    assert top.attributes[sc.ERROR_TYPE] == "ProviderError"
    assert top.attributes[sc.LOGPOSE_RUN_OUTCOME] == "error"


async def test_tool_failure_and_gate_block() -> None:
    @tool
    def broken() -> str:
        """Always fails."""
        raise RuntimeError("nope")

    provider = FakeProvider(
        [
            ScriptedTurn.tool_use(
                tool_call("broken", {}, id="c1"), tool_call("add", {"a": 1, "b": 1}, id="c2")
            ),
            ScriptedTurn.text("done"),
        ]
    )

    def gate(call: Any) -> str | None:
        return "DENIED-TEXT" if call.name == "add" else None

    agent = Agent(provider, model="m", tools=[broken, add], on_tool_call=gate, tracing=Tracing())
    await agent.run("go")
    tools = {
        s.attributes[sc.GEN_AI_TOOL_CALL_ID]: s for s in by_op(spans_of(agent), "execute_tool")
    }
    assert tools["c1"].status_code == ERROR
    assert tools["c1"].attributes[sc.ERROR_TYPE] == "RuntimeError"
    assert tools["c1"].attributes[sc.LOGPOSE_TOOL_IS_ERROR] is True
    gated = tools["c2"]
    assert [e["name"] for e in gated.events] == [sc.EVENT_GATE_BLOCKED]
    assert "DENIED-TEXT" not in _dump([gated])
    assert gated.attributes[sc.LOGPOSE_TOOL_IS_ERROR] is True


async def test_run_id_override_and_duplicate_rejected() -> None:
    seen: list[Any] = []
    events: list[RuntimeEvent] = []

    @tool
    def peek() -> str:
        """Look at the runtime context."""
        seen.append(current_runtime_context())
        return "ok"

    provider = FakeProvider(
        [ScriptedTurn.tool_use(tool_call("peek", {}, id="c1")), ScriptedTurn.text("done")]
    )
    agent = Agent(provider, model="m", tools=[peek], observers=[events.append], tracing=Tracing())
    await agent.run("go", run_id="eval-1.item_7", metadata={"item": "q-17"})
    (context,) = seen
    assert context.run_id == "eval-1.item_7"
    assert dict(context.metadata) == {"item": "q-17"}
    assert events and all(e.run_id == "eval-1.item_7" for e in events)
    spans = spans_of(agent)
    assert all(s.run_id == "eval-1.item_7" for s in spans)
    assert root(spans).attributes[sc.LOGPOSE_RUN_ID] == "eval-1.item_7"

    calls = provider.call_count
    with pytest.raises(LogposeError, match="run_id"):
        await agent.run("again", run_id="eval-1.item_7")
    assert provider.call_count == calls

    for bad in ("", "has space", "x" * 129, "slash/no"):
        with pytest.raises(LogposeError):
            agent.stream("hi", run_id=bad)
    for meta in ({"k": [1]}, {1: "v"}, {"k": None}, "notamapping"):
        with pytest.raises(LogposeError):
            agent.stream("hi", metadata=meta)  # type: ignore[arg-type]


async def test_duplicate_run_id_rejected_with_session_only(db: Path) -> None:
    def guard(inp: GuardrailInput) -> GuardrailVerdict:
        guarded.append(inp.run_id)
        return GuardrailVerdict.allow()

    guarded: list[str] = []
    provider = FakeProvider([ScriptedTurn.text("a"), ScriptedTurn.text("b")])
    agent = Agent(provider, model="m", input_guardrails=[guard], tracing=None)
    await agent.run("one", session_id="s1", run_id="r-1")
    assert _rows(db, "SELECT run_id FROM logpose_runs") == [("r-1",)]
    with pytest.raises(LogposeError, match="run_id"):
        await agent.run("two", session_id="s2", run_id="r-1")
    assert provider.call_count == 1
    assert guarded == ["r-1"]
    # Without tracing or a session, nothing is checked (no store is touched).
    plain = Agent(FakeProvider([ScriptedTurn.text("c")]), model="m", tracing=None)
    await plain.run("three", run_id="r-1")
    await agent.aclose()


async def test_tracing_off_by_default_and_none_forces_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert Agent(FakeProvider(), model="m").tracing is None
    monkeypatch.setenv("LOGPOSE_TRACE", "  ")
    assert Agent(FakeProvider(), model="m").tracing is None
    monkeypatch.setenv("LOGPOSE_TRACE_CONTENT", "1")
    assert Agent(FakeProvider(), model="m").tracing is None  # content alone does not enable
    monkeypatch.setenv("LOGPOSE_TRACE", "TRUE")
    assert Agent(FakeProvider(), model="m").tracing == Tracing(capture_content=True)
    monkeypatch.setenv("LOGPOSE_TRACE_CONTENT", "no")
    assert Agent(FakeProvider(), model="m").tracing == Tracing()
    monkeypatch.setenv("LOGPOSE_TRACE", "yes")
    assert Agent(FakeProvider(), model="m", tracing=None).tracing is None
    explicit = Tracing(service_name="svc")
    assert Agent(FakeProvider(), model="m", tracing=explicit).tracing is explicit
    agent = Agent(FakeProvider(ScriptedTurn.text("x")), model="m", tracing=None)
    await agent.run("hi")
    assert "_test_sink" not in agent.__dict__


def test_tracing_validates_its_fields() -> None:
    for kwargs in (
        {"capture_content": "yes"},
        {"service_name": ""},
        {"content_max_chars": 0},
        {"content_max_chars": True},
        {"flush_timeout": -1},
    ):
        with pytest.raises(LogposeError):
            Tracing(**kwargs)  # type: ignore[arg-type]
    with pytest.raises(LogposeError):
        Agent(FakeProvider(), model="m", tracing="on")  # type: ignore[arg-type]


async def test_builder_failure_disables_tracing_without_affecting_run(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from logpose.tracing import spans as spans_module

    def explode(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("builder bug")

    monkeypatch.setattr(spans_module._SpanBuilder, "_chat_attributes", explode)
    provider = FakeProvider(
        [ScriptedTurn.tool_use(tool_call("add", {"a": 1, "b": 2}, id="c1")), ScriptedTurn.text("3")]
    )
    agent = Agent(provider, model="m", tools=[add], tracing=Tracing())
    with caplog.at_level(logging.WARNING, logger="logpose.trace"):
        result = await agent.run("1+2?")
    assert result.text == "3"
    assert result.iterations == 2
    warnings = [r for r in caplog.records if r.name == "logpose.trace"]
    assert len(warnings) == 1
    # Tracing stopped for that run: nothing after the failure was recorded.
    assert not by_op(spans_of(agent), "chat")


async def test_parallel_block_marks_discarded_chat_span() -> None:
    async def slow_block(inp: GuardrailInput) -> GuardrailVerdict:
        await asyncio.sleep(0.05)
        return GuardrailVerdict.block("unsafe", "No.")

    provider = FakeProvider(ScriptedTurn.text("quick"))
    agent = Agent(
        provider,
        model="m",
        input_guardrails=[slow_block],
        guardrail_mode="parallel",
        tracing=Tracing(),
    )
    result = await agent.run("bad")
    assert result.guardrail is not None
    assert provider.call_count == 1
    spans = spans_of(agent)
    (chat,) = by_op(spans, "chat")
    assert [e["name"] for e in chat.events] == [sc.EVENT_DISCARDED]
    (guard,) = named(spans, sc.SPAN_GUARDRAIL)
    assert sc.EVENT_DISCARDED not in [e["name"] for e in guard.events]
    assert root(spans).attributes[sc.LOGPOSE_RUN_OUTCOME] == "blocked"


async def test_parallel_block_while_turn_in_flight_still_records_chat() -> None:
    async def slow_block(inp: GuardrailInput) -> GuardrailVerdict:
        await asyncio.sleep(0.02)
        return GuardrailVerdict.block("unsafe", "No.")

    provider = FakeProvider(ScriptedTurn.text("slow", delay=5))
    agent = Agent(
        provider,
        model="m",
        input_guardrails=[slow_block],
        guardrail_mode="parallel",
        tracing=Tracing(),
    )
    await agent.run("bad")
    (chat,) = by_op(spans_of(agent), "chat")
    assert [e["name"] for e in chat.events] == [sc.EVENT_DISCARDED]


async def test_cancelled_and_max_iterations_outcomes() -> None:
    agent = Agent(
        FakeProvider(ScriptedTurn.tool_use(tool_call("add", {"a": 1, "b": 1})), repeat_last=True),
        model="m",
        tools=[add],
        max_iterations=2,
        tracing=Tracing(),
    )
    with pytest.raises(LogposeError):
        await agent.run("loop")
    top = root(spans_of(agent))
    assert top.attributes[sc.LOGPOSE_RUN_OUTCOME] == "max_iterations"
    assert top.attributes[sc.LOGPOSE_RUN_ITERATIONS] == 2
    assert top.status_code == ERROR

    agent = Agent(
        FakeProvider(ScriptedTurn.text("a", chunks=["a", "b"])), model="m", tracing=Tracing()
    )
    stream = agent.stream("hi")
    await stream.__anext__()
    await stream.aclose()  # type: ignore[attr-defined]
    spans = spans_of(agent)
    assert root(spans).attributes[sc.LOGPOSE_RUN_OUTCOME] == "cancelled"
    (chat,) = by_op(spans, "chat")
    assert chat.end_ns >= chat.start_ns


def test_message_conversion_shapes() -> None:
    from logpose.messages import ToolResultBlock, ToolUseBlock
    from logpose.tracing.spans import _messages_json

    messages = [
        Message(role="user", content=[TextBlock(text="q")]),
        Message(role="assistant", content=[ToolUseBlock(id="t", name="add", input={"a": 1})]),
        Message(role="user", content=[ToolResultBlock(tool_use_id="t", content="2")]),
    ]
    assert json.loads(_messages_json(messages)) == [
        {"role": "user", "parts": [{"type": "text", "content": "q"}]},
        {
            "role": "assistant",
            "parts": [{"type": "tool_call", "id": "t", "name": "add", "arguments": {"a": 1}}],
        },
        {
            "role": "tool",
            "parts": [{"type": "tool_call_response", "id": "t", "response": "2"}],
        },
    ]


# -- fix round 1 ------------------------------------------------------------


async def test_concurrent_duplicate_run_id_runs_once(db: Path) -> None:
    provider = FakeProvider(ScriptedTurn.text("ok", delay=0.05), repeat_last=True)
    agent = Agent(provider, model="m", tracing=None)
    outcomes = await asyncio.gather(
        agent.run("a", session_id="s1", run_id="dup"),
        agent.run("b", session_id="s2", run_id="dup"),
        return_exceptions=True,
    )
    kinds = sorted(type(o).__name__ for o in outcomes)
    assert kinds == ["LogposeError", "RunResult"]
    assert provider.call_count == 1
    # Completed IDs stay rejected.
    with pytest.raises(LogposeError, match="run_id"):
        await agent.run("c", session_id="s3", run_id="dup")
    assert provider.call_count == 1
    await agent.aclose()


async def test_concurrent_duplicate_run_id_rejected_with_tracing_only() -> None:
    provider = FakeProvider(ScriptedTurn.text("ok", delay=0.05), repeat_last=True)
    agent = Agent(provider, model="m", tracing=Tracing())
    other = Agent(provider, model="m", tracing=Tracing())
    outcomes = await asyncio.gather(
        agent.run("a", run_id="dup-t"), other.run("b", run_id="dup-t"), return_exceptions=True
    )
    assert sorted(type(o).__name__ for o in outcomes) == ["LogposeError", "RunResult"]
    assert provider.call_count == 1


async def test_run_id_reservation_released_when_run_fails_early(db: Path) -> None:
    provider = FakeProvider([ScriptedTurn.text("ok")])
    agent = Agent(provider, model="m", tracing=Tracing())
    with pytest.raises(LogposeError, match="Nothing to send"):
        await agent.run(session_id="empty", run_id="never-ran")
    from logpose.agent import _ACTIVE_RUN_IDS

    assert "never-ran" not in _ACTIVE_RUN_IDS
    await agent.run("go", session_id="s1", run_id="never-ran")
    assert "never-ran" not in _ACTIVE_RUN_IDS
    await agent.aclose()


async def test_record_run_duplicate_maps_to_logpose_error(tmp_path: Path) -> None:
    from logpose.store.base import RunRecord
    from logpose.store.sqlite import SQLiteStore

    record = RunRecord(
        run_id="r1",
        session_id="s1",
        model="m",
        provider="p",
        started_at_ms=1,
        ended_at_ms=2,
        outcome="completed",
        iterations=1,
        input_tokens=0,
        output_tokens=0,
        reasoning_tokens=0,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
    )
    async with SQLiteStore(tmp_path / "r.db") as store:
        async with store.session("s1"):
            pass
        await store.record_run(record)
        with pytest.raises(LogposeError, match="run_id"):
            await store.record_run(record)


async def test_all_content_shares_one_truncation_rule() -> None:
    cap = 60
    provider = FakeProvider(ScriptedTurn.failure(ProviderError("E" * 500 + "TAIL")))
    agent = Agent(provider, model="m", tracing=Tracing(capture_content=True, content_max_chars=cap))
    with pytest.raises(ProviderError):
        await agent.run("hi")
    (chat,) = by_op(spans_of(agent), "chat")
    assert chat.status_message is not None
    assert len(chat.status_message) <= cap
    assert chat.status_message.startswith("…[truncated ")
    assert chat.status_message.endswith("TAIL")
    assert chat.attributes[sc.LOGPOSE_CONTENT_TRUNCATED] is True

    provider = FakeProvider(
        [ScriptedTurn.tool_use(tool_call("add", {"a": 1, "b": 1}, id="c1")), ScriptedTurn.text("x")]
    )
    agent = Agent(
        provider,
        model="m",
        tools=[add],
        on_tool_call=lambda call: "D" * 500 + "GATE-END",
        tracing=Tracing(capture_content=True, content_max_chars=cap),
    )
    await agent.run("go")
    (gated,) = by_op(spans_of(agent), "execute_tool")
    replacement = gated.events[0]["attributes"][sc.LOGPOSE_GATE_REPLACEMENT]
    assert len(replacement) <= cap
    assert replacement.startswith("…[truncated ")
    assert replacement.endswith("GATE-END")
    assert gated.attributes[sc.LOGPOSE_CONTENT_TRUNCATED] is True


async def test_cap_smaller_than_marker_never_exceeds_cap() -> None:
    agent = Agent(
        FakeProvider(ScriptedTurn.text("y" * 500)),
        model="m",
        tracing=Tracing(capture_content=True, content_max_chars=5),
    )
    await agent.run("a much longer prompt")
    for span in spans_of(agent):
        for key in (sc.GEN_AI_INPUT_MESSAGES, sc.GEN_AI_OUTPUT_MESSAGES):
            if key in span.attributes:
                assert len(span.attributes[key]) <= 5


async def test_tool_span_errors_when_observer_intent_fails() -> None:
    async def observer(observation: Any) -> None:
        raise RuntimeError("journal down")

    provider = FakeProvider(
        [ScriptedTurn.tool_use(tool_call("add", {"a": 1, "b": 1}, id="c1")), ScriptedTurn.text("x")]
    )
    agent = Agent(provider, model="m", tools=[add], tool_observer=observer, tracing=Tracing())
    with pytest.raises(LogposeError):
        await agent.run("go")
    (tool_span,) = by_op(spans_of(agent), "execute_tool")
    assert tool_span.status_code == ERROR
    assert tool_span.attributes[sc.ERROR_TYPE] == "ToolObservationError"


def test_run_metadata_is_bounded() -> None:
    agent = Agent(FakeProvider(), model="m")
    for meta in (
        {f"k{i}": i for i in range(65)},
        {"k" * 129: 1},
        {"k": "v" * 1025},
        {"k": float("nan")},
        {"k": float("inf")},
    ):
        with pytest.raises(LogposeError):
            agent.stream("hi", metadata=meta)
    agent.stream("hi", metadata={f"k{i}": "v" * 1024 for i in range(64)})
    agent.stream("hi", metadata={"k" * 128: 1.5})


async def test_model_error_is_not_logged_as_tracing_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class NoModel(Agent):
        def _model(self) -> str:
            raise LogposeError("no model could be resolved")

    agent = NoModel(FakeProvider(ScriptedTurn.text("x")), model="m", tracing=Tracing())
    with caplog.at_level(logging.WARNING, logger="logpose.trace"):
        with pytest.raises(LogposeError, match="no model"):
            await agent.run("hi")
    assert not [r for r in caplog.records if r.name == "logpose.trace"]


# -- fix round 2 ------------------------------------------------------------


async def _close_then_cancel(stream: Any, delay: float) -> None:
    closer = asyncio.ensure_future(stream.aclose())
    await asyncio.sleep(delay)
    closer.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await closer


async def test_session_run_id_released_when_closer_cancelled_during_exit(
    db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from logpose.agent import _ACTIVE_RUN_IDS
    from logpose.store.sqlite import SQLiteStore

    original = SQLiteStore.record_run
    writing = asyncio.Event()

    async def slow_record(self: SQLiteStore, record: Any) -> None:
        writing.set()
        await asyncio.sleep(0.2)
        await original(self, record)

    monkeypatch.setattr(SQLiteStore, "record_run", slow_record)
    provider = FakeProvider(ScriptedTurn.text("ab", chunks=["a", "b"]))
    agent = Agent(provider, model="m", tracing=None)
    stream = agent.stream("hi", session_id="s1", run_id="x1")
    await stream.__anext__()
    closer = asyncio.ensure_future(stream.aclose())  # type: ignore[attr-defined]
    await writing.wait()
    assert "x1" in _ACTIVE_RUN_IDS  # held while the record is being written
    closer.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await closer
    await asyncio.sleep(0)
    assert "x1" not in _ACTIVE_RUN_IDS
    assert _rows(db, "SELECT run_id FROM logpose_runs") == [("x1",)]
    await agent.aclose()


class SlowCloseProvider(FakeProvider):
    async def stream(self, req: Any) -> Any:  # type: ignore[override]
        self.requests.append(req)
        try:
            yield ProviderTextDelta(text="a")
            yield ProviderTextDelta(text="b")
        finally:
            await asyncio.sleep(0.5)


async def test_traced_run_id_released_when_closer_cancelled_during_close() -> None:
    from logpose.agent import _ACTIVE_RUN_IDS

    agent = Agent(SlowCloseProvider(), model="m", tracing=Tracing())
    stream = agent.stream("hi", run_id="x2")
    await stream.__anext__()
    assert "x2" in _ACTIVE_RUN_IDS
    await _close_then_cancel(stream, 0.05)
    assert "x2" not in _ACTIVE_RUN_IDS


async def test_system_prompt_hash_only_with_a_system_prompt() -> None:
    cfg = sc.LOGPOSE_CONFIG_PREFIX + "system_prompt_sha256"
    plain = Agent(FakeProvider(ScriptedTurn.text("ok")), model="m", tracing=Tracing())
    await plain.run("hi")
    assert cfg not in root(spans_of(plain)).attributes
    prompted = Agent(
        FakeProvider(ScriptedTurn.text("ok")), model="m", system="Be terse.", tracing=Tracing()
    )
    await prompted.run("hi")
    assert len(root(spans_of(prompted)).attributes[cfg]) == 64
