"""Per-run ``extra=`` on ``run``/``stream`` and the sync facade."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from logpose import Agent, Conversation, tool
from logpose.errors import LogposeError
from logpose.guardrails import GuardrailInput, GuardrailVerdict, openai_decision_guard
from logpose.providers.openai_compat import OpenAICompatProvider
from logpose.sync import SyncAgent, close_sync, run_sync, stream_sync
from logpose.tracing import semconv as sc
from logpose.tracing.spans import SpanRecord, Tracing
from tests.fake_provider import FakeProvider, ScriptedTurn, tool_call

BASE = "http://test.local/v1"


@tool
async def add(a: int, b: int) -> str:
    """Add two numbers.

    Args:
        a: First.
        b: Second.
    """
    await asyncio.sleep(0)
    return str(a + b)


class Answer(BaseModel):
    value: int


def _tool_round() -> list[ScriptedTurn]:
    return [ScriptedTurn.tool_use(tool_call("add", {"a": 1, "b": 2})), ScriptedTurn.text("3")]


# ---------------------------------------------------------------------------
# merge and scope
# ---------------------------------------------------------------------------


async def test_run_extra_merges_over_agent_extra() -> None:
    provider = FakeProvider([ScriptedTurn.text("hi")])
    agent = Agent(provider, model="m", extra={"a": 1, "b": 2})
    await agent.run("hello", extra={"b": 3, "c": 4})
    assert provider.last_request.extra == {"a": 1, "b": 3, "c": 4}
    assert agent.extra == {"a": 1, "b": 2}


async def test_run_extra_applies_to_every_turn() -> None:
    provider = FakeProvider(_tool_round())
    agent = Agent(provider, model="m", tools=[add], extra={"a": 1})
    await agent.run("1+2?", extra={"c": 4})
    assert provider.call_count == 2
    assert [r.extra for r in provider.requests] == [{"a": 1, "c": 4}] * 2


async def test_run_extra_applies_to_correction_turns() -> None:
    provider = FakeProvider([ScriptedTurn.text("not json"), ScriptedTurn.text('{"value": 3}')])
    agent = Agent(provider, model="m")
    result = await agent.run("hi", output_type=Answer, extra={"c": 4})
    assert result.output == Answer(value=3)
    assert provider.call_count == 2
    assert all(r.extra.get("c") == 4 for r in provider.requests)


async def test_run_extra_does_not_leak() -> None:
    provider = FakeProvider([ScriptedTurn.text("hi")], repeat_last=True)
    agent = Agent(provider, model="m", extra={"a": 1, "b": 2})
    await agent.run("one", extra={"b": 3, "c": 4})
    await agent.run("two")
    assert provider.last_request.extra == {"a": 1, "b": 2}

    # Concurrent runs on one shared agent, each with a tool round so turns interleave.
    provider = FakeProvider(
        [ScriptedTurn.tool_use(tool_call("add", {"a": 1, "b": 2})), ScriptedTurn.text("3")] * 2
    )
    agent = Agent(provider, model="m", tools=[add], extra={"a": 1})
    await asyncio.gather(
        agent.run("run-x", extra={"who": "x"}),
        agent.run("run-y", extra={"who": "y"}),
    )
    assert provider.call_count == 4
    for request in provider.requests:
        prompt = request.messages[0].text
        assert request.extra == {"a": 1, "who": prompt.removeprefix("run-")}


@pytest.mark.parametrize("given", [None, {}])
async def test_no_extra_unchanged(given: dict[str, Any] | None) -> None:
    baseline = FakeProvider(_tool_round())
    await Agent(baseline, model="m", tools=[add], extra={"a": 1}).run("1+2?")
    provider = FakeProvider(_tool_round())
    await Agent(provider, model="m", tools=[add], extra={"a": 1}).run("1+2?", extra=given)
    assert [r.extra for r in provider.requests] == [r.extra for r in baseline.requests]
    assert provider.requests[0].extra == {"a": 1}

    bare = FakeProvider([ScriptedTurn.text("hi")])
    await Agent(bare, model="m").run("hello", extra=given)
    assert bare.last_request.extra == {}


async def test_run_extra_is_copied() -> None:
    provider = FakeProvider([ScriptedTurn.text("hi")])
    agent = Agent(provider, model="m")
    mine = {"c": 4}
    events = agent.stream("hello", extra=mine)
    mine["c"] = 5
    async for _ in events:
        pass
    assert provider.last_request.extra == {"c": 4}


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["model", "messages", "tools", "stream", "stream_options"])
async def test_run_extra_reserved_keys_rejected(key: str) -> None:
    provider = FakeProvider([ScriptedTurn.text("hi")])
    agent = Agent(provider, model="m")
    with pytest.raises(LogposeError, match=key):
        await agent.run("hello", extra={key: 1})
    with pytest.raises(LogposeError, match=key):
        agent.stream("hello", extra={key: 1})  # raises before iteration
    assert provider.call_count == 0


async def test_run_extra_bad_shapes_rejected() -> None:
    provider = FakeProvider([ScriptedTurn.text("hi")])
    agent = Agent(provider, model="m")
    with pytest.raises(LogposeError):
        await agent.run("hello", extra={1: "x"})  # type: ignore[dict-item]
    with pytest.raises(LogposeError):
        agent.stream("hello", extra={1: "x"})  # type: ignore[dict-item]
    with pytest.raises(LogposeError):
        await agent.run("hello", extra=[("a", 1)])  # type: ignore[arg-type]
    with pytest.raises(LogposeError):
        agent.stream("hello", extra="a=1")  # type: ignore[arg-type]
    assert provider.call_count == 0


async def test_run_extra_response_format_conflicts_with_output_type() -> None:
    fmt = {"type": "json_object"}
    provider = FakeProvider([ScriptedTurn.text("{}")], repeat_last=True)
    agent = Agent(provider, model="m")
    with pytest.raises(LogposeError, match="response_format"):
        await agent.run("hi", output_type=Answer, extra={"response_format": fmt})
    with pytest.raises(LogposeError, match="response_format"):
        agent.stream("hi", output_type=Answer, extra={"response_format": fmt})
    # The agent default output_type counts too; disabling it for the call lifts the conflict.
    typed = Agent(provider, model="m", output_type=Answer)
    with pytest.raises(LogposeError, match="response_format"):
        typed.stream("hi", extra={"response_format": fmt})
    assert provider.call_count == 0

    await agent.run("hi", extra={"response_format": fmt})
    assert provider.last_request.extra == {"response_format": fmt}
    await typed.run("hi", output_type=None, extra={"response_format": fmt})
    assert provider.last_request.extra == {"response_format": fmt}


# ---------------------------------------------------------------------------
# wire body
# ---------------------------------------------------------------------------


def _sse_stop() -> bytes:
    chunks = [
        {"choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    ]
    return "".join(f"data: {json.dumps(c)}\n\n" for c in chunks).encode() + b"data: [DONE]\n\n"


class _Recorder:
    def __init__(self) -> None:
        self.bodies: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.bodies.append(json.loads(request.content))
        return httpx.Response(200, content=_sse_stop())


def _wire_agent(recorder: _Recorder, **kwargs: Any) -> Agent:
    client = httpx.AsyncClient(transport=httpx.MockTransport(recorder))
    provider = OpenAICompatProvider(base_url=BASE, client=client)
    return Agent(provider, model="m", max_tokens=128, context_policy=None, **kwargs)


async def test_run_extra_ceiling_field_replaces_logposes() -> None:
    recorder = _Recorder()
    agent = _wire_agent(recorder)
    await agent.run("hi", extra={"max_completion_tokens": 50})
    assert recorder.bodies[-1]["max_completion_tokens"] == 50
    assert "max_tokens" not in recorder.bodies[-1]

    await agent.run("hi", extra={"max_tokens": 40})
    assert recorder.bodies[-1]["max_tokens"] == 40
    assert "max_completion_tokens" not in recorder.bodies[-1]

    await agent.run("hi")
    assert recorder.bodies[-1]["max_completion_tokens"] == 128
    assert "max_tokens" not in recorder.bodies[-1]


async def test_run_extra_reaches_the_wire_body() -> None:
    recorder = _Recorder()
    agent = _wire_agent(recorder, extra={"temperature": 0.2})
    await agent.run("hi", extra={"safety_identifier": "h", "prompt_cache_key": "k"})
    body = recorder.bodies[-1]
    assert body["safety_identifier"] == "h"
    assert body["prompt_cache_key"] == "k"
    assert body["temperature"] == 0.2


# ---------------------------------------------------------------------------
# summary call and guardrails
# ---------------------------------------------------------------------------


async def test_summary_call_gets_merged_extra_minus_turn_only() -> None:
    from tests.test_compaction import chat_history, is_summary_call, small

    provider = FakeProvider([ScriptedTurn.text("SUMMARY"), ScriptedTurn.text("answer")])
    agent = Agent(
        provider, model="m", tools=[small], context_budget=2_000, extra={"temperature": 0.1}
    )
    conv = Conversation(chat_history(10))
    await agent.run(
        "latest question",
        conversation=conv,
        extra={"safety_identifier": "h", "tool_choice": "none"},
    )
    summary_call, turn = provider.requests
    assert is_summary_call(summary_call)
    assert summary_call.extra == {"temperature": 0.1, "safety_identifier": "h"}
    assert turn.extra == {"temperature": 0.1, "safety_identifier": "h", "tool_choice": "none"}


async def test_guardrails_never_see_extra() -> None:
    guard_bodies: list[dict[str, Any]] = []

    def guard_handler(request: httpx.Request) -> httpx.Response:
        guard_bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": "gpt-6-luna",
                "answers": [
                    {"name": "on_topic", "type": "predicate", "probability": 0.9},
                    {"name": "unsafe", "type": "predicate", "probability": 0.1},
                ],
            },
        )

    decision = openai_decision_guard(
        "billing support",
        replies={"off_topic": "OFF", "unsafe": "UNSAFE"},
        api_key="sk-test",
        client=httpx.AsyncClient(transport=httpx.MockTransport(guard_handler)),
    )
    seen: list[GuardrailInput] = []

    async def recording_guard(inp: GuardrailInput) -> GuardrailVerdict:
        seen.append(inp)
        return GuardrailVerdict.allow()

    provider = FakeProvider([ScriptedTurn.text("hi")])
    agent = Agent(provider, model="m", input_guardrails=[decision, recording_guard])
    await agent.run("hello", extra={"safety_identifier": "SAFETY-ID-42"})

    assert provider.last_request.extra == {"safety_identifier": "SAFETY-ID-42"}
    (inp,) = seen
    assert set(vars(inp)) == {"message", "recent", "metadata", "run_id"}
    assert "SAFETY-ID-42" not in repr(inp)
    (body,) = guard_bodies
    assert "safety_identifier" not in body
    assert "SAFETY-ID-42" not in json.dumps(body)


# ---------------------------------------------------------------------------
# tracing
# ---------------------------------------------------------------------------


class _MemorySink:
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


@pytest.fixture
def memory_spans(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LOGPOSE_TRACE", raising=False)
    monkeypatch.delenv("LOGPOSE_TRACE_CONTENT", raising=False)

    async def open_sink(self: Agent) -> _MemorySink | None:
        if self.tracing is None:
            return None
        sink = self.__dict__.get("_test_sink")
        if sink is None:
            sink = self.__dict__["_test_sink"] = _MemorySink()
        return sink

    monkeypatch.setattr(Agent, "_open_trace_sink", open_sink)


def _spans(agent: Agent) -> list[SpanRecord]:
    return list(agent.__dict__["_test_sink"].spans)


def _root(spans: list[SpanRecord]) -> SpanRecord:
    (span,) = [
        s for s in spans if s.attributes.get(sc.GEN_AI_OPERATION_NAME) == sc.OPERATION_INVOKE_AGENT
    ]
    return span


@pytest.mark.usefixtures("memory_spans")
async def test_tracing_records_extra_keys_not_values() -> None:
    assert sc.LOGPOSE_REQUEST_EXTRA_KEYS == "logpose.request.extra_keys"
    secret = "SECRET-VALUE-123"
    provider = FakeProvider(_tool_round())
    agent = Agent(
        provider,
        model="m",
        tools=[add],
        extra={"temperature": 0.3},
        tracing=Tracing(capture_content=True),
    )
    await agent.run("1+2?", extra={"safety_identifier": secret})
    spans = _spans(agent)
    assert _root(spans).attributes[sc.LOGPOSE_REQUEST_EXTRA_KEYS] == [
        "safety_identifier",
        "temperature",
    ]
    assert provider.requests[0].extra["safety_identifier"] == secret
    for span in spans:
        dumped = json.dumps(
            [span.name, span.attributes, span.events, span.status_message], default=str
        )
        assert secret not in dumped

    # Only the keys of this run: the next run on the same agent has no per-run key.
    plain = FakeProvider([ScriptedTurn.text("hi")], repeat_last=True)
    agent = Agent(plain, model="m", tracing=Tracing(capture_content=True))
    await agent.run("hello")
    assert sc.LOGPOSE_REQUEST_EXTRA_KEYS not in _root(_spans(agent)).attributes
    await agent.run("hello", extra={})
    roots = [
        s
        for s in _spans(agent)
        if s.attributes.get(sc.GEN_AI_OPERATION_NAME) == sc.OPERATION_INVOKE_AGENT
    ]
    assert all(sc.LOGPOSE_REQUEST_EXTRA_KEYS not in r.attributes for r in roots)


async def test_session_runs_carry_extra_and_do_not_persist_it(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sqlite3

    from logpose.store import manager

    db = tmp_path / "s.db"
    monkeypatch.setenv("LOGPOSE_DB_PATH", str(db))
    manager.reset_stores_for_tests()
    secret = "PER-RUN-SECRET-77"
    try:
        provider = FakeProvider([ScriptedTurn.text("hi")], repeat_last=True)
        agent = Agent(provider, model="m", extra={"a": 1})
        await agent.run("one", session_id="s1", extra={"safety_identifier": secret})
        assert provider.last_request.extra == {"a": 1, "safety_identifier": secret}
        async with agent.session("s1") as handle:
            async for _ in agent.stream("two", session=handle, extra={"c": 2}):
                pass
        assert provider.last_request.extra == {"a": 1, "c": 2}
        await agent.run("three", session_id="s1")
        assert provider.last_request.extra == {"a": 1}
        await agent.aclose()
        with sqlite3.connect(db) as conn:
            tables = [
                r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            ]
            for table in tables:
                for row in conn.execute(f"SELECT * FROM {table}"):  # noqa: S608
                    assert secret not in repr(row)
    finally:
        manager.reset_stores_for_tests()


@pytest.mark.usefixtures("memory_spans")
@pytest.mark.parametrize("traced", [False, True])
async def test_mixed_key_agent_extra_still_runs(traced: bool) -> None:
    # Agent(extra=) is not validated; a non-str key must not break a run (traced or not).
    provider = FakeProvider([ScriptedTurn.text("hi")], repeat_last=True)
    tracing = Tracing() if traced else None
    agent = Agent(provider, model="m", extra={1: "x", "a": 2}, tracing=tracing)  # type: ignore[dict-item]
    result = await agent.run("hi")
    assert result.text == "hi"
    assert provider.last_request.extra == {1: "x", "a": 2}
    await agent.run("hi", extra={"b": 3})
    assert provider.last_request.extra == {1: "x", "a": 2, "b": 3}
    if traced:
        roots = [
            s
            for s in _spans(agent)
            if s.attributes.get(sc.GEN_AI_OPERATION_NAME) == sc.OPERATION_INVOKE_AGENT
        ]
        assert roots[-1].attributes[sc.LOGPOSE_REQUEST_EXTRA_KEYS] == ["1", "a", "b"]


async def test_context_length_retry_carries_merged_extra() -> None:
    from logpose.errors import ProviderError
    from tests.test_compaction import CONTEXT_ERROR, chat_history, is_summary_call

    rejection = ProviderError(CONTEXT_ERROR, status_code=400)
    provider = FakeProvider(
        [ScriptedTurn.failure(rejection), ScriptedTurn.text("S"), ScriptedTurn.text("ok")]
    )
    agent = Agent(provider, model="m", extra={"temperature": 0.1})
    conv = Conversation(chat_history(4))
    result = await agent.run(
        "latest", conversation=conv, extra={"safety_identifier": "h", "tool_choice": "auto"}
    )
    assert result.text == "ok"
    first, summary, retried = provider.requests
    merged = {"temperature": 0.1, "safety_identifier": "h", "tool_choice": "auto"}
    assert first.extra == merged
    assert retried.extra == merged
    assert is_summary_call(summary)
    assert summary.extra == {"temperature": 0.1, "safety_identifier": "h"}


async def test_parallel_guardrail_first_turn_carries_merged_extra() -> None:
    async def slow_allow(inp: GuardrailInput) -> GuardrailVerdict:
        await asyncio.sleep(0.05)
        return GuardrailVerdict.allow()

    provider = FakeProvider([ScriptedTurn.text("hi")])
    agent = Agent(
        provider,
        model="m",
        extra={"a": 1},
        input_guardrails=[slow_allow],
        guardrail_mode="parallel",
    )
    result = await agent.run("hello", extra={"c": 4})
    assert result.guardrail is None
    assert provider.call_count == 1
    assert provider.requests[0].extra == {"a": 1, "c": 4}


# ---------------------------------------------------------------------------
# sync facade
# ---------------------------------------------------------------------------


def test_sync_facade_forwards_extra() -> None:
    provider = FakeProvider([ScriptedTurn.text("hi")], repeat_last=True)
    agent = Agent(provider, model="m", extra={"a": 1})
    try:
        run_sync(agent, "one", extra={"c": "run_sync"})
        assert provider.last_request.extra == {"a": 1, "c": "run_sync"}
        list(stream_sync(agent, "two", extra={"c": "stream_sync"}))
        assert provider.last_request.extra == {"a": 1, "c": "stream_sync"}
        sync_agent = SyncAgent(agent)
        sync_agent.run("three", extra={"c": "SyncAgent.run"})
        assert provider.last_request.extra == {"a": 1, "c": "SyncAgent.run"}
        list(sync_agent.stream("four", extra={"c": "SyncAgent.stream"}))
        assert provider.last_request.extra == {"a": 1, "c": "SyncAgent.stream"}
        run_sync(agent, "five")
        assert provider.last_request.extra == {"a": 1}
        with pytest.raises(LogposeError, match="model"):
            stream_sync(agent, "six", extra={"model": "x"})  # eager, before next()
    finally:
        close_sync(agent)
