"""Tests for the synchronous facade.

The async loop is already covered by ``tests/test_agent.py``; nothing here
re-tests loop semantics. What is tested is the bridge itself:

* results, events, and exceptions cross the thread boundary intact;
* one private loop is reused across calls on the same agent (a fresh loop per
  call would break any object that binds itself to a loop, such as the
  ``asyncio.Lock`` inside ``CredentialProvider``);
* breaking out of ``stream_sync`` closes the async generator, which cancels
  in-flight tools and closes the provider stream;
* calling any sync entry point from a running event loop is refused;
* closing releases the thread, and a later call transparently starts a new one.
"""

from __future__ import annotations

import asyncio
import contextlib
import gc
import threading
import time
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest

from logpose import (
    Agent,
    Conversation,
    LogposeError,
    MaxIterationsError,
    Message,
    ProviderError,
    RunEnd,
    RunResult,
    SyncAgent,
    TextDelta,
    ToolCall,
    ToolResult,
    TurnEnd,
    Usage,
    close_sync,
    run_sync,
    stream_sync,
    tool,
)
from logpose.providers.base import CompletionRequest, Provider, ProviderEvent
from logpose.sync import _RUNNERS
from tests.fake_provider import FakeProvider, ScriptedTurn, tool_call

# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------


@tool
def add(a: int, b: int) -> str:
    """Add two numbers.

    Args:
        a: First addend.
        b: Second addend.
    """
    return str(a + b)


@tool
async def slow_echo(text: str) -> str:
    """Echo text after a short await.

    Args:
        text: What to echo.
    """
    await asyncio.sleep(0.01)
    return text


class LoopProbeProvider:
    """Wraps a provider and records the loop each turn ran on."""

    name = "loop-probe"
    model_default = "loop-probe-1"

    def __init__(self, inner: Provider) -> None:
        self.inner = inner
        self.loops: list[asyncio.AbstractEventLoop] = []
        self.threads: list[int] = []

    async def stream(self, req: CompletionRequest) -> AsyncIterator[ProviderEvent]:
        self.loops.append(asyncio.get_running_loop())
        self.threads.append(threading.get_ident())
        async for event in self.inner.stream(req):
            yield event


class BlockingToolProvider:
    """Asks for one tool forever; used to test cancellation of in-flight tools."""

    name = "blocking"
    model_default = "blocking-1"

    def __init__(self) -> None:
        self.calls = 0

    async def stream(self, req: CompletionRequest) -> AsyncIterator[ProviderEvent]:
        self.calls += 1
        turn = ScriptedTurn.tool_use(tool_call("block", {}), text="working")
        for delta in turn.deltas:
            yield delta
        yield turn.completion()


def _fresh_agent(*turns: ScriptedTurn, **kwargs: Any) -> tuple[Agent, FakeProvider]:
    """Build an agent over a scripted FakeProvider."""
    provider = FakeProvider(list(turns))
    return Agent(provider, **kwargs), provider


@contextlib.contextmanager
def _registered(name: str, factory: Any) -> Iterator[None]:
    """Temporarily register a provider factory, restoring the registry after."""
    from logpose.providers import _REGISTRY, register

    previous = _REGISTRY.get(name)
    register(name, factory)
    try:
        yield
    finally:
        if previous is None:
            _REGISTRY.pop(name, None)
        else:  # pragma: no cover - only if a real provider ever takes this name
            register(name, previous)


@pytest.fixture(autouse=True)
def _no_leaked_runners() -> Any:
    """Fail a test that leaves a private loop thread behind."""
    before = threading.active_count()
    yield
    gc.collect()
    deadline = time.monotonic() + 5.0
    while threading.active_count() > before and time.monotonic() < deadline:
        time.sleep(0.01)
    leaked = [t.name for t in threading.enumerate() if t.name.startswith("logpose-sync")]
    assert not leaked, f"sync loop threads leaked: {leaked}"


# ---------------------------------------------------------------------------
# run_sync
# ---------------------------------------------------------------------------


def test_run_sync_returns_the_result() -> None:
    agent, provider = _fresh_agent(ScriptedTurn.text("hello there"))
    try:
        result = run_sync(agent, "hi")
    finally:
        close_sync(agent)

    assert isinstance(result, RunResult)
    assert result.text == "hello there"
    assert result.iterations == 1
    assert provider.call_count == 1
    assert provider.last_request.messages[0].text == "hi"


def test_run_sync_executes_tools_and_aggregates_usage() -> None:
    agent, provider = _fresh_agent(
        ScriptedTurn.tool_use(
            tool_call("add", {"a": 2, "b": 3}, id="t1"), usage=Usage(input_tokens=5)
        ),
        ScriptedTurn.text("5", usage=Usage(output_tokens=7)),
        tools=[add],
    )
    try:
        result = run_sync(agent, "2+3?")
    finally:
        close_sync(agent)

    assert result.text == "5"
    assert result.iterations == 2
    assert result.usage.input_tokens == 5
    assert result.usage.output_tokens == 7
    tool_results = provider.requests[1].messages[-1].content
    assert tool_results[0].content == "5"  # type: ignore[union-attr]
    assert tool_results[0].is_error is False  # type: ignore[union-attr]


def test_run_sync_runs_async_tool_handlers() -> None:
    agent, _ = _fresh_agent(
        ScriptedTurn.tool_use(tool_call("slow_echo", {"text": "pong"}, id="t1")),
        ScriptedTurn.text("pong"),
        tools=[slow_echo],
    )
    try:
        assert run_sync(agent, "ping").text == "pong"
    finally:
        close_sync(agent)


def test_run_sync_accepts_a_message_and_a_conversation() -> None:
    agent, provider = _fresh_agent(
        ScriptedTurn.text("first"),
        ScriptedTurn.text("second"),
    )
    conversation = Conversation()
    try:
        run_sync(agent, Message.user("one"), conversation=conversation)
        result = run_sync(agent, "two", conversation=conversation)
    finally:
        close_sync(agent)

    assert result.text == "second"
    assert [m.text for m in conversation] == ["one", "first", "two", "second"]
    assert len(provider.requests[1].messages) == 3


def test_run_sync_propagates_provider_errors() -> None:
    agent, _ = _fresh_agent(ScriptedTurn.failure(ProviderError("boom", status_code=500)))
    try:
        with pytest.raises(ProviderError) as excinfo:
            run_sync(agent, "hi")
    finally:
        close_sync(agent)
    assert excinfo.value.status_code == 500


def test_run_sync_propagates_max_iterations_with_partial_history() -> None:
    provider = FakeProvider(
        [ScriptedTurn.tool_use(tool_call("add", {"a": 1, "b": 1}))],
        repeat_last=True,
    )
    agent = Agent(provider, tools=[add], max_iterations=2)
    try:
        with pytest.raises(MaxIterationsError) as excinfo:
            run_sync(agent, "loop forever")
    finally:
        close_sync(agent)
    assert excinfo.value.max_iterations == 2
    assert excinfo.value.messages


def test_run_sync_rejects_a_blank_prompt() -> None:
    agent, _ = _fresh_agent(ScriptedTurn.text("unused"))
    try:
        with pytest.raises(LogposeError, match="blank"):
            run_sync(agent, "   ")
    finally:
        close_sync(agent)


def test_run_sync_reuses_one_loop_across_calls() -> None:
    """A fresh loop per call would break loop-bound objects on the second call."""
    probe = LoopProbeProvider(
        FakeProvider([ScriptedTurn.text("one"), ScriptedTurn.text("two")]),
    )
    agent = Agent(probe)
    try:
        run_sync(agent, "a")
        run_sync(agent, "b")
    finally:
        close_sync(agent)

    assert len(probe.loops) == 2
    assert probe.loops[0] is probe.loops[1]
    assert probe.threads[0] == probe.threads[1]
    assert probe.threads[0] != threading.get_ident()


def test_each_agent_gets_its_own_loop() -> None:
    probe_a = LoopProbeProvider(FakeProvider([ScriptedTurn.text("a")]))
    probe_b = LoopProbeProvider(FakeProvider([ScriptedTurn.text("b")]))
    agent_a, agent_b = Agent(probe_a), Agent(probe_b)
    try:
        run_sync(agent_a, "x")
        run_sync(agent_b, "y")
    finally:
        close_sync(agent_a)
        close_sync(agent_b)

    assert probe_a.loops[0] is not probe_b.loops[0]


# ---------------------------------------------------------------------------
# stream_sync
# ---------------------------------------------------------------------------


def test_stream_sync_yields_events_in_order_and_ends_with_run_end() -> None:
    agent, _ = _fresh_agent(
        ScriptedTurn.tool_use(tool_call("add", {"a": 1, "b": 2}, id="t1"), text="let me add"),
        ScriptedTurn.text("3", chunks=["3"]),
        tools=[add],
    )
    try:
        events = list(stream_sync(agent, "1+2?"))
    finally:
        close_sync(agent)

    kinds = [type(event).__name__ for event in events]
    assert kinds == [
        "TextDelta",
        "TurnEnd",
        "ToolCall",
        "ToolResult",
        "TextDelta",
        "TurnEnd",
        "RunEnd",
    ]
    assert isinstance(events[0], TextDelta) and events[0].text == "let me add"
    assert isinstance(events[2], ToolCall) and events[2].name == "add"
    assert isinstance(events[3], ToolResult) and events[3].content == "3"
    assert isinstance(events[5], TurnEnd)
    assert isinstance(events[-1], RunEnd) and events[-1].result.text == "3"


def test_stream_sync_is_lazy_until_iterated() -> None:
    agent, provider = _fresh_agent(ScriptedTurn.text("hi"))
    try:
        events = stream_sync(agent, "hello")
        assert provider.call_count == 0
        first = next(events)
        assert provider.call_count == 1
        assert isinstance(first, TextDelta)
        events.close()
    finally:
        close_sync(agent)


def test_stream_sync_validates_eagerly() -> None:
    agent, provider = _fresh_agent(ScriptedTurn.text("unused"))
    try:
        with pytest.raises(LogposeError, match="must be a str or a Message"):
            stream_sync(agent, 42)  # type: ignore[arg-type]
        with pytest.raises(LogposeError, match="Nothing to send"):
            stream_sync(agent)
    finally:
        close_sync(agent)
    assert provider.call_count == 0


def test_stream_sync_propagates_exceptions_at_the_for_statement() -> None:
    agent, _ = _fresh_agent(ScriptedTurn.failure(RuntimeError("upstream exploded")))
    try:
        with pytest.raises(RuntimeError, match="upstream exploded"):
            list(stream_sync(agent, "hi"))
    finally:
        close_sync(agent)


def test_early_break_closes_the_provider_stream() -> None:
    agent, provider = _fresh_agent(
        ScriptedTurn.text("a long answer", chunks=["a ", "long ", "answer"]),
    )
    try:
        events = stream_sync(agent, "hi")
        for _event in events:
            break
        events.close()
        assert provider.closed == 1
    finally:
        close_sync(agent)


def test_early_break_cleans_up_when_the_generator_is_collected() -> None:
    agent, provider = _fresh_agent(
        ScriptedTurn.text("a long answer", chunks=["a ", "long ", "answer"]),
    )
    try:
        for _event in stream_sync(agent, "hi"):
            break  # the generator is a temporary; CPython collects it here
        gc.collect()
        assert provider.closed == 1
    finally:
        close_sync(agent)


def test_breaking_at_a_tool_call_closes_without_running_the_tool() -> None:
    """The loop yields ToolCall *before* invoking; abandoning there must not hang."""
    ran = threading.Event()

    @tool
    async def block() -> str:
        """Block for a long time if it ever starts."""
        ran.set()
        await asyncio.sleep(30)
        return "never"  # pragma: no cover

    agent = Agent(BlockingToolProvider(), tools=[block])
    try:
        events = stream_sync(agent, "go")
        seen: list[Any] = []
        while not seen or not isinstance(seen[-1], ToolCall):
            seen.append(next(events))

        closer = threading.Thread(target=events.close, daemon=True)
        closer.start()
        closer.join(5.0)
        assert not closer.is_alive(), "closing the stream hung"
        assert not ran.is_set(), "the tool started even though the stream was abandoned"
    finally:
        close_sync(agent)


def test_iterating_after_close_raises_a_clear_error() -> None:
    agent, _ = _fresh_agent(
        ScriptedTurn.text("a long answer", chunks=["a ", "long ", "answer"]),
    )
    events = stream_sync(agent, "hi")
    assert isinstance(next(events), TextDelta)
    close_sync(agent)
    with pytest.raises(LogposeError, match="sync runtime has been closed"):
        next(events)
    events.close()


def test_stream_sync_after_close_starts_a_new_loop() -> None:
    probe = LoopProbeProvider(FakeProvider([ScriptedTurn.text("a"), ScriptedTurn.text("b")]))
    agent = Agent(probe)
    try:
        run_sync(agent, "one")
        close_sync(agent)
        run_sync(agent, "two")
    finally:
        close_sync(agent)

    assert probe.loops[0] is not probe.loops[1]


# ---------------------------------------------------------------------------
# SyncAgent
# ---------------------------------------------------------------------------


def test_sync_agent_run_and_stream() -> None:
    agent, _ = _fresh_agent(ScriptedTurn.text("one"), ScriptedTurn.text("two"))
    with SyncAgent(agent) as sync_agent:
        assert sync_agent.run("a").text == "one"
        assert [type(e).__name__ for e in sync_agent.stream("b")][-1] == "RunEnd"
        assert sync_agent.provider_name == "fake"
        assert "Agent(provider='fake'" in repr(sync_agent)


def test_sync_agent_shares_the_runner_with_the_free_functions() -> None:
    probe = LoopProbeProvider(FakeProvider([ScriptedTurn.text("a"), ScriptedTurn.text("b")]))
    agent = Agent(probe)
    sync_agent = SyncAgent(agent)
    try:
        run_sync(agent, "one")
        sync_agent.run("two")
    finally:
        sync_agent.close()

    assert probe.loops[0] is probe.loops[1]


def test_sync_agent_rejects_a_non_agent() -> None:
    with pytest.raises(LogposeError, match="SyncAgent wraps an Agent"):
        SyncAgent("anthropic")  # type: ignore[arg-type]


def test_close_is_idempotent_and_safe_on_an_unused_agent() -> None:
    agent, _ = _fresh_agent(ScriptedTurn.text("unused"))
    close_sync(agent)  # never ran
    sync_agent = SyncAgent(agent)
    sync_agent.close()
    sync_agent.close()


def test_close_closes_a_provider_the_agent_built() -> None:
    closed: list[str] = []

    class ClosableProvider(FakeProvider):
        async def aclose(self) -> None:
            closed.append("yes")

    provider = ClosableProvider([ScriptedTurn.text("hi")])
    with _registered("closable-test", lambda **kwargs: provider):
        agent = Agent("closable-test")
        with SyncAgent(agent) as sync_agent:
            sync_agent.run("hello")
    assert closed == ["yes"]


def test_close_leaves_an_injected_provider_alone() -> None:
    closed: list[str] = []

    class ClosableProvider(FakeProvider):
        async def aclose(self) -> None:
            closed.append("yes")

    agent = Agent(ClosableProvider([ScriptedTurn.text("hi")]))
    with SyncAgent(agent) as sync_agent:
        sync_agent.run("hello")
    assert closed == []


def test_context_manager_does_not_swallow_exceptions() -> None:
    agent, _ = _fresh_agent(ScriptedTurn.text("hi"))
    with pytest.raises(ValueError, match="from the body"):
        with SyncAgent(agent) as sync_agent:
            sync_agent.run("hello")
            raise ValueError("from the body")


# ---------------------------------------------------------------------------
# running-loop rejection
# ---------------------------------------------------------------------------


async def test_run_sync_from_a_running_loop_is_refused() -> None:
    agent, provider = _fresh_agent(ScriptedTurn.text("unused"))
    with pytest.raises(LogposeError) as excinfo:
        run_sync(agent, "hi")
    assert "running event loop" in str(excinfo.value)
    assert "await agent.run" in str(excinfo.value)
    assert provider.call_count == 0


async def test_stream_sync_from_a_running_loop_is_refused() -> None:
    agent, provider = _fresh_agent(ScriptedTurn.text("unused"))
    with pytest.raises(LogposeError, match="running event loop"):
        stream_sync(agent, "hi")
    assert provider.call_count == 0


async def test_sync_agent_methods_from_a_running_loop_are_refused() -> None:
    agent, _ = _fresh_agent(ScriptedTurn.text("unused"))
    sync_agent = SyncAgent(agent)
    with pytest.raises(LogposeError, match="running event loop"):
        sync_agent.run("hi")
    with pytest.raises(LogposeError, match="running event loop"):
        sync_agent.stream("hi")
    with pytest.raises(LogposeError, match="running event loop"):
        sync_agent.close()


async def test_the_async_api_still_works_normally() -> None:
    """The sync facade must not disturb the async path it wraps."""
    agent, _ = _fresh_agent(ScriptedTurn.text("async answer"))
    result = await agent.run("hi")
    assert result.text == "async answer"
    assert agent not in _RUNNERS


# ---------------------------------------------------------------------------
# lifecycle
# ---------------------------------------------------------------------------


def test_runner_is_registered_per_agent_and_dropped_on_close() -> None:
    agent, _ = _fresh_agent(ScriptedTurn.text("hi"))
    run_sync(agent, "hello")
    assert agent in _RUNNERS
    close_sync(agent)
    assert agent not in _RUNNERS


def test_abandoning_an_agent_stops_its_loop_thread() -> None:
    agent, _ = _fresh_agent(ScriptedTurn.text("hi"))
    run_sync(agent, "hello")
    names = {t.name for t in threading.enumerate() if t.name.startswith("logpose-sync")}
    assert names

    del agent
    gc.collect()

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        alive = {t.name for t in threading.enumerate() if t.name.startswith("logpose-sync")}
        if not (names & alive):
            break
        time.sleep(0.01)
    alive = {t.name for t in threading.enumerate() if t.name.startswith("logpose-sync")}
    assert not (names & alive), "the private loop thread outlived its agent"


def test_sync_calls_do_not_run_on_the_calling_thread() -> None:
    seen: list[int] = []

    @tool
    def where() -> str:
        """Report the thread the handler ran on."""
        seen.append(threading.get_ident())
        return "ok"

    agent, _ = _fresh_agent(
        ScriptedTurn.tool_use(tool_call("where", {}, id="t1")),
        ScriptedTurn.text("done"),
        tools=[where],
    )
    try:
        run_sync(agent, "go")
    finally:
        close_sync(agent)

    assert seen and seen[0] != threading.get_ident()


def test_concurrent_callers_share_one_agent_safely() -> None:
    """A service will call one agent from several worker threads."""
    probe = LoopProbeProvider(
        FakeProvider([ScriptedTurn.text(f"answer {i}", delay=0.01) for i in range(6)]),
    )
    agent = Agent(probe)
    results: dict[int, str] = {}
    errors: list[BaseException] = []

    def worker(index: int) -> None:
        try:
            results[index] = run_sync(agent, f"question {index}").text
        except BaseException as exc:  # noqa: BLE001 - reported, not swallowed
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10.0)
    finally:
        close_sync(agent)

    assert not errors, errors
    assert len(results) == 6
    assert sorted(results.values()) == sorted(f"answer {i}" for i in range(6))
    assert len(set(map(id, probe.loops))) == 1
