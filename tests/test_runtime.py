"""Tests for metadata-only runtime context, logging, and observers."""

from __future__ import annotations

import asyncio
import logging

from logpose import Agent, RuntimeContext, RuntimeEvent, current_runtime_context, tool
from tests.fake_provider import FakeProvider, ScriptedTurn, tool_call


async def test_runtime_context_reaches_async_and_sync_tools_and_is_restored() -> None:
    seen: list[RuntimeContext | None] = []

    @tool
    async def async_context() -> str:
        """Record the runtime context."""
        seen.append(current_runtime_context())
        return "async result"

    @tool
    def sync_context() -> str:
        """Record the runtime context from a worker thread."""
        seen.append(current_runtime_context())
        return "sync result"

    provider = FakeProvider(
        [
            ScriptedTurn.tool_use(
                tool_call("async_context", id="a"), tool_call("sync_context", id="s")
            ),
            ScriptedTurn.text("done"),
        ]
    )

    assert current_runtime_context() is None
    await Agent(provider, tools=[async_context, sync_context]).run("run prompt")
    assert current_runtime_context() is None
    assert len(seen) == 2
    assert all(context is not None for context in seen)
    contexts = [context for context in seen if context is not None]
    assert {context.tool_name for context in contexts} == {"async_context", "sync_context"}
    assert {context.tool_call_id for context in contexts} == {"a", "s"}
    assert len({context.run_id for context in contexts}) == 1
    assert len({context.turn_id for context in contexts}) == 1
    assert all(context.provider == "fake" and context.model == "fake-model" for context in contexts)


async def test_runtime_context_is_isolated_between_concurrent_runs() -> None:
    seen: list[RuntimeContext | None] = []

    @tool
    async def capture() -> str:
        """Capture this run's context after yielding to a concurrent task."""
        await asyncio.sleep(0)
        seen.append(current_runtime_context())
        return "ok"

    def make_agent() -> Agent:
        return Agent(
            FakeProvider([ScriptedTurn.tool_use(tool_call("capture")), ScriptedTurn.text("done")]),
            tools=[capture],
        )

    await asyncio.gather(make_agent().run("one"), make_agent().run("two"))

    contexts = [context for context in seen if context is not None]
    assert len(contexts) == 2
    assert len({context.run_id for context in contexts}) == 2
    assert current_runtime_context() is None


async def test_observers_receive_ordered_safe_events_and_are_isolated() -> None:
    received: list[RuntimeEvent] = []
    later: list[str] = []
    observer_contexts: list[RuntimeContext | None] = []

    def first(event: RuntimeEvent) -> None:
        received.append(event)
        observer_contexts.append(current_runtime_context())

    def broken(event: RuntimeEvent) -> None:
        raise RuntimeError("observer secret must not escape")

    def final(event: RuntimeEvent) -> None:
        later.append(event.name)

    secret = "PROMPT_SECRET_TOOL_ARGUMENT_RESULT"

    @tool
    def reveal(value: str) -> str:
        """Return a deliberately confidential-looking result."""
        return value

    provider = FakeProvider(
        [
            ScriptedTurn.tool_use(tool_call("reveal", {"value": secret}, id="tool-secret")),
            ScriptedTurn.text("final secret response"),
        ]
    )
    result = await Agent(provider, tools=[reveal], observers=[first, broken, final]).run(secret)

    assert result.text == "final secret response"
    names = [event.name for event in received]
    assert names == later
    assert names[0] == "run.started"
    assert names[-1] == "run.completed"
    assert {"tool.queued", "tool.started", "tool.completed"} <= set(names)
    assert all(event.schema_version == 1 for event in received)
    assert [context.run_id if context is not None else None for context in observer_contexts] == [
        event.run_id for event in received
    ]
    assert all(secret not in repr(event) for event in received)
    assert all("final secret response" not in repr(event) for event in received)
    assert all(event.__dict__.get("tool_name") != secret for event in received)


async def test_runtime_records_are_published_through_standard_logging(
    caplog: object,
) -> None:
    secret = "DO_NOT_LOG_THIS"
    with caplog.at_level(logging.DEBUG, logger="logpose.runtime"):  # type: ignore[attr-defined]
        await Agent(FakeProvider([ScriptedTurn.text("model " + secret)])).run("prompt " + secret)

    records = [
        record
        for record in caplog.records  # type: ignore[attr-defined]
        if record.name == "logpose.runtime"
    ]
    assert [record.getMessage() for record in records] == [
        "run.started",
        "provider.attempt.started",
        "provider.attempt.completed",
        "run.completed",
    ]
    assert all(record.logpose_run_id.startswith("run_") for record in records)  # type: ignore[attr-defined]
    assert [record.logpose_event for record in records] == [  # type: ignore[attr-defined]
        record.getMessage() for record in records
    ]
    assert all(secret not in record.getMessage() for record in records)
    assert all(not hasattr(record, "logpose_prompt") for record in records)


async def test_broken_logging_handlers_are_isolated() -> None:
    class BrokenHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            raise RuntimeError("logging sink is unavailable")

    logger = logging.getLogger("logpose.runtime")
    handler = BrokenHandler()
    logger.addHandler(handler)
    try:
        result = await Agent(FakeProvider([ScriptedTurn.text("still succeeds")])).run("go")
    finally:
        logger.removeHandler(handler)

    assert result.text == "still succeeds"
