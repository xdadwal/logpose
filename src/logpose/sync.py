"""Synchronous facade over the async core.

logpose is async-first: :class:`~logpose.agent.Agent` owns the loop and every
entry point on it is a coroutine or an async iterator. Plenty of callers are not
async — a Django view, a Celery task, a script, a notebook cell — so this module
bridges the gap without duplicating a single line of loop logic.

Two equivalent spellings, pick whichever reads better:

.. code-block:: python

    from logpose import Agent, SyncAgent, run_sync

    agent = Agent("anthropic", tools=[get_weather])

    # 1. wrapper object — best when you make several calls
    with SyncAgent(agent) as sync_agent:
        result = sync_agent.run("What's the weather in Pune?")
        for event in sync_agent.stream("And in Goa?"):
            ...

    # 2. free functions — best for a one-off
    result = run_sync(agent, "What's the weather in Pune?")

How the bridge works
--------------------
Each :class:`~logpose.agent.Agent` gets **one** private event loop running on a
dedicated daemon thread, created on first use and reused for every later sync
call on that agent. Reusing one loop is not an optimization; it is required.
Objects the async stack builds lazily — notably the :class:`asyncio.Lock` inside
:class:`~logpose.auth.claude_code.CredentialProvider` — bind themselves to the
first loop that touches them and raise if a second loop touches them later. A
fresh loop per call would therefore work once and fail on the second call
against a real provider.

:func:`stream_sync` drives the async generator one step at a time
(``__anext__`` submitted to the private loop, result handed back) rather than
draining it into a queue. Nothing buffers, so backpressure is preserved, the
consumer's ``break`` reaches the async generator as a ``GeneratorExit`` at the
next step, and exceptions surface at the ``for`` statement with their original
traceback.

Never from inside a running loop
--------------------------------
Every entry point here raises :class:`~logpose.errors.LogposeError` when called
from a thread that already has a running event loop. Blocking that thread on
another loop's result cannot work — at best it deadlocks the caller's loop — so
it is rejected loudly, with a pointer to the async API.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import weakref
from collections.abc import AsyncIterator, Coroutine, Iterator
from typing import Any, TypeVar

from logpose.agent import Agent, Conversation
from logpose.errors import LogposeError
from logpose.events import Event, RunResult
from logpose.messages import Message

__all__ = ["SyncAgent", "close_sync", "run_sync", "stream_sync"]

_T = TypeVar("_T")

_START_TIMEOUT = 30.0
"""Seconds to wait for the private loop's thread to come up."""

_CLOSE_TIMEOUT = 30.0
"""Seconds to wait for that thread to wind down."""


# ---------------------------------------------------------------------------
# the private loop
# ---------------------------------------------------------------------------


class _LoopRunner:
    """An event loop running forever on a dedicated daemon thread.

    Coroutines are submitted from the calling thread and awaited there via
    :class:`concurrent.futures.Future`, so exceptions (and their tracebacks)
    cross the thread boundary intact.
    """

    def __init__(self, name: str = "logpose-sync") -> None:
        """Start the loop and wait until it is ready to accept work.

        Args:
            name: Thread name, useful when reading a traceback or a profiler.

        Raises:
            LogposeError: If the thread does not come up in time.
        """
        self._loop = asyncio.new_event_loop()
        self._started = threading.Event()
        self._lock = threading.Lock()
        self._closed = False
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()
        if not self._started.wait(_START_TIMEOUT):  # pragma: no cover - defensive
            self._closed = True
            raise LogposeError("Timed out starting the private event loop for sync execution.")

    @property
    def closed(self) -> bool:
        """Whether this runner can no longer accept work."""
        return self._closed or not self._thread.is_alive()

    def run(self, coro: Coroutine[Any, Any, _T]) -> _T:
        """Run ``coro`` on the private loop and return its result.

        Args:
            coro: The coroutine to run. It is closed without running if this
                runner is already shut down, so no "never awaited" warning is
                emitted.

        Returns:
            Whatever the coroutine returned.

        Raises:
            LogposeError: If the runner has been closed.
            BaseException: Whatever the coroutine raised, re-raised in the
                calling thread.
        """
        if self.closed:
            coro.close()
            raise LogposeError(
                "This agent's sync runtime has been closed. Build a new SyncAgent "
                "(or call run_sync again) to start a fresh one."
            )
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def close(self) -> None:
        """Stop the loop and join its thread. Idempotent and never raises."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        if not self._thread.is_alive():  # pragma: no cover - already gone
            return
        with contextlib.suppress(RuntimeError):
            self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(_CLOSE_TIMEOUT)

    def _run(self) -> None:
        """Thread body: own the loop, run it, then drain it on the way out."""
        asyncio.set_event_loop(self._loop)
        try:
            self._started.set()
            self._loop.run_forever()
        finally:
            try:
                _cancel_pending(self._loop)
                self._loop.run_until_complete(self._loop.shutdown_asyncgens())
                # Sync tool handlers run in asyncio.to_thread, i.e. on the loop's
                # default executor; it has to be drained or its workers outlive us.
                self._loop.run_until_complete(self._loop.shutdown_default_executor())
            finally:
                asyncio.set_event_loop(None)
                self._loop.close()

    def __repr__(self) -> str:
        """Return a representation carrying the thread name and liveness."""
        state = "closed" if self.closed else "running"
        return f"_LoopRunner(thread={self._thread.name!r}, {state})"


def _cancel_pending(loop: asyncio.AbstractEventLoop) -> None:
    """Cancel and await every task still pending on ``loop``.

    Args:
        loop: The stopped-but-not-closed loop to drain.
    """
    pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
    if not pending:
        return
    for task in pending:
        task.cancel()
    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))


# ---------------------------------------------------------------------------
# one runner per agent
# ---------------------------------------------------------------------------

_RUNNERS: weakref.WeakKeyDictionary[Agent, _LoopRunner] = weakref.WeakKeyDictionary()
_RUNNERS_LOCK = threading.Lock()


def _runner_for(agent: Agent) -> _LoopRunner:
    """Return the private loop bound to ``agent``, starting one if needed.

    The runner is dropped (and its thread stopped) when the agent is garbage
    collected or at interpreter exit, whichever comes first.

    Args:
        agent: The agent whose sync calls need a loop.

    Returns:
        A live :class:`_LoopRunner`.
    """
    with _RUNNERS_LOCK:
        existing = _RUNNERS.get(agent)
        if existing is not None and not existing.closed:
            return existing
        runner = _LoopRunner()
        _RUNNERS[agent] = runner
        # Fires on collection *and* via atexit, so an abandoned agent never
        # leaves a live thread behind.
        weakref.finalize(agent, runner.close)
        return runner


def _discard_runner(agent: Agent) -> _LoopRunner | None:
    """Remove and return the runner bound to ``agent``, if any.

    Args:
        agent: The agent to unbind.

    Returns:
        The runner that was bound, or ``None``.
    """
    with _RUNNERS_LOCK:
        return _RUNNERS.pop(agent, None)


def _reject_running_loop(entry_point: str) -> None:
    """Refuse to block the calling thread when it already runs a loop.

    Args:
        entry_point: Name of the sync API being called, used in the message.

    Raises:
        LogposeError: If an event loop is running in this thread.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return
    raise LogposeError(
        f"{entry_point} cannot be called from inside a running event loop, because it "
        "would have to block that loop. Use the async API instead: "
        "`await agent.run(...)` or `async for event in agent.stream(...)`."
    )


# ---------------------------------------------------------------------------
# async plumbing driven from the sync side
# ---------------------------------------------------------------------------


async def _next_event(events: AsyncIterator[Event]) -> Event | None:
    """Pull one event, mapping exhaustion to ``None``.

    ``StopAsyncIteration`` is a control-flow signal, not a result, so it is
    translated rather than thrown across the thread boundary. No
    :data:`~logpose.events.Event` is ever ``None``, so the sentinel is unambiguous.

    Args:
        events: The stream being driven.

    Returns:
        The next event, or ``None`` once the stream is exhausted.
    """
    try:
        return await events.__anext__()
    except StopAsyncIteration:
        return None


async def _close_events(events: AsyncIterator[Event]) -> None:
    """Close an event stream that supports it.

    Args:
        events: The stream to close. Closing an already-finished async generator
            is a no-op, so this is safe on every exit path.
    """
    closer = getattr(events, "aclose", None)
    if closer is None:  # pragma: no cover - Agent.stream is always a generator
        return
    await closer()


def _drive(runner: _LoopRunner, events: AsyncIterator[Event]) -> Iterator[Event]:
    """Turn an async event stream into a plain generator.

    Args:
        runner: The private loop to run each step on.
        events: The async stream produced by :meth:`~logpose.agent.Agent.stream`.

    Yields:
        Each :data:`~logpose.events.Event`, in order.
    """
    try:
        while True:
            event = runner.run(_next_event(events))
            if event is None:
                return
            yield event
    finally:
        # Reached on exhaustion, on an exception, and on the consumer's `break`
        # (which arrives here as GeneratorExit). The async generator must be
        # closed inside its own loop so its `finally` blocks — cancelling
        # in-flight tools, closing the provider stream — actually run.
        if not runner.closed:
            with contextlib.suppress(Exception):
                runner.run(_close_events(events))


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def run_sync(
    agent: Agent,
    prompt: str | Message | None = None,
    *,
    conversation: Conversation | None = None,
) -> RunResult:
    """Run ``agent`` to completion, blocking until it finishes.

    The synchronous twin of :meth:`~logpose.agent.Agent.run`.

    Args:
        agent: The agent to drive.
        prompt: The user turn to send. A ``str`` becomes a user message; a
            :class:`~logpose.messages.Message` is sent as-is; ``None`` continues
            from ``conversation`` without adding a turn.
        conversation: History to read from and append to. Omit for a single-shot
            run.

    Returns:
        The completed :class:`~logpose.events.RunResult`.

    Raises:
        LogposeError: If called from a thread with a running event loop, or for
            any of the reasons :meth:`~logpose.agent.Agent.run` raises it.
        MaxIterationsError: If the run exceeds the agent's iteration cap.
        ProviderError: If the provider fails or violates its contract.
    """
    _reject_running_loop("run_sync()")
    runner = _runner_for(agent)
    return runner.run(agent.run(prompt, conversation=conversation))


def stream_sync(
    agent: Agent,
    prompt: str | Message | None = None,
    *,
    conversation: Conversation | None = None,
) -> Iterator[Event]:
    """Run ``agent``, yielding events as they happen.

    The synchronous twin of :meth:`~logpose.agent.Agent.stream`. The returned
    generator is lazy: nothing is sent to the provider until it is iterated.

    Breaking out early is safe. The generator's cleanup closes the underlying
    async stream inside the private loop, which cancels in-flight tools and
    closes the provider stream. CPython runs that cleanup as soon as the
    generator is collected; call ``.close()`` on it yourself if you want to pin
    down exactly when.

    Args:
        agent: The agent to drive.
        prompt: The user turn to send. A ``str`` becomes a user message; a
            :class:`~logpose.messages.Message` is sent as-is; ``None`` continues
            from ``conversation`` without adding a turn.
        conversation: History to read from and append to. Omit for a single-shot
            run.

    Returns:
        An iterator of :data:`~logpose.events.Event` values, ending with
        :class:`~logpose.events.RunEnd`.

    Raises:
        LogposeError: If called from a thread with a running event loop, if
            ``prompt`` has an unsupported type, or if there is nothing to send.
            These are raised by the call itself, before any provider request.
    """
    _reject_running_loop("stream_sync()")
    runner = _runner_for(agent)
    # Eager: Agent.stream validates its arguments at call time, and those errors
    # should surface here rather than at the first `next()`.
    events = agent.stream(prompt, conversation=conversation)
    return _drive(runner, events)


def close_sync(agent: Agent) -> None:
    """Release the sync runtime and provider resources held by ``agent``.

    Closes the agent's provider (only if the agent built it) and stops the
    private loop thread. Safe to call on an agent that never ran synchronously,
    and safe to call twice. A later sync call on the same agent transparently
    starts a fresh loop.

    Args:
        agent: The agent to shut down.

    Raises:
        LogposeError: If called from a thread with a running event loop.
    """
    _reject_running_loop("close_sync()")
    runner = _discard_runner(agent)
    if runner is None or runner.closed:
        return
    try:
        runner.run(agent.aclose())
    finally:
        runner.close()


class SyncAgent:
    """A blocking view over an :class:`~logpose.agent.Agent`.

    Holds no state of its own beyond the wrapped agent: ``SyncAgent(agent)`` and
    :func:`run_sync` share the same private loop, so mixing them is fine.

    .. code-block:: python

        with SyncAgent(Agent("anthropic", tools=[get_weather])) as agent:
            print(agent.run("What's the weather in Pune?").text)

    Using it as a context manager closes the provider and stops the private loop
    on exit. Without the ``with``, that happens when the underlying agent is
    garbage collected or at interpreter exit.

    Attributes:
        agent: The wrapped async agent. Use it directly for async calls.
    """

    __slots__ = ("agent",)

    def __init__(self, agent: Agent) -> None:
        """Wrap an agent.

        Args:
            agent: The agent to drive synchronously.

        Raises:
            LogposeError: If ``agent`` is not an :class:`~logpose.agent.Agent`.
        """
        if not isinstance(agent, Agent):
            raise LogposeError(
                f"SyncAgent wraps an Agent, got {type(agent).__name__}. "
                "Build one first: SyncAgent(Agent('anthropic', tools=[...]))."
            )
        self.agent = agent

    def run(
        self,
        prompt: str | Message | None = None,
        *,
        conversation: Conversation | None = None,
    ) -> RunResult:
        """Run to completion, blocking until it finishes.

        Args:
            prompt: The user turn to send, or ``None`` to continue from
                ``conversation``.
            conversation: History to read from and append to.

        Returns:
            The completed :class:`~logpose.events.RunResult`.

        Raises:
            LogposeError: If called from a thread with a running event loop, or
                if there is nothing to send.
            MaxIterationsError: If the run exceeds the agent's iteration cap.
            ProviderError: If the provider fails or violates its contract.
        """
        return run_sync(self.agent, prompt, conversation=conversation)

    def stream(
        self,
        prompt: str | Message | None = None,
        *,
        conversation: Conversation | None = None,
    ) -> Iterator[Event]:
        """Run, yielding events as they happen.

        Args:
            prompt: The user turn to send, or ``None`` to continue from
                ``conversation``.
            conversation: History to read from and append to.

        Returns:
            An iterator of :data:`~logpose.events.Event` values, ending with
            :class:`~logpose.events.RunEnd`.

        Raises:
            LogposeError: If called from a thread with a running event loop, or
                if the prompt is unusable.
        """
        return stream_sync(self.agent, prompt, conversation=conversation)

    def close(self) -> None:
        """Close the provider and stop the private loop. Idempotent."""
        close_sync(self.agent)

    def __enter__(self) -> SyncAgent:
        """Return ``self`` so the wrapper can be used with ``with``."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close on the way out of a ``with`` block.

        Args:
            *exc_info: Standard exception triple; ignored, and never suppressed.
        """
        self.close()

    @property
    def provider_name(self) -> str:
        """The wrapped agent's provider name."""
        return self.agent.provider_name

    def __repr__(self) -> str:
        """Return a representation carrying the wrapped agent's configuration."""
        return f"SyncAgent({self.agent!r})"
