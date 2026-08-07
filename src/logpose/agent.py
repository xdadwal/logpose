"""The agentic loop.

:class:`Agent` is the one place in logpose where side effects happen. It drives a
:class:`~logpose.providers.base.Provider`, executes consumer-defined tools, and
streams normalized :data:`~logpose.events.Event` values.

.. code-block:: python

    from logpose import Agent, tool


    @tool
    def get_weather(city: str) -> str:
        '''Get the current weather.

        Args:
            city: City name.
        '''
        return f"22C and sunny in {city}"


    agent = Agent("anthropic", tools=[get_weather], system="Be concise.")
    result = await agent.run("What's the weather in Pune?")
    print(result.text, result.usage)

The loop
--------
One iteration is one provider round trip:

#. Build a :class:`~logpose.providers.base.CompletionRequest` from the history,
   the system prompt, and the tool specs.
#. Stream the provider, re-emitting its deltas as :class:`~logpose.events.TextDelta`
   and :class:`~logpose.events.ThinkingDelta`.
#. Append the assembled assistant message to the history **verbatim** — thinking
   blocks and their signatures included, because the API rejects a later turn
   whose thinking blocks were altered — and emit :class:`~logpose.events.TurnEnd`.
   The one exception is a turn with *no* content blocks (the documented shape of
   a pre-output ``refusal``): providers reject an empty content array, so
   appending it would poison every later turn on the same
   :class:`Conversation`. Such a turn is reported but never stored.
#. On ``stop_reason == "tool_use"``, offer each call to the optional
   ``on_tool_call`` gate, run *every* surviving tool concurrently, and append
   **one** user message holding all the results in request order. Splitting
   results across several user messages measurably degrades parallel tool
   calling, so it is treated as a bug.
#. On ``stop_reason == "pause_turn"``, re-issue transparently.
#. Otherwise finish: :class:`~logpose.events.RunEnd` carries the
   :class:`~logpose.events.RunResult`.

A failing tool is never an exception: it becomes a
:class:`~logpose.messages.ToolResultBlock` with ``is_error=True`` so the model
can read the error and adapt. The same is true of a call to a tool that does not
exist. Only the iteration cap (:class:`~logpose.errors.MaxIterationsError`), a
provider failure, a credential failure, or a gate that raises can end a run
abnormally.

Gating tool calls
-----------------
``Agent(on_tool_call=...)`` installs a gate that sees every requested call
*before* any handler starts, in wire order, one at a time. Returning ``None``
lets the call through; returning a string or a :class:`ToolGateResult`
substitutes that text as the call's result and the handler never runs. That is
the seam for a permission prompt, a policy check, or a dry run — and within a
turn it is deliberately sequential, because a gate that asks a human cannot be
asked several things at once. An :class:`Agent` holds no per-run state, so a
gate shared across *concurrent* runs still needs its own lock.

:meth:`Agent.run` is implemented by draining :meth:`Agent.stream`, so there is
exactly one loop implementation to reason about.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Union

from logpose.errors import LogposeError, MaxIterationsError, ProviderError, ToolExecutionError
from logpose.events import (
    Event,
    RunEnd,
    RunResult,
    TextDelta,
    ThinkingDelta,
    ToolCall,
    ToolResult,
    TurnEnd,
)
from logpose.messages import Message, ToolResultBlock, ToolUseBlock, Usage
from logpose.providers import resolve
from logpose.providers.base import (
    CompletionDone,
    CompletionRequest,
    Provider,
    ProviderEvent,
    ProviderTextDelta,
    ProviderThinkingDelta,
    ToolSpec,
)
from logpose.retry import DEFAULT_RETRY_POLICY, RetryPolicy
from logpose.tools import ToolDef

__all__ = [
    "Agent",
    "Conversation",
    "ToolGate",
    "ToolGateOutcome",
    "ToolGateResult",
    "DEFAULT_MAX_ITERATIONS",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_PROVIDER_TURN_TIMEOUT",
    "DEFAULT_MAX_CONCURRENT_TOOLS",
    "DEFAULT_TOOL_TIMEOUT",
    "EMPTY_TOOL_RESULT",
    "RetryPolicy",
    "DEFAULT_RETRY_POLICY",
]

DEFAULT_MAX_ITERATIONS = 25
"""Provider round trips a single run may take before it gives up."""

DEFAULT_MAX_TOKENS = 16000
"""Output-token ceiling used when neither the agent nor the provider names one.

An explicit ``Agent(max_tokens=...)`` wins; failing that the provider's own
``max_tokens`` is honoured, so configuring a provider to cap output cost is not
silently overridden.
"""

DEFAULT_PROVIDER_TURN_TIMEOUT = 900.0
"""Fallback deadline, in seconds, for one complete cloud provider turn."""

DEFAULT_MAX_CONCURRENT_TOOLS = 8
"""Maximum tool handlers one agent executes at once across concurrent runs."""

DEFAULT_TOOL_TIMEOUT = 300.0
"""Default execution deadline, in seconds, for one tool after queueing."""

EMPTY_TOOL_RESULT = "(no output)"
"""Stand-in for a tool result that is empty or only whitespace.

Providers reject empty content blocks, so a tool that returns ``""`` would
otherwise fail the *next* request rather than its own.
"""


@dataclass(frozen=True)
class ToolGateResult:
    """The result a gate substitutes for a call it blocked.

    Returning a bare ``str`` from a gate is shorthand for
    ``ToolGateResult(content)``, which reports the block as an error. Build the
    dataclass explicitly when the block is *not* a failure — a permission layer
    that declines a call while telling the model what to do instead wants
    ``is_error=False``, so the model reads it as a redirection rather than
    something that went wrong.

    The gate never names the call it answers: the loop pairs the result with the
    :class:`~logpose.messages.ToolUseBlock` it was handed.

    Attributes:
        content: The text handed back to the model in place of the tool's output.
        is_error: Whether the model should read this as a failure.
    """

    content: str
    is_error: bool = True

    def __post_init__(self) -> None:
        """Reject a non-string result at the point of construction.

        Raises:
            LogposeError: If ``content`` is not a string. Caught here rather than
                deep in the loop, where it would surface as an ``AttributeError``
                naming nothing the caller wrote.
        """
        if not isinstance(self.content, str):
            raise LogposeError(
                f"ToolGateResult.content must be a str, got {type(self.content).__name__}."
            )


ToolGateOutcome = Union[str, ToolGateResult, None]  # noqa: UP007 - runtime-importable alias
"""What a gate may return: ``None`` to allow, otherwise the substituted result."""

ToolGate = Callable[[ToolUseBlock], Union[ToolGateOutcome, Awaitable[ToolGateOutcome]]]  # noqa: UP007
"""A pre-execution gate over tool calls.

Called once per requested call, in wire order, before any handler starts. It may
be a coroutine function or an ordinary one; a synchronous gate runs inline on the
event loop, so anything blocking (a prompt, a file read, a network check) belongs
in :func:`asyncio.to_thread`.

An exception raised by a gate propagates out of the run rather than being turned
into a tool result: a permission layer that breaks must not fail open.
"""


class Conversation:
    """A mutable multi-turn message history.

    Passing a conversation to :meth:`Agent.run` or :meth:`Agent.stream` makes the
    run read from and append to *this* object, so a follow-up run continues where
    the last one stopped::

        conv = Conversation()
        await agent.run("My name is Ada.", conversation=conv)
        await agent.run("What is my name?", conversation=conv)  # remembers

    Appends happen as the run progresses, not at the end, so a run that fails
    part way through leaves the partial history in place for inspection. A run
    without a conversation is single-shot and shares nothing.

    Attributes:
        messages: The turns so far, oldest first. Mutating this list directly is
            supported; the loop only ever appends to it.
    """

    __slots__ = ("messages",)

    def __init__(self, messages: Iterable[Message] | None = None) -> None:
        """Initialize the history.

        Args:
            messages: Turns to start from. Copied into a new list, so the caller
                keeps ownership of whatever they passed.
        """
        self.messages: list[Message] = list(messages) if messages is not None else []

    def append(self, message: Message) -> None:
        """Add one turn to the end of the history.

        Args:
            message: The turn to add.
        """
        self.messages.append(message)

    def extend(self, messages: Iterable[Message]) -> None:
        """Add several turns to the end of the history.

        Args:
            messages: The turns to add, in order.
        """
        self.messages.extend(messages)

    def clear(self) -> None:
        """Drop every turn, keeping the same object usable."""
        self.messages.clear()

    @property
    def text(self) -> str:
        """The text of the most recent assistant turn, or ``""`` if there is none."""
        for message in reversed(self.messages):
            if message.role == "assistant":
                return message.text
        return ""

    def __len__(self) -> int:
        """Return the number of turns in the history."""
        return len(self.messages)

    def __iter__(self) -> Iterator[Message]:
        """Iterate the turns, oldest first."""
        return iter(self.messages)

    def __repr__(self) -> str:
        """Return a representation carrying the turn count, never the content."""
        return f"Conversation(messages={len(self.messages)})"


class Agent:
    """Runs the agentic loop against one provider with one set of tools.

    An agent holds no per-run state, so a single instance can serve concurrent
    runs (subject to the provider's own concurrency limits) and is safe to keep
    for the lifetime of a service.

    Attributes:
        provider: The backend this agent drives.
        model: Model override, or ``None`` to use the provider's default.
        system: System prompt sent with every turn.
        tools: The tools advertised to the model.
        max_iterations: Provider round trips allowed per run.
        max_tokens: Output-token ceiling override, or ``None`` to use the
            provider's own default.
        extra: Provider-specific request parameters merged into every turn.
        on_tool_call: Optional gate consulted before each tool runs.
    """

    def __init__(
        self,
        provider: Provider | str = "anthropic",
        *,
        model: str | None = None,
        system: str | None = None,
        tools: Sequence[ToolDef] = (),
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        max_tokens: int | None = None,
        retry_policy: RetryPolicy = DEFAULT_RETRY_POLICY,
        provider_turn_timeout: float | None | Literal["default"] = "default",
        max_concurrent_tools: int = DEFAULT_MAX_CONCURRENT_TOOLS,
        tool_timeout: float | None = DEFAULT_TOOL_TIMEOUT,
        extra: dict[str, Any] | None = None,
        on_tool_call: ToolGate | None = None,
        **provider_kwargs: Any,
    ) -> None:
        """Build an agent.

        Args:
            provider: A provider instance, or the name of a registered one
                (see :func:`logpose.providers.resolve`).
            model: Model identifier. Defaults to the provider's own
                ``model_default``.
            system: System prompt sent with every turn.
            tools: Tools the model may call, built with :func:`logpose.tools.tool`.
            max_iterations: Maximum provider round trips per run. Exceeding it
                raises :class:`~logpose.errors.MaxIterationsError`.
            max_tokens: Output-token ceiling for each turn. Defaults to the
                provider's own ``max_tokens`` when it has one, and to
                :data:`DEFAULT_MAX_TOKENS` otherwise — so
                ``Agent(AnthropicProvider(max_tokens=2048))`` is honoured rather
                than silently overridden.
            retry_policy: Provider retry timing. Retryable failures are replayed
                only before a text or reasoning delta reaches the caller;
                ``RetryPolicy(max_attempts=1)`` disables retries.
            provider_turn_timeout: Complete-turn deadline in seconds. The
                default uses the provider's recommendation; ``None`` disables
                the deadline.
            max_concurrent_tools: Tool handlers this agent may execute at once,
                across all of its concurrent runs. Extra calls wait for capacity.
            tool_timeout: Execution deadline in seconds after a call acquires
                capacity. ``None`` disables the tool deadline.
            extra: Provider-specific request parameters, merged into every wire
                request (:attr:`CompletionRequest.extra`). This is the escape
                hatch for options logpose does not model — ``tool_choice``,
                ``stop_sequences``, ``metadata``, ``service_tier``. Copied, so
                later mutation of the caller's dict is not observed.
            on_tool_call: Gate consulted once per requested tool call, in wire
                order, before any handler starts. Return ``None`` to let the call
                run, or a ``str`` / :class:`ToolGateResult` to block it and hand
                that text back to the model instead. See :data:`ToolGate`.
            **provider_kwargs: Forwarded to the named provider's factory, e.g.
                ``Agent("anthropic", api_key=...)``. Only valid when ``provider``
                is a name.

        Raises:
            LogposeError: If the provider name is unknown, if ``provider_kwargs``
                are passed alongside an already-built provider, if the provider
                does not implement the protocol, if two tools share a name, if a
                tool is not a :class:`~logpose.tools.ToolDef`, if
                ``on_tool_call`` is not callable, or if ``max_iterations`` /
            ``max_tokens`` are not positive, or if ``retry_policy`` is not a
            :class:`~logpose.retry.RetryPolicy`, or if
            ``provider_turn_timeout``, ``max_concurrent_tools``, or
            ``tool_timeout`` is invalid.
        """
        if max_iterations < 1:
            raise LogposeError(f"max_iterations must be at least 1, got {max_iterations}.")
        if max_tokens is not None and max_tokens < 1:
            raise LogposeError(f"max_tokens must be at least 1, got {max_tokens}.")
        if not isinstance(retry_policy, RetryPolicy):
            raise LogposeError(
                f"retry_policy must be a RetryPolicy, got {type(retry_policy).__name__}."
            )
        if provider_turn_timeout != "default" and (
            provider_turn_timeout is not None
            and (not isinstance(provider_turn_timeout, (int, float)) or provider_turn_timeout <= 0)
        ):
            raise LogposeError(
                "provider_turn_timeout must be a positive number, None, or 'default'; "
                f"got {provider_turn_timeout!r}."
            )
        if (
            isinstance(max_concurrent_tools, bool)
            or not isinstance(max_concurrent_tools, int)
            or max_concurrent_tools < 1
        ):
            raise LogposeError(
                "max_concurrent_tools must be at least 1, "
                f"got {max_concurrent_tools!r}."
            )
        if tool_timeout is not None and (
            not isinstance(tool_timeout, (int, float)) or tool_timeout <= 0
        ):
            raise LogposeError(
                f"tool_timeout must be a positive number or None, got {tool_timeout!r}."
            )
        if on_tool_call is not None and not callable(on_tool_call):
            raise LogposeError(
                f"on_tool_call must be callable, got {type(on_tool_call).__name__}."
            )

        if isinstance(provider, str):
            self.provider: Provider = resolve(provider, **provider_kwargs)
            self._owns_provider = True
        else:
            if provider_kwargs:
                unexpected = ", ".join(sorted(provider_kwargs))
                raise LogposeError(
                    f"Cannot pass provider arguments ({unexpected}) alongside an "
                    "already-built provider; configure the provider itself instead."
                )
            if not isinstance(provider, Provider):
                raise LogposeError(
                    f"{type(provider).__name__} does not implement the Provider protocol; "
                    "it needs 'name' and 'model_default' attributes and a 'stream' method."
                )
            self.provider = provider
            self._owns_provider = False

        self.model = model
        self.system = system
        self.tools: tuple[ToolDef, ...] = tuple(tools)
        self.max_iterations = max_iterations
        self.max_tokens = max_tokens
        self.retry_policy = retry_policy
        self.provider_turn_timeout = provider_turn_timeout
        self.max_concurrent_tools = max_concurrent_tools
        self.tool_timeout = tool_timeout
        # Created once per agent so capacity is shared by concurrent runs. Like
        # the provider's existing locks, it binds to the event loop on first use.
        self._tool_slots = asyncio.Semaphore(max_concurrent_tools)
        self.extra: dict[str, Any] = dict(extra) if extra else {}
        self.on_tool_call = on_tool_call

        by_name: dict[str, ToolDef] = {}
        for tool_def in self.tools:
            if not isinstance(tool_def, ToolDef):
                raise LogposeError(
                    f"Tool {tool_def!r} is a {type(tool_def).__name__}, not a ToolDef; "
                    "decorate it with @logpose.tool first."
                )
            if tool_def.name in by_name:
                raise LogposeError(
                    f"Duplicate tool name {tool_def.name!r}; every tool must be uniquely named."
                )
            by_name[tool_def.name] = tool_def
        self._tools_by_name = by_name
        self._specs: tuple[ToolSpec, ...] = tuple(tool_def.spec() for tool_def in self.tools)

    # -- public API ---------------------------------------------------------

    async def run(
        self,
        prompt: str | Message | None = None,
        *,
        conversation: Conversation | None = None,
    ) -> RunResult:
        """Run the loop to completion and return its result.

        Implemented by draining :meth:`stream`, so both entry points share one
        loop implementation.

        Args:
            prompt: The user turn to send. A ``str`` becomes a user message; a
                :class:`~logpose.messages.Message` is sent as-is; ``None``
                continues from ``conversation`` without adding a turn.
            conversation: History to read from and append to. Omit for a
                single-shot run.

        Returns:
            The completed :class:`~logpose.events.RunResult`.

        Raises:
            LogposeError: If there is nothing to send, if no model could be
                resolved (no ``model=`` and no provider ``model_default``), or if
                ``on_tool_call`` returns an unsupported type.
            MaxIterationsError: If the run exceeds ``max_iterations``. The
                partial conversation is attached to the error.
            ProviderError: If the provider fails or violates its contract.
            Exception: Whatever ``on_tool_call`` raises, propagated unchanged
                rather than converted into a tool result.
        """
        result: RunResult | None = None
        async for event in self.stream(prompt, conversation=conversation):
            if isinstance(event, RunEnd):
                result = event.result
        if result is None:  # pragma: no cover - the loop always ends with RunEnd
            raise LogposeError("The agentic loop finished without producing a result.")
        return result

    def stream(
        self,
        prompt: str | Message | None = None,
        *,
        conversation: Conversation | None = None,
    ) -> AsyncIterator[Event]:
        """Run the loop, yielding events as they happen.

        The return value is an async iterator, not a coroutine, so it is used
        directly::

            async for event in agent.stream("hello"):
                ...

        The stream always ends with :class:`~logpose.events.RunEnd` unless it
        raises. Breaking out early is safe: in-flight tools are cancelled and
        awaited, and the provider stream is closed.

        Args:
            prompt: The user turn to send. A ``str`` becomes a user message; a
                :class:`~logpose.messages.Message` is sent as-is; ``None``
                continues from ``conversation`` without adding a turn.
            conversation: History to read from and append to. Omit for a
                single-shot run.

        Returns:
            An async iterator of :data:`~logpose.events.Event` values.

        Raises:
            LogposeError: If ``prompt`` is not a supported type, or if there is
                nothing to send. Iterating can raise the same errors
                :meth:`run` documents, including whatever ``on_tool_call``
                raises.
        """
        first = _as_message(prompt) if prompt is not None else None
        if first is None and not (conversation is not None and conversation.messages):
            raise LogposeError(
                "Nothing to send: pass a prompt, or a conversation that already has messages."
            )
        return self._loop(first, conversation)

    async def aclose(self) -> None:
        """Release provider resources owned by this agent.

        Closes the provider only if the agent built it (``provider=`` was a
        name). A provider handed in by the caller is left alone.
        """
        if not self._owns_provider:
            return
        closer = getattr(self.provider, "aclose", None)
        if closer is not None:
            await closer()

    def __repr__(self) -> str:
        """Return a representation of the agent's configuration."""
        return (
            f"Agent(provider={self.provider_name!r}, model={self.model!r}, "
            f"tools={len(self.tools)}, max_iterations={self.max_iterations})"
        )

    @property
    def provider_name(self) -> str:
        """The provider's short name, used in error messages."""
        return str(getattr(self.provider, "name", type(self.provider).__name__))

    # -- the loop -----------------------------------------------------------

    async def _loop(
        self,
        prompt: Message | None,
        conversation: Conversation | None,
    ) -> AsyncIterator[Event]:
        """Drive provider turns and tool executions until the run ends.

        Args:
            prompt: The already-normalized opening turn, if any.
            conversation: History to append to; a private list is used when omitted.

        Yields:
            :data:`~logpose.events.Event` values, ending with
            :class:`~logpose.events.RunEnd`.

        Raises:
            LogposeError: If no model could be resolved for the request.
            MaxIterationsError: If the run exceeds ``max_iterations``.
            ProviderError: If the provider fails or violates its contract.
        """
        history: list[Message] = conversation.messages if conversation is not None else []
        if prompt is not None:
            history.append(prompt)

        total_usage = Usage()
        iterations = 0
        last: CompletionDone | None = None

        while True:
            if iterations >= self.max_iterations:
                raise MaxIterationsError(
                    f"Gave up after {iterations} provider iterations without a final answer "
                    f"(max_iterations={self.max_iterations}).",
                    list(history),
                    self.max_iterations,
                )
            iterations += 1

            done: CompletionDone | None = None
            turn_stream = self._provider_turn(self._request(history))
            try:
                async for event in turn_stream:
                    if isinstance(event, ProviderTextDelta):
                        yield TextDelta(text=event.text)
                    elif isinstance(event, ProviderThinkingDelta):
                        yield ThinkingDelta(text=event.text)
                    else:
                        done = event
            finally:
                await _aclose(turn_stream)

            if done is None:
                raise ProviderError(
                    f"Provider {self.provider_name!r} ended its stream without a "
                    "CompletionDone event."
                )

            # Verbatim: thinking blocks and their signatures must survive to the
            # next request untouched or the provider rejects the turn. The one
            # thing that must NOT be appended is a turn with no content blocks —
            # the documented shape of a pre-output refusal, and of any turn the
            # provider assembled with nothing representable in it. Providers
            # reject an empty content array, so storing it would break every
            # later turn on this conversation with an error naming a message the
            # consumer never wrote. `last` still carries the stop reason.
            stored = bool(done.message.content)
            if stored:
                history.append(done.message)
            total_usage = total_usage + done.usage
            last = done
            yield TurnEnd(stop_reason=done.stop_reason, usage=done.usage)

            if done.stop_reason == "pause_turn":
                # Not a termination: re-issue with the paused turn in history.
                # Unless there was no paused turn to store — re-issuing an
                # identical request would spin against the API until the
                # iteration cap, so end the run instead.
                if not stored:
                    break
                continue

            if done.stop_reason == "tool_use":
                calls = [block for block in done.message.content if isinstance(block, ToolUseBlock)]
                if not calls:
                    # The model asked for tools without naming any. Re-sending
                    # would loop forever, so treat it as the end of the run.
                    break
                for call in calls:
                    yield ToolCall(id=call.id, name=call.name, input=call.input)
                results = await self._execute(calls)
                for call, result in zip(calls, results, strict=True):
                    yield ToolResult(
                        id=result.tool_use_id,
                        name=call.name,
                        content=result.content,
                        is_error=result.is_error,
                    )
                # ONE user message holding every result, in request order.
                history.append(Message(role="user", content=list(results)))
                continue

            break

        yield RunEnd(
            result=RunResult(
                text=last.message.text,
                messages=list(history),
                usage=total_usage,
                stop_reason=last.stop_reason,
                iterations=iterations,
            )
        )

    async def _provider_turn(self, request: CompletionRequest) -> AsyncIterator[ProviderEvent]:
        """Stream one provider turn with safe, pre-delta retries.

        A retry can replay a request only while nothing has reached the caller.
        Once a text or reasoning delta was yielded, replaying would duplicate
        visible output, so the failure is marked partial and propagated.

        Args:
            request: Fully built immutable-by-convention request for this turn.

        Yields:
            Provider deltas followed by exactly one completion event.

        Raises:
            ProviderError: If the provider fails permanently, violates its
                streaming contract, exhausts retry attempts, or fails after an
                emitted delta.
        """
        for attempt in range(1, self.retry_policy.max_attempts + 1):
            done: CompletionDone | None = None
            emitted = False
            provider_stream = self.provider.stream(request)
            try:
                async with _deadline(self._turn_timeout()):
                    async for event in provider_stream:
                        if done is not None:
                            raise ProviderError(
                                f"Provider {self.provider_name!r} yielded "
                                f"{type(event).__name__} after CompletionDone; a provider stream "
                                "must end with exactly one CompletionDone.",
                            )
                        if isinstance(event, (ProviderTextDelta, ProviderThinkingDelta)):
                            emitted = True
                            yield event
                        elif isinstance(event, CompletionDone):
                            done = event
                        else:
                            raise ProviderError(
                                f"Provider {self.provider_name!r} yielded an unsupported event "
                                f"of type {type(event).__name__}."
                            )
            except TimeoutError:
                exc = ProviderError(
                    f"Provider {self.provider_name!r} exceeded its complete-turn deadline.",
                    retryable=True,
                    error_code="turn_timeout",
                )
                exc.attempts = attempt
                if emitted:
                    exc.partial = True
                if not self._should_retry(exc, emitted=emitted, attempt=attempt):
                    raise exc from None
                await asyncio.sleep(self.retry_policy.delay(attempt))
                continue
            except ProviderError as exc:
                exc.attempts = attempt
                if emitted:
                    exc.partial = True
                if not self._should_retry(exc, emitted=emitted, attempt=attempt):
                    raise
                await asyncio.sleep(self.retry_policy.delay(attempt, retry_after=exc.retry_after))
                continue
            finally:
                await _aclose(provider_stream)

            if done is None:
                raise ProviderError(
                    f"Provider {self.provider_name!r} ended its stream without a "
                    "CompletionDone event."
                )
            yield done
            return

        raise AssertionError(  # pragma: no cover
            "Retry loop exhausted without returning or raising."
        )

    def _should_retry(self, error: ProviderError, *, emitted: bool, attempt: int) -> bool:
        """Decide whether a failed provider turn can be replayed safely."""
        return (
            error.retryable
            and not error.partial
            and not emitted
            and attempt < self.retry_policy.max_attempts
        )

    def _turn_timeout(self) -> float | None:
        """Resolve the complete-turn deadline for the selected provider."""
        if self.provider_turn_timeout != "default":
            return self.provider_turn_timeout
        candidate = getattr(self.provider, "turn_timeout", DEFAULT_PROVIDER_TURN_TIMEOUT)
        if isinstance(candidate, (int, float)) and candidate > 0:
            return float(candidate)
        return DEFAULT_PROVIDER_TURN_TIMEOUT

    def _request(self, history: Sequence[Message]) -> CompletionRequest:
        """Build the request for the next turn.

        Args:
            history: The conversation so far. Copied, so later appends cannot
                mutate a request the provider is still working with.

        Returns:
            The :class:`~logpose.providers.base.CompletionRequest` to send.

        Raises:
            LogposeError: If no model could be resolved. Sending an empty model
                would fail as an opaque provider 400 on the first live request.
        """
        return CompletionRequest(
            messages=list(history),
            model=self._model(),
            max_tokens=self._max_tokens(),
            system=self.system,
            tools=list(self._specs),
            extra=dict(self.extra),
        )

    def _model(self) -> str:
        """Resolve the model identifier for a request.

        Returns:
            The explicit ``model=`` override, else the provider's
            ``model_default``.

        Raises:
            LogposeError: If neither yields a non-empty string.
        """
        model = self.model or str(getattr(self.provider, "model_default", "") or "")
        if not model.strip():
            raise LogposeError(
                f"Provider {self.provider_name!r} has no usable model_default; "
                f"pass model=... to Agent, or give the provider a non-empty "
                "model_default attribute."
            )
        return model

    def _max_tokens(self) -> int:
        """Resolve the output-token ceiling for a request.

        Returns:
            The explicit ``max_tokens=`` override, else the provider's own
            ``max_tokens`` if it has a positive one, else
            :data:`DEFAULT_MAX_TOKENS`.
        """
        if self.max_tokens is not None:
            return self.max_tokens
        try:
            provider_default = int(getattr(self.provider, "max_tokens", 0) or 0)
        except (TypeError, ValueError):
            provider_default = 0
        return provider_default if provider_default > 0 else DEFAULT_MAX_TOKENS

    # -- tools --------------------------------------------------------------

    async def _execute(self, calls: Sequence[ToolUseBlock]) -> list[ToolResultBlock]:
        """Gate every requested call, then run the survivors concurrently.

        Gating is a sequential pre-pass and running is concurrent: a gate that
        asks a human cannot be asked several things at once, while the tools it
        cleared are independent by construction.

        Args:
            calls: The tool-use blocks from the assistant turn, in wire order.

        Returns:
            One :class:`~logpose.messages.ToolResultBlock` per call, in the same
            order. Failures are reported as error results, never raised.
        """
        blocked = await self._gate(calls)
        runnable = [(index, call) for index, call in enumerate(calls) if index not in blocked]
        tasks = [asyncio.create_task(self._invoke(call)) for _, call in runnable]
        try:
            ran: list[ToolResultBlock] = list(await asyncio.gather(*tasks))
        finally:
            # If the consumer abandoned the stream (or a sibling blew up), no
            # tool task may outlive this call.
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        results = dict(blocked)
        for (index, _call), result in zip(runnable, ran, strict=True):
            results[index] = result
        return [results[index] for index in range(len(calls))]

    async def _gate(self, calls: Sequence[ToolUseBlock]) -> dict[int, ToolResultBlock]:
        """Offer each call to ``on_tool_call`` and collect the ones it blocked.

        Runs one call at a time, in wire order, and entirely before any handler
        starts — so a gate can prompt on stdin, and a call it blocks has no
        chance to have already run.

        Args:
            calls: The tool-use blocks from the assistant turn, in wire order.

        Returns:
            The substituted result for each blocked call, keyed by its position
            in ``calls``. Empty when no gate is installed.

        Raises:
            LogposeError: If the gate returns something that is neither ``None``,
                a ``str``, nor a :class:`ToolGateResult`.
            Exception: Whatever the gate itself raises, unchanged — a permission
                layer that breaks must not fail open.
        """
        if self.on_tool_call is None:
            return {}
        blocked: dict[int, ToolResultBlock] = {}
        for index, call in enumerate(calls):
            returned = self.on_tool_call(call)
            outcome = await returned if isinstance(returned, Awaitable) else returned
            if outcome is None:
                continue
            if isinstance(outcome, str):
                outcome = ToolGateResult(content=outcome)
            elif not isinstance(outcome, ToolGateResult):
                # LogposeError, not ToolExecutionError: this is the embedder's
                # contract being violated, like a bad prompt type or a
                # non-callable gate. ToolExecutionError means "fold me into a
                # tool result", which is the opposite of what must happen here.
                raise LogposeError(
                    f"on_tool_call returned {type(outcome).__name__} for tool {call.name!r}; "
                    "it must return None, a str, or a ToolGateResult."
                )
            blocked[index] = ToolResultBlock(
                tool_use_id=call.id,
                # Same reason as a tool that returns "": providers reject an
                # empty content block, so a silent gate would fail the *next*
                # request rather than this one.
                content=outcome.content if outcome.content.strip() else EMPTY_TOOL_RESULT,
                is_error=outcome.is_error,
            )
        return blocked

    async def _invoke(self, call: ToolUseBlock) -> ToolResultBlock:
        """Execute one tool call and render the outcome for the model.

        Args:
            call: The tool-use block to satisfy.

        Returns:
            The matching :class:`~logpose.messages.ToolResultBlock`. Unknown
            tools and handler failures produce ``is_error=True`` results so the
            model can correct itself; they never raise.
        """
        tool_def = self._tools_by_name.get(call.name)
        if tool_def is None:
            return ToolResultBlock(
                tool_use_id=call.id,
                content=self._unknown_tool_message(call.name),
                is_error=True,
            )
        try:
            async with self._tool_slots:
                if self.tool_timeout is None:
                    content = await tool_def.invoke(call.input)
                else:
                    content = await asyncio.wait_for(
                        tool_def.invoke(call.input),
                        timeout=self.tool_timeout,
                    )
        except asyncio.TimeoutError:
            return ToolResultBlock(
                tool_use_id=call.id,
                content=(
                    f"Tool {call.name!r} exceeded its {self.tool_timeout:g}-second execution limit."
                ),
                is_error=True,
            )
        except ToolExecutionError as exc:
            return ToolResultBlock(
                tool_use_id=call.id,
                content=str(exc) or f"Tool {call.name!r} failed.",
                is_error=True,
            )
        except Exception as exc:  # noqa: BLE001 - a tool must never crash the loop
            return ToolResultBlock(
                tool_use_id=call.id,
                content=f"Tool {call.name!r} failed: {type(exc).__name__}: {exc}",
                is_error=True,
            )
        return ToolResultBlock(
            tool_use_id=call.id,
            content=content if content.strip() else EMPTY_TOOL_RESULT,
            is_error=False,
        )

    def _unknown_tool_message(self, name: str) -> str:
        """Explain to the model that it called a tool that does not exist.

        Args:
            name: The tool name the model asked for.

        Returns:
            An error string naming the tools that do exist.
        """
        if not self._tools_by_name:
            return f"Unknown tool {name!r}: this agent has no tools."
        available = ", ".join(sorted(self._tools_by_name))
        return f"Unknown tool {name!r}. Available tools: {available}."


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _as_message(prompt: str | Message) -> Message:
    """Normalize a prompt into a :class:`~logpose.messages.Message`.

    Args:
        prompt: A string (becomes a user turn) or a message (used as-is).

    Returns:
        The message to append to the history.

    Raises:
        LogposeError: If ``prompt`` is neither a string nor a message, or if it
            is a blank string. Providers reject empty content blocks, so a blank
            prompt would otherwise fail as an opaque 400.
    """
    if isinstance(prompt, Message):
        return prompt
    if isinstance(prompt, str):
        if not prompt.strip():
            raise LogposeError(
                "prompt must not be blank; pass None to continue from a conversation."
            )
        return Message.user(prompt)
    raise LogposeError(f"prompt must be a str or a Message, got {type(prompt).__name__}.")


async def _aclose(iterator: AsyncIterator[Any]) -> None:
    """Close a provider stream if it supports being closed.

    Providers are typed as plain async iterators, so ``aclose`` is optional.
    Closing an async generator that is suspended at a ``yield`` is what lets it
    run its own cleanup when a consumer walks away mid-turn.

    Args:
        iterator: The provider stream to close.
    """
    closer = getattr(iterator, "aclose", None)
    if closer is None:
        return
    with contextlib.suppress(RuntimeError):
        await closer()


@contextlib.asynccontextmanager
async def _deadline(seconds: float | None) -> AsyncIterator[None]:
    """Cancel the current task at a complete-operation deadline.

    ``asyncio.timeout`` would provide this directly, but logpose supports Python
    3.10. The context distinguishes its own scheduled cancellation from a
    caller's cancellation and turns only the former into ``TimeoutError``.
    """
    if seconds is None:
        yield
        return
    task = asyncio.current_task()
    if task is None:  # pragma: no cover - async code always has a task
        yield
        return
    expired = False

    def expire() -> None:
        nonlocal expired
        expired = True
        task.cancel()

    handle = asyncio.get_running_loop().call_later(seconds, expire)
    try:
        yield
    except asyncio.CancelledError:
        if expired:
            raise TimeoutError from None
        raise
    finally:
        handle.cancel()
