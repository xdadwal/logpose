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


    agent = Agent("openai-compat", model="gpt-5.1", tools=[get_weather], system="Be concise.")
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
import copy
import enum
import inspect
import json
import logging
import time
import traceback
import weakref
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Coroutine,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field, replace
from time import monotonic
from typing import TYPE_CHECKING, Any, Literal, TypeVar, Union, overload
from uuid import uuid4

from pydantic import BaseModel, ValidationError

from logpose.cancellation import (
    ToolDeadlineHandle,
    _clear_deadline_handle,
    _install_deadline_handle,
)
from logpose.context import ContextInput, ContextPolicy, ContextPreparationError, PreparedContext
from logpose.errors import (
    LogposeError,
    MaxIterationsError,
    OutputRefusalError,
    OutputValidationError,
    ProviderError,
    RunTimeoutError,
    SessionBusyError,
    ToolExecutionError,
)
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
from logpose.lifecycle import ToolObservation, ToolObservationError, ToolObserver
from logpose.messages import Message, TextBlock, ToolResultBlock, ToolUseBlock, Usage
from logpose.output import (
    OutputParseFailure,
    format_errors,
    parse_output,
    response_format_for,
    strict_json_schema,
)
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
from logpose.runtime import (
    RuntimeContext,
    RuntimeEvent,
    RuntimeObserver,
    _reset_runtime_context,
    _set_runtime_context,
    current_runtime_context,
)
from logpose.tools import ToolDef

if TYPE_CHECKING:  # pragma: no cover - store modules import this one
    from logpose.store.base import Outcome, SessionHandle, SessionStore

__all__ = [
    "Agent",
    "Conversation",
    "ToolGate",
    "ToolGateOutcome",
    "ToolGateResult",
    "ToolErrorContext",
    "ToolErrorFormatter",
    "ToolErrorMode",
    "RuntimeObserver",
    "DEFAULT_MAX_ITERATIONS",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_MAX_CONCURRENT_TOOLS",
    "DEFAULT_PROVIDER_TURN_TIMEOUT",
    "DEFAULT_RUN_TIMEOUT",
    "DEFAULT_TOOL_TIMEOUT",
    "EMPTY_TOOL_RESULT",
    "RetryPolicy",
    "DEFAULT_RETRY_POLICY",
]

DEFAULT_MAX_ITERATIONS = 10
"""Provider round trips a single run may take before it gives up."""

DEFAULT_MAX_TOKENS = 16384
"""Output-token ceiling used when neither the agent nor the provider names one.

An explicit ``Agent(max_tokens=...)`` wins; failing that the provider's own
``max_tokens`` is honoured, so configuring a provider to cap output cost is not
silently overridden. Room for reasoning plus a long answer. A full 16k-token
reply can take longer than :data:`DEFAULT_PROVIDER_TURN_TIMEOUT` on slower
models; raise the turn and run deadlines too if replies run that long.
"""

DEFAULT_PROVIDER_TURN_TIMEOUT = 120.0
"""Deadline, in seconds, for one complete provider turn. ``None`` disables it."""

DEFAULT_RUN_TIMEOUT = 320.0
"""Wall-clock deadline, in seconds, for a whole run. ``None`` disables it."""

DEFAULT_MAX_CONCURRENT_TOOLS = 8
"""Maximum tool handlers one agent executes at once across concurrent runs."""

DEFAULT_TOOL_TIMEOUT = 30.0
"""Default execution deadline, in seconds, for one tool after queueing."""

EMPTY_TOOL_RESULT = "(no output)"
"""Stand-in for a tool result that is empty or only whitespace.

Providers reject empty content blocks, so a tool that returns ``""`` would
otherwise fail the *next* request rather than its own.
"""

SESSION_EXIT_TIMEOUT = 8.0
"""Seconds a session run's exit (record the run, save, release) may take once shielded.

Longer than the store's 5-second SQLite ``busy_timeout``, so one wait on a busy
database cannot use up the whole exit. A session block closes the streams still
open on its handle first, each bounded by its own exit; the save gets a fresh
budget afterwards.
"""

_STREAM_CLOSE_MARGIN = 2.0
"""Seconds past :data:`SESSION_EXIT_TIMEOUT` that closing a block's open streams
may take before it is abandoned (a stream's run record already has its own
bound; this backstops closing the loop itself)."""

_ABANDONED: set[asyncio.Future[Any]] = set()
"""Exit tasks given up on, kept referenced until they finish so none is
destroyed while pending."""

_RUNTIME_LOGGER = logging.getLogger("logpose.runtime")

_M = TypeVar("_M", bound=BaseModel)


class _Unset(enum.Enum):
    """Sentinel type: a per-call keyword the caller did not pass."""

    UNSET = enum.auto()

    def __repr__(self) -> str:
        return "<unset>"


_UNSET = _Unset.UNSET
"""Means "use the agent default" for ``output_type`` / ``output_retries``."""

_CORRECTION_PREFIX = "Your previous answer did not match the required JSON schema:\n"
_CORRECTION_SUFFIX = "\nReply again with only the corrected JSON."


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


ToolErrorMode = Literal["safe", "message", "traceback"]
"""How a failed tool is rendered to the model.

``"safe"`` exposes only the tool name, exception type, and an error ID;
``"message"`` also exposes the exception message; ``"traceback"`` exposes a
formatted traceback. The latter two modes are opt-in because tool failures can
contain confidential application data.
"""


@dataclass(frozen=True)
class ToolErrorContext:
    """Raw context passed only to an explicitly configured error formatter.

    Attributes:
        error_id: Unique identifier shared with the model-facing error result.
        tool_name: Name of the tool that failed.
        exception: Original handler or invocation exception.
    """

    error_id: str
    tool_name: str
    exception: BaseException


ToolErrorFormatter = Callable[[ToolErrorContext], str]
"""An opt-in function that renders a failed tool result for the model.

The formatter receives raw exception data and is responsible for any redaction.
If it raises or returns a non-string, logpose falls back to its safe message.
"""


_CONVERSATION_JSON_VERSION = 1


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

    def to_json(self) -> str:
        """Serialize the history to a versioned JSON document.

        Returns:
            ``{"version": 1, "messages": [...]}`` as a JSON string; lossless for
            every content block type.
        """
        messages = [m.model_dump(mode="json") for m in self.messages]
        return json.dumps({"version": _CONVERSATION_JSON_VERSION, "messages": messages})

    @classmethod
    def from_json(cls, text: str) -> Conversation:
        """Rebuild a conversation from :meth:`to_json` output.

        Args:
            text: The JSON document.

        Returns:
            A new :class:`Conversation`.

        Raises:
            LogposeError: If the document is not valid JSON, its ``version`` is
                not supported, or its messages are malformed.
        """
        try:
            data = json.loads(text)
        except (TypeError, ValueError) as exc:
            raise LogposeError(f"conversation JSON is not valid JSON: {exc}") from None
        version = data.get("version") if isinstance(data, dict) else None
        if version != _CONVERSATION_JSON_VERSION:
            raise LogposeError(f"unsupported conversation JSON version: {version!r}")
        raw = data.get("messages")
        if not isinstance(raw, list):
            raise LogposeError("conversation JSON must have a 'messages' list")
        try:
            return cls(Message.model_validate(m) for m in raw)
        except ValidationError as exc:
            raise LogposeError(f"conversation JSON has a malformed message: {exc}") from None

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
        model: Model identifier sent with every request.
        system: System prompt sent with every turn.
        tools: The tools advertised to the model.
        max_iterations: Provider round trips allowed per run, or ``None`` for no cap.
        max_tokens: Output-token ceiling override, or ``None`` to use the
            provider's own default.
        extra: Provider-specific request parameters merged into every turn.
        on_tool_call: Optional gate consulted before each tool runs.
        output_type: Default structured-output model for runs, or ``None``.
        output_retries: Default number of correction turns for invalid answers.
    """

    def __init__(
        self,
        provider: Provider | str,
        *,
        model: str,
        system: str | None = None,
        tools: Sequence[ToolDef] = (),
        max_iterations: int | None = DEFAULT_MAX_ITERATIONS,
        max_tokens: int | None = None,
        retry_policy: RetryPolicy = DEFAULT_RETRY_POLICY,
        provider_turn_timeout: float | None = DEFAULT_PROVIDER_TURN_TIMEOUT,
        run_timeout: float | None = DEFAULT_RUN_TIMEOUT,
        max_concurrent_tools: int = DEFAULT_MAX_CONCURRENT_TOOLS,
        tool_timeout: float | None = DEFAULT_TOOL_TIMEOUT,
        tool_error_mode: ToolErrorMode = "safe",
        tool_error_formatter: ToolErrorFormatter | None = None,
        extra: dict[str, Any] | None = None,
        on_tool_call: ToolGate | None = None,
        observers: Sequence[RuntimeObserver] = (),
        tool_observer: ToolObserver | None = None,
        context_policy: ContextPolicy | None = None,
        output_type: type[BaseModel] | None = None,
        output_retries: int = 1,
        session_wait: float = 0.0,
        **provider_kwargs: Any,
    ) -> None:
        """Build an agent.

        Args:
            provider: A provider instance, or the name of a registered one
                (see :func:`logpose.providers.resolve`).
            model: Model identifier sent with every request. Required;
                ``"auto"`` asks an OpenAI-compatible server for its first model.
            system: System prompt sent with every turn.
            tools: Tools the model may call, built with :func:`logpose.tools.tool`.
            max_iterations: Maximum provider round trips per run. Exceeding it
                raises :class:`~logpose.errors.MaxIterationsError`. ``None``
                disables the step cap; callers should retain a run deadline
                or cancellation mechanism for unattended work.
            max_tokens: Output-token ceiling for each turn. Defaults to the
                provider's own ``max_tokens`` when it has one, and to
                :data:`DEFAULT_MAX_TOKENS` otherwise — so
                ``Agent(OpenAICompatProvider(max_tokens=2048))`` is honoured rather
                than silently overridden.
            retry_policy: Provider retry timing. Retryable failures are replayed
                only before a text or reasoning delta reaches the caller;
                ``RetryPolicy(max_attempts=1)`` disables retries.
            provider_turn_timeout: Complete-turn deadline in seconds (default
                120), or ``None`` for no deadline.
            run_timeout: Wall-clock deadline in seconds for a whole run
                (default 320), measured from the start of ``run``/``stream``
                and including time the caller spends between streamed events.
                It interrupts only awaits logpose makes (provider, tools,
                ``on_tool_call``, ``tool_observer``, ``context_policy``), never
                the caller's own code. Expiry raises
                :class:`~logpose.errors.RunTimeoutError` and is never retried.
                ``None`` disables it.
            max_concurrent_tools: Tool handlers this agent may execute at once,
                across all of its concurrent runs. Extra calls wait for capacity.
            tool_timeout: Execution deadline in seconds after a call acquires
                capacity. ``None`` disables the tool deadline.
            tool_error_mode: Model-facing detail for tool failures. ``"safe"``
                (the default) omits exception messages; ``"message"`` and
                ``"traceback"`` expose progressively more diagnostic detail.
            tool_error_formatter: Optional formatter for failed tool results.
                It receives raw exception data and must redact anything
                confidential. It overrides ``tool_error_mode`` when it returns
                a string.
            extra: Provider-specific request parameters, merged into every wire
                request (:attr:`CompletionRequest.extra`). This is the escape
                hatch for options logpose does not model — ``tool_choice``,
                ``stop_sequences``, ``metadata``, ``service_tier``. Copied, so
                later mutation of the caller's dict is not observed.
            on_tool_call: Gate consulted once per requested tool call, in wire
                order, before any handler starts. Return ``None`` to let the call
                run, or a ``str`` / :class:`ToolGateResult` to block it and hand
                that text back to the model instead. See :data:`ToolGate`.
            observers: Synchronous callbacks that receive content-free
                :class:`~logpose.runtime.RuntimeEvent` lifecycle records. They
                run in registration order; a callback failure is isolated and
                never affects the agent run.
            tool_observer: Optional async application persistence callback. Receives
                detached approved tool intent and normalized outcome. Failures stop
                the run with ToolObservationError rather than becoming tool results.
                Contains application content; never attach it to telemetry logging.
            context_policy: Optional sync/async request preparation policy. Runs before
                every provider iteration on detached data; never rewrites history.
                Applications own selection and summaries. Failure stops the run.
            output_type: Default pydantic model the final answer must satisfy.
                When set, every turn requests strict ``json_schema`` output and
                the run returns the validated object as ``RunResult.output``.
                ``run``/``stream`` can override it per call.
            output_retries: Default number of correction turns sent when the
                final answer fails validation (default 1). ``0`` fails at once.
            session_wait: Seconds to keep retrying a busy session's lease before
                raising :class:`~logpose.errors.SessionBusyError` (default 0).
            **provider_kwargs: Forwarded to the named provider's factory, e.g.
                ``Agent("openai-compat", api_key=...)``. Only valid when ``provider``
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
            ``tool_timeout`` is invalid, or if a tool-error option is invalid,
            or if ``output_type`` is not a ``BaseModel`` subclass or
            ``output_retries`` is not an int >= 0.
        """
        _check_output_type(output_type)
        _check_output_retries(output_retries)
        if max_iterations is not None and max_iterations < 1:
            raise LogposeError(f"max_iterations must be at least 1, got {max_iterations}.")
        if max_tokens is not None and max_tokens < 1:
            raise LogposeError(f"max_tokens must be at least 1, got {max_tokens}.")
        if not isinstance(retry_policy, RetryPolicy):
            raise LogposeError(
                f"retry_policy must be a RetryPolicy, got {type(retry_policy).__name__}."
            )
        if not isinstance(model, str) or not model.strip():
            raise LogposeError(f"model must be a non-empty string, got {model!r}.")
        if provider_turn_timeout is not None and (
            isinstance(provider_turn_timeout, bool)
            or not isinstance(provider_turn_timeout, (int, float))
            or provider_turn_timeout <= 0
        ):
            raise LogposeError(
                "provider_turn_timeout must be a positive number or None; "
                f"got {provider_turn_timeout!r}."
            )
        if run_timeout is not None and (
            isinstance(run_timeout, bool)
            or not isinstance(run_timeout, (int, float))
            or run_timeout <= 0
        ):
            raise LogposeError(
                f"run_timeout must be a positive number or None; got {run_timeout!r}."
            )
        if (
            isinstance(max_concurrent_tools, bool)
            or not isinstance(max_concurrent_tools, int)
            or max_concurrent_tools < 1
        ):
            raise LogposeError(
                f"max_concurrent_tools must be at least 1, got {max_concurrent_tools!r}."
            )
        if tool_timeout is not None and (
            not isinstance(tool_timeout, (int, float)) or tool_timeout <= 0
        ):
            raise LogposeError(
                f"tool_timeout must be a positive number or None, got {tool_timeout!r}."
            )
        if tool_error_mode not in ("safe", "message", "traceback"):
            raise LogposeError(
                "tool_error_mode must be 'safe', 'message', or 'traceback', "
                f"got {tool_error_mode!r}."
            )
        if tool_error_formatter is not None and not callable(tool_error_formatter):
            raise LogposeError(
                f"tool_error_formatter must be callable, got {type(tool_error_formatter).__name__}."
            )
        if on_tool_call is not None and not callable(on_tool_call):
            raise LogposeError(f"on_tool_call must be callable, got {type(on_tool_call).__name__}.")
        if not all(callable(observer) for observer in observers):
            raise LogposeError("observers must contain only callable callbacks.")
        if (
            isinstance(session_wait, bool)
            or not isinstance(session_wait, (int, float))
            or session_wait < 0
        ):
            raise LogposeError(f"session_wait must be a number >= 0, got {session_wait!r}.")

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
                    "it needs a 'name' attribute and a 'stream' method."
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
        self.run_timeout = run_timeout
        self.max_concurrent_tools = max_concurrent_tools
        self.tool_timeout = tool_timeout
        self.tool_error_mode = tool_error_mode
        self.tool_error_formatter = tool_error_formatter
        # Created once per agent so capacity is shared by concurrent runs. Like
        # the provider's existing locks, it binds to the event loop on first use.
        self._tool_slots = asyncio.Semaphore(max_concurrent_tools)
        self.extra: dict[str, Any] = dict(extra) if extra else {}
        self.on_tool_call = on_tool_call
        self.observers: tuple[RuntimeObserver, ...] = tuple(observers)
        self.tool_observer = tool_observer
        self.context_policy = context_policy
        self.last_context_diagnostics: dict[str, int | str | bool] = {}
        self.output_type = output_type
        self.output_retries = output_retries
        self.session_wait = float(session_wait)
        self._uses_store = False

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

    @overload
    async def run(
        self,
        prompt: str | Message | None = None,
        *,
        conversation: Conversation | None = None,
        output_type: type[_M],
        output_retries: int | _Unset = _UNSET,
        session_id: str | None = None,
        session_metadata: Mapping[str, str] | None = None,
        session: SessionHandle | None = None,
    ) -> RunResult[_M]: ...

    @overload
    async def run(
        self,
        prompt: str | Message | None = None,
        *,
        conversation: Conversation | None = None,
        output_type: None | _Unset = _UNSET,
        output_retries: int | _Unset = _UNSET,
        session_id: str | None = None,
        session_metadata: Mapping[str, str] | None = None,
        session: SessionHandle | None = None,
    ) -> RunResult[Any]: ...

    async def run(
        self,
        prompt: str | Message | None = None,
        *,
        conversation: Conversation | None = None,
        output_type: type[BaseModel] | None | _Unset = _UNSET,
        output_retries: int | _Unset = _UNSET,
        session_id: str | None = None,
        session_metadata: Mapping[str, str] | None = None,
        session: SessionHandle | None = None,
    ) -> RunResult[Any]:
        """Run the loop to completion and return its result.

        Implemented by draining :meth:`stream`, so both entry points share one
        loop implementation.

        Args:
            prompt: The user turn to send. A ``str`` becomes a user message; a
                :class:`~logpose.messages.Message` is sent as-is; ``None``
                continues from ``conversation`` without adding a turn.
            conversation: History to read from and append to. Omit for a
                single-shot run.
            output_type: Pydantic model the final answer must satisfy; the
                validated object is returned as ``RunResult.output``. Omit to
                use the agent default; ``None`` disables it for this call.
            output_retries: Correction turns allowed for an invalid final
                answer. Omit to use the agent default.
            session_id: Run on this stored session: its lease is taken, its
                history loaded, the new messages saved and a run record written,
                exactly as if the call were wrapped in :meth:`session`.
            session_metadata: Metadata merged into the session on lease; only
                with ``session_id``.
            session: A handle from an open :meth:`session` block to run on. The
                run is recorded at once; messages are saved when the block exits.

        Returns:
            The completed :class:`~logpose.events.RunResult`.

        Raises:
            LogposeError: If there is nothing to send, if
                ``on_tool_call`` returns an unsupported type, or if the output
                options are invalid or conflict with ``extra["response_format"]``.
            OutputValidationError: If the final answer never validates against
                ``output_type``, or was cut off at ``max_tokens``.
            OutputRefusalError: If the model refused to give a structured answer.
            MaxIterationsError: If the run exceeds ``max_iterations``. The
                partial conversation is attached to the error.
            ProviderError: If the provider fails or violates its contract.
            SessionBusyError: If the session is leased elsewhere, or its lease
                was lost during the run (the run record is still written).
            Exception: Whatever ``on_tool_call`` raises, propagated unchanged
                rather than converted into a tool result.
        """
        result: RunResult[Any] | None = None
        events = self.stream(
            prompt,
            conversation=conversation,
            output_type=output_type,
            output_retries=output_retries,
            session_id=session_id,
            session_metadata=session_metadata,
            session=session,
        )
        async for event in events:
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
        output_type: type[BaseModel] | None | _Unset = _UNSET,
        output_retries: int | _Unset = _UNSET,
        session_id: str | None = None,
        session_metadata: Mapping[str, str] | None = None,
        session: SessionHandle | None = None,
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
            output_type: As for :meth:`run`; ``RunEnd.result.output`` carries
                the validated object.
            output_retries: As for :meth:`run`.
            session_id: As for :meth:`run`. The lease is taken when iteration
                starts, so a busy session raises from the first ``__anext__``;
                use :meth:`session` to learn that before streaming.
            session_metadata: As for :meth:`run`.
            session: As for :meth:`run`.

        Returns:
            An async iterator of :data:`~logpose.events.Event` values.

        Raises:
            LogposeError: If ``prompt`` is not a supported type, if there is
                nothing to send, if the output options are invalid or
                conflict with ``extra["response_format"]``, or if the session
                arguments are invalid or combined. Iterating can raise
                the same errors :meth:`run` documents, including whatever
                ``on_tool_call`` raises.
        """
        resolved_type = self.output_type if output_type is _UNSET else output_type
        resolved_retries = self.output_retries if output_retries is _UNSET else output_retries
        _check_output_type(resolved_type)
        _check_output_retries(resolved_retries)
        if resolved_type is not None and "response_format" in self.extra:
            raise LogposeError(
                "Agent(extra=...) already sets 'response_format', which output_type also "
                "sets; remove one of them."
            )
        given = [
            name
            for name, value in (
                ("session_id", session_id),
                ("session", session),
                ("conversation", conversation),
            )
            if value is not None
        ]
        if len(given) > 1:
            raise LogposeError(f"Pass at most one of {', '.join(given)}.")
        if session_metadata is not None and session_id is None:
            raise LogposeError("session_metadata is only valid with session_id.")
        first = _as_message(prompt) if prompt is not None else None
        if session_id is not None:
            from logpose.store.base import validate_session_id

            validate_session_id(session_id)
            return self._session_loop(
                first,
                resolved_type,
                resolved_retries,
                session_id=session_id,
                metadata=_check_metadata(session_metadata),
            )
        if session is not None:
            self._active_store(session)
            if first is None and not session.conversation.messages:
                raise LogposeError("Nothing to send: pass a prompt, or a session with history.")
            from logpose.store import manager

            stream = self._session_loop(first, resolved_type, resolved_retries, handle=session)
            # Registered so the block can close it before saving: a stream left
            # suspended mid-run must answer its tool calls before the history is stored.
            manager._STREAMS.setdefault(session, weakref.WeakSet()).add(stream)
            return stream
        if first is None and not (conversation is not None and conversation.messages):
            raise LogposeError(
                "Nothing to send: pass a prompt, or a conversation that already has messages."
            )
        return self._loop(first, conversation, resolved_type, resolved_retries)

    def session(
        self, session_id: str, *, metadata: Mapping[str, str] | None = None
    ) -> AbstractAsyncContextManager[SessionHandle]:
        """Lease a stored session for the duration of an ``async with`` block.

        Entering takes the lease (waiting up to ``session_wait`` seconds) and
        loads the history, so a busy session raises before anything is
        streamed — a web app can still answer 409::

            async with agent.session("chat-42") as handle:
                async for event in agent.stream("hi", session=handle):
                    ...

        Each run on the handle writes its run record when it ends. Leaving the
        block saves the messages appended to ``handle.conversation`` and
        releases the lease, inside a shield bounded to
        :data:`SESSION_EXIT_TIMEOUT` seconds so a cancellation cannot leave the
        lease held until it expires. Streams still open on the handle are closed
        before that budget starts, each within its own bounded exit.

        Args:
            session_id: The session to lease; ``[A-Za-z0-9._-]{1,128}``.
            metadata: String metadata merged into the session's metadata.

        Returns:
            An async context manager yielding the
            :class:`~logpose.store.base.SessionHandle`.

        Raises:
            LogposeError: If ``session_id`` or ``metadata`` is invalid.
            SessionBusyError: On enter, if the session stays leased elsewhere;
                on exit, if the lease was lost and the messages were not saved.
        """
        from logpose.store.base import validate_session_id

        validate_session_id(session_id)
        return self._session_block(session_id, _check_metadata(metadata))

    @contextlib.asynccontextmanager
    async def _session_block(
        self, session_id: str, metadata: dict[str, str] | None
    ) -> AsyncIterator[SessionHandle]:
        from logpose.store import manager

        store = await self._store()
        cm = store.session(session_id, wait=self.session_wait, metadata=metadata)
        handle = await cm.__aenter__()
        manager._ACTIVE_HANDLES[handle] = store
        try:
            yield handle
        except BaseException as exc:
            await _finish_session(_exit_session(handle, cm, exc), exc, close=handle)
            raise
        await _finish_session(_exit_session(handle, cm, None), None, close=handle)

    async def purge_expired_sessions(self) -> int:
        """Delete every session idle past its time-to-live; return how many.

        logpose never purges on its own; applications call this when they choose.
        """
        store = await self._store()
        return await store.purge_expired()

    async def delete_session(self, session_id: str) -> None:
        """Delete a stored session with its messages and run records."""
        from logpose.store.base import validate_session_id

        validate_session_id(session_id)
        store = await self._store()
        await store.delete_session(session_id)

    async def _store(self) -> SessionStore:
        from logpose.store import manager

        self._uses_store = True
        return await manager.get_store(owner=self)

    @staticmethod
    def _active_store(handle: SessionHandle) -> SessionStore:
        """Return the store that leased ``handle``, if its block is still open."""
        from logpose.store import manager

        store = manager._ACTIVE_HANDLES.get(handle)
        if store is None:
            raise LogposeError(
                "session= takes a handle from an open 'async with agent.session(...)' block."
            )
        if handle in manager._RUNNING:
            raise LogposeError(f"Session {handle.id!r} already has a run in progress.")
        return store

    async def _session_loop(
        self,
        first: Message | None,
        output_type: type[BaseModel] | None,
        output_retries: int,
        *,
        session_id: str | None = None,
        metadata: dict[str, str] | None = None,
        handle: SessionHandle | None = None,
    ) -> AsyncIterator[Event]:
        """Run :meth:`_loop` on a session and record the run on every outcome.

        With ``session_id`` the lease is taken here, at the first ``__anext__``,
        and the session is saved and released when the run ends. With ``handle``
        only the run record is written; the session block saves and releases.
        ``RunEnd`` is yielded only after the run is recorded (and, with
        ``session_id``, saved), so a lost lease raises instead of ending normally.
        """
        from logpose.store import manager

        cm: AbstractAsyncContextManager[SessionHandle] | None = None
        if handle is None:
            assert session_id is not None
            store = await self._store()
            cm = store.session(session_id, wait=self.session_wait, metadata=metadata)
            handle = await cm.__aenter__()
            manager._ACTIVE_HANDLES[handle] = store
            if first is None and not handle.conversation.messages:
                error = LogposeError("Nothing to send: pass a prompt, or a session with history.")
                await _finish_session(_exit_session(handle, cm, error), error)
                raise error
        else:
            store = self._active_store(handle)
        manager._RUNNING.add(handle)
        handle.idle_since = None
        if handle.lapsed:
            # The block sat idle and stopped renewing: take the lease back if it is
            # still ours, else the session moved on and this run must not start.
            revived = False
            try:
                revived = handle.revive is not None and await handle.revive()
            finally:
                if not revived:
                    manager._RUNNING.discard(handle)
                    handle.idle_since = monotonic()
            if not revived:
                raise SessionBusyError(
                    f"lease on session {handle.id!r} lapsed while its block sat idle "
                    "and was taken elsewhere; open a new session block"
                )
        stats = _RunStats()
        started_ms = _now_ms()
        run_end: RunEnd | None = None
        try:
            inner = self._loop(first, handle.conversation, output_type, output_retries, stats)
            try:
                async for event in inner:
                    if isinstance(event, RunEnd):
                        run_end = event
                    else:
                        yield event
            finally:
                # Closing the loop first cancels in-flight tools and closes the
                # provider stream before the session is saved.
                await _aclose(inner)
        except BaseException as exc:
            manager._RUNNING.discard(handle)
            handle.idle_since = monotonic()
            work = self._end_session_run(store, handle, cm, stats, started_ms, exc)
            await _finish_session(work, exc)
            raise
        manager._RUNNING.discard(handle)
        handle.idle_since = monotonic()
        await _finish_session(
            self._end_session_run(store, handle, cm, stats, started_ms, None), None
        )
        if run_end is None:  # pragma: no cover - the loop always ends with RunEnd
            raise LogposeError("The agentic loop finished without producing a result.")
        yield run_end

    async def _end_session_run(
        self,
        store: SessionStore,
        handle: SessionHandle,
        cm: AbstractAsyncContextManager[SessionHandle] | None,
        stats: _RunStats,
        started_ms: int,
        exc: BaseException | None,
    ) -> None:
        """Write the run record, then (``session_id`` form) save and release."""
        from logpose.store.base import RunRecord

        outcome, error_type = _outcome(exc)
        record = RunRecord(
            run_id=stats.run_id or self._runtime_id("run"),
            session_id=handle.id,
            model=self._model(),
            provider=self.provider_name,
            started_at_ms=started_ms,
            ended_at_ms=_now_ms(),
            outcome=outcome,
            iterations=stats.iterations,
            input_tokens=stats.usage.input_tokens,
            output_tokens=stats.usage.output_tokens,
            reasoning_tokens=stats.usage.reasoning_tokens,
            cache_read_input_tokens=stats.usage.cache_read_input_tokens,
            cache_creation_input_tokens=stats.usage.cache_creation_input_tokens,
            error_type=error_type,
        )
        try:
            await store.record_run(record)
        finally:
            if cm is not None:
                await _exit_session(handle, cm, exc)

    async def aclose(self) -> None:
        """Release provider and session-store resources owned by this agent.

        Closes the provider only if the agent built it (``provider=`` was a
        name). A provider handed in by the caller is left alone. Session stores
        this agent used are closed once no other agent uses them.
        """
        if self._uses_store:
            from logpose.store import manager

            self._uses_store = False
            await manager.release_stores(self)
        if not self._owns_provider:
            return
        closer = getattr(self.provider, "aclose", None)
        if closer is not None:
            await closer()

    def __enter__(self) -> Agent:
        """Enter a sync context that releases the agent's resources on exit.

        The alternate spelling for callers who hold an agent as a bare
        instance and remember to call :func:`close_sync` themselves. Exit
        closes the owned provider and stops the private loop thread; a later
        sync call transparently starts a fresh loop, so exiting is not
        destructive to reuse — but the intended shape is one scope per agent.
        """
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        from logpose.sync import close_sync

        close_sync(self)

    async def __aenter__(self) -> Agent:
        """Enter an async context that releases the agent's resources on exit."""
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.aclose()
        # An agent may also hold a sync-facade loop thread; stop it too. The
        # runner close never blocks on this loop — we are inside a loop here,
        # and the runner's daemon thread winds down on its own.
        from logpose.sync import _discard_runner

        runner = _discard_runner(self)
        if runner is not None:
            # close() joins the runner's thread; do it off the event loop so
            # a slow wind-down cannot stall this loop.
            await asyncio.to_thread(runner.close)

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

    def _emit_runtime(
        self,
        name: str,
        *,
        context: RuntimeContext | None = None,
        **fields: Any,
    ) -> RuntimeEvent | None:
        """Publish one content-free runtime event to logging and observers.

        Reporting is deliberately best-effort: a consumer's logging handler or
        observer must not turn a successful model run into a failed one. The
        event type exposes only primitive metadata fields, so this method never
        serializes prompts, model output, tool payloads, or exception messages.
        """
        runtime_context = context if context is not None else current_runtime_context()
        if runtime_context is None:  # pragma: no cover - every internal call has a run context
            return None
        token = _set_runtime_context(runtime_context)
        try:
            event = RuntimeEvent(
                name=name,
                run_id=runtime_context.run_id,
                provider=runtime_context.provider,
                model=runtime_context.model,
                turn_id=runtime_context.turn_id,
                attempt_id=runtime_context.attempt_id,
                request_id=runtime_context.request_id,
                tool_call_id=runtime_context.tool_call_id,
                tool_name=runtime_context.tool_name,
                **fields,
            )
            level = (
                logging.INFO
                if name == "run.completed"
                else logging.WARNING
                if name in {"provider.retry.scheduled", "tool.failed", "tool.timed_out"}
                else logging.DEBUG
            )
            try:
                _RUNTIME_LOGGER.log(level, name, extra=event.log_fields())
            except Exception:  # noqa: BLE001 - application-owned handlers are isolated
                pass
            for observer in self.observers:
                try:
                    observer(event)
                except Exception:  # noqa: BLE001 - observers are isolated by contract
                    pass
        finally:
            _reset_runtime_context(token)
        return event

    @staticmethod
    def _runtime_id(kind: str) -> str:
        """Return an opaque identifier without embedding application data."""
        return f"{kind}_{uuid4().hex}"

    # -- the loop -----------------------------------------------------------

    async def _loop(
        self,
        prompt: Message | None,
        conversation: Conversation | None,
        output_type: type[BaseModel] | None = None,
        output_retries: int = 0,
        stats: _RunStats | None = None,
    ) -> AsyncIterator[Event]:
        """Drive provider turns and tool executions until the run ends.

        Args:
            prompt: The already-normalized opening turn, if any.
            conversation: History to append to; a private list is used when omitted.
            output_type: Model the final answer must validate against, or ``None``.
            output_retries: Correction turns allowed for an invalid final answer.
            stats: Run-scoped accumulator kept current with the run id, usage and
                iterations, so a caller can read them whatever the outcome.

        Yields:
            :data:`~logpose.events.Event` values, ending with
            :class:`~logpose.events.RunEnd`.

        Raises:
            LogposeError: If no model could be resolved for the request.
            MaxIterationsError: If the run exceeds ``max_iterations``.
            RunTimeoutError: If the run exceeds ``run_timeout``.
            ProviderError: If the provider fails or violates its contract.
            OutputValidationError: If the final answer never validates.
            OutputRefusalError: If the model refused the structured answer.
        """
        model = self._model()
        response_format = response_format_for(output_type) if output_type is not None else None
        run_context = RuntimeContext(
            run_id=self._runtime_id("run"), provider=self.provider_name, model=model
        )
        started_at = monotonic()
        run_deadline = started_at + self.run_timeout if self.run_timeout is not None else None
        self._emit_runtime("run.started", context=run_context)
        history: list[Message] = conversation.messages if conversation is not None else []
        iterations = 0
        if stats is None:
            stats = _RunStats()
        stats.run_id = run_context.run_id
        # Tool calls appended to history whose results are not yet appended.
        # Whatever ends the run meanwhile (timeout, cancellation, an abandoned
        # stream, a raising gate), each one gets an error result, or every later
        # request on this conversation is rejected by the provider.
        unanswered: list[ToolUseBlock] = []

        def answer_unanswered(text: str) -> None:
            if unanswered:
                history.append(_error_results(unanswered, text))
                unanswered.clear()

        try:
            if prompt is not None:
                history.append(prompt)

            total_usage = Usage()
            last: CompletionDone | None = None
            output: BaseModel | None = None
            corrections = 0
            # Indices of invalid answers and correction prompts this run added;
            # dropped from history once a corrected answer validates.
            discard: set[int] = set()

            while True:
                if _remaining(run_deadline) == 0.0:
                    raise _RunExpired
                if self.max_iterations is not None and iterations >= self.max_iterations:
                    raise MaxIterationsError(
                        f"Gave up after {iterations} provider iterations without a final answer "
                        f"(max_iterations={self.max_iterations}).",
                        list(history),
                        self.max_iterations,
                    )
                iterations += 1
                stats.iterations = iterations
                turn_context = replace(
                    run_context,
                    turn_id=self._runtime_id("turn"),
                    request_id=self._runtime_id("request"),
                )
                done: CompletionDone | None = None
                async with _run_budget(run_deadline):
                    request = await self._prepare_request(history, response_format)
                turn_stream = self._provider_turn(request, turn_context, run_deadline)
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
                # The model asking for tools without naming any is the end of
                # the run: re-sending would loop forever.
                calls = (
                    [block for block in done.message.content if isinstance(block, ToolUseBlock)]
                    if done.stop_reason == "tool_use"
                    else []
                )
                if stored:
                    answer_index = len(history)
                    history.append(done.message)
                    unanswered[:] = calls
                total_usage = total_usage + done.usage
                stats.usage = total_usage
                last = done
                yield TurnEnd(stop_reason=done.stop_reason, usage=done.usage)

                if done.stop_reason == "pause_turn" and stored:
                    # Not a termination: re-issue with the paused turn in history.
                    # Unless there was no paused turn to store — re-issuing an
                    # identical request would spin against the API until the
                    # iteration cap, so that case falls through and ends the run.
                    continue

                if calls:
                    for call in calls:
                        yield ToolCall(id=call.id, name=call.name, input=call.input)
                    async with _run_budget(run_deadline):
                        results = await self._execute(calls, turn_context)
                    # ONE user message holding every result, in request order —
                    # appended before the results are yielded, so a consumer that
                    # walks away mid-results still leaves a replayable history.
                    history.append(Message(role="user", content=list(results)))
                    unanswered.clear()
                    for call, tool_result in zip(calls, results, strict=True):
                        yield ToolResult(
                            id=tool_result.tool_use_id,
                            name=call.name,
                            content=tool_result.content,
                            is_error=tool_result.is_error,
                        )
                    continue

                if output_type is not None:
                    try:
                        output = self._final_output(done, output_type, history)
                    except OutputParseFailure as failure:
                        if corrections >= output_retries:
                            raise OutputValidationError(
                                f"The final answer did not validate against "
                                f"{output_type.__name__} after {corrections + 1} attempt(s): "
                                f"{format_errors(failure.errors)}. If the answer ignored the "
                                "schema, the server may not enforce strict json_schema; use a "
                                "server and model that support structured output.",
                                raw_text=done.message.text,
                                errors=failure.errors,
                                messages=list(history),
                                output_type=output_type,
                            ) from None
                        corrections += 1
                        if stored:
                            discard.add(answer_index)
                        discard.add(len(history))
                        history.append(
                            Message(
                                role="user",
                                content=[
                                    TextBlock(
                                        text=_CORRECTION_PREFIX
                                        + format_errors(failure.errors)
                                        + _CORRECTION_SUFFIX
                                    )
                                ],
                            )
                        )
                        continue
                    if discard:
                        history[:] = [m for i, m in enumerate(history) if i not in discard]
                break

            if last is None:  # pragma: no cover - a provider turn always completes
                raise LogposeError("The agentic loop finished without a provider completion.")
            result = RunResult(
                text=last.message.text,
                messages=list(history),
                usage=total_usage,
                stop_reason=last.stop_reason,
                iterations=iterations,
                output=output,
            )
            self._emit_runtime(
                "run.completed",
                context=run_context,
                duration_seconds=monotonic() - started_at,
                stop_reason=result.stop_reason,
                iterations=result.iterations,
                input_tokens=result.usage.input_tokens,
                output_tokens=result.usage.output_tokens,
                cache_read_input_tokens=result.usage.cache_read_input_tokens,
                cache_creation_input_tokens=result.usage.cache_creation_input_tokens,
            )
            yield RunEnd(result=result)
        except _RunExpired:
            answer_unanswered("Run deadline exceeded before this tool finished.")
            self._emit_runtime(
                "run.timed_out",
                context=run_context,
                duration_seconds=monotonic() - started_at,
                iterations=iterations,
            )
            raise RunTimeoutError(
                f"Run exceeded run_timeout={self.run_timeout}s.",
                list(history),
                float(self.run_timeout or 0.0),
            ) from None
        except asyncio.CancelledError:
            self._emit_runtime(
                "run.cancelled", context=run_context, duration_seconds=monotonic() - started_at
            )
            raise
        finally:
            answer_unanswered("Run ended before this tool finished.")

    async def _provider_turn(
        self,
        request: CompletionRequest,
        turn_context: RuntimeContext,
        run_deadline: float | None = None,
    ) -> AsyncIterator[ProviderEvent]:
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
            attempt_context = replace(turn_context, attempt_id=self._runtime_id("attempt"))
            started_at = monotonic()
            self._emit_runtime("provider.attempt.started", context=attempt_context)
            provider_token = _set_runtime_context(attempt_context)
            try:
                provider_stream = self.provider.stream(request)
            finally:
                _reset_runtime_context(provider_token)
            turn_timeout = self._turn_timeout()
            turn_deadline = started_at + turn_timeout if turn_timeout is not None else None
            try:
                # Deadlines are absolute and enforced only around each provider
                # await, never across a ``yield``: the caller's own code between
                # events must not be cancelled by logpose.
                while True:
                    try:
                        event = await _next_within(
                            provider_stream, attempt_context, turn_deadline, run_deadline
                        )
                    except StopAsyncIteration:
                        break
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
                    self._emit_runtime(
                        "provider.attempt.failed",
                        context=attempt_context,
                        duration_seconds=monotonic() - started_at,
                        error_id=self._runtime_id("error"),
                        exception_type=type(exc).__name__,
                        status_code=exc.status_code,
                    )
                    raise exc from None
                delay = self.retry_policy.delay(attempt)
                if _outlasts_run(delay, run_deadline):
                    self._emit_runtime(
                        "provider.attempt.failed",
                        context=attempt_context,
                        duration_seconds=monotonic() - started_at,
                        error_id=self._runtime_id("error"),
                        exception_type=type(exc).__name__,
                        status_code=exc.status_code,
                    )
                    raise exc from None
                self._emit_runtime(
                    "provider.attempt.failed",
                    context=attempt_context,
                    duration_seconds=monotonic() - started_at,
                    error_id=self._runtime_id("error"),
                    exception_type=type(exc).__name__,
                    status_code=exc.status_code,
                )
                self._emit_runtime(
                    "provider.retry.scheduled",
                    context=attempt_context,
                    retry_delay_seconds=delay,
                )
                await asyncio.sleep(delay)
                continue
            except ProviderError as exc:
                exc.attempts = attempt
                if emitted:
                    exc.partial = True
                if not self._should_retry(exc, emitted=emitted, attempt=attempt):
                    self._emit_runtime(
                        "provider.attempt.failed",
                        context=attempt_context,
                        duration_seconds=monotonic() - started_at,
                        error_id=self._runtime_id("error"),
                        exception_type=type(exc).__name__,
                        status_code=exc.status_code,
                    )
                    raise
                delay = self.retry_policy.delay(attempt, retry_after=exc.retry_after)
                if _outlasts_run(delay, run_deadline):
                    self._emit_runtime(
                        "provider.attempt.failed",
                        context=attempt_context,
                        duration_seconds=monotonic() - started_at,
                        error_id=self._runtime_id("error"),
                        exception_type=type(exc).__name__,
                        status_code=exc.status_code,
                    )
                    raise
                self._emit_runtime(
                    "provider.attempt.failed",
                    context=attempt_context,
                    duration_seconds=monotonic() - started_at,
                    error_id=self._runtime_id("error"),
                    exception_type=type(exc).__name__,
                    status_code=exc.status_code,
                )
                self._emit_runtime(
                    "provider.retry.scheduled",
                    context=attempt_context,
                    retry_delay_seconds=delay,
                )
                await asyncio.sleep(delay)
                continue
            except _RunExpired:
                raise
            except Exception as exc:
                self._emit_runtime(
                    "provider.attempt.failed",
                    context=attempt_context,
                    duration_seconds=monotonic() - started_at,
                    error_id=self._runtime_id("error"),
                    exception_type=type(exc).__name__,
                )
                raise
            finally:
                await _aclose_with_runtime_context(provider_stream, attempt_context)

            if done is None:
                error = ProviderError(
                    f"Provider {self.provider_name!r} ended its stream without a "
                    "CompletionDone event."
                )
                self._emit_runtime(
                    "provider.attempt.failed",
                    context=attempt_context,
                    duration_seconds=monotonic() - started_at,
                    error_id=self._runtime_id("error"),
                    exception_type=type(error).__name__,
                )
                raise error
            self._emit_runtime(
                "provider.attempt.completed",
                context=attempt_context,
                duration_seconds=monotonic() - started_at,
                input_tokens=done.usage.input_tokens,
                output_tokens=done.usage.output_tokens,
                cache_read_input_tokens=done.usage.cache_read_input_tokens,
                cache_creation_input_tokens=done.usage.cache_creation_input_tokens,
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
        """Return the complete-turn deadline, or ``None`` for no deadline."""
        return self.provider_turn_timeout

    async def _prepare_request(
        self,
        history: Sequence[Message],
        response_format: dict[str, Any] | None = None,
    ) -> CompletionRequest:
        """Await selection before inference; isolate policy mutation from canonical data."""
        request = self._request(history)
        if response_format is not None:
            request.extra["response_format"] = copy.deepcopy(response_format)
        self.last_context_diagnostics = {}
        if self.context_policy is None:
            return request
        try:
            selected = self.context_policy(
                ContextInput(
                    tuple(message.model_copy(deep=True) for message in request.messages),
                    request.system,
                    tuple(copy.deepcopy(request.tools)),
                    request.model,
                    request.max_tokens,
                )
            )
            prepared = await selected if inspect.isawaitable(selected) else selected
            if not isinstance(prepared, PreparedContext):
                raise TypeError("Policy must return PreparedContext")
            if not isinstance(prepared.messages, tuple) or not all(
                isinstance(m, Message) for m in prepared.messages
            ):
                raise TypeError("Prepared messages must be a tuple of Message objects")
            if prepared.system is not None and not isinstance(prepared.system, str):
                raise TypeError("Prepared system must be text or None")
            if not all(
                isinstance(k, str) and type(v) in (int, str, bool)
                for k, v in prepared.diagnostics.items()
            ):
                raise TypeError("Diagnostics must contain metadata scalars")
            self.last_context_diagnostics = dict(prepared.diagnostics)
            return replace(
                request,
                messages=[m.model_copy(deep=True) for m in prepared.messages],
                system=prepared.system,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise ContextPreparationError(
                "Context preparation failed before provider inference"
            ) from exc

    @staticmethod
    def _final_output(
        done: CompletionDone, output_type: type[BaseModel], history: Sequence[Message]
    ) -> BaseModel:
        """Validate the final turn against ``output_type``.

        Args:
            done: The final provider completion.
            output_type: The model the answer must satisfy.
            history: The conversation so far, attached to raised errors.

        Returns:
            The validated output object.

        Raises:
            OutputRefusalError: If the model refused.
            OutputValidationError: If the answer was cut off at ``max_tokens``.
            OutputParseFailure: If the answer is not valid for ``output_type``;
                the caller decides whether to send a correction turn.
        """
        if done.refusal or done.stop_reason == "refusal":
            raise OutputRefusalError(
                f"The model refused to produce a {output_type.__name__} answer.",
                refusal=done.refusal or "",
                messages=list(history),
            )
        if done.stop_reason == "max_tokens":
            raise OutputValidationError(
                f"The {output_type.__name__} answer was cut off at max_tokens before it was "
                "complete; raise max_tokens for this agent.",
                raw_text=done.message.text,
                errors=[],
                messages=list(history),
                output_type=output_type,
                reason="max_tokens",
            )
        return parse_output(output_type, done.message.text)

    def _request(self, history: Sequence[Message]) -> CompletionRequest:
        """Build the request for the next turn.

        Args:
            history: The conversation so far. Copied, so later appends cannot
                mutate a request the provider is still working with.

        Returns:
            The :class:`~logpose.providers.base.CompletionRequest` to send.

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
        """Return the model identifier sent with every request."""
        return self.model

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

    async def _execute(
        self, calls: Sequence[ToolUseBlock], turn_context: RuntimeContext
    ) -> list[ToolResultBlock]:
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
        queued: list[tuple[int, ToolUseBlock, float]] = []
        for index, call in runnable:
            queued_at = monotonic()
            context = replace(turn_context, tool_call_id=call.id, tool_name=call.name)
            self._emit_runtime("tool.queued", context=context)
            queued.append((index, call, queued_at))
        tasks = [
            asyncio.create_task(self._invoke(call, queued_at, turn_context))
            for _, call, queued_at in queued
        ]
        try:
            ran: list[ToolResultBlock] = list(await asyncio.gather(*tasks))
        except ToolObservationError:
            # Persistence failures must not abandon approved sibling effects.
            # A cancelled to_thread await cannot stop its synchronous handler.
            # Shield/drain the batch so every surviving invocation can journal
            # its outcome before the original persistence error escapes.
            settling = asyncio.gather(*tasks, return_exceptions=True)
            while not settling.done():
                try:
                    await asyncio.shield(settling)
                except asyncio.CancelledError:
                    continue
            raise
        finally:
            # If the consumer abandoned the stream (or a sibling blew up), no
            # tool task may outlive this call.
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        results = dict(blocked)
        for (index, _call, _queued_at), result in zip(queued, ran, strict=True):
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

    async def _invoke(
        self, call: ToolUseBlock, queued_at: float, turn_context: RuntimeContext
    ) -> ToolResultBlock:
        """Journal approved known tools without converting persistence failures into results."""
        if self.tool_observer is None or call.name not in self._tools_by_name:
            return await self._invoke_handler(call, queued_at, turn_context)
        context = replace(turn_context, tool_call_id=call.id, tool_name=call.name)
        try:
            await self.tool_observer(ToolObservation("intent", context, call.model_copy(deep=True)))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise ToolObservationError("intent") from exc
        try:
            result = await self._invoke_handler(call, queued_at, turn_context)
        except asyncio.CancelledError:
            # Run deadline or abandoned stream: close the journal entry so it
            # matches the error result the conversation records for this call.
            await _settle(
                self.tool_observer(
                    ToolObservation(
                        "outcome",
                        context,
                        call.model_copy(deep=True),
                        ToolResultBlock(
                            tool_use_id=call.id,
                            content="Tool call was cancelled before it finished.",
                            is_error=True,
                        ),
                    )
                )
            )
            raise
        try:
            await self.tool_observer(
                ToolObservation(
                    "outcome", context, call.model_copy(deep=True), result.model_copy(deep=True)
                )
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise ToolObservationError("outcome") from exc
        return result

    async def _invoke_handler(
        self, call: ToolUseBlock, queued_at: float, turn_context: RuntimeContext
    ) -> ToolResultBlock:
        """Execute one tool call and render the outcome for the model.

        Args:
            call: The tool-use block to satisfy.

        Returns:
            The matching :class:`~logpose.messages.ToolResultBlock`. Unknown
            tools and handler failures produce ``is_error=True`` results so the
            model can correct itself; they never raise.
        """
        context = replace(turn_context, tool_call_id=call.id, tool_name=call.name)
        token = _set_runtime_context(context)
        # A handle exists whenever any deadline can stop this tool, so a run
        # deadline can terminate it just like a tool deadline.
        has_deadline = self.tool_timeout is not None or self.run_timeout is not None
        deadline = ToolDeadlineHandle() if has_deadline else None
        if deadline is not None:
            # Installed before invoke() so the to_thread copy of this context
            # carries the handle into the handler thread.
            _install_deadline_handle(deadline)
        try:
            tool_def = self._tools_by_name.get(call.name)
            if tool_def is None:
                result = ToolResultBlock(
                    tool_use_id=call.id,
                    content=self._unknown_tool_message(call.name),
                    is_error=True,
                )
                self._emit_runtime(
                    "tool.failed",
                    result_bytes=len(result.content.encode()),
                    exception_type="UnknownTool",
                )
                return result
            async with self._tool_slots:
                started_at = monotonic()
                self._emit_runtime("tool.started", queue_seconds=started_at - queued_at)
                try:
                    if self.tool_timeout is None:
                        content = await tool_def.invoke(call.input)
                    else:
                        content = await asyncio.wait_for(
                            tool_def.invoke(call.input),
                            timeout=self.tool_timeout,
                        )
                except asyncio.CancelledError:
                    # Run deadline or abandoned stream: stop what can be stopped
                    # before the slot is released.
                    if deadline is not None:
                        deadline.terminate()
                    raise
                except asyncio.TimeoutError:
                    if deadline is not None:
                        deadline.terminate()
                    result = ToolResultBlock(
                        tool_use_id=call.id,
                        content=(
                            f"Tool {call.name!r} exceeded its "
                            f"{self.tool_timeout:g}-second execution limit."
                        ),
                        is_error=True,
                    )
                    self._emit_runtime(
                        "tool.timed_out",
                        execution_seconds=monotonic() - started_at,
                        tool_timed_out=True,
                        result_bytes=len(result.content.encode()),
                    )
                    return result
                except ToolExecutionError as exc:
                    error_id = self._runtime_id("toolerr")
                    result = self._tool_error_result(call, exc, error_id=error_id)
                    self._emit_runtime(
                        "tool.failed",
                        execution_seconds=monotonic() - started_at,
                        error_id=error_id,
                        exception_type=type(exc.__cause__ or exc).__name__,
                        result_bytes=len(result.content.encode()),
                    )
                    return result
                except Exception as exc:  # noqa: BLE001 - a tool must never crash the loop
                    error_id = self._runtime_id("toolerr")
                    result = self._tool_error_result(call, exc, error_id=error_id)
                    self._emit_runtime(
                        "tool.failed",
                        execution_seconds=monotonic() - started_at,
                        error_id=error_id,
                        exception_type=type(exc).__name__,
                        result_bytes=len(result.content.encode()),
                    )
                    return result
            result = ToolResultBlock(
                tool_use_id=call.id,
                content=content if content.strip() else EMPTY_TOOL_RESULT,
                is_error=False,
            )
            self._emit_runtime(
                "tool.completed",
                execution_seconds=monotonic() - started_at,
                result_bytes=len(result.content.encode()),
            )
            return result
        except asyncio.CancelledError:
            self._emit_runtime("tool.cancelled")
            raise
        finally:
            _reset_runtime_context(token)
            if deadline is not None:
                _clear_deadline_handle()

    def _tool_error_result(
        self,
        call: ToolUseBlock,
        exc: BaseException,
        *,
        error_id: str | None = None,
    ) -> ToolResultBlock:
        """Render a tool failure without leaking handler data by default."""
        error_id = error_id or self._runtime_id("toolerr")
        original = exc.__cause__ if isinstance(exc, ToolExecutionError) and exc.__cause__ else exc
        context = ToolErrorContext(error_id=error_id, tool_name=call.name, exception=original)

        if self.tool_error_formatter is not None:
            try:
                content = self.tool_error_formatter(context)
            except Exception:  # noqa: BLE001 - reporting must not break the run
                content = ""
            if isinstance(content, str) and content.strip():
                return ToolResultBlock(tool_use_id=call.id, content=content, is_error=True)

        if isinstance(exc, ToolExecutionError) and exc.safe_to_expose:
            content = f"{str(exc) or f'Tool {call.name!r} could not run.'} Error ID: {error_id}."
        elif self.tool_error_mode == "message":
            content = (
                f"Tool {call.name!r} failed with {type(original).__name__}: {original}. "
                f"Error ID: {error_id}."
            )
        elif self.tool_error_mode == "traceback":
            content = (
                f"Tool {call.name!r} failed. Error ID: {error_id}.\n"
                f"{''.join(traceback.format_exception(exc))}"
            )
        else:
            content = (
                f"Tool {call.name!r} failed with {type(original).__name__}. Error ID: {error_id}."
            )
        return ToolResultBlock(tool_use_id=call.id, content=content, is_error=True)

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


def _check_output_type(output_type: object) -> None:
    """Reject an ``output_type`` strict structured output cannot serve.

    Besides the type check, the strict schema is built once so models it cannot
    express (open ``dict`` fields, non-object roots) fail before any request.
    """
    if output_type is None:
        return
    if not (isinstance(output_type, type) and issubclass(output_type, BaseModel)):
        shown = (output_type if isinstance(output_type, type) else type(output_type)).__name__
        raise LogposeError(
            f"output_type must be a pydantic BaseModel subclass or None, got {shown}."
        )
    strict_json_schema(output_type)


def _check_output_retries(output_retries: object) -> None:
    """Reject an ``output_retries`` that is not an int >= 0."""
    if (
        isinstance(output_retries, bool)
        or not isinstance(output_retries, int)
        or output_retries < 0
    ):
        raise LogposeError(f"output_retries must be an int >= 0, got {output_retries!r}.")


@dataclass
class _RunStats:
    """Run-scoped accumulator that :meth:`Agent._loop` keeps current."""

    run_id: str = ""
    iterations: int = 0
    usage: Usage = field(default_factory=Usage)


def _now_ms() -> int:
    return int(time.time() * 1000)


def _outcome(exc: BaseException | None) -> tuple[Outcome, str | None]:
    """Map how a run ended to its ``RunRecord`` outcome and ``error_type``."""
    if exc is None:
        return "completed", None
    if isinstance(exc, MaxIterationsError):
        return "max_iterations", None
    if isinstance(exc, RunTimeoutError):
        return "run_timeout", None
    if isinstance(exc, (asyncio.CancelledError, GeneratorExit)):
        return "cancelled", None
    return "error", type(exc).__name__


def _check_metadata(metadata: Mapping[str, str] | None) -> dict[str, str] | None:
    if metadata is None:
        return None
    if not isinstance(metadata, Mapping) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in metadata.items()
    ):
        raise LogposeError("session metadata must be a mapping of str to str.")
    return dict(metadata)


async def _exit_session(
    handle: SessionHandle,
    cm: AbstractAsyncContextManager[SessionHandle],
    exc: BaseException | None,
) -> None:
    """Leave a store session block: save new messages and release the lease.

    A session block closes its open streams first (:func:`_close_streams`).
    Any tool call still unanswered at the end of the history (a stream that
    could not be closed because it is running elsewhere, or whose close was
    abandoned) then gets an error result in the saved copy, so the stored
    session stays replayable.
    """
    from logpose.store import manager

    manager._ACTIVE_HANDLES.pop(handle, None)
    messages = handle.conversation.messages
    if messages and messages[-1].role == "assistant":
        calls = [b for b in messages[-1].content if isinstance(b, ToolUseBlock)]
        if calls:
            # A copy: a run still going in another task keeps appending to the old
            # list (and answers these calls itself); only the saved history changes.
            repair = _error_results(calls, "Run ended before this tool finished.")
            handle.conversation = Conversation([*messages, repair])
    if exc is None:
        await cm.__aexit__(None, None, None)
    else:
        await cm.__aexit__(type(exc), exc, exc.__traceback__)


async def _close_streams(handle: SessionHandle) -> None:
    """Close the streams still open on a session block's handle.

    A run suspended after a ``ToolCall`` answers its calls and writes its run
    record (inside its own bounded exit) before the block saves. A stream
    running in another task cannot be closed and is skipped.
    """
    from logpose.store import manager

    # No new run may start on the handle once its streams are being closed.
    manager._ACTIVE_HANDLES.pop(handle, None)
    for stream in list(manager._STREAMS.pop(handle, ())):
        try:
            await _aclose(stream)
        except Exception as error:  # noqa: BLE001 - the save must still happen
            _RUNTIME_LOGGER.warning("closing a session stream failed: %s", type(error).__name__)


async def _wait_shielded(task: asyncio.Future[Any], timeout: float) -> bool:
    """Wait up to ``timeout`` seconds for ``task``; return whether the caller was cancelled.

    A cancellation of the caller does not stop the wait; it is reported instead.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    cancelled = False
    while not task.done():
        remaining = deadline - loop.time()
        if remaining <= 0:
            break
        try:
            await asyncio.wait({task}, timeout=remaining)
        except asyncio.CancelledError:
            cancelled = True
    return cancelled


def _abandon(task: asyncio.Future[Any]) -> None:
    """Cancel an exit task that overran, keeping it referenced until it ends."""
    task.cancel()
    _ABANDONED.add(task)

    def done(t: asyncio.Future[Any]) -> None:
        _ABANDONED.discard(t)
        if not t.cancelled() and t.exception() is not None:
            _RUNTIME_LOGGER.warning(
                "abandoned session exit failed: %s", type(t.exception()).__name__
            )

    task.add_done_callback(done)


async def _finish_session(
    work: Coroutine[Any, Any, None],
    exc: BaseException | None,
    *,
    close: SessionHandle | None = None,
) -> None:
    """Run a session exit inside a bounded shield.

    The work runs as its own task and is waited on for up to
    :data:`SESSION_EXIT_TIMEOUT` seconds even if the caller is cancelled again
    meanwhile; a cancellation received while waiting is re-raised afterwards.
    When the run itself was cancelled or abandoned (``exc`` is a
    ``CancelledError`` or ``GeneratorExit``), failures of the exit are logged
    rather than raised, so the cancellation propagates unchanged; otherwise
    they propagate (for example :class:`~logpose.errors.SessionBusyError` for
    a lost lease).

    With ``close``, the streams still open on that handle are closed first, in
    a shielded phase of their own (each stream's exit is already bounded; the
    phase is abandoned after :data:`SESSION_EXIT_TIMEOUT` plus a margin), so
    the save's budget starts only once closing is over.
    """
    cancelled = False
    if close is not None:
        closing = asyncio.ensure_future(_close_streams(close))
        cancelled = await _wait_shielded(closing, SESSION_EXIT_TIMEOUT + _STREAM_CLOSE_MARGIN)
        if not closing.done():
            _abandon(closing)
            _RUNTIME_LOGGER.warning(
                "closing session streams did not finish within %ss; saving without "
                "their run records",
                SESSION_EXIT_TIMEOUT + _STREAM_CLOSE_MARGIN,
            )
        elif not closing.cancelled() and closing.exception() is not None:
            _RUNTIME_LOGGER.warning(
                "closing session streams failed: %s", type(closing.exception()).__name__
            )
    task = asyncio.ensure_future(work)
    cancelled = await _wait_shielded(task, SESSION_EXIT_TIMEOUT) or cancelled
    quiet = isinstance(exc, (asyncio.CancelledError, GeneratorExit))
    if not task.done():
        _abandon(task)
        _RUNTIME_LOGGER.warning(
            "session exit did not finish within %ss; the lease will lapse on its own",
            SESSION_EXIT_TIMEOUT,
        )
        if cancelled:
            raise asyncio.CancelledError
        if quiet:
            return
        raise LogposeError(
            f"Saving the session did not finish within {SESSION_EXIT_TIMEOUT}s."
        ) from exc
    if cancelled:
        if not task.cancelled() and task.exception() is not None:
            _RUNTIME_LOGGER.warning("session exit failed: %s", type(task.exception()).__name__)
        raise asyncio.CancelledError
    if quiet:
        if not task.cancelled() and task.exception() is not None:
            _RUNTIME_LOGGER.warning("session exit failed: %s", type(task.exception()).__name__)
        return
    task.result()


def _error_results(calls: list[ToolUseBlock], text: str) -> Message:
    """Build the user turn answering every call in ``calls`` with an error result."""
    return Message(
        role="user",
        content=[
            ToolResultBlock(tool_use_id=call.id, content=text, is_error=True) for call in calls
        ],
    )


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


async def _next_with_runtime_context(
    iterator: AsyncIterator[ProviderEvent], context: RuntimeContext
) -> ProviderEvent:
    """Advance a provider stream while making its correlation context current.

    The context is reset before a provider delta is re-yielded to the agent's
    caller. That boundary is important for the synchronous streaming facade,
    which resumes an async generator from a fresh task for each event.
    """
    token = _set_runtime_context(context)
    try:
        return await iterator.__anext__()
    finally:
        _reset_runtime_context(token)


async def _aclose_with_runtime_context(
    iterator: AsyncIterator[ProviderEvent], context: RuntimeContext
) -> None:
    """Close a provider stream while preserving its application log context."""
    token = _set_runtime_context(context)
    try:
        await _aclose(iterator)
    finally:
        _reset_runtime_context(token)


class _DeadlineExpired(TimeoutError):
    """A ``_deadline`` timer fired (as opposed to a TimeoutError raised by user code)."""


class _RunExpired(Exception):
    """Internal signal: the whole-run deadline passed. Becomes RunTimeoutError."""


def _remaining(run_deadline: float | None) -> float | None:
    """Seconds left before ``run_deadline`` (never negative), or ``None`` if unbounded."""
    if run_deadline is None:
        return None
    return max(0.0, run_deadline - monotonic())


def _outlasts_run(delay: float, run_deadline: float | None) -> bool:
    """Whether a retry sleep of ``delay`` would outlast the run deadline."""
    left = _remaining(run_deadline)
    return left is not None and delay >= left


async def _settle(awaitable: Awaitable[None]) -> None:
    """Run ``awaitable`` to completion despite repeated cancellation; swallow its errors.

    Used only while already unwinding a cancellation, where the original
    cancellation must stay the reported outcome.
    """
    task = asyncio.ensure_future(awaitable)
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
        except Exception:  # noqa: BLE001 - the cancellation is the outcome that matters
            return
    if not task.cancelled():
        task.exception()  # mark retrieved


async def _next_within(
    iterator: AsyncIterator[ProviderEvent],
    context: RuntimeContext,
    turn_deadline: float | None,
    run_deadline: float | None,
) -> ProviderEvent:
    """Await the next provider event within whichever deadline comes first.

    Raises ``_RunExpired`` when the run deadline is the one that ran out, and
    ``TimeoutError`` when the turn deadline did.
    """
    run_left = _remaining(run_deadline)
    turn_left = _remaining(turn_deadline)
    if run_left == 0.0:
        raise _RunExpired
    if turn_left == 0.0:
        raise TimeoutError
    run_binding = run_left is not None and (turn_left is None or run_left <= turn_left)
    budget = run_left if run_binding else turn_left
    try:
        async with _deadline(budget):
            return await _next_with_runtime_context(iterator, context)
    except _DeadlineExpired:
        # Only our own timer is a deadline. A TimeoutError raised by the provider
        # itself propagates unchanged and takes the turn-timeout path.
        if run_binding:
            raise _RunExpired from None
        raise


@contextlib.asynccontextmanager
async def _run_budget(run_deadline: float | None) -> AsyncIterator[None]:
    """Bound a block by the time left in the run, signalling expiry as _RunExpired."""
    if run_deadline is None:
        yield
        return
    left = _remaining(run_deadline)
    if left == 0.0:
        raise _RunExpired
    try:
        async with _deadline(left):
            yield
    except _DeadlineExpired:
        # A TimeoutError from a gate, observer, or context policy is theirs, not ours.
        raise _RunExpired from None


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
            raise _DeadlineExpired from None
        raise
    finally:
        handle.cancel()
