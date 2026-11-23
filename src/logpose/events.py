"""Public streaming events yielded by the agentic loop.

These are plain frozen dataclasses (not pydantic models) so consumers can pattern
match on them cheaply. Anything a caller can observe while a run is in flight is
one of :data:`Event`; the terminal :class:`RunEnd` carries the same
:class:`RunResult` that the non-streaming entry point returns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, TypeVar, Union

from logpose.messages import Message, StopReason, Usage

if TYPE_CHECKING:
    from logpose.guardrails import GuardrailResult

__all__ = [
    "TextDelta",
    "ThinkingDelta",
    "ToolCall",
    "ToolResult",
    "TurnEnd",
    "RunResult",
    "RunEnd",
    "Event",
]

T = TypeVar("T")


@dataclass(frozen=True)
class TextDelta:
    """An incremental chunk of assistant-visible text.

    Attributes:
        text: The newly generated text fragment.
    """

    text: str


@dataclass(frozen=True)
class ThinkingDelta:
    """An incremental chunk of model reasoning.

    Attributes:
        text: The newly generated reasoning fragment.
    """

    text: str


@dataclass(frozen=True)
class ToolCall:
    """The loop is about to invoke a tool the model requested.

    Attributes:
        id: Provider-assigned tool-use identifier.
        name: Name of the tool being invoked.
        input: Already-parsed arguments for the tool.
    """

    id: str
    name: str
    input: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    """A tool finished and its result is being returned to the model.

    Attributes:
        id: The ``id`` of the :class:`ToolCall` this answers.
        name: Name of the tool that ran.
        content: The result rendered as text.
        is_error: Whether the tool failed.
    """

    id: str
    name: str
    content: str
    is_error: bool


@dataclass(frozen=True)
class TurnEnd:
    """One provider turn completed.

    Attributes:
        stop_reason: Why the model stopped generating this turn.
        usage: Token usage for this turn only.
    """

    stop_reason: StopReason
    usage: Usage


@dataclass(frozen=True)
class RunResult(Generic[T]):
    """The outcome of a complete run.

    Attributes:
        text: The final assistant text.
        messages: The full conversation history, including the input messages.
        usage: Token usage aggregated across every turn of the run.
        stop_reason: Why the final turn stopped.
        iterations: Number of provider turns the loop took.
        output: The validated structured answer when the run had an
            ``output_type``; ``None`` otherwise. ``text`` keeps the raw JSON.
        guardrail: When an input guardrail blocked the run, the guard name,
            category, reason and scores; ``text`` is then the placeholder reply
            and ``stop_reason`` is ``"refusal"``. ``None`` otherwise.
    """

    text: str
    messages: list[Message] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    stop_reason: StopReason = "end_turn"
    iterations: int = 0
    output: T | None = None
    guardrail: GuardrailResult | None = None


@dataclass(frozen=True)
class RunEnd:
    """The run finished; the terminal event of every stream.

    Attributes:
        result: The completed :class:`RunResult`.
    """

    result: RunResult[Any]


Event = Union[  # noqa: UP007 - explicit Union keeps the alias importable at runtime
    TextDelta,
    ThinkingDelta,
    ToolCall,
    ToolResult,
    TurnEnd,
    RunEnd,
]
"""Union of every event the public stream can yield."""
