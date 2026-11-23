"""The provider seam.

A provider turns a :class:`CompletionRequest` into a stream of
:data:`ProviderEvent` values and nothing else. It never executes tools — note
that :class:`ToolSpec` deliberately carries no handler — so the loop stays the
single place where side effects happen.

The built-in backends all speak the OpenAI Chat Completions API; a custom
backend implements this same protocol and registers itself by name.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol, Union, runtime_checkable

from logpose.messages import Message, StopReason, Usage

__all__ = [
    "ToolSpec",
    "CompletionRequest",
    "ProviderTextDelta",
    "ProviderThinkingDelta",
    "CompletionDone",
    "ProviderEvent",
    "Provider",
]

@dataclass(frozen=True)
class ToolSpec:
    """The provider-visible description of a tool.

    Deliberately carries no handler: providers can advertise a tool to the model
    but cannot execute it. Only the loop can.

    Attributes:
        name: Tool name the model uses to call it.
        description: What the tool does and when to use it.
        input_schema: JSON Schema describing the tool's arguments.
    """

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class CompletionRequest:
    """One request for a single assistant turn.

    Attributes:
        messages: Full conversation history to send.
        model: Provider-specific model identifier.
        max_tokens: Maximum tokens to generate for this turn.
        system: Optional system prompt.
        tools: Tools to advertise to the model.
        extra: Provider-specific escape hatch merged into the wire request.
    """

    messages: list[Message]
    model: str
    max_tokens: int
    system: str | None = None
    tools: list[ToolSpec] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProviderTextDelta:
    """An incremental chunk of assistant-visible text.

    Attributes:
        text: The newly generated text fragment.
    """

    text: str


@dataclass(frozen=True)
class ProviderThinkingDelta:
    """An incremental chunk of model reasoning.

    Attributes:
        text: The newly generated reasoning fragment.
    """

    text: str


@dataclass(frozen=True)
class CompletionDone:
    """Terminal event of a provider stream.

    Attributes:
        message: The fully assembled assistant message, including any
            :class:`~logpose.messages.ThinkingBlock` and
            :class:`~logpose.messages.ToolUseBlock` content in wire order.
        stop_reason: Why the model stopped generating.
        usage: Token usage for this turn.
        refusal: The model's refusal text, if it refused; otherwise ``None``.
    """

    message: Message
    stop_reason: StopReason
    usage: Usage
    refusal: str | None = None


ProviderEvent = Union[  # noqa: UP007 - explicit Union keeps the alias importable at runtime
    ProviderTextDelta,
    ProviderThinkingDelta,
    CompletionDone,
]
"""Union of every event a provider stream can yield."""


@runtime_checkable
class Provider(Protocol):
    """What logpose needs from a model backend.

    ``name`` and ``stream`` are the contract checked by
    ``isinstance(obj, Provider)``. The model is not part of it: the caller
    names one on every :attr:`CompletionRequest.model`.

    A provider may also expose an optional ``max_tokens: int`` attribute holding
    its own output-token ceiling. It is *not* part of the protocol — the loop
    has a sane fallback — but when present, an ``Agent`` constructed without an
    explicit ``max_tokens=`` will honour it.

    Likewise an optional ``preserve_reasoning: bool`` attribute says whether the
    provider re-sends earlier turns' reasoning (thinking blocks). The built-in
    context policy counts reasoning toward its token estimate only when it is
    true; when absent, reasoning is treated as not sent.

    Attributes:
        name: Short identifier for the provider, e.g. ``"openai-compat"``.
    """

    name: str

    def stream(self, req: CompletionRequest) -> AsyncIterator[ProviderEvent]:
        """Run one turn and stream its events.

        The returned value is an async iterator, not a coroutine, so callers
        write ``async for ev in provider.stream(req)`` without awaiting first.
        An ``async def`` generator function satisfies this signature.

        The stream must end with exactly one :class:`CompletionDone`.

        Args:
            req: The turn to run.

        Returns:
            An async iterator of :data:`ProviderEvent` values.
        """
        ...
