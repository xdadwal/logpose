"""Application-owned model-context selection without rewriting retained history."""

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field

from logpose.errors import LogposeError
from logpose.messages import Message
from logpose.providers.base import ToolSpec


@dataclass(frozen=True)
class ContextInput:
    """Detached inputs for one provider iteration, including opaque message blocks."""

    messages: tuple[Message, ...]
    system: str | None
    tools: tuple[ToolSpec, ...]
    model: str
    max_output_tokens: int


@dataclass(frozen=True)
class PreparedContext:
    """Selected request content; diagnostics are application-owned metadata.

    Policies own protocol validity, instruction retention, source provenance, and
    provider-specific opaque-block preservation. This type is not a compactor.
    Tool schemas, model, output limit, and extra parameters remain caller-owned.
    """

    messages: tuple[Message, ...]
    system: str | None
    diagnostics: Mapping[str, int | str | bool] = field(default_factory=dict)


ContextPolicy = Callable[[ContextInput], PreparedContext | Awaitable[PreparedContext]]


class ContextPreparationError(LogposeError):
    """Request preparation failed; no provider request was sent for that iteration."""
