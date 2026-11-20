"""Content-free runtime telemetry and correlation context.

The agent loop publishes :class:`RuntimeEvent` values for lifecycle operations.
They intentionally contain identifiers, timings, and other operational metadata
only: application prompts, model output, tool arguments and results, exception
messages, headers, and credentials have no field in this module.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Literal

__all__ = [
    "RuntimeContext",
    "RuntimeEvent",
    "RuntimeObserver",
    "current_runtime_context",
]

@dataclass(frozen=True)
class RuntimeContext:
    """Correlation identifiers for the operation currently executing.

    A context is local to the current async task (and is copied into synchronous
    tools run through :func:`asyncio.to_thread`). It is ``None`` outside an
    agent run.

    ``metadata`` is the app-supplied run metadata (``run(..., metadata=...)``),
    read-only; it never appears on :class:`RuntimeEvent`.
    """

    run_id: str
    provider: str
    model: str
    turn_id: str | None = None
    attempt_id: str | None = None
    request_id: str | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    metadata: Mapping[str, str | int | float | bool] | None = field(
        default=None, hash=False, compare=False
    )


@dataclass(frozen=True)
class RuntimeEvent:
    """An immutable, metadata-only record of an agent lifecycle operation.

    ``name`` is one of the documented ``run.*``, ``provider.*``, ``tool.*``,
    ``context.*``, or ``guardrail.*`` event names. Optional fields are populated
    only where they apply. ``guard_scores`` is a JSON object of a guard's scores.
    """

    name: str
    run_id: str
    provider: str
    model: str
    schema_version: Literal[1] = 1
    turn_id: str | None = None
    attempt_id: str | None = None
    request_id: str | None = None
    error_id: str | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    duration_seconds: float | None = None
    queue_seconds: float | None = None
    execution_seconds: float | None = None
    retry_delay_seconds: float | None = None
    status_code: int | None = None
    tool_timed_out: bool | None = None
    result_bytes: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    exception_type: str | None = None
    stop_reason: str | None = None
    iterations: int | None = None
    pointers_applied: int | None = None
    summarized: bool | None = None
    estimated_tokens_before: int | None = None
    estimated_tokens_after: int | None = None
    input_budget: int | None = None
    guard_name: str | None = None
    guard_allowed: bool | None = None
    guard_category: str | None = None
    guard_scores: str | None = None

    def log_fields(self) -> dict[str, str | int | float | bool | None]:
        """Return primitive, namespaced fields safe to attach to a LogRecord."""
        fields: dict[str, str | int | float | bool | None] = {"logpose_event": self.name}
        fields.update(
            {
                f"logpose_{name}": value
                for name, value in self.__dict__.items()
                if name != "name"
            }
        )
        return fields


RuntimeObserver = Callable[[RuntimeEvent], None]
"""A synchronous callback receiving one metadata-only runtime event."""

_CURRENT_RUNTIME_CONTEXT: ContextVar[RuntimeContext | None] = ContextVar(
    "logpose_runtime_context", default=None
)


def current_runtime_context() -> RuntimeContext | None:
    """Return the context for the current agent operation, if one is active."""
    return _CURRENT_RUNTIME_CONTEXT.get()


def _set_runtime_context(context: RuntimeContext) -> Token[RuntimeContext | None]:
    """Install a context at an internal lifecycle boundary."""
    return _CURRENT_RUNTIME_CONTEXT.set(context)


def _reset_runtime_context(token: Token[RuntimeContext | None]) -> None:
    """Restore the context that preceded an internal lifecycle boundary."""
    _CURRENT_RUNTIME_CONTEXT.reset(token)
