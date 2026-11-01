"""Opt-in application persistence hooks, separate from metadata telemetry."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal

from logpose.errors import LogposeError
from logpose.messages import ToolResultBlock, ToolUseBlock
from logpose.runtime import RuntimeContext


@dataclass(frozen=True)
class ToolObservation:
    """Detached tool data for an application-owned durable journal.

    Intent follows approval but may precede waiting for an execution slot.
    Outcome contains the normalized tool result, including error status.
    Cancellation may leave an intent without an outcome; never infer success.
    This object contains application content and must not enter telemetry logs.
    """

    phase: Literal["intent", "outcome"]
    context: RuntimeContext
    call: ToolUseBlock
    result: ToolResultBlock | None = None


ToolObserver = Callable[[ToolObservation], Awaitable[None]]


class ToolObservationError(LogposeError):
    """Persistence failed; callers must reconcile effects before resuming."""

    def __init__(self, phase: Literal["intent", "outcome"]) -> None:
        self.phase = phase
        self.effect_may_have_run = phase == "outcome"
        super().__init__(
            f"Tool {phase} persistence failed; execution stopped. "
            "Reconcile stored observations before retrying."
        )
