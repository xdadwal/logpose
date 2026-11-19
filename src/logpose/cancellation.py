"""Deadline-scoped cancellation for synchronous tool handlers.

When a tool exceeds ``Agent(tool_timeout=...)``, ``asyncio.wait_for`` cancels
the *await*, but a handler running via :func:`asyncio.to_thread` cannot be
stopped from outside. The deadline handle closes that gap cooperatively: the
handler registers terminator callbacks (close a socket, kill a process group,
release a lock) and, when the deadline fires, the agent loop invokes them from
the event-loop thread. Terminators must therefore be thread-safe and must not
block; they exist to unblock or kill work, not to finish it.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from contextvars import ContextVar

_CURRENT_DEADLINE: ContextVar[ToolDeadlineHandle | None] = ContextVar(
    "logpose_tool_deadline", default=None
)


class ToolDeadlineHandle:
    """The cancellation surface for one in-flight tool invocation."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._terminators: list[Callable[[], None]] = []
        self.expired = False

    def register_terminator(self, terminator: Callable[[], None]) -> None:
        """Arrange for ``terminator`` to run if this invocation times out."""
        with self._lock:
            self._terminators.append(terminator)

    def terminate(self) -> None:
        """Run every registered terminator once; swallow their failures."""
        self.expired = True
        with self._lock:
            terminators = list(self._terminators)
            self._terminators.clear()
        for terminator in terminators:
            try:
                terminator()
            except Exception:  # noqa: BLE001 - a failing terminator must not mask the timeout
                pass


def current_tool_deadline() -> ToolDeadlineHandle | None:
    """Return the handle for the tool invocation running in this context.

    ``None`` when no deadline can stop the tool (both ``tool_timeout`` and
    ``run_timeout`` are ``None``) or when called outside a tool invocation.
    Terminators run when either deadline fires. The handle is shared between the event
    loop and the handler thread, so registrations made inside the handler are
    visible to the loop when the deadline fires.
    """
    return _CURRENT_DEADLINE.get()


def _install_deadline_handle(handle: ToolDeadlineHandle) -> None:
    """Install the handle at the internal invocation boundary."""
    _CURRENT_DEADLINE.set(handle)


def _clear_deadline_handle() -> None:
    """Remove the handle once the invocation has settled."""
    _CURRENT_DEADLINE.set(None)
