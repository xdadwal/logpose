"""Store interface. No database library is imported here."""

from __future__ import annotations

import re
from collections.abc import Mapping
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Literal, Protocol

from logpose.agent import Conversation
from logpose.errors import LogposeError
from logpose.messages import Message

__all__ = ["RunRecord", "SessionHandle", "SessionStore", "validate_session_id"]

_SESSION_ID_RE = re.compile(r"[A-Za-z0-9._-]{1,128}")

Outcome = Literal["completed", "max_iterations", "run_timeout", "cancelled", "error"]


def validate_session_id(session_id: str) -> str:
    """Return ``session_id`` if it matches ``[A-Za-z0-9._-]{1,128}``, else raise."""
    if not isinstance(session_id, str) or not _SESSION_ID_RE.fullmatch(session_id):
        raise LogposeError("session_id must match [A-Za-z0-9._-]{1,128}")
    return session_id


@dataclass(frozen=True)
class RunRecord:
    """One recorded agent run."""

    run_id: str
    session_id: str
    model: str
    provider: str
    started_at_ms: int
    ended_at_ms: int
    outcome: Outcome
    iterations: int
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    cache_read_input_tokens: int
    cache_creation_input_tokens: int
    error_type: str | None = None


class SessionHandle:
    """A leased session: loaded history plus metadata.

    Attributes:
        id: The session ID.
        conversation: The history loaded on enter; append to it inside the block.
        metadata: The session metadata after merging any metadata given on enter.
        created: True if the session was new (or had expired) on enter.
        lost: True once the lease is known to be lost; the final save is then refused.
        loaded_count: How many messages were loaded on enter. Messages after this
            index in ``conversation.messages`` are new and are saved on exit.
        next_seq: The stored ``seq`` the first new message will get.
    """

    def __init__(
        self,
        id: str,
        conversation: Conversation,
        metadata: dict[str, str],
        created: bool,
        lost: bool = False,
        loaded_count: int = 0,
        next_seq: int = 0,
    ) -> None:
        self.id = id
        self.conversation = conversation
        self.metadata = metadata
        self.created = created
        self.lost = lost
        self.loaded_count = loaded_count
        self.next_seq = next_seq

    def new_messages(self) -> list[Message]:
        """Return the messages appended to ``conversation`` since it was loaded."""
        return self.conversation.messages[self.loaded_count :]


class SessionStore(Protocol):
    """Persistence interface for sessions, messages and run records."""

    async def open(self) -> None: ...

    async def close(self) -> None: ...

    def session(
        self,
        session_id: str,
        *,
        wait: float = 0.0,
        metadata: Mapping[str, str] | None = None,
    ) -> AbstractAsyncContextManager[SessionHandle]: ...

    async def load_messages(self, session_id: str) -> list[Message]: ...

    async def record_run(self, record: RunRecord) -> None: ...

    async def delete_session(self, session_id: str) -> None: ...

    async def purge_expired(self, now_ms: int | None = None) -> int: ...
