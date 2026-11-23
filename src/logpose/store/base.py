"""Store interface. No database library is imported here."""

from __future__ import annotations

import re
import time
from collections.abc import Awaitable, Callable, Mapping
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from logpose.agent import Conversation
from logpose.compaction import _CompactionState, fingerprint
from logpose.errors import LogposeError
from logpose.messages import Message

__all__ = ["RunRecord", "SessionHandle", "SessionStore", "SummaryStore", "validate_session_id"]

_SESSION_ID_RE = re.compile(r"[A-Za-z0-9._-]{1,128}")

Outcome = Literal["completed", "max_iterations", "run_timeout", "cancelled", "error", "blocked"]


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
        stored_summary: The latest stored summary as ``(upto_seq, text)``, if any.
        idle_since: ``time.monotonic()`` when the handle last went idle (opened,
            or a run on it ended), or ``None`` while a run is in progress. The
            store stops renewing a lease idle for longer than three lease TTLs.
        lapsed: True once renewal stopped because the handle sat idle; the
            lease then expires on its own. A new run on the handle first tries
            to take the lease back (see ``revive``).

    Stored messages are append-only from ``seq`` 0 with no gaps, so a loaded
    message's ``seq`` equals its index in ``conversation.messages``. A summary's
    ``upto_seq`` is the ``seq`` of the last message it covers; the built-in
    context policy's cut index (the first message sent verbatim after the
    summary) is therefore ``upto_seq + 1``. A stored summary that fits the
    loaded history seeds the conversation's compaction state on enter, with a
    calibration ratio of 1.0 (the ratio is not persisted).
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
        stored_summary: tuple[int, str] | None = None,
    ) -> None:
        self.id = id
        self.conversation = conversation
        self.metadata = metadata
        self.created = created
        self.lost = lost
        self.loaded_count = loaded_count
        self.next_seq = next_seq
        self.stored_summary = stored_summary
        self.idle_since: float | None = time.monotonic()
        self.lapsed = False
        # Set by the store: re-extend a lapsed lease if its stored token is still
        # this holder's, and resume renewal. Returns False (and marks the handle
        # lost) if another holder took the lease meanwhile.
        self.revive: Callable[[], Awaitable[bool]] | None = None
        self._seed_summary()

    def _seed_summary(self) -> None:
        """Seed the conversation's compaction state from ``stored_summary`` if it fits."""
        stored_summary = self.stored_summary
        conversation = self.conversation
        if stored_summary is not None and self.next_seq == self.loaded_count:
            upto_seq, text = stored_summary
            cut = upto_seq + 1
            if 0 < cut <= self.loaded_count:
                conversation._compaction = _CompactionState(
                    summary=text,
                    cut_index=cut,
                    fingerprint=fingerprint(conversation.messages[:cut]),
                )

    def summary_to_save(self) -> tuple[int, str] | None:
        """Return ``(upto_seq, text)`` for a summary to store on exit, else ``None``.

        Only a summary that differs from ``stored_summary`` and still matches
        the history it was made from (its fingerprint) is returned, and only
        when stored ``seq`` values equal message indices.
        """
        state = self.conversation._compaction
        if state is None or state.summary is None or self.next_seq != self.loaded_count:
            return None
        messages = self.conversation.messages
        cut = state.cut_index
        if not 0 < cut <= len(messages) or fingerprint(messages[:cut]) != state.fingerprint:
            return None
        pending = (cut - 1, state.summary)
        return None if pending == self.stored_summary else pending

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


@runtime_checkable
class SummaryStore(Protocol):
    """Optional extra a :class:`SessionStore` may implement: direct summary access.

    The session path persists summaries through :class:`SessionHandle`
    (``stored_summary`` and :meth:`SessionHandle.summary_to_save`), so a custom
    store need not implement these. The SQLite store does.
    """

    async def save_summary(self, session_id: str, upto_seq: int, text: str) -> None: ...

    async def load_summary(self, session_id: str) -> tuple[int, str] | None: ...
