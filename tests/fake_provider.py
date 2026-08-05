"""A scriptable :class:`~logpose.providers.base.Provider` test double.

The loop is the only place in logpose where side effects happen, so testing it
means controlling exactly what the model "said" on every turn. :class:`FakeProvider`
replays a list of :class:`ScriptedTurn` values — one per provider round trip — and
records every :class:`~logpose.providers.base.CompletionRequest` it was handed so
tests can assert on what the loop actually sent.

.. code-block:: python

    provider = FakeProvider(
        [
            ScriptedTurn.tool_use(tool_call("add", {"a": 1, "b": 2}, id="t1")),
            ScriptedTurn.text("The answer is 3."),
        ]
    )
    agent = Agent(provider, tools=[add])
    result = await agent.run("what is 1 + 2?")

    assert provider.call_count == 2
    assert provider.requests[1].messages[-1].content[0].content == "3"

Scripting rules
---------------
* Turn ``n`` answers request ``n``; running out of script is an ``AssertionError``
  (a loop that spins forever should fail loudly, not hang). Pass
  ``repeat_last=True`` to replay the final turn indefinitely, which is how the
  ``max_iterations`` tests keep the model asking for tools forever.
* Every turn ends with exactly one :class:`~logpose.providers.base.CompletionDone`,
  as the provider contract requires. ``emit_done=False`` deliberately violates
  that so the loop's handling of a broken provider can be tested.
"""

from __future__ import annotations

import asyncio
import itertools
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from logpose.messages import (
    Message,
    StopReason,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    Usage,
)
from logpose.providers.base import (
    CompletionDone,
    CompletionRequest,
    ProviderEvent,
    ProviderTextDelta,
    ProviderThinkingDelta,
)

__all__ = ["FakeProvider", "ScriptedTurn", "tool_call"]

_TOOL_USE_IDS = itertools.count(1)


def tool_call(
    name: str,
    arguments: dict[str, Any] | None = None,
    *,
    id: str | None = None,  # noqa: A002 - mirrors ToolUseBlock.id
) -> ToolUseBlock:
    """Build a :class:`~logpose.messages.ToolUseBlock` for a scripted turn.

    Args:
        name: Tool name the model is asking for.
        arguments: Already-parsed arguments; defaults to ``{}``.
        id: Tool-use id. Defaults to a process-unique ``"toolu_N"``. Pass one
            explicitly whenever a test asserts on the id.

    Returns:
        The tool-use block to hand to :meth:`ScriptedTurn.tool_use`.
    """
    return ToolUseBlock(
        id=id if id is not None else f"toolu_{next(_TOOL_USE_IDS)}",
        name=name,
        input=dict(arguments or {}),
    )


@dataclass(frozen=True)
class ScriptedTurn:
    """One provider round trip: the deltas to emit, then how the turn ended.

    Build these with :meth:`text`, :meth:`tool_use`, or :meth:`failure` rather
    than by hand unless a test needs an unusual shape.

    Attributes:
        message: The assembled assistant message the turn finishes with.
        stop_reason: Why the model stopped.
        usage: Token usage reported for this turn.
        deltas: Streaming events emitted before the terminal
            :class:`~logpose.providers.base.CompletionDone`.
        error: Exception to raise instead of completing the turn.
        delay: Seconds to sleep before emitting anything.
        emit_done: When ``False``, the stream ends without a
            :class:`~logpose.providers.base.CompletionDone`, violating the
            provider contract on purpose.
    """

    message: Message = field(default_factory=lambda: Message(role="assistant", content=[]))
    stop_reason: StopReason = "end_turn"
    usage: Usage = field(default_factory=Usage)
    deltas: tuple[ProviderEvent, ...] = ()
    error: BaseException | None = None
    delay: float = 0.0
    emit_done: bool = True

    @classmethod
    def text(
        cls,
        text: str,
        *,
        chunks: Sequence[str] | None = None,
        thinking: str | None = None,
        signature: str | None = None,
        stop_reason: StopReason = "end_turn",
        usage: Usage | None = None,
        delay: float = 0.0,
    ) -> ScriptedTurn:
        """Script a plain text turn.

        Args:
            text: The assistant text of the assembled message.
            chunks: Text fragments to stream as deltas; defaults to ``[text]``.
                Their concatenation is not required to equal ``text``, so a test
                can check that the loop forwards deltas rather than re-deriving
                them from the final message.
            thinking: Reasoning text; adds a leading
                :class:`~logpose.messages.ThinkingBlock` and a thinking delta.
            signature: Opaque signature carried by that thinking block.
            stop_reason: Why the model stopped.
            usage: Token usage for the turn.
            delay: Seconds to sleep before emitting anything.

        Returns:
            The scripted turn.
        """
        blocks: list[Any] = []
        deltas: list[ProviderEvent] = []
        if thinking is not None:
            blocks.append(ThinkingBlock(thinking=thinking, signature=signature))
            deltas.append(ProviderThinkingDelta(text=thinking))
        blocks.append(TextBlock(text=text))
        for chunk in [text] if chunks is None else chunks:
            if chunk:
                deltas.append(ProviderTextDelta(text=chunk))
        return cls(
            message=Message(role="assistant", content=blocks),
            stop_reason=stop_reason,
            usage=usage or Usage(),
            deltas=tuple(deltas),
            delay=delay,
        )

    @classmethod
    def tool_use(
        cls,
        *calls: ToolUseBlock,
        text: str | None = None,
        thinking: str | None = None,
        signature: str | None = None,
        usage: Usage | None = None,
        stop_reason: StopReason = "tool_use",
        delay: float = 0.0,
    ) -> ScriptedTurn:
        """Script a turn in which the model asks for one or more tools.

        Args:
            *calls: The tool-use blocks, in wire order. Build them with
                :func:`tool_call`.
            text: Optional preamble text emitted before the tool calls.
            thinking: Optional reasoning text, placed first in the message.
            signature: Opaque signature carried by that thinking block.
            usage: Token usage for the turn.
            stop_reason: Overridable so a test can script the pathological
                ``tool_use`` stop reason with no tool-use blocks.
            delay: Seconds to sleep before emitting anything.

        Returns:
            The scripted turn.
        """
        blocks: list[Any] = []
        deltas: list[ProviderEvent] = []
        if thinking is not None:
            blocks.append(ThinkingBlock(thinking=thinking, signature=signature))
            deltas.append(ProviderThinkingDelta(text=thinking))
        if text is not None:
            blocks.append(TextBlock(text=text))
            deltas.append(ProviderTextDelta(text=text))
        blocks.extend(calls)
        return cls(
            message=Message(role="assistant", content=blocks),
            stop_reason=stop_reason,
            usage=usage or Usage(),
            deltas=tuple(deltas),
            delay=delay,
        )

    @classmethod
    def failure(cls, error: BaseException, *, delay: float = 0.0) -> ScriptedTurn:
        """Script a turn that raises instead of completing.

        Args:
            error: The exception to raise from the provider stream.
            delay: Seconds to sleep before raising.

        Returns:
            The scripted turn.
        """
        return cls(error=error, delay=delay)

    def completion(self) -> CompletionDone:
        """Return the terminal event this turn finishes with.

        Returns:
            The :class:`~logpose.providers.base.CompletionDone` carrying the
            assembled message, stop reason, and usage.
        """
        return CompletionDone(
            message=self.message,
            stop_reason=self.stop_reason,
            usage=self.usage,
        )


class FakeProvider:
    """A provider that replays scripted turns and records what it was sent.

    Satisfies the :class:`~logpose.providers.base.Provider` protocol; executes
    nothing.

    Attributes:
        name: Provider name reported to the loop.
        model_default: Model reported to the loop when the agent has no
            ``model=`` override. Part of the ``Provider`` protocol.
        turns: The scripted turns, one per request.
        repeat_last: Whether to replay the final turn once the script runs out.
        requests: Every :class:`~logpose.providers.base.CompletionRequest`
            received, in order.
        closed: How many times a consumer closed the stream early.
    """

    name: str = "fake"
    model_default: str = "fake-model"

    def __init__(
        self,
        turns: Sequence[ScriptedTurn] | ScriptedTurn = (),
        *,
        name: str = "fake",
        model_default: str = "fake-model",
        repeat_last: bool = False,
    ) -> None:
        """Initialize the double.

        Args:
            turns: Turns to replay, one per request. A bare
                :class:`ScriptedTurn` is accepted for the single-turn case.
            name: Provider name reported to the loop.
            model_default: Model reported when the agent has no override. Pass
                ``""`` to script a provider that declares no default.
            repeat_last: Replay the final turn forever once the script is
                exhausted, instead of raising ``AssertionError``.
        """
        self.turns: list[ScriptedTurn] = [turns] if isinstance(turns, ScriptedTurn) else list(turns)
        self.name = name
        self.model_default = model_default
        self.repeat_last = repeat_last
        self.requests: list[CompletionRequest] = []
        self.closed = 0

    @property
    def call_count(self) -> int:
        """How many times :meth:`stream` has been started."""
        return len(self.requests)

    @property
    def last_request(self) -> CompletionRequest:
        """The most recent request.

        Returns:
            The last recorded request.

        Raises:
            AssertionError: If the provider was never called.
        """
        assert self.requests, "FakeProvider was never called"
        return self.requests[-1]

    def _turn_for(self, index: int) -> ScriptedTurn:
        """Return the turn answering request ``index``.

        Args:
            index: Zero-based request number.

        Returns:
            The scripted turn to replay.

        Raises:
            AssertionError: If the script is exhausted and ``repeat_last`` is off.
        """
        if index < len(self.turns):
            return self.turns[index]
        if self.repeat_last and self.turns:
            return self.turns[-1]
        raise AssertionError(
            f"FakeProvider ran out of scripted turns: request #{index + 1} arrived "
            f"but only {len(self.turns)} turn(s) were scripted."
        )

    async def stream(self, req: CompletionRequest) -> AsyncIterator[ProviderEvent]:
        """Record ``req`` and replay the matching scripted turn.

        Args:
            req: The turn the loop wants run.

        Yields:
            The scripted deltas, then exactly one
            :class:`~logpose.providers.base.CompletionDone` (unless the turn
            sets ``emit_done=False``).

        Raises:
            BaseException: Whatever ``ScriptedTurn.error`` holds, if set.
        """
        turn = self._turn_for(len(self.requests))
        self.requests.append(req)
        try:
            if turn.delay:
                await asyncio.sleep(turn.delay)
            if turn.error is not None:
                raise turn.error
            for delta in turn.deltas:
                yield delta
            if turn.emit_done:
                yield turn.completion()
        except GeneratorExit:
            # The consumer stopped iterating early; recorded so cancellation
            # tests can prove the loop closed us rather than leaking us.
            self.closed += 1
            raise
