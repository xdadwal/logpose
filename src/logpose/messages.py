"""Provider-neutral message model.

Everything above ``logpose.providers`` speaks this model; providers translate it
to and from their native wire format.

Round-tripping requirement
--------------------------
:class:`ThinkingBlock` carries ``signature`` and :class:`RedactedThinkingBlock`
carries ``data`` because the Anthropic API **requires thinking blocks to be
echoed back unchanged** on subsequent turns of the same conversation. Editing,
dropping, or reconstructing those fields makes the API reject the turn, so this
model must round-trip them losslessly (``Message.model_validate(msg.model_dump())``
is an identity for every block type).

The same requirement is why :class:`RawBlock` exists: a provider may return a
block type logpose has no model for (server tool use, search results, ...), and
an assistant turn that lost those blocks cannot be re-sent — the API reads the
remainder as a prefill and rejects it. :class:`RawBlock` keeps the payload
opaque and verbatim so every assistant turn survives a round trip.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, TypeAdapter

__all__ = [
    "TextBlock",
    "ThinkingBlock",
    "RedactedThinkingBlock",
    "ToolUseBlock",
    "ToolResultBlock",
    "RawBlock",
    "ContentBlock",
    "CONTENT_BLOCK_ADAPTER",
    "Message",
    "Usage",
    "StopReason",
]


class TextBlock(BaseModel):
    """Assistant- or user-authored plain text.

    Attributes:
        type: Discriminator, always ``"text"``.
        text: The text content.
    """

    type: Literal["text"] = "text"
    text: str


class ThinkingBlock(BaseModel):
    """A block of model reasoning.

    Attributes:
        type: Discriminator, always ``"thinking"``.
        thinking: The reasoning text (may be empty when the provider omits it).
        signature: Opaque provider signature. Must be preserved verbatim and
            echoed back unchanged on later turns of the same conversation.
    """

    type: Literal["thinking"] = "thinking"
    thinking: str
    signature: str | None = None


class RedactedThinkingBlock(BaseModel):
    """Reasoning the provider returned in encrypted form.

    Attributes:
        type: Discriminator, always ``"redacted_thinking"``.
        data: Opaque encrypted payload. Must be preserved verbatim and echoed
            back unchanged on later turns of the same conversation.
    """

    type: Literal["redacted_thinking"] = "redacted_thinking"
    data: str


class ToolUseBlock(BaseModel):
    """A model request to invoke a tool.

    Attributes:
        type: Discriminator, always ``"tool_use"``.
        id: Provider-assigned identifier; the matching result must echo it.
        name: Name of the tool to invoke.
        input: Already-parsed arguments for the tool.
    """

    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any]


class ToolResultBlock(BaseModel):
    """The outcome of a tool invocation, sent back to the model.

    Attributes:
        type: Discriminator, always ``"tool_result"``.
        tool_use_id: The ``id`` of the :class:`ToolUseBlock` this answers.
        content: The result rendered as text.
        is_error: Whether the tool failed. Errors are reported to the model
            rather than raised, so it can adapt.
    """

    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str
    is_error: bool = False


class RawBlock(BaseModel):
    """A provider block logpose has no model for, preserved verbatim.

    Providers put unmodelled block types (server tool use, web-search results,
    code-execution output, ...) in here rather than dropping them, because an
    assistant turn that lost blocks cannot be re-sent: the API reads the
    remainder as a prefill and rejects it. That matters most for
    ``stop_reason == "pause_turn"``, whose documented resume is to re-send the
    paused assistant turn unchanged.

    ``data`` is opaque. logpose never inspects it beyond its ``"type"`` key and
    emits it back on the wire exactly as it came in.

    Attributes:
        type: Discriminator, always ``"raw"``. This is logpose's own tag; the
            provider's block type lives under ``data["type"]``.
        data: The provider's block, as a JSON-compatible dict.
    """

    type: Literal["raw"] = "raw"
    data: dict[str, Any]

    @property
    def block_type(self) -> str:
        """The provider's own block type, or ``""`` when the payload lacks one."""
        value = self.data.get("type")
        return value if isinstance(value, str) else ""


ContentBlock = Annotated[
    Union[  # noqa: UP007 - explicit Union keeps the discriminator readable
        TextBlock,
        ThinkingBlock,
        RedactedThinkingBlock,
        ToolUseBlock,
        ToolResultBlock,
        RawBlock,
    ],
    Field(discriminator="type"),
]
"""Discriminated union of every content block type, keyed on ``type``."""

CONTENT_BLOCK_ADAPTER: TypeAdapter[ContentBlock] = TypeAdapter(ContentBlock)
"""Validator for a single :data:`ContentBlock` (useful when parsing provider payloads)."""


class Message(BaseModel):
    """One turn of a conversation.

    Attributes:
        role: Who authored the turn.
        content: The ordered content blocks making up the turn.
    """

    role: Literal["user", "assistant"]
    content: list[ContentBlock]

    @classmethod
    def user(cls, text: str) -> Message:
        """Build a user turn containing a single text block.

        Args:
            text: The user's message.

        Returns:
            A ``user`` message with one :class:`TextBlock`.
        """
        return cls(role="user", content=[TextBlock(text=text)])

    @classmethod
    def assistant_text(cls, text: str) -> Message:
        """Build an assistant turn containing a single text block.

        Args:
            text: The assistant's message.

        Returns:
            An ``assistant`` message with one :class:`TextBlock`.
        """
        return cls(role="assistant", content=[TextBlock(text=text)])

    @property
    def text(self) -> str:
        """The concatenation of every :class:`TextBlock` in the turn.

        Non-text blocks (thinking, tool use, tool results) are ignored.

        Returns:
            All text-block contents joined with no separator.
        """
        return "".join(block.text for block in self.content if isinstance(block, TextBlock))


class Usage(BaseModel):
    """Token accounting for one or more turns.

    Attributes:
        input_tokens: Uncached prompt tokens billed at full rate.
        output_tokens: Tokens generated by the model.
        cache_read_input_tokens: Prompt tokens served from the cache.
        cache_creation_input_tokens: Prompt tokens written to the cache.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        """Aggregate two usage records field by field.

        Args:
            other: The usage to add to this one.

        Returns:
            A new :class:`Usage` holding the per-field sums.
        """
        if not isinstance(other, Usage):
            return NotImplemented
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_input_tokens=self.cache_read_input_tokens + other.cache_read_input_tokens,
            cache_creation_input_tokens=(
                self.cache_creation_input_tokens + other.cache_creation_input_tokens
            ),
        )

    def __radd__(self, other: object) -> Usage:
        """Support ``sum(usages)`` by treating ``0`` as the empty usage.

        Args:
            other: The left-hand operand; only ``0`` and :class:`Usage` are supported.

        Returns:
            A new :class:`Usage` holding the per-field sums.
        """
        if other == 0:
            return self
        if isinstance(other, Usage):
            return other.__add__(self)
        return NotImplemented


StopReason = Literal[
    "end_turn",
    "tool_use",
    "max_tokens",
    "stop_sequence",
    "refusal",
    "pause_turn",
]
"""Why the model stopped generating a turn."""
