"""Built-in context compaction: pure transforms and the recall tool.

Nothing here touches stored history. Every function works on what is *sent*
to the model and returns fresh copies, so the caller's
:class:`~logpose.agent.Conversation`, the session store and
``RunResult.messages`` keep the full transcript.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Sequence

from logpose.messages import ContentBlock, Message, ToolResultBlock, ToolUseBlock
from logpose.providers.base import ToolSpec
from logpose.tools import ToolDef, tool

__all__ = [
    "RECALL_TOOL_NAME",
    "RECALL_PAGE_CHARS",
    "estimate_tokens",
    "apply_pointers",
    "make_recall_tool",
]

RECALL_TOOL_NAME = "recall_tool_output"
"""Name of the tool the built-in policy adds so the model can read pointed outputs."""

RECALL_PAGE_CHARS = 8000
"""Characters one ``recall_tool_output`` call returns at most."""

_CHARS_PER_TOKEN = 4


def estimate_tokens(
    messages: Sequence[Message], system: str | None, tools: Sequence[ToolSpec]
) -> int:
    """Estimate a request's input tokens as characters of its JSON pieces ÷ 4.

    Args:
        messages: The messages the request carries.
        system: The system prompt, if any.
        tools: The tool schemas the request advertises.

    Returns:
        The estimated token count, rounded up.
    """
    chars = len(json.dumps([message.model_dump(mode="json") for message in messages]))
    if system:
        chars += len(system)
    if tools:
        chars += len(json.dumps([dataclasses.asdict(spec) for spec in tools]))
    return -(-chars // _CHARS_PER_TOKEN)


def _pointer_text(tool_use_id: str, chars: int) -> str:
    return (
        f"[tool output stored: {tool_use_id}, {chars} chars. "
        f'Call {RECALL_TOOL_NAME}("{tool_use_id}") to read it.]'
    )


def apply_pointers(
    messages: Sequence[Message], *, current_start: int, min_chars: int
) -> tuple[list[Message], int]:
    """Replace long tool outputs from earlier runs with short pointers.

    A :class:`~logpose.messages.ToolResultBlock` becomes a pointer only when it
    is longer than ``min_chars``, comes before ``current_start``, and the model
    has already answered it (an assistant message follows it before
    ``current_start``). Results a resumed run has not shown the model yet, and
    results of ``recall_tool_output`` itself, stay in full. Each pointer keeps
    its ``tool_use_id`` and ``is_error``, so tool pairing stays valid.

    Args:
        messages: The request's messages. Never mutated.
        current_start: Index of the first message appended by the current run;
            outputs at or after it are sent in full.
        min_chars: Outputs at or below this length stay inline.

    Returns:
        New message copies and the number of pointers applied.
    """
    # Only results already followed by an assistant message may be pointed at.
    boundary = 0
    for index in range(min(current_start, len(messages)) - 1, -1, -1):
        if messages[index].role == "assistant":
            boundary = index + 1
            break
    recall_ids = {
        block.id
        for message in messages[:boundary]
        for block in message.content
        if isinstance(block, ToolUseBlock) and block.name == RECALL_TOOL_NAME
    }

    def pointed(block: object) -> bool:
        return (
            isinstance(block, ToolResultBlock)
            and len(block.content) > min_chars
            and block.tool_use_id not in recall_ids
        )

    out: list[Message] = []
    applied = 0
    for index, message in enumerate(messages):
        if index >= boundary or not any(pointed(block) for block in message.content):
            out.append(message.model_copy(deep=True))
            continue
        blocks: list[ContentBlock] = []
        for block in message.content:
            if isinstance(block, ToolResultBlock) and pointed(block):
                blocks.append(
                    block.model_copy(
                        update={"content": _pointer_text(block.tool_use_id, len(block.content))}
                    )
                )
                applied += 1
            else:
                blocks.append(block.model_copy(deep=True))
        out.append(message.model_copy(update={"content": blocks}))
    return out, applied


def _find_output(messages: Sequence[Message], call_id: str) -> str | None:
    for message in reversed(messages):
        for block in message.content:
            if isinstance(block, ToolResultBlock) and block.tool_use_id == call_id:
                return block.content
    return None


def make_recall_tool(history: Callable[[], Sequence[Message]]) -> ToolDef:
    """Build the ``recall_tool_output`` tool over a run's full history.

    Args:
        history: Returns the current run's full (un-pointered) history when called.

    Returns:
        The :class:`~logpose.tools.ToolDef` to advertise to the model.
    """

    @tool(
        name=RECALL_TOOL_NAME,
        description=(
            "Read the full text of an earlier tool output that was replaced by a "
            "'[tool output stored: ...]' pointer. Returns up to "
            f"{RECALL_PAGE_CHARS} characters starting at offset."
        ),
    )
    async def recall_tool_output(call_id: str, offset: int = 0) -> str:
        """Read a stored tool output.

        Args:
            call_id: The tool call id named in the pointer.
            offset: Character offset to start reading from (default 0).
        """
        content = _find_output(history(), call_id)
        if content is None:
            return f"No stored tool output with id '{call_id}'."
        total = len(content)
        if offset < 0 or (offset > 0 and offset >= total):
            return f"Offset {offset} is outside tool output '{call_id}' ({total} chars)."
        end = min(offset + RECALL_PAGE_CHARS, total)
        page = content[offset:end]
        if end < total:
            page += (
                f"\n\n[Showing characters {offset}-{end} of {total}. "
                f'Call {RECALL_TOOL_NAME}("{call_id}", offset={end}) to read more.]'
            )
        return page

    return recall_tool_output
