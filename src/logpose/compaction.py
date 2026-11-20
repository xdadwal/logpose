"""Built-in context compaction: pure transforms and the recall tool.

Nothing here touches stored history. Every function works on what is *sent*
to the model and returns fresh copies, so the caller's
:class:`~logpose.agent.Conversation`, the session store and
``RunResult.messages`` keep the full transcript.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Callable, Sequence

from logpose.messages import ContentBlock, Message, TextBlock, ToolResultBlock, ToolUseBlock
from logpose.providers.base import ToolSpec
from logpose.tools import ToolDef, tool

__all__ = [
    "RECALL_TOOL_NAME",
    "RECALL_PAGE_CHARS",
    "estimate_tokens",
    "apply_pointers",
    "make_recall_tool",
    "turn_starts",
    "SUMMARY_PREFIX",
    "SUMMARY_MAX_TOKENS",
]

RECALL_TOOL_NAME = "recall_tool_output"
"""Name of the tool the built-in policy adds so the model can read pointed outputs."""

RECALL_PAGE_CHARS = 8000
"""Characters one ``recall_tool_output`` call returns at most."""

_CHARS_PER_TOKEN = 4

SUMMARY_PREFIX = "[Summary of earlier conversation]\n"
"""Text that starts the assistant-role message carrying the summary."""

SUMMARY_MAX_TOKENS = 2048
"""Floor for the summary call's output ceiling (the agent's ``max_tokens`` if larger)."""

SUMMARY_INSTRUCTION = (
    "You compress the earlier part of a conversation between a user and an AI "
    "assistant so the assistant can continue it without the original messages.\n"
    "Write a concise summary that keeps every fact, decision, identifier (IDs, "
    "names, numbers, file paths, URLs, tool call ids) and every open or "
    "unresolved question. Note which tools were called and what they returned, "
    "briefly. Do not add any information that is not in the transcript, do not "
    "answer open questions, and do not address the user. Output only the summary."
)
"""System prompt for the summary call."""

_SMOOTHING = 0.7
"""Weight the calibration ratio keeps on its old value after each turn."""

RATIO_MIN = 0.5
RATIO_MAX = 2.0
"""Bounds on the calibration ratio, so a server that under- or over-reports
prompt tokens (prompt-cache accounting, for example) cannot disable the
threshold or the overflow guard."""


@dataclasses.dataclass
class _CompactionState:
    """Per-conversation compaction state (private; lives on ``Conversation``).

    Attributes:
        ratio: Calibration factor applied to raw estimates: smoothed
            ``reported input tokens ÷ estimate``, starting at 1.0.
        summary: Cached summary of ``messages[first turn + 1 : cut_index]``.
        cut_index: Index of the first message sent verbatim after the summary.
        fingerprint: Identifies ``messages[:cut_index]`` the summary was made
            from; a mismatch means the caller edited history.
    """

    ratio: float = 1.0
    summary: str | None = None
    cut_index: int = 0
    fingerprint: str = ""

    def calibrate(self, reported: int, estimate: int) -> None:
        """Fold one turn's reported input tokens into the ratio.

        Args:
            reported: Input tokens the provider reported for the request
                (uncached + cache reads + cache writes). ``0`` means unknown.
            estimate: The raw (uncalibrated) estimate of that request.

        The result is clamped to [``RATIO_MIN``, ``RATIO_MAX``].
        """
        if reported > 0 and estimate > 0:
            ratio = _SMOOTHING * self.ratio + (1 - _SMOOTHING) * (reported / estimate)
            self.ratio = min(RATIO_MAX, max(RATIO_MIN, ratio))

    def clear_summary(self) -> None:
        """Forget the cached summary and cut point."""
        self.summary = None
        self.cut_index = 0
        self.fingerprint = ""


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
    chars = len(_dumps([message.model_dump(mode="json") for message in messages]))
    return tokens_for(chars + overhead_chars(system, tools))


def overhead_chars(system: str | None, tools: Sequence[ToolSpec]) -> int:
    """Characters the system prompt and tool schemas add to an estimate."""
    chars = len(system) if system else 0
    if tools:
        chars += len(_dumps([dataclasses.asdict(spec) for spec in tools]))
    return chars


def _dumps(value: object) -> str:
    """JSON text an estimate counts: non-ASCII kept as-is, not ``\\uXXXX`` escaped.

    An escape is six characters for one code point, which would inflate the
    estimate for CJK or emoji text about sixfold.
    """
    return json.dumps(value, ensure_ascii=False)


def tokens_for(chars: int) -> int:
    """Convert a character count to estimated tokens, rounding up."""
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


def turn_starts(messages: Sequence[Message]) -> list[int]:
    """Return the indices of caller-authored user turns.

    A turn starts at a user message that carries a :class:`TextBlock` and no
    :class:`ToolResultBlock`. Cutting the history at one of these never splits
    a tool call from its result: the results answering an assistant's calls
    sit in a tool-result message, which is never a turn start.

    Args:
        messages: The history to scan.

    Returns:
        Indices in ascending order.
    """
    return [
        index
        for index, message in enumerate(messages)
        if message.role == "user"
        and any(isinstance(block, TextBlock) for block in message.content)
        and not any(isinstance(block, ToolResultBlock) for block in message.content)
    ]


def fingerprint(messages: Sequence[Message]) -> str:
    """Identify a run of messages: their count plus a hash of their content."""
    digest = hashlib.sha256()
    for message in messages:
        digest.update(message.model_dump_json().encode())
        digest.update(b"\x00")
    return f"{len(messages)}:{digest.hexdigest()}"


def summary_message(summary: str) -> Message:
    """The message that stands in for the summarized span.

    Assistant role: it sits between the pinned first user turn and a user turn
    start, so the request keeps strict user/assistant alternation, which chat
    templates on vLLM, llama.cpp and Ollama enforce.
    """
    return Message(role="assistant", content=[TextBlock(text=SUMMARY_PREFIX + summary)])


def message_chars(message: Message) -> int:
    """Characters ``message`` adds to :func:`estimate_tokens`'s JSON list."""
    return len(_dumps(message.model_dump(mode="json")))


def choose_cut(
    messages: Sequence[Message],
    candidates: Sequence[int],
    *,
    fixed_chars: int,
    ratio: float,
    target_tokens: float,
) -> int:
    """Pick the earliest cut whose remaining request fits ``target_tokens``.

    The kept request is the fixed part (system, tools, the pinned first turn
    and the summary message) plus ``messages[cut:]``. The span grows by whole
    turns from the oldest end; if no candidate reaches the target, the last
    (largest) one is used.

    Args:
        messages: The request's messages (already pointered).
        candidates: Allowed cut indices, ascending and non-empty.
        fixed_chars: Characters of everything kept besides ``messages[cut:]``.
        ratio: Calibration ratio applied to the raw estimate.
        target_tokens: The estimate to get down to.

    Returns:
        The chosen cut index.
    """
    sizes = [message_chars(message) for message in messages]
    suffix = [0] * (len(sizes) + 1)
    for index in range(len(sizes) - 1, -1, -1):
        # ", " separates list items in _dumps output.
        suffix[index] = suffix[index + 1] + sizes[index] + 2
    for cut in candidates:
        if tokens_for(fixed_chars + suffix[cut]) * ratio <= target_tokens:
            return cut
    return candidates[-1]


def render_transcript(messages: Sequence[Message], previous: str | None) -> str:
    """Render the span to summarize as plain text for the summary call.

    Args:
        messages: The span (pointers already applied).
        previous: An earlier summary this one extends, if any.

    Returns:
        The user message text for the summary call.
    """
    lines: list[str] = []
    if previous:
        lines += ["Earlier summary (extend it):", previous, ""]
    lines.append("Transcript to summarize:")
    for message in messages:
        speaker = "User" if message.role == "user" else "Assistant"
        for block in message.content:
            if isinstance(block, TextBlock):
                lines.append(f"{speaker}: {block.text}")
            elif isinstance(block, ToolUseBlock):
                lines.append(
                    f"Assistant called tool {block.name} (id {block.id}) with "
                    f"{json.dumps(block.input, sort_keys=True)}"
                )
            elif isinstance(block, ToolResultBlock):
                label = "error" if block.is_error else "result"
                lines.append(f"Tool {label} (id {block.tool_use_id}): {block.content}")
    return "\n".join(lines)
