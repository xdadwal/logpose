"""Tolerant parsing of tool arguments that arrive as a JSON string.

Anthropic returns tool input already parsed. Every OpenAI-shaped API — Chat
Completions and the Responses API alike — sends a *string* the client has to
parse, and models routinely get that string wrong. Both providers must fail the
same way when they do, and must use the **same** sentinel key: consumer code
that special-cases :data:`UNPARSED_ARGUMENTS_KEY` would otherwise silently miss
one backend.

Shared here rather than imported across sibling providers for the same reason
:mod:`logpose.providers._redact` exists — a provider module must never import
another provider module, or resolving one backend drags in the other's SDK.
"""

from __future__ import annotations

import json
from typing import Any

__all__ = ["UNPARSED_ARGUMENTS_KEY", "parse_tool_arguments"]

UNPARSED_ARGUMENTS_KEY = "__logpose_unparsed_arguments__"
"""Key holding tool arguments the model emitted as invalid JSON.

Surfacing the raw string as an unexpected argument makes the loop's schema
validation fail, which comes back to the model as a tool error it can retry —
strictly better than raising (kills the run) or silently substituting ``{}``
(the tool runs with wrong arguments).
"""


def parse_tool_arguments(raw: str) -> dict[str, Any]:
    """Parse a tool call's arguments, tolerating malformed JSON.

    Args:
        raw: The ``arguments`` string as the model produced it.

    Returns:
        The parsed object, or ``{UNPARSED_ARGUMENTS_KEY: raw}`` when the model
        produced something that is not a JSON object.
    """
    if not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return {UNPARSED_ARGUMENTS_KEY: raw}
    return parsed if isinstance(parsed, dict) else {UNPARSED_ARGUMENTS_KEY: raw}
