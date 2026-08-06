"""End-to-end smoke example: one tool, one agent, streamed events.

This example talks to a real model and doubles as a manual verification for the
selected authentication path.

Run it
------
Anthropic API key (the default)::

    export ANTHROPIC_API_KEY=sk-ant-api03-...
    uv run python examples/weather.py

Experimental Claude Code subscription auth::

    claude setup-token
    export CLAUDE_CODE_OAUTH_TOKEN=sk-ant-oat01-...
    uv run python examples/weather.py --claude-code

If Claude Code is already installed and logged in, the export is optional.
This path depends on unstable CLI authentication details and may stop working
without notice.

Expected output: a couple of thinking/text deltas, a ``get_weather`` tool call
and its result, then a final answer and a non-zero token count.
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Literal

from logpose import (
    Agent,
    AuthError,
    LogposeError,
    RunEnd,
    TextDelta,
    ThinkingDelta,
    ToolCall,
    ToolResult,
    TurnEnd,
    tool,
)

FAKE_WEATHER = {
    "pune": (31, "hazy sunshine"),
    "goa": (29, "humid, light rain"),
    "reykjavik": (3, "sleet"),
}


@tool
def get_weather(city: str, unit: Literal["c", "f"] = "c") -> str:
    """Get the current weather in a city.

    Args:
        city: City name, e.g. "Pune".
        unit: Temperature unit — "c" for Celsius, "f" for Fahrenheit.
    """
    celsius, description = FAKE_WEATHER.get(city.strip().lower(), (22, "clear"))
    degrees = celsius if unit == "c" else round(celsius * 9 / 5 + 32)
    return f"{degrees}{unit.upper()}, {description}"


def build_agent(*, claude_code: bool) -> Agent:
    """Build the agent for this example.

    Args:
        claude_code: When ``True``, use the experimental ``claude-code``
            subscription provider. Otherwise use ``anthropic`` with an API key.

    Returns:
        A configured :class:`~logpose.Agent`.
    """
    if claude_code:
        return Agent(
            "claude-code",
            tools=[get_weather],
            system="You are a terse weather assistant.",
            max_iterations=8,
        )

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit(
            "Set ANTHROPIC_API_KEY, or pass --claude-code for the experimental path."
        )
    return Agent(
        "anthropic",
        api_key=api_key,
        tools=[get_weather],
        system="You are a terse weather assistant.",
        max_iterations=8,
    )


async def main() -> int:
    """Run one streamed request and print what happened.

    Returns:
        A process exit code.
    """
    use_claude_code = "--claude-code" in sys.argv
    agent = build_agent(claude_code=use_claude_code)
    prompt = "What's the weather in Pune and in Reykjavik? Answer in one sentence."

    print(f"> {prompt}\n")
    try:
        async for event in agent.stream(prompt):
            if isinstance(event, ThinkingDelta):
                print(f"\033[2m{event.text}\033[0m", end="", flush=True)
            elif isinstance(event, TextDelta):
                print(event.text, end="", flush=True)
            elif isinstance(event, ToolCall):
                print(f"\n  -> {event.name}({event.input})", flush=True)
            elif isinstance(event, ToolResult):
                marker = "!!" if event.is_error else "<-"
                print(f"  {marker} {event.content}\n", flush=True)
            elif isinstance(event, TurnEnd):
                print(f"\n  [turn ended: {event.stop_reason}]", flush=True)
            elif isinstance(event, RunEnd):
                usage = event.result.usage
                print(f"\n\n--- {event.result.iterations} iteration(s) ---")
                print(f"answer: {event.result.text}")
                print(
                    f"tokens: in={usage.input_tokens} out={usage.output_tokens} "
                    f"cache_read={usage.cache_read_input_tokens} "
                    f"cache_write={usage.cache_creation_input_tokens}"
                )
    except AuthError as exc:
        print(f"\nauth failed: {exc}", file=sys.stderr)
        if use_claude_code:
            print("Run `claude setup-token` and export CLAUDE_CODE_OAUTH_TOKEN.", file=sys.stderr)
        else:
            print("Export ANTHROPIC_API_KEY.", file=sys.stderr)
        return 2
    except LogposeError as exc:
        print(f"\n{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        await agent.aclose()
    return 0


# The blocking spelling of the same thing, for callers that are not async:
#
#     from logpose import SyncAgent
#
#     with SyncAgent(build_agent(claude_code=False)) as agent:
#         for event in agent.stream(prompt):
#             ...

if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
