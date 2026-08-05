"""End-to-end smoke example: one tool, one agent, streamed events.

This is the only place in the repo that talks to a real model, so it doubles as
the manual verification for the auth path.

Run it
------
Subscription auth (the default for v0.1)::

    claude setup-token                       # prints a long-lived OAuth token
    export CLAUDE_CODE_OAUTH_TOKEN=sk-ant-oat01-...
    uv run python examples/weather.py

If Claude Code is already installed and logged in on this machine, the export is
optional: logpose discovers the credential store read-only (macOS Keychain, or
``~/.claude/.credentials.json``).

Bring-your-own-key instead::

    export ANTHROPIC_API_KEY=sk-ant-api03-...
    uv run python examples/weather.py --byok

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


def build_agent(*, byok: bool) -> Agent:
    """Build the agent for this example.

    Args:
        byok: When ``True``, force the bring-your-own-key path by passing
            ``ANTHROPIC_API_KEY`` explicitly. When ``False``, let logpose resolve
            a credential itself (subscription token first).

    Returns:
        A configured :class:`~logpose.Agent`.
    """
    # --- BYOK variant -------------------------------------------------------
    # An explicit api_key outranks every environment variable and every stored
    # credential, so this is how a service pins itself to per-token billing:
    #
    #     return Agent(
    #         "anthropic",
    #         api_key=os.environ["ANTHROPIC_API_KEY"],
    #         model="claude-opus-5",
    #         tools=[get_weather],
    #         system="You are a terse weather assistant.",
    #     )
    if byok:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise SystemExit("--byok needs ANTHROPIC_API_KEY to be set.")
        return Agent(
            "anthropic",
            api_key=api_key,
            tools=[get_weather],
            system="You are a terse weather assistant.",
        )

    # --- default: subscription token, then BYOK, then the credential store ---
    return Agent(
        "anthropic",
        model="claude-opus-5",
        tools=[get_weather],
        system="You are a terse weather assistant.",
        max_iterations=8,
    )


async def main() -> int:
    """Run one streamed request and print what happened.

    Returns:
        A process exit code.
    """
    agent = build_agent(byok="--byok" in sys.argv)
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
        print("Run `claude setup-token` and export CLAUDE_CODE_OAUTH_TOKEN.", file=sys.stderr)
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
#     with SyncAgent(build_agent(byok=False)) as agent:
#         for event in agent.stream(prompt):
#             ...

if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
