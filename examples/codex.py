"""End-to-end Codex example: one tool, one agent, streamed events.

Together with ``examples/weather.py`` this is the only code in the repo that talks
to a real model, so it doubles as the manual verification for the Codex auth path
and — more importantly — for the reasoning round trip, which no mocked test can
fully prove.

Run it
------
Subscription auth::

    codex login                              # from the Codex CLI
    uv run python examples/codex.py

Nothing to export: logpose reads ``~/.codex/auth.json`` read-only.

Bring-your-own-key instead::

    export OPENAI_API_KEY=sk-proj-...
    uv run python examples/codex.py --byok

Expected output: some dim reasoning text, a ``get_weather`` call and its result,
then a final answer, a non-zero token count, and — the thing actually being
verified — ``reasoning items resent: 1`` or more. That line proves the second turn
carried the model's opaque reasoning back in position; if it reads ``0`` while
reasoning events did stream, the round trip is broken even though the run
succeeded.
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
    RawBlock,
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
        byok: When ``True``, use the ``openai`` provider, which accepts only an
            API key and talks to ``api.openai.com``. When ``False``, use
            ``codex``, which accepts only a subscription token and talks to the
            Codex backend. The two are separate providers, so the choice is made
            here rather than inferred from whatever credential turns up.

    Returns:
        A configured :class:`~logpose.Agent`.
    """
    if byok:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit("--byok needs OPENAI_API_KEY to be set.")
        return Agent(
            "openai",
            api_key=api_key,
            tools=[get_weather],
            system="You are a terse weather assistant.",
        )

    # Default: the Codex subscription store. reasoning_effort is left at "medium";
    # "low" makes reasoning summaries mostly empty, which makes this example look
    # broken when it is not.
    return Agent(
        "codex",
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
                result = event.result
                usage = result.usage
                # Reasoning lives in a RawBlock, not a ThinkingBlock: it carries an
                # opaque encrypted_content that has to go back byte-exact.
                reasoning = [
                    block
                    for message in result.messages
                    for block in message.content
                    if isinstance(block, RawBlock) and block.block_type == "reasoning"
                ]
                print(f"\n\n--- {result.iterations} iteration(s) ---")
                print(f"answer: {result.text}")
                print(
                    f"tokens: in={usage.input_tokens} out={usage.output_tokens} "
                    f"cache_read={usage.cache_read_input_tokens}"
                )
                print(f"reasoning items resent: {len(reasoning)}")
    except AuthError as exc:
        print(f"\nauth failed: {exc}", file=sys.stderr)
        print("Run `codex login`, or export OPENAI_API_KEY.", file=sys.stderr)
        return 2
    except LogposeError as exc:
        print(f"\n{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        await agent.aclose()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
