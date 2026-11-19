"""Run the logpose loop against a local model served by Docker Model Runner.

No API key, no cloud call, no extra dependency.

Prerequisites::

    docker desktop enable model-runner --tcp 12434
    docker model pull ai/gemma4        # or any tool-capable model
    docker model ls                    # confirm it is there

Then::

    uv run python examples/local_docker.py

Docker Model Runner speaks the OpenAI Chat Completions API, so this uses the
``openai-compat`` provider with the runner's address passed explicitly.

Not every local model can call tools. If the run finishes without the tool being
invoked, the model — not logpose — declined to use it; try a larger or more
tool-tuned model by passing its name instead of ``"auto"``.
"""

from __future__ import annotations

import asyncio
from typing import Literal

from logpose import Agent, TextDelta, ThinkingDelta, ToolCall, ToolResult, TurnEnd, tool


@tool
def get_weather(city: str, unit: Literal["c", "f"] = "c") -> str:
    """Get the current weather for a city.

    Args:
        city: The city to look up, e.g. "Pune".
        unit: Temperature unit — "c" for celsius, "f" for fahrenheit.
    """
    # A real implementation would call a weather API here.
    return f"18 degrees {unit}, light rain, humidity 82%"


async def main() -> None:
    """Stream one tool-using turn against the local model."""
    # model="auto" asks the runner for its first model; name one to pin it.
    agent = Agent(
        "openai-compat",
        model="auto",
        base_url="http://localhost:12434/engines/v1",
        # Small local models have small context windows; the 32000 default
        # output ceiling would exceed them.
        max_tokens=4096,
        tools=[get_weather],
        system="You are terse. Use the tools available to you. Answer in one sentence.",
        max_iterations=6,
    )

    print(f"provider: {agent.provider!r}\n")

    # Reasoning and answer are separate channels; label each run so they do not
    # read as one paragraph when the model switches between them.
    channel: str | None = None

    def switch_to(name: str, label: str) -> None:
        nonlocal channel
        if channel != name:
            print(f"\n{label} " if channel else f"{label} ", end="", flush=True)
            channel = name

    async for event in agent.stream("What's the weather in Pune right now?"):
        if isinstance(event, ThinkingDelta):
            switch_to("thinking", "[thinking]")
            print(event.text, end="", flush=True)
        elif isinstance(event, TextDelta):
            switch_to("text", "[answer]")
            print(event.text, end="", flush=True)
        elif isinstance(event, ToolCall):
            channel = None
            print(f"\n\n[tool call] {event.name}({event.input})")
        elif isinstance(event, ToolResult):
            channel = None
            print(f"[tool result] {event.content}")
        elif isinstance(event, TurnEnd):
            channel = None
            print(f"\n[turn end] stop={event.stop_reason} usage={event.usage.model_dump()}")

    # Local servers are long-lived HTTP clients; close it when you are done.
    await agent.provider.aclose()


if __name__ == "__main__":
    asyncio.run(main())
