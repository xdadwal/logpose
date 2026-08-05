"""logpose — a universal agentic loop that services embed.

logpose owns the loop: it drives a provider, executes consumer-defined tools, and
streams normalized events. It does not wrap any vendor agent harness, so nothing
provider-specific leaks above :mod:`logpose.providers`.

.. code-block:: python

    import asyncio

    from logpose import Agent, tool


    @tool
    def get_weather(city: str) -> str:
        '''Get the current weather.

        Args:
            city: City name, e.g. "Pune".
        '''
        return f"22C and sunny in {city}"


    async def main() -> None:
        agent = Agent("anthropic", tools=[get_weather], system="Be concise.")
        result = await agent.run("What's the weather in Pune?")
        print(result.text, result.usage)


    asyncio.run(main())

Not async? :class:`SyncAgent`, :func:`run_sync`, and :func:`stream_sync` wrap the
same loop for blocking callers.

Importing this package is cheap and side-effect free: no provider SDK is
imported, no credential is read, and no network call is made. Backends are
resolved by name (:func:`resolve`) and import their SDK lazily, so
``import logpose`` works on a machine with no configuration at all.
"""

from __future__ import annotations

from logpose.agent import (
    Agent,
    Conversation,
    ToolGate,
    ToolGateOutcome,
    ToolGateResult,
)
from logpose.errors import (
    AuthError,
    LogposeError,
    MaxIterationsError,
    ProviderError,
    ToolExecutionError,
    ToolSchemaError,
)
from logpose.events import (
    Event,
    RunEnd,
    RunResult,
    TextDelta,
    ThinkingDelta,
    ToolCall,
    ToolResult,
    TurnEnd,
)
from logpose.messages import (
    ContentBlock,
    Message,
    RawBlock,
    RedactedThinkingBlock,
    StopReason,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
)
from logpose.providers import (
    CompletionDone,
    CompletionRequest,
    Provider,
    ProviderEvent,
    ProviderTextDelta,
    ProviderThinkingDelta,
    ToolSpec,
    known_providers,
    register,
    resolve,
)
from logpose.sync import SyncAgent, close_sync, run_sync, stream_sync
from logpose.tools import ToolDef, discover_tools, tool

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # the loop
    "Agent",
    "Conversation",
    "ToolGate",
    "ToolGateOutcome",
    "ToolGateResult",
    # tools
    "tool",
    "ToolDef",
    "discover_tools",
    # sync facade
    "SyncAgent",
    "run_sync",
    "stream_sync",
    "close_sync",
    # messages
    "Message",
    "Usage",
    "StopReason",
    "ContentBlock",
    "TextBlock",
    "ThinkingBlock",
    "RedactedThinkingBlock",
    "ToolUseBlock",
    "ToolResultBlock",
    "RawBlock",
    # events
    "Event",
    "RunResult",
    "RunEnd",
    "TextDelta",
    "ThinkingDelta",
    "ToolCall",
    "ToolResult",
    "TurnEnd",
    # errors
    "LogposeError",
    "AuthError",
    "ProviderError",
    "MaxIterationsError",
    "ToolSchemaError",
    "ToolExecutionError",
    # provider seam (for writing your own backend)
    "Provider",
    "ProviderEvent",
    "ProviderTextDelta",
    "ProviderThinkingDelta",
    "CompletionDone",
    "CompletionRequest",
    "ToolSpec",
    "register",
    "resolve",
    "known_providers",
]
