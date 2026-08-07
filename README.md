# logpose

[![CI](https://github.com/xdadwal/logpose/actions/workflows/ci.yml/badge.svg)](https://github.com/xdadwal/logpose/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Checked with mypy](https://img.shields.io/badge/mypy-checked-blue)](https://mypy-lang.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**A provider-neutral agent loop for Python applications.**

logpose helps applications run tool-using language models through one small,
typed API. It manages the conversation loop, validates and executes application
tools, streams normalized events, and returns complete message history and token
usage.

Use it to build assistants, workflow automation, developer tools, support
experiences, or any service where a model needs to call Python functions and
continue working with their results.

```text
your application -> logpose Agent -> model provider
                         |
                         +-> your Python tools
```

## What it provides

- One async-first loop for model turns, tool calls, tool results, and follow-up
  turns, plus a synchronous facade.
- Typed Python tools with JSON Schema generated from annotations and Google-style
  docstrings.
- Streaming events for text, reasoning summaries, tool activity, turn completion,
  and final results.
- Concurrent execution for parallel tool calls, with an optional gate before any
  tool starts.
- Mutable multi-turn conversations with provider reasoning state preserved when
  the backend supports it.
- Provider discovery, readiness checks, model listing, and a protocol for adding
  custom backends.
- Credential redaction in provider errors and tracebacks.

logpose is currently **alpha software**. The public surface is usable, but APIs
may change before 1.0.

## Install

Python 3.10 or newer is required. The project is not published on PyPI yet, so
install it directly from GitHub:

```bash
uv add git+https://github.com/xdadwal/logpose.git
```

or:

```bash
python -m pip install "git+https://github.com/xdadwal/logpose.git"
```

> The `logpose` name on PyPI currently belongs to an unrelated project. Until a
> distribution name is announced here, use the Git URL above.

Runtime dependencies are `anthropic`, `pydantic`, and `httpx`. Docker Model
Runner and OpenAI-compatible servers require no additional Python packages.

## Quickstart

Set an API key:

```bash
export ANTHROPIC_API_KEY=sk-ant-api03-...
```

Then create a tool and run an agent:

```python
import asyncio
from typing import Literal

from logpose import Agent, tool


@tool
def get_weather(city: str, unit: Literal["c", "f"] = "c") -> str:
    """Get the current weather for a city.

    Args:
        city: City name, for example "Pune".
        unit: Temperature unit.
    """
    return f"31 degrees {unit.upper()} and hazy in {city}"


async def main() -> None:
    agent = Agent(
        "anthropic",
        tools=[get_weather],
        system="Answer in one concise sentence.",
    )
    try:
        result = await agent.run("What is the weather in Pune?")
        print(result.text)
        print(result.usage)
    finally:
        await agent.aclose()


asyncio.run(main())
```

The same agent loop works with another backend by changing the provider name and
supplying that provider's configuration. See [Authentication and
providers](#authentication-and-providers).

A runnable streaming version is available in
[`examples/weather.py`](examples/weather.py).

## Authentication and providers

Each provider name selects an explicit backend and credential type.

| Provider | Use case | Authentication |
| --- | --- | --- |
| `anthropic` | Anthropic Messages API | `ANTHROPIC_API_KEY` or `api_key=` |
| `openai` | OpenAI Responses API | `OPENAI_API_KEY` or `api_key=` |
| `docker` | Models hosted locally by Docker Model Runner | None |
| `openai-compat` | A configured Chat Completions-compatible server | Optional, server-dependent |
| `claude-code` | Claude Code subscription session | Local CLI login or OAuth token; experimental |
| `codex` | Codex subscription session | Local CLI login; experimental |

Credentials may be passed directly to `Agent`, which takes precedence over the
corresponding environment variable. Credential stores are read-only; refreshed
subscription credentials are kept in memory and are not written back to disk.

### Anthropic API key

```bash
export ANTHROPIC_API_KEY=sk-ant-api03-...
```

```python
agent = Agent("anthropic")
```

### OpenAI API key

```bash
export OPENAI_API_KEY=sk-proj-...
```

```python
agent = Agent("openai")
```

The OpenAI provider uses the Responses API. Select a model with `model=` or
`OPENAI_RESPONSES_MODEL` when you do not want the provider default.

### Docker Model Runner

Enable Docker Model Runner and pull a tool-capable model:

```bash
docker desktop enable model-runner --tcp 12434
docker model pull ai/gemma4
docker model ls
```

```python
agent = Agent("docker", tools=[get_weather])
```

With no `model=`, logpose selects the first model returned by the runner. Use
`DOCKER_MODEL` to pin a model and `DOCKER_MODEL_RUNNER_URL` to change the endpoint.
See [`examples/local_docker.py`](examples/local_docker.py) for a complete example.

### OpenAI-compatible servers

Use `openai-compat` for servers exposing `POST /chat/completions`, including local
or hosted deployments:

```python
agent = Agent(
    "openai-compat",
    base_url="http://localhost:8000/v1",
    model="my-tool-capable-model",
    api_key="optional-server-key",
    tools=[get_weather],
)
```

The same values can be set with `OPENAI_BASE_URL`, `OPENAI_MODEL`, and
`OPENAI_API_KEY`. The `docker` provider intentionally does not read
`OPENAI_API_KEY`.

### Experimental: Claude Code subscription

```bash
claude setup-token
export CLAUDE_CODE_OAUTH_TOKEN=sk-ant-oat01-...
```

```python
agent = Agent("claude-code")
```

If Claude Code is already logged in, logpose can read its local credential store
without the export. This integration relies on unstable CLI authentication details
and may stop working without notice. Use it for experimentation, not workloads
that require a stable authentication path.

### Experimental: Codex subscription

```bash
codex login
```

```python
agent = Agent("codex")
```

logpose reads `~/.codex/auth.json`, or `$CODEX_HOME/auth.json`, without modifying
it. This integration relies on unstable CLI authentication details and may stop
working without notice. A complete example is in
[`examples/codex.py`](examples/codex.py).

## Streaming

`Agent.stream()` yields normalized events as a run progresses:

```python
from logpose import RunEnd, TextDelta, ToolCall, ToolResult

async for event in agent.stream("Compare the weather in Pune and Goa."):
    if isinstance(event, TextDelta):
        print(event.text, end="", flush=True)
    elif isinstance(event, ToolCall):
        print(f"\ncalling {event.name}: {event.input}")
    elif isinstance(event, ToolResult):
        print(f"result: {event.content}")
    elif isinstance(event, RunEnd):
        print(f"\n{event.result.iterations} model turn(s)")
```

| Event | Meaning |
| --- | --- |
| `TextDelta` | Assistant-visible text arrived. |
| `ThinkingDelta` | A provider-supplied reasoning summary arrived. |
| `ToolCall` | A tool is about to run. |
| `ToolResult` | A tool completed and its result will be returned to the model. |
| `TurnEnd` | One provider round trip completed. |
| `RunEnd` | The run completed and includes its `RunResult`. |

`RunResult` contains the final text, complete message history, aggregated usage,
final stop reason, and iteration count.

### Provider retries

logpose retries temporary provider failures before the first text or reasoning
delta reaches your application. The default policy makes three total attempts,
uses bounded exponential backoff with jitter, and honors a provider's
`Retry-After` response when available.

```python
from logpose import Agent, RetryPolicy

agent = Agent(
    "anthropic",
    retry_policy=RetryPolicy(max_attempts=3),
)
```

Set `max_attempts=1` to disable retries. Once a provider has emitted a delta,
logpose does not replay the turn because doing so could duplicate streamed output.
The resulting `ProviderError` is marked `partial=True` and carries its attempt
count, request ID, retry delay, and provider error code when available.

### Provider deadlines

Each provider turn has a complete-turn deadline in addition to its transport
timeouts: 15 minutes for cloud and generic providers, and 30 minutes for Docker
Model Runner. A timeout before any delta can use the retry policy; a timeout
after output is partial and is not replayed.

Override the selected provider's recommendation, or disable the complete-turn
deadline explicitly:

```python
agent = Agent("anthropic", provider_turn_timeout=600)
agent = Agent("docker", provider_turn_timeout=None)
```

HTTPX-backed providers also accept `timeout=` for connection, pool, write, and
idle-read settings. Anthropic providers accept a transport timeout in seconds.

## Defining tools

The `@tool` decorator keeps the function callable in normal Python while adding
the schema and handler metadata the agent needs:

```python
from typing import Literal

from logpose import tool


@tool
def search_orders(customer_id: str, status: Literal["open", "closed"] = "open") -> dict:
    """Find orders for a customer.

    Args:
        customer_id: Stable customer identifier.
        status: Order status to include.
    """
    return {"customer_id": customer_id, "status": status, "orders": []}
```

- Type annotations become JSON Schema and are validated with Pydantic.
- The docstring summary describes the tool; the Google-style `Args:` section
  describes its parameters.
- `async def` tools are awaited. Synchronous tools run in a worker thread.
- Dictionaries and lists are JSON-encoded; other return values are converted to
  strings.
- Handler failures become error tool results so the model can respond or retry.

Tool names must match `[A-Za-z0-9_-]{1,128}`. Unsupported signatures fail at
decoration time with `ToolSchemaError`.

### Discovering tools

`discover_tools()` imports modules and collects their module-level decorated
tools:

```python
from logpose import Agent, discover_tools

tools = discover_tools("myapp.tools", "myapp.integrations")
agent = Agent("anthropic", tools=tools)
```

It deduplicates re-exported tools and reports conflicting names. Because discovery
imports the target modules and executes their top-level code, only use trusted,
static module paths and run discovery during application startup.

## Gating tool calls

Use `on_tool_call` to put approval, policy, or dry-run logic in front of tool
execution:

```python
def approve(call):
    if call.name == "delete_record":
        return "This operation requires explicit user approval."
    return None


agent = Agent("anthropic", tools=[...], on_tool_call=approve)
```

The gate sees calls in wire order before any tool in that turn starts. Return
`None` to allow a call, a string to block it with an error result, or
`ToolGateResult(..., is_error=False)` to redirect the model without reporting a
failure. Gates may be synchronous or asynchronous. An exception from a gate ends
the run.

## Multi-turn conversations

Pass a `Conversation` to retain history across calls:

```python
from logpose import Agent, Conversation

agent = Agent("anthropic", tools=[get_weather])
conversation = Conversation()

await agent.run("My name is Ada.", conversation=conversation)
result = await agent.run("What is my name?", conversation=conversation)
```

A run without a conversation is independent. A conversation should remain on the
backend that produced it because some providers include backend-specific reasoning
state in the history.

## Synchronous usage

`SyncAgent` exposes the same loop to blocking applications:

```python
from logpose import Agent, SyncAgent

with SyncAgent(Agent("anthropic", tools=[get_weather])) as agent:
    result = agent.run("What is the weather in Pune?")
    print(result.text)
```

For one-off calls, `run_sync()` and `stream_sync()` are also available. Do not use
the synchronous API from a thread already running an event loop; use `await` and
`async for` there.

## Provider discovery

Applications can build provider pickers without hardcoding logpose's registry:

```python
from logpose import provider_catalog, provider_status, resolve

for info in provider_catalog():
    print(info.name, info.credential, info.default_model)

for status in await provider_status():
    print(status.name, status.ready, status.detail)

models = await resolve("anthropic").list_models()
```

- `provider_catalog()` returns static metadata without reading credentials or
  importing provider SDKs.
- `provider_status()` checks local configuration without making a network request.
- `provider.list_models()` asks the selected backend for its current model list.

The experimental subscription providers have `officially_supported=False` in
their catalog metadata so applications can label them appropriately.

## Configuration reference

| Provider | Model | Endpoint | Other |
| --- | --- | --- | --- |
| `anthropic` | `model=` | provider default | `ANTHROPIC_API_KEY` |
| `openai` | `OPENAI_RESPONSES_MODEL` | `OPENAI_RESPONSES_BASE_URL` | `OPENAI_API_KEY` |
| `docker` | `DOCKER_MODEL` | `DOCKER_MODEL_RUNNER_URL` | no credential |
| `openai-compat` | `OPENAI_MODEL` | `OPENAI_BASE_URL` | optional `OPENAI_API_KEY` |
| `claude-code` | `model=` | provider default | `CLAUDE_CODE_OAUTH_TOKEN`, `CLAUDE_CONFIG_DIR` |
| `codex` | `CODEX_MODEL` | `CODEX_BASE_URL` | `CODEX_HOME`, `CHATGPT_ACCOUNT_ID` |

Provider-specific request fields that logpose does not model can be passed with
`extra=`:

```python
agent = Agent("anthropic", extra={"tool_choice": {"type": "any"}})
```

## Errors

All library-defined errors inherit from `LogposeError`.

| Error | Meaning |
| --- | --- |
| `AuthError` | A usable credential could not be resolved or refreshed. |
| `ProviderError` | The upstream provider failed; includes status and retryability when known. |
| `MaxIterationsError` | The run reached its iteration limit; includes partial messages. |
| `ToolSchemaError` | A tool signature could not be represented as JSON Schema. |
| `ToolExecutionError` | The tool execution machinery failed. |

Ordinary exceptions raised inside a tool are returned to the model as error tool
results rather than raised from the run.

## Extending logpose

Custom providers implement the `Provider` protocol and can be registered by name:

```python
from collections.abc import AsyncIterator

from logpose import CompletionDone, CompletionRequest, ProviderEvent, register


class MyProvider:
    name = "mine"
    model_default = "my-model-1"

    async def stream(self, request: CompletionRequest) -> AsyncIterator[ProviderEvent]:
        ...
        yield CompletionDone(
            message=...,
            stop_reason="end_turn",
            usage=...,
        )


register("mine", lambda **kwargs: MyProvider(**kwargs))
```

A provider translates a `CompletionRequest` into streaming provider events and
finishes with exactly one `CompletionDone`. Tool execution remains in `Agent`.
See [CONTRIBUTING.md](CONTRIBUTING.md) for the provider contract and test
expectations.

## Roadmap

The current release focuses on a reliable, embeddable loop: typed tools,
streaming, multi-turn state, usage accounting, tool gating, provider discovery,
and cloud or local model backends.

Planned areas of expansion include:

- context compaction for long-running conversations;
- post-tool-call hooks and richer execution policies;
- MCP tool ingestion;
- improved portability of conversations across compatible backends;
- additional provider integrations and conformance tests;
- production-oriented observability and retry controls.

Roadmap items are directional and may evolve with community feedback. Feature
requests and focused proposals are welcome in GitHub issues.

## Contributing

Contributions are welcome. To set up a development environment:

```bash
git clone https://github.com/xdadwal/logpose.git
cd logpose
uv sync
uv run pytest -q
uv run ruff check
uv run mypy src/
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for architecture, conventions, tests, and
pull request guidance. Please report security issues privately as described in
[SECURITY.md](SECURITY.md).

## License

logpose is available under the [MIT License](LICENSE).
