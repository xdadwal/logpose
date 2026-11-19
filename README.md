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

Runtime dependencies are `pydantic` and `httpx`. Every built-in provider speaks
the OpenAI Chat Completions API, so no vendor SDK is needed.

## Quickstart

Point logpose at any Chat Completions server, such as OpenAI, vLLM, Ollama, LM
Studio, llama.cpp, or OpenRouter:

```bash
export LOGPOSE_BASE_URL=https://api.openai.com/v1
export LOGPOSE_API_KEY=sk-...   # omit for a local server that needs no key
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
        "openai-compat",
        model="gpt-5.1",
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

logpose ships one provider, `openai-compat`, for any server speaking the OpenAI
Chat Completions API, local or hosted (including OpenAI).

logpose makes no assumptions about your server. The application supplies:

| Setting | How | If missing |
| --- | --- | --- |
| Model | `Agent(model=...)` (required) | `TypeError`; `model="auto"` uses the server's first listed model |
| Endpoint | `base_url=` or `LOGPOSE_BASE_URL` | `LogposeError` at construction |
| API key | `api_key=` or `LOGPOSE_API_KEY` | no `Authorization` header (fine for local servers) |

Arguments take precedence over environment variables. Blank environment values
count as unset.

### OpenAI-compatible servers

Use `openai-compat` for servers exposing `POST /chat/completions`, including
OpenAI itself and local or hosted deployments:

```python
agent = Agent(
    "openai-compat",
    model="my-tool-capable-model",
    base_url="http://localhost:8000/v1",
    api_key="optional-server-key",
    tools=[get_weather],
)
```

Docker Model Runner, Ollama, and other local servers work the same way. For
example, with Docker Model Runner enabled (`docker desktop enable model-runner
--tcp 12434`):

```python
agent = Agent(
    "openai-compat",
    model="auto",
    base_url="http://localhost:12434/engines/v1",
    api_key="",  # never send an ambient LOGPOSE_API_KEY to a local server
    max_tokens=4096,
)
```

See [`examples/local_docker.py`](examples/local_docker.py) for a complete example.

### Output limit

Each turn's output ceiling defaults to 8192 tokens: room for moderate reasoning
plus an answer inside the 120-second turn deadline. For long, high-effort
outputs raise `max_tokens` together with `provider_turn_timeout` and
`run_timeout`. Small local models often need it lower; set it with
`Agent(max_tokens=...)` or `OpenAICompatProvider(max_tokens=...)`.

Chat Completions has no standard field for a reasoning model's chain of thought,
so by default reasoning is not sent back on later turns. For servers that accept
it as `reasoning_content`, pass `preserve_reasoning=True` to replay it unchanged
across tool calls.

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
delta reaches your application. The default policy makes two total attempts
(one retry), uses bounded exponential backoff with jitter (at most 4 s), and
honors a provider's `Retry-After` of up to 10 s. A retry never waits past the
run deadline.

```python
from logpose import Agent, RetryPolicy

agent = Agent(
    "openai-compat",
    model="gpt-5.1",
    retry_policy=RetryPolicy(max_attempts=3),
)
```

Set `max_attempts=1` to disable retries. Once a provider has emitted a delta,
logpose does not replay the turn because doing so could duplicate streamed output.
The resulting `ProviderError` is marked `partial=True` and carries its attempt
count, request ID, retry delay, and provider error code when available.

### Limits and deadlines

logpose is tuned for web apps: a user waits on one request, so every value that
could otherwise run unbounded has a safe default. Each accepts `None` for
"unlimited", and any can be raised.

| Setting | Default | Bounds |
| --- | --- | --- |
| `Agent(run_timeout=)` | 320 s | the whole run, wall clock, including time between streamed events |
| `Agent(provider_turn_timeout=)` | 120 s | one model turn |
| `OpenAICompatProvider(timeout=)` | connect 5 s, read 60 s, write 10 s, pool 5 s | HTTP; `read` is the longest silence between streamed chunks |
| `Agent(max_iterations=)` | 10 | model round trips per run |
| `Agent(max_tokens=)` | 8192 | output tokens per turn |
| `Agent(tool_timeout=)` | 30 s | one tool call, after it gets a slot |
| `RetryPolicy()` | 2 attempts, backoff up to 4 s, `Retry-After` up to 10 s | provider retries |

Tool output is budgeted to about 4k tokens per call: shell, `read_file` pages,
`list_dir`, grep, and page fetches return at most 16,000 characters, and web
search 8,000. `read_file` has no line cap and `list_dir` no entry cap; both page
by characters with a continuation the model can follow.

```python
import httpx
from logpose import Agent, RunTimeoutError

agent = Agent(
    "openai-compat",
    model="gpt-5.1",
    run_timeout=600,
    provider_turn_timeout=300,
    max_tokens=32000,
    timeout=httpx.Timeout(connect=5, read=120, write=10, pool=5),
)

try:
    result = await agent.run("Write the full report.")
except RunTimeoutError as exc:
    partial_history = exc.messages
```

A run that exceeds `run_timeout` raises `RunTimeoutError` with the partial
history; it is never retried. A provider error that arrives when too little time
is left to retry is raised as itself (with its status and `retry_after`), not as
a run timeout.

Deadlines interrupt only work logpose awaits on your behalf: the provider, your
tools, and the callbacks it runs for them (`on_tool_call`, `tool_observer`,
`context_policy`). Your code between streamed events is never interrupted; the
deadline is checked when logpose next waits. If the run deadline passes while
tools run, they are cancelled (registered terminators run, so shell commands are
killed), `tool_observer` records an error outcome for each, and each unfinished
call gets an error tool result so the conversation stays valid for the next run.
A `TimeoutError` raised by your own gate, observer, or policy propagates as
itself. A turn timeout before any delta can use the retry
policy; a timeout after output is partial and is not replayed.

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
agent = Agent("openai-compat", model="gpt-5.1", tools=tools)
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


agent = Agent("openai-compat", model="gpt-5.1", tools=[...], on_tool_call=approve)
```

The gate sees calls in wire order before any tool in that turn starts. Return
`None` to allow a call, a string to block it with an error result, or
`ToolGateResult(..., is_error=False)` to redirect the model without reporting a
failure. Gates may be synchronous or asynchronous. An exception from a gate ends
the run.

### Tool capacity and timeouts

An agent runs up to eight tool handlers at once across all of its concurrent
runs. Additional calls wait for capacity and still return results in the model's
original request order. Each tool has a 30-second execution deadline after it
acquires a slot:

```python
agent = Agent(
    "openai-compat",
    model="gpt-5.1",
    tools=[get_weather],
    max_concurrent_tools=8,
    tool_timeout=30,
)
```

Set `tool_timeout=None` for tools that intentionally run without a deadline.
When a call times out, logpose returns an error result to the model and continues
the run. Asynchronous tools are cancelled; a synchronous tool may continue in
its worker thread, but any late result is discarded and never sent to the model.

### Tool error detail

Tool errors are safe by default: logpose returns the tool name, exception type,
and a generated error ID, without returning an exception message or traceback to
the model. Argument-validation feedback remains actionable but does not echo
the rejected values.

Use `tool_error_mode="message"` or `tool_error_mode="traceback"` only when the
model is allowed to see that diagnostic detail:

```python
agent = Agent("openai-compat", model="gpt-5.1", tools=[get_weather], tool_error_mode="message")
```

For application-specific redaction, supply a formatter. It receives raw failure
context, so the formatter is responsible for keeping secrets out of its result.
If it fails, logpose uses the safe default instead.

```python
from logpose import Agent, ToolErrorContext


def tool_error(context: ToolErrorContext) -> str:
    return f"{context.tool_name} failed; reference {context.error_id}."


agent = Agent("openai-compat", model="gpt-5.1", tools=[get_weather], tool_error_formatter=tool_error)
```

## Runtime observability

Every agent run publishes metadata-only lifecycle records to the standard
`logpose.runtime` logger. logpose adds no handlers, formatters, destinations, or
global logging configuration; configure that logger through your application's
normal logging setup.

For application-owned context selection, pass a sync or async `context_policy=`
callback. Each provider iteration receives a detached `ContextInput` and expects
`PreparedContext(messages=..., system=..., diagnostics=...)`. The prepared view is
sent to the model; canonical `Conversation` and `RunResult.messages` retain the
original history. Tool schemas, model, output limit and extra options stay unchanged.
The application owns budgets, retrieval, summaries, instruction retention and valid
tool-call/result groups, including provider-specific opaque blocks. Preparation
failures raise `ContextPreparationError` before that iteration's inference; there
is no fallback that silently drops history. Cancellation propagates normally.
Synchronous policies must avoid blocking I/O; async policies own their I/O deadlines.
`Agent.last_context_diagnostics` holds the most recently prepared metadata and is
not automatically logged. With no policy, full-history behavior is unchanged.

For durable application journals, pass an async `tool_observer=` callback. It
receives `ToolObservation` values containing detached approved call data and
normalized outcomes. Callback failures raise `ToolObservationError` and stop the
run. Intent failures prevent that handler from executing; outcome failures mean
the effect may already have occurred. Reconcile application records before retrying.
Intents can precede waiting for a tool slot, and cancellation can leave an intent
without an outcome. This callback does not provide exactly-once execution or
automatic replay. It contains application content and must not be logged as telemetry.
Applications own storage, timeouts, and recovery; the callback is disabled by default.

For metrics, tracing, or audit integration, pass synchronous `observers=`. They
receive immutable `RuntimeEvent` values in registration order. Observer and
logging-handler failures are isolated from the agent run. Runtime records contain
identifiers, provider/model metadata, timings, status, usage, and tool result
sizes—but never prompts, model output, tool arguments/results, exception
messages, headers, or credentials.

```python
from logpose import Agent, RuntimeEvent, current_runtime_context


def metrics(event: RuntimeEvent) -> None:
    if event.name == "run.completed":
        metrics_client.count("agent.run", provider=event.provider)


@tool
def lookup_customer(customer_id: str) -> str:
    context = current_runtime_context()
    app_logger.info("customer lookup", extra={"run_id": context.run_id})
    return "..."


agent = Agent("openai-compat", model="gpt-5.1", tools=[lookup_customer], observers=[metrics])
```

`current_runtime_context()` returns `None` outside runtime execution. Within a
provider, tool, observer, or logging handler, it supplies opaque run, turn,
attempt, request, and tool identifiers.

## Multi-turn conversations

Pass a `Conversation` to retain history across calls:

```python
from logpose import Agent, Conversation

agent = Agent("openai-compat", model="gpt-5.1", tools=[get_weather])
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

with SyncAgent(Agent("openai-compat", model="gpt-5.1", tools=[get_weather])) as agent:
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

models = await resolve("openai-compat").list_models()
```

- `provider_catalog()` returns static metadata without reading credentials or
  importing provider modules.
- `provider_status()` checks local configuration without making a network request.
- `provider.list_models()` asks the selected backend for its current model list.

## Configuration reference

| Provider | Model | Endpoint | Credential |
| --- | --- | --- | --- |
| `openai-compat` | `Agent(model=)` | `base_url=` / `LOGPOSE_BASE_URL` | optional `api_key=` / `LOGPOSE_API_KEY` |

Provider-specific request fields that logpose does not model can be passed with
`extra=`:

```python
agent = Agent("openai-compat", model="gpt-5.1", extra={"tool_choice": "required"})
```

## Errors

All library-defined errors inherit from `LogposeError`.

| Error | Meaning |
| --- | --- |
| `AuthError` | A credential was missing or rejected; its message names the setting to check. |
| `ProviderError` | The upstream provider failed; includes status and retryability when known. |
| `MaxIterationsError` | The run reached its iteration limit; includes partial messages. |
| `RunTimeoutError` | The run exceeded `run_timeout`; includes partial messages. |
| `ToolSchemaError` | A tool signature could not be represented as JSON Schema. |
| `ToolExecutionError` | The tool execution machinery failed. |

Ordinary exceptions raised inside a tool are returned to the model as safe error
results rather than raised from the run. Configure `tool_error_mode` or
`tool_error_formatter` only when exposing additional diagnostic detail is
appropriate for that model-facing context.

An HTTP authentication rejection is never retried automatically. `AuthError`
tells you which key setting to check.

## Extending logpose

Custom providers implement the `Provider` protocol and can be registered by name:

```python
from collections.abc import AsyncIterator

from logpose import CompletionDone, CompletionRequest, ProviderEvent, register


class MyProvider:
    name = "mine"

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

The approved runtime-hardening milestones, defaults, acceptance criteria, and PR
sequence are documented in the
[robustness and release-readiness plan](docs/robustness-plan.md).

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
