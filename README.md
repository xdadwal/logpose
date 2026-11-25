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
- Input guardrails that screen user messages before the agent acts, blocking or
  racing the first model turn.
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

Each turn's output ceiling defaults to 16384 tokens: room for reasoning plus a
long answer. Small local models often need it lower; set it with
`Agent(max_tokens=...)` or `OpenAICompatProvider(max_tokens=...)`. A reply that
long can take longer than the 120-second turn deadline on slower models; if
your replies run that long, raise `provider_turn_timeout` and `run_timeout` as
well.

Some servers hard-reject a large ceiling with a 400. Examples: DeepSeek (limit
8192 on `max_tokens`), some Groq models (8192 on `max_completion_tokens`), and
vLLM when the prompt plus the ceiling exceeds `max_model_len` (so a 16k-context
deployment fails every turn). For these, lower `max_tokens=` (8192 or less).

The ceiling is sent as `max_completion_tokens`, the field OpenAI (and vLLM and
llama.cpp) now expect; OpenAI's reasoning models reject the older `max_tokens`.
A server that only understands `max_tokens` may ignore the new field, so the
output is uncapped (only the turn and run deadlines bound it), or it may reject
it. For such servers pass `OpenAICompatProvider(max_tokens_field="max_tokens")`,
or `Agent("openai-compat", ..., max_tokens_field="max_tokens")`.

If you set `max_tokens` or `max_completion_tokens` yourself in `extra_body` or
`Agent(extra=...)` (or a per-run `extra=`, see
[Configuration reference](#configuration-reference)), your value replaces logpose's, and
only your field is sent.

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
| `Agent(max_tokens=)` | 16384 | output tokens per turn |
| `Agent(tool_timeout=)` | 30 s | one tool call, after it gets a slot |
| `RetryPolicy()` | 2 attempts, backoff up to 4 s, `Retry-After` up to 10 s | provider retries |

Tool output is budgeted to about 2k tokens per call. Page fetches and web
search return about 8,000 characters, and older tool output that no longer fits
the context is replaced by a pointer the model can expand with
`recall_tool_output`, one page of about 8,000 characters at a time.

### Built-in tools

`current_time` is the one built-in tool: it returns the current date and time in
an IANA zone (default UTC), for example
`Thursday, 26 November 2026, 14:05 IST (Asia/Kolkata)`. Pass it as a tool, and
tell the model your zone in the system prompt so it asks for the right one:

```python
from logpose import Agent, current_time

agent = Agent(
    "openai-compat",
    model="gpt-5.1",
    tools=[current_time],
    system="The user is in India; use the Asia/Kolkata time zone.",
)
```

For a fixed zone, wrap it with your own `@tool`:

```python
from logpose import tool, current_time

@tool
def now() -> str:
    """Get the current date and time in India."""
    return current_time.handler(timezone="Asia/Kolkata")
```

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
tools run, they are cancelled (registered terminators run), `tool_observer` records an error outcome for each, and each unfinished
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
not automatically logged. Without a custom policy, the built-in compaction described
in [Long conversations](#long-conversations) applies; `context_policy=None` sends full history.

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

## Sessions

Pass a `session_id` and logpose keeps the conversation in a local SQLite database:
history is loaded before the run, new messages and a run record are saved after it,
and a per-session lease stops two requests from running on one session at once.

```python
async with agent.session("chat-42") as handle:
    async for event in agent.stream("hi", session=handle):
        ...
```

Entering the block takes the lease and loads the history, so a busy session raises
`SessionBusyError` before anything is streamed. That makes a clean HTTP answer
possible:

```python
from fastapi import HTTPException
from logpose import SessionBusyError

@app.post("/chat/{sid}")
async def chat(sid: str, body: ChatIn):
    try:
        async with agent.session(sid) as handle:
            result = await agent.run(body.text, session=handle)
    except SessionBusyError:
        raise HTTPException(status_code=409, detail="session is busy") from None
    return {"text": result.text}
```

To stream the reply, the session must stay open until the body has been sent. Do
not `return StreamingResponse(...)` from inside `async with agent.session(...)`:
the block exits (saving and releasing the session) before the first chunk is
streamed, and the stream then fails. Enter the session before building the
response, so a busy session is still a 409 before any headers go out, and leave it
when the body finishes:

```python
from contextlib import AsyncExitStack, aclosing

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask
from logpose import SessionBusyError, TextDelta

@app.post("/chat/{sid}/stream")
async def chat_stream(sid: str, body: ChatIn):
    stack = AsyncExitStack()
    try:
        handle = await stack.enter_async_context(agent.session(sid))
    except SessionBusyError:
        raise HTTPException(status_code=409, detail="session is busy") from None

    async def chunks():
        try:
            # aclosing ends the run (answering any tool call it was waiting on)
            # before the session is saved, also when the client leaves mid-stream.
            async with aclosing(agent.stream(body.text, session=handle)) as events:
                async for event in events:
                    if isinstance(event, TextDelta):
                        yield event.text
        finally:
            await stack.aclose()  # saves and releases, also when the client leaves

    # The background task covers a body that is never iterated (the client left
    # before streaming began); closing the stack a second time does nothing.
    return StreamingResponse(chunks(), background=BackgroundTask(stack.aclose))
```

Leaving a session block also closes any stream still open on its handle before
saving, and answers a trailing unanswered tool call with an error result, so a
stored session always stays replayable.

For a single call, the convenience form does the same in one step:

```python
result = await agent.run("hi", session_id="chat-42", session_metadata={"user": "ada"})
```

Blocking code uses `with sync_agent.session(sid) as handle:`, and `run_sync`,
`stream_sync` and `SyncAgent` accept `session_id`, `session_metadata` and `session`.

- **Where it lives:** `Agent(db_path=...)` when given, otherwise `LOGPOSE_DB_PATH`
  (when set and not blank), otherwise `logpose.db` in the current directory. An
  explicit `db_path` is fixed when the agent is constructed; without it the
  environment variable is read at each use. `agent.db_path` returns the resolved
  absolute path. The file is created on first session use; runs without a session
  never touch disk. A `session=` handle always saves to the database it was leased
  from; if its agent has a different `db_path`, that run's traces still go to the
  running agent's database.
- **Session IDs:** anyone who knows a session ID can load that conversation, so use
  unguessable IDs (for example `secrets.token_urlsafe(32)`) and check that the
  caller owns the session before using it.
- **Lease:** a lease lasts 60 seconds and is renewed every 20 seconds while its run
  or block is active, so a crashed process frees its sessions within a minute. A
  block that has had no run in progress for 3 minutes stops renewing: its lease
  expires on its own (so a dropped, never-run streaming response cannot hold a
  session forever). A block held open across turns (a WebSocket chat) keeps
  working: the next run on its handle takes the lease back if nobody else took it
  meanwhile, and raises `SessionBusyError` if someone did.
- **Waiting:** `Agent(session_wait=...)` seconds to retry a busy lease (default 0).
- **Expiry:** a session expires after `session_ttl` without use (default 30 days; a
  `timedelta` or seconds; values above `timedelta.max` raise `LogposeError`). Every
  write stamps `expires_at = now + that agent's session_ttl`, so changing the TTL
  affects a session from its next write. An expired session starts empty on its
  next use. logpose never deletes rows on its own: call
  `await agent.purge_expired_sessions()` to delete expired sessions, or
  `await agent.delete_session(sid)` to remove one.
- **Run records:** each run writes a row to `logpose_runs`. Token counts cover
  completed turns only. A stream still open when its `agent.session(...)` block
  exits is closed first and writes its record within its own time limit (about
  10 s); the block then waits up to 6 s for any record still writing, and its
  save gets a full 8 s of its own. So a busy database can lose that stream's run
  record (logged) but not the block's messages; such an exit can take up to about
  24 s. Wrapping the stream in `contextlib.aclosing(...)` closes it before the
  block exits and avoids this.
- **Exit and cancellation:** the exit's save runs past a cancellation, so a server
  cancelling the handler on client disconnect does not shorten it. If streams may
  be left open on a handle, set the ASGI server's graceful-shutdown timeout to at
  least 25 s, or use `aclosing`.

### Showing a stored conversation

`await agent.load_session(sid)` (`SyncAgent.load_session(sid)` for blocking code)
returns a session's saved messages, oldest first, so a page can render the history
on load:

- It is the full stored history, including turns a compaction summary covers.
- An unknown or expired session returns `[]`. It never deletes, resets or extends
  a session. A session still under a live lease is not treated as expired.
- It takes no lease, so it works while a run is streaming in another tab and never
  raises `SessionBusyError`. A turn still in progress is not visible until it
  saves.
- An invalid session ID raises `LogposeError`. Check that the caller owns the
  session first (see Session IDs above).

`.env` files read by pydantic-settings (or similar) do not populate `os.environ`,
so logpose cannot see them. Pass `base_url=`, `api_key=`, `db_path=`, `session_ttl=` and
`tracing=Tracing(...)` explicitly:

```python
from contextlib import asynccontextmanager
from datetime import timedelta

from fastapi import FastAPI, Request
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from logpose import Agent, LogposeError, Tracing

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")
    base_url: str
    openai_api_key: SecretStr
    db_path: str = "/var/lib/myapp/logpose.db"
    session_ttl_days: int = 7

settings = Settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.agent = Agent(
        "openai-compat",
        model="gpt-5.1",
        base_url=settings.base_url,
        api_key=settings.openai_api_key.get_secret_value(),
        db_path=settings.db_path,
        session_ttl=timedelta(days=settings.session_ttl_days),
        tracing=Tracing(),
    )
    yield
    await app.state.agent.aclose()

app = FastAPI(lifespan=lifespan)

@app.get("/chat/history")
async def history(request: Request):
    # The session ID comes from the visitor's own cookie, never from the URL, so
    # a visitor can only read the conversation they own.
    sid = request.cookies.get("chat_sid")
    if not sid:
        return []
    try:
        messages = await app.state.agent.load_session(sid)
    except LogposeError:  # malformed cookie value
        return []
    # Send the browser only user and assistant text, not tool calls or results.
    return [{"role": m.role, "text": m.text} for m in messages if m.text]
```

Known limits:

- Sessions are single-host. SQLite in WAL mode supports several worker processes on
  one machine; several hosts need a networked database (MySQL or Postgres support is
  planned).
- A run cancelled after its lease was lost surfaces the cancellation; the failed save
  is logged.
- Abandoned streams release the lease when they are closed. Wrap them to make that
  prompt: `async with contextlib.aclosing(agent.stream(..., session_id=sid)) as events:`.

## Long conversations

By default logpose keeps long conversations inside the model's context window.
Only the request that is sent is changed. The `Conversation`, the session store and
`RunResult.messages` always keep the full, original history.

**What the default changes for existing apps.** This is on unless you pass
`context_policy=None`:

- **Set `context_window` to the model's window.** Unset, logpose budgets for
  128,000 tokens but refuses nothing before sending; the provider decides, and a
  context-length rejection triggers one summary and retry. Once set, a request
  estimated above `context_window - max_tokens` is refused before it is sent, with
  `ContextOverflowError`. On a small local model set it, so logpose compacts before
  the server rejects the request.
- **`context_budget` (default 32,000) caps input per request**, whatever the
  window: summaries start at about 27,000 estimated tokens even on a 1M-token
  model. Raise it (or pass `None`) if you want long conversations sent verbatim.
- **A `recall_tool_output` tool appears in every request that has tools.** Allow-list
  gates in `on_tool_call` must allow it, and tests that assert the exact tool specs
  sent to the provider will see it. `context_policy=None` removes it.
- **Summaries are extra, billed model calls** on the same provider and model. Their
  usage is counted in the run's usage.

```python
agent = Agent(
    "openai-compat", model="gpt-6-luna", tools=[get_weather],
    context_window=1_050_000,  # the model's window
    context_budget=200_000,  # keep more verbatim history than the 32k default
)
```

**Tool-output pointers.** A tool result longer than `pointer_min_chars` (default
1,000) that the model has already answered, in an earlier run, is sent as a short
pointer instead of its text:

```text
[tool output stored: <id>, <N> chars. Call recall_tool_output("<id>") to read it.]
```

Results the model has not yet seen (trailing results of a resumed run) and results
of `recall_tool_output` itself are never pointed. When the agent has tools, logpose
adds a `recall_tool_output(call_id, offset=0)` tool that returns the stored text in
pages of 8,000 characters. The tool passes through `on_tool_call` and
`tool_observer` like any app tool, so allow it in gates. An app tool may not use
that name.

**Summaries.** The input budget is
`min(context_window - max_tokens, context_budget)`. Defaults: `context_window`
128,000, `context_budget` 32,000 (`None` removes the cap), with `max_tokens`
resolved as in [Output limit](#output-limit). Input size is estimated as
characters / 4, multiplied by a calibration ratio learned from the usage the
provider reports (smoothed, clamped to [0.5, 2.0]). When the estimate reaches 85%
of the budget (`compaction_threshold`), older turns are summarized down to about
60% (`compaction_target`): one extra model call (no tools, output ceiling the
larger of 2,048 and the agent's `max_tokens` so reasoning models have room to think,
counted in the run's usage) whose result is inserted as an assistant message after
the first user turn, which stays pinned. The summary is asked to stay under 15% of
the input budget and is truncated to that length (ending with `…`) if it does not.
Recent turns, and everything from the current run, are sent verbatim. The summary
is cached and reused until the threshold is reached again. Estimates leave out
reasoning (thinking) blocks unless the provider re-sends them
(`OpenAICompatProvider(preserve_reasoning=True)`).

**The system prompt is never compacted.** It is sent unchanged on every request,
but it and the tool schemas count toward the budget. When they, or a large first
user turn (a pasted document), already fill the target on their own, no summary
is made, since one could not help: the request is sent as is. The summary call does
not use your system prompt: it sends its own instruction, which treats the
transcript as data, does not carry forward instructions found in it (for example in
fetched web pages), and reports such content as attributed facts. The summary is
inserted after the `[Summary of earlier conversation]` prefix.

The summary cache lives on the `Conversation` (and, with sessions, is persisted in
the session store: schema v2, migrated automatically, reused across requests). An
app that builds a fresh `Conversation(messages)` for every request without sessions
pays for a new summary each time; use sessions, or keep the `Conversation` object
between requests.

Prompt-cache stability is per run. Pointers are decided at run boundaries: the
previous run's long tool outputs are sent in full during that run and become
pointers at the next run, so the provider's prompt cache misses from that point
once per run (and again whenever a summary is made or extended). The system prompt
and tool schemas never change, so the cached prefix up to the first changed message
still hits.

If the summary call fails or returns no text, the request is sent uncompacted when
it fits the window (`context_window - max_tokens`) or when `context_window` is not
set; otherwise the run raises `ContextOverflowError`. A failed summary is not
retried within the same run; after repeated failures later runs back off
exponentially, waiting 1, 2, 4, 8 and at most 16 new user turns between attempts
(a success resets it; the reactive retry below still forces one).
If the provider rejects a request with an HTTP 400 that mentions the context length
(`context_length`, `maximum context length`, `context window`, `context size`,
`context length` or `exceed_context`, covering OpenAI-style servers, vLLM,
llama.cpp and LM Studio), logpose forces one summary and retries once.

| Error | Raised when |
| --- | --- |
| `ContextOverflowError` | `context_window` was set and the request cannot fit `context_window - max_tokens` even after compaction (or after a failed summary call). Raised before the request is sent; carries `estimated_tokens` and `input_budget`. |

```python
agent = Agent(
    "openai-compat", model="gpt-5.1", tools=[get_weather],
    context_window=200_000, context_budget=64_000,
    compaction_threshold=0.85, compaction_target=0.6, pointer_min_chars=1_000,
)
```

**Observing it.** Each prepared request that applied pointers or attempted a
summary publishes a metadata-only `context.compacted` runtime event (pointer count,
whether a summary is in use, estimated tokens before and after, input budget). A
`context.summarizing` event (estimated tokens, input budget) fires just before a
summary call, so a streaming UI can show why the reply is paused. Like every
runtime event it goes to `observers` (and logging), not to the `stream()` iterator:
pass an observer that forwards it to your UI, for example
`Agent(..., observers=[lambda e: e.name == "context.summarizing" and notify(e.run_id)])`.
`agent.last_context_diagnostics` holds the same numbers for the latest request.

**Opting out.** `context_policy=None` disables all of this and sends full history.
A custom `context_policy` function (see
[Runtime observability](#runtime-observability)) replaces all of it: no pointers,
recall tool, summary, retry, or event.

## Structured output

Pass a pydantic model as `output_type` and the run returns a validated instance
on `result.output`:

```python
from pydantic import BaseModel
from logpose import Agent


class LineItem(BaseModel):
    description: str
    amount: float


class Invoice(BaseModel):
    customer: str
    items: list[LineItem]
    total: float
    notes: str | None = None


agent = Agent("openai-compat", model="gpt-5.1")
result = await agent.run("Invoice Acme for two widgets at 10 each.", output_type=Invoice)
invoice: Invoice | None = result.output
```

`output_type` must be an object model (not a `RootModel` or list) and cannot
contain open-ended `dict` fields such as `dict[str, int]`: strict mode cannot
express arbitrary keys, so use a nested model with named fields instead. Such
models raise `LogposeError`, naming the field, when `output_type` is set and
before any request is sent.

`output_type` can be set per call or as an `Agent` default. Passing
`output_type=None` on a call disables an agent default for that call. Runs
without `output_type` are unchanged, apart from `result.output` being `None`.

If the reply does not validate, logpose sends one correction turn describing the
problem. `output_retries` sets that budget (default `1`; `0` fails immediately,
which is useful in tests). A `null` for a field that has a default takes the
default, while `X | None` fields keep `None`.

| Error | Raised when |
| --- | --- |
| `OutputValidationError` | The reply is not valid for the model after the retry budget, including `reason="max_tokens"` when output was cut off. |
| `OutputRefusalError` | The model refused. There is no correction turn. |
| `StructuredOutputUnsupportedError` | The server rejected the strict `json_schema` request (HTTP 400/422). The message includes the provider's reason when it gave one. Not retried. |

### Provider compatibility (as of 2026-11)

logpose enforces structure only with a strict `json_schema` `response_format`. It
does not fall back to `json_object` or prompt-only modes, so providers that lack
strict `json_schema` support cannot be used with `output_type`.

| Provider | Strict `json_schema` |
| --- | --- |
| OpenAI | ✅ |
| vLLM | ✅ |
| Ollama | ✅ |
| llama.cpp | ✅ |
| LM Studio | ✅ |
| Sarvam | ✅ |
| OpenRouter | ⚠️ depends on the model (use models listing `structured_outputs`) |
| Z.ai GLM API | ❌ documents only `json_object` |
| DeepSeek API | ❌ documents only `json_object` |

## Input guardrails

Input guardrails check each run's new user message before the agent acts on it.
They are configured on the agent, so `run`, `stream` and the sync facade need no
extra arguments:

```python
import httpx

from logpose import Agent, openai_decision_guard, pattern_guard

# Create once and keep for the app's lifetime; close it at shutdown.
shared_httpx_client = httpx.AsyncClient()

agent = Agent(
    "openai-compat",
    model="gpt-5.1",
    input_guardrails=[
        pattern_guard(max_chars=4000, reply="That message is too long."),
        openai_decision_guard(
            "billing questions for Acme",
            replies={
                "off_topic": "I can only help with Acme billing.",
                "unsafe": "I can't help with that.",
            },
            # key: LOGPOSE_GUARD_API_KEY env var, or pass api_key= explicitly
            client=shared_httpx_client,  # one httpx.AsyncClient for the app
        ),
    ],
    guardrail_mode="blocking",  # or "parallel"
)
```

A guard is any sync or async callable taking a `GuardrailInput` (the new
`message`, a few `recent` messages, session `metadata`, `run_id`) and returning
a `GuardrailVerdict`. Guards see the prompt's text exactly as it is sent to
the provider (text blocks joined with no separator). Guards are text-only: a prompt
with any non-text content (tool results, raw provider blocks) is blocked with
category `"error"`, guard `"content_check"` and reason `"unsupported content"`
before any guard runs,
rather than sent unscreened. A run with an empty text prompt, or with no new
prompt, skips the guards. A guard that raises or returns the wrong type
counts as a block with category `"error"`, as does a verdict with malformed `scores` or `usage`, or whose `allowed` is not a `bool`. A sync callable that returns a coroutine is treated as a sync-tier guard (awaited in sequence, so it does not race the turn); write async guards as `async def`.

**Order.** Sync guards run first, in list order, and the first block ends the run
before any async guard starts (in parallel mode too: sync guards run before the
model call begins). Async guards then run concurrently; the first block wins
(list order among guards that finish together) and the rest are cancelled. A
slow sync guard runs inline and cannot be interrupted by `run_timeout`; the
deadline is checked when it returns, so keep sync guards fast.

**Blocking versus parallel.** In `"blocking"` mode (the default) the guards
finish before the main model is called. In `"parallel"` mode async guards race
the first model turn, which hides their latency, with these trade-offs:

- Text and thinking output is buffered until the guards allow. Tools never run
  before then.
- On a block the turn is cancelled. If the turn had already finished, its
  reported usage counts in the run's usage and `RunRecord`. A turn cancelled
  mid-stream reports no usage, so its main-model tokens are not counted, but a
  blocked request may still cost some main-model tokens.
- `context.compacted` and `context.summarizing` events from the discarded turn
  remain emitted, but a summary it made (or a failed summary attempt) is not
  kept: the conversation's compaction state is left as it was.
- If the run times out or is cancelled before the verdict, the prompt is dropped
  and `iterations` is 0.

**What a block returns.** The run ends with the guard's placeholder reply as the
text, `stop_reason="refusal"`, and `RunResult.guardrail` (a `GuardrailResult`
with `guard`, `category`, `reason`, `scores`). In blocking mode no model call is
made. Neither the prompt nor the placeholder is added to the `Conversation` or
the session, and the session's `RunRecord` has outcome `"blocked"`. With
`output_type`, a block raises `GuardrailTripped` instead; the result is on
`.result` (a `GuardrailResult`). Each decided guard emits a `guardrail.decided` runtime event with the
guard name, verdict, category, and scores, never message text. Usage a guard
reports on `GuardrailVerdict.usage` counts in the run's usage and session
records; a custom async guard that calls a model should return its usage there.

**`pattern_guard`** is free and deterministic: a `max_chars` cap (default 4000,
`None` disables) and case-insensitive regexes in `block_patterns`, compiled at
construction (an invalid regex raises `LogposeError`). Patterns you supply can
backtrack catastrophically on adversarial input, so set `max_chars`.

**`openai_decision_guard`** judges "on topic" and "unsafe" in one call to
OpenAI's Decisions API (`POST /v1/decisions`, model `gpt-6-luna`), billed for
input tokens only. It is OpenAI-specific: for other providers, write a custom
guard. It opens a new httpx client per call unless you pass a shared `client=`,
which is recommended in web apps for connection reuse. The key is
`api_key=` if non-blank, else the `LOGPOSE_GUARD_API_KEY` environment variable;
if neither is set, construction raises `LogposeError`. It never falls back to
`LOGPOSE_API_KEY` (which may be another server's key) or `OPENAI_API_KEY`. `timeout` (default 5 s) bounds the whole call. Pass `name=`
to tell two decision guards apart in results and events. `on_error` is `"block"` (the default, fail closed)
or `"allow"`. A fail-open allow is observable: the verdict has
`category="fail_open"` and the failure `reason` (no body text), and the
`guardrail.decided` event has `guard_allowed=True` with
`guard_category="fail_open"`. `history` (default 4) is how many prior messages
the judge sees; the runner supplies at most 4, so values outside 0 to 4 raise
`LogposeError`. A malformed or non-finite probability is handled as an
error under `on_error`.

**Tuning thresholds.** Start with `on_topic_min=0.5` and `unsafe_max=0.5`, then
review the `scores` in `guardrail.decided` events from real traffic and adjust.

**Limits.** Input guardrails only see the user's message. They do not see
injection inside tool outputs or fetched documents. Pair them with read-only
tools, and use `on_tool_call` to gate actions.

## Synchronous usage

`SyncAgent` exposes the same loop to blocking applications:

```python
from logpose import Agent, SyncAgent

with SyncAgent(Agent("openai-compat", model="gpt-5.1", tools=[get_weather])) as agent:
    result = agent.run("What is the weather in Pune?")
    print(result.text)
```

For one-off calls, `run_sync()` and `stream_sync()` are also available. All of them accept
`output_type`, `output_retries` and the session arguments. Do not use
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

## Tracing

logpose can record each run as an OpenTelemetry trace using the GenAI semantic
conventions, store it in the same SQLite database as sessions, and export it as
OTLP/JSON. It is off by default and has no OpenTelemetry dependency.

```python
from logpose import Agent, Tracing

agent = Agent("openai-compat", model="gpt-5.1", tracing=Tracing())
```

`LOGPOSE_TRACE=1` (also `true` or `yes`) turns tracing on for agents that do not
pass `tracing=`; `LOGPOSE_TRACE_CONTENT=1` then also captures content.
`tracing=None` forces tracing off whatever the environment says.

A trace has an `invoke_agent` root span per run with `chat`, `execute_tool`,
`guardrail` and `compaction` child spans. Without `capture_content` spans carry
metadata only: models, token counts, finish reasons, tool names, outcomes and
your run metadata. With `Tracing(capture_content=True)` they also carry prompts,
outputs, reasoning, tool arguments and results, and exception messages.

A run that passes `extra=` (or whose agent sets `extra`) also records
`logpose.request.extra_keys` on the `invoke_agent` span: the sorted key names of
the run's merged `extra`. Values are never recorded, even with `capture_content`
on.

What capture stores and scrubs:

- The configured provider API key and `LOGPOSE_API_KEY` are replaced in captured
  content. Request headers are never recorded. An `api_key` you pass directly to
  `openai_decision_guard` is not known to the scrubber, although the guard never
  puts it in content. Values shorter than 12 characters (placeholders such as
  `EMPTY` or `ollama` that local servers accept) are not scrubbed, so ordinary
  words are not mangled; a real credential that short would be stored as is.
- `logpose.config.system_prompt_sha256` (a SHA-256 of the system prompt) is
  stored even with capture off. It does not reveal the prompt, but a short or
  guessable system prompt can be confirmed by hashing candidates.
- Anything else the model or your tools see is stored as is. Treat the database
  as sensitive when capturing content.
- Each content value is capped at `content_max_chars` (default 64,000). A longer
  value keeps its end and starts with `…[truncated N chars]`, and the span gets
  `logpose.content.truncated`.

### Where spans live

Spans are rows in the SQLite store (`LOGPOSE_DB_PATH`; schema v3, migrated
automatically on open). A background writer saves them as they end. At the end of
a run logpose waits at most `flush_timeout` seconds (default 1) for that run's
spans; spans not yet written stay queued and are written later. The queue holds
up to 10,000 spans or 64 MiB; beyond that new spans are dropped and one `WARNING`
is logged on the `logpose.trace` logger. Tracing never fails or changes a run: a
store or writer error is logged and the run continues.

If another process holds the database write lock, a traced run without a session
does not wait for it. Opening the store for the first time (per database path and
event loop) and checking a caller-supplied `run_id` against stored traces share a
wait of at most `max(flush_timeout, 0.25)` seconds. When the open takes longer, the
run is untraced and the open continues in the background for later runs, which
do not wait for it again: until it finishes they run untraced at once. When the
check takes longer, it is skipped. Each logs one `WARNING` per agent on
`logpose.trace`. Session runs wait for their lease as usual.

### Run IDs and metadata (evals)

Give a run your own ID and labels to join traces to eval results:

```python
await agent.run(
    "question", run_id="eval-2026-11-20-case-17", metadata={"suite": "smoke", "case": 17}
)
```

`run_id` matches `[A-Za-z0-9._-]{1,128}`. With tracing on or a session, reusing an
ID that is already recorded (in a run record or in stored spans) or in use by a
running run in this process raises `LogposeError` before any provider call. With
neither, reuse is not detected. Known gaps:

- Several worker processes: traced runs without a session see other processes'
  runs only once those are stored, so two workers starting the same `run_id` at
  once can both write a trace with that ID.
- When the database is busy (see above) the check against stored traces is skipped
  for that run, with a `WARNING`; runs in this process are still checked.
- Session runs in several worker processes: a `run_id` another process is running
  right now (not yet recorded) is caught only when this run's record is written,
  at the end of the run, after its work is done; the run then raises
  `LogposeError`.
- Reuse is rejected only while a run record or stored spans exist: purging a
  sessionless run's trace (`purge_traces`) frees its `run_id`.

`metadata` takes at most 64 keys, keys up to 128 characters, string values
up to 1,024 characters, and `str`, `int`, `float` (finite) or `bool` values. It
appears on spans as `logpose.metadata.<key>`, on `RuntimeContext`, and on the
session `RunRecord`, but not on `RuntimeEvent`.

### Export

```python
from datetime import datetime, timezone

from logpose import export_traces

result = await export_traces("traces.jsonl")
print(result.traces, result.through)
```

`export_traces(..., db_path=agent.db_path)` reads a specific database; omitted, it
resolves the path as `Agent(db_path=...)` does (`LOGPOSE_DB_PATH`, then
`logpose.db`). Pass the agent's `db_path` when the agent was given one explicitly.

`export_traces` writes one OTLP `ExportTraceServiceRequest` per line (OTLP/JSON,
one trace per line) for traces whose root span has finished; runs still in
progress are skipped until they end. `since` (exclusive) and `until` (inclusive)
are timezone-aware datetimes compared with the root span's start at microsecond
resolution; naive datetimes raise `LogposeError`. `session_id=` limits the export
to one session. The resource `service.name` is the run's `Tracing.service_name`
(stored on the root span as `logpose.service.name`; `"logpose"` if absent). Metadata
values are exported as strings on spans (`bool` as `true`/`false`, as in the run
record's JSON). The file is replaced atomically (written
to a temporary file in the same directory, then renamed), so use a new path per
incremental batch.

For incremental export pass the previous `cursor`. It is a position in the order
roots were written to the store, not a start time, so a run that finishes after a
later-started run is still exported exactly once, and re-writing a root never
re-exports it. When a batch exports nothing, `cursor` is the `after` you passed.
`after=` cannot be combined with `since=`, `until=` or `session_id=` (that raises
`LogposeError`), and a filtered export returns `cursor=None`: a cursor taken from a
filtered export would step over the traces the filter left out:

```python
first = await export_traces("batch-1.jsonl")
second = await export_traces("batch-2.jsonl", after=first.cursor)  # no repeats, no gaps
```

`through` (the latest root start exported) is informational.

A trace is exported as the spans stored at that moment. If a batch of child spans
fails to write while the root's batch succeeds (rare; the writer logs a `WARNING`),
the exported trace is partial.

Feed the files to an OpenTelemetry Collector with the `otlpjsonfile` receiver:

```yaml
receivers:
  otlpjsonfile:
    include: [/var/lib/logpose/traces/*.jsonl]
    start_at: beginning
exporters:
  otlphttp:
    endpoint: https://collector.example.com:4318
service:
  pipelines:
    traces:
      receivers: [otlpjsonfile]
      exporters: [otlphttp]
```

### Purging

logpose never deletes spans on a schedule. Deleting a session also deletes its
spans: `delete_session`, `purge_expired_sessions`, and an expired session being
reset on its next use. Export first if you keep eval traces.

```python
from datetime import timedelta

deleted = await agent.purge_traces(older_than=timedelta(days=30))
```

`purge_traces` deletes whole traces whose root span started more than `older_than`
ago, in every session, plus orphan spans (no root stored) that started before the
cutoff, and returns the span count. Session run records stay, so a purged session
run's `run_id` still cannot be reused. A run without a session has no run record:
its stored spans are the only trace of its `run_id`, so purging them frees that ID
for reuse. A cutoff shorter than a still-running run can remove that run's
already-written child spans, leaving a partial trace.

### Convention status

The GenAI semantic conventions are still in Development. logpose pins release
1.45.0 and stamps it on exports (`logpose.semconv.version` and the schema URL),
so attribute names may change in a future logpose release.
`gen_ai.usage.input_tokens` includes cache reads; cache reads and writes are also
reported as `gen_ai.usage.cache_read.input_tokens` and
`gen_ai.usage.cache_creation.input_tokens` (and as `logpose.usage.*`).
`gen_ai.provider.name` is logpose's provider name (`openai-compat`), not a
well-known value such as `openai`: an OpenAI-compatible endpoint can be any
server.

## Configuration reference

| Provider | Model | Endpoint | Credential |
| --- | --- | --- | --- |
| `openai-compat` | `Agent(model=)` | `base_url=` / `LOGPOSE_BASE_URL` | optional `api_key=` / `LOGPOSE_API_KEY` |

| Setting | Argument | Environment | Default |
| --- | --- | --- | --- |
| Database file | `Agent(db_path=)`, `export_traces(db_path=)` | `LOGPOSE_DB_PATH` | `logpose.db` in the working directory |
| Session lifetime | `Agent(session_ttl=)` (`timedelta` or seconds) | none | 30 days |

An explicit argument wins over the environment. Settings loaded from a `.env` file
by another library are not in `os.environ`; pass them as arguments.

Provider-specific request fields that logpose does not model can be passed with
`extra=`:

```python
agent = Agent("openai-compat", model="gpt-5.1", extra={"tool_choice": "required"})
```

### Per-run `extra`

`run`, `stream`, `run_sync`, `stream_sync`, `SyncAgent.run` and `SyncAgent.stream`
accept `extra=` for request fields that belong to one run, so one shared agent can
serve many callers:

```python
import hashlib

agent = Agent("openai-compat", model="gpt-5.1")  # shared by every request

async def answer(visitor_id: str, question: str):
    return await agent.run(
        question,
        extra={
            "safety_identifier": hashlib.sha256(visitor_id.encode()).hexdigest(),
            "prompt_cache_key": "support-bot-v3",
        },
    )
```

- The run's `extra` is `{**agent.extra, **extra}`: per-run keys win. It is sent
  with every model call of that run: each turn, output-correction turns and the
  retry after a context-length error.
- `extra` must be a mapping with `str` keys. `model`, `messages`, `tools`,
  `stream` and `stream_options` are built by logpose and raise `LogposeError`
  if set. Setting `response_format` here (or in `Agent(extra=...)`) together with
  `output_type` also raises `LogposeError`.
- A per-run `max_tokens` or `max_completion_tokens` replaces logpose's ceiling
  field, as it does in `Agent(extra=...)`. Compaction budgets and the chat span's
  `gen_ai.request.max_tokens` still use the agent's `max_tokens`.
- The compaction summary call gets the merged `extra` without `response_format`,
  `tool_choice` and `parallel_tool_calls`. A ceiling field in `extra` therefore
  also caps the summary; set a small ceiling with `Agent(max_tokens=...)` instead
  if long chats are compacted.
- Input guardrails never receive it. It is not saved in sessions: pass it on each
  run.

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
| `OutputValidationError` | Structured output did not validate after the retry budget (or was cut off by `max_tokens`). |
| `OutputRefusalError` | The model refused a structured-output request. |
| `StructuredOutputUnsupportedError` | The server rejected the strict `json_schema` request (a `ProviderError`). |
| `ContextOverflowError` | The request cannot fit the explicitly set `context_window` even after compaction; see [Long conversations](#long-conversations). |
| `GuardrailTripped` | An input guardrail blocked a run that uses `output_type`; `.result` is the `GuardrailResult`. See [Input guardrails](#input-guardrails). |

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
