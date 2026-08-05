# logpose

[![CI](https://github.com/xdadwal/logpose/actions/workflows/ci.yml/badge.svg)](https://github.com/xdadwal/logpose/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Checked with mypy](https://img.shields.io/badge/mypy-checked-blue)](https://mypy-lang.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**A universal agentic loop that your service embeds.**

logpose owns the loop. It is not a wrapper around `claude-agent-sdk`, the Codex
harness, or anyone else's agent framework — it drives the model API directly,
executes your tools, and streams normalized events back to you. That means one
loop, one message model, and one set of events no matter which provider is
behind it.

Three things follow from that:

- **Provider-agnostic by construction.** Everything above `logpose.providers`
  speaks a neutral message model. A provider's only job is turning a
  `CompletionRequest` into a stream of events — it cannot execute tools, so the
  loop stays the single place where side effects happen.
- **You own your tools.** logpose ships *no* built-in bash, file, or shell
  tools. It gives you a `@tool` decorator and runs exactly what you hand it.
  Nothing touches your machine that you did not write.
- **Async-first, sync-friendly.** The core is `async`; a thin, correct sync
  facade wraps it for callers that are not.

Backends today: **Anthropic** (Claude Code subscription token or BYOK API key),
**Docker Model Runner** for local models, and a generic **OpenAI-compatible**
backend covering OpenAI, Kimi, vLLM, Ollama, and LM Studio. The local and
OpenAI-compatible paths add no dependency and need no credential.

If you plan to use a Claude Code subscription token, read the
[disclaimer](#subscription-auth-disclaimer) first.

---

## Contents

- [Install](#install)
- [Quickstart](#quickstart)
- [Auth](#auth)
- [Streaming](#streaming)
- [Defining tools](#defining-tools)
- [Multi-turn conversations](#multi-turn-conversations)
- [Sync usage](#sync-usage)
- [Errors](#errors)
- [Providers](#providers) — [local models](#local-models-via-docker-model-runner),
  [OpenAI-compatible](#any-openai-compatible-server),
  [writing your own](#writing-your-own)
- [⚠️ Subscription auth disclaimer](#subscription-auth-disclaimer)
- [Contributing](#contributing)
- [License](#license)

---

## Install

```bash
uv add logpose
# or
pip install logpose
```

Python 3.10+.

## Quickstart

```python
import asyncio
from logpose import Agent, tool


@tool
def get_weather(city: str) -> str:
    """Get the current weather in a city.

    Args:
        city: City name, e.g. "Pune".
    """
    return f"31C and hazy in {city}"


async def main() -> None:
    agent = Agent("anthropic", tools=[get_weather], system="Be terse.")
    result = await agent.run("What's the weather in Pune?")
    print(result.text)                # "It's 31C and hazy in Pune."
    print(result.usage.output_tokens) # token accounting for the whole run
    await agent.aclose()


asyncio.run(main())
```

A runnable version with streaming and a BYOK variant lives in
[`examples/weather.py`](examples/weather.py).

## Auth

### Subscription (default path in v0.1)

```bash
claude setup-token                        # from the Claude Code CLI
export CLAUDE_CODE_OAUTH_TOKEN=sk-ant-oat01-...
```

### BYOK

```bash
export ANTHROPIC_API_KEY=sk-ant-api03-...
```

### Auto-discovery

If Claude Code is installed and logged in, you do not have to export anything.
logpose reads its credential store **read-only**:

- macOS: the Keychain entry `Claude Code-credentials`;
- elsewhere (and as a macOS fallback): `~/.claude/.credentials.json`, or
  `$CLAUDE_CONFIG_DIR/.credentials.json` when that variable is set.

An expired OAuth token is refreshed in memory for the life of the process.
logpose **never writes back** to Claude Code's store.

### Precedence

First match wins:

| # | Source | Mode |
|---|--------|------|
| 1 | `Agent("anthropic", auth_token=...)` | subscription |
| 2 | `Agent("anthropic", api_key=...)` | API key |
| 3 | `CLAUDE_CODE_OAUTH_TOKEN` | subscription |
| 4 | `ANTHROPIC_API_KEY` | API key |
| 5 | Claude Code credential store | subscription |

Blank and whitespace-only values count as absent. Nothing found raises
`AuthError` telling you to run `claude setup-token`.

Note that **`CLAUDE_CODE_OAUTH_TOKEN` outranks `ANTHROPIC_API_KEY`**. A stray
`ANTHROPIC_API_KEY` set for unrelated tooling is common, and silently letting it
win would bill per token someone who deliberately set up a subscription. Pass
`api_key=` explicitly (and no `auth_token=`) to force BYOK regardless of the
environment.

Credential values never appear in a log line, a `repr`, an exception message, or
a **traceback** — anything that must reference one redacts it to a prefix plus a
length. That includes a credential a gateway echoes back in an error body, and
includes a client you injected yourself via `Agent("anthropic", client=...)`.

## Streaming

`agent.stream(...)` yields normalized events as they happen:

| Event | Fields | When |
|---|---|---|
| `TextDelta` | `text` | A chunk of assistant-visible text arrived. |
| `ThinkingDelta` | `text` | A chunk of model reasoning arrived. |
| `TurnEnd` | `stop_reason`, `usage` | One provider round trip finished. |
| `ToolCall` | `id`, `name`, `input` | A tool is about to run. Arguments are already parsed. |
| `ToolResult` | `id`, `name`, `content`, `is_error` | A tool finished; the result is going back to the model. |
| `RunEnd` | `result` | Terminal event, carrying the `RunResult`. |

```python
from logpose import RunEnd, TextDelta, ToolCall

async for event in agent.stream("What's the weather in Pune and Goa?"):
    match event:
        case TextDelta(text=text):
            print(text, end="", flush=True)
        case ToolCall(name=name, input=args):
            print(f"\n[{name}({args})]")
        case RunEnd(result=result):
            print(f"\n{result.iterations} turns, {result.usage.output_tokens} tokens out")
```

`RunEnd.result` is the same `RunResult` that `agent.run(...)` returns: final
`text`, the full `messages` history, aggregated `usage`, the final
`stop_reason`, and the `iterations` count.

### What the loop does

One iteration is one provider round trip. On `stop_reason == "tool_use"`, every
requested tool runs **concurrently** and all results go back in a **single** user
message (splitting them measurably degrades parallel tool calling). `pause_turn`
is re-issued transparently. `end_turn`, `refusal`, `max_tokens`, and
`stop_sequence` end the run. Exceeding `max_iterations` (default 25) raises
`MaxIterationsError` with the partial conversation attached.

`max_tokens` defaults to the provider's own ceiling — `Agent(AnthropicProvider(
max_tokens=2048))` sends 2048, not the loop default. Provider-specific request
options logpose does not model go through `extra=`:

```python
agent = Agent("anthropic", extra={"tool_choice": {"type": "any"}})
```

It is merged into every wire request, so it reaches `CompletionRequest.extra`.

**A failing tool is never an exception.** A handler that raises, or a call to a
tool that does not exist, becomes a `ToolResultBlock(is_error=True)` so the model
reads the error and adapts. Only the iteration cap, a provider failure, or a
credential failure ends a run abnormally.

## Defining tools

```python
from typing import Literal
from logpose import tool


@tool
def get_weather(city: str, unit: Literal["c", "f"] = "c") -> str:
    """Get the current weather in a city.

    Args:
        city: City name, e.g. "Pune".
        unit: Temperature unit.
    """
    return f"31{unit.upper()} and hazy in {city}"
```

- The JSON Schema comes from your annotations (pydantic models, `Literal`,
  `Optional`, nested models — all fine). Parameters without a default are
  required; `additionalProperties` is always `false`.
- The tool description comes from the docstring summary, and **per-parameter
  descriptions come from the Google-style `Args:` block** — the model leans on
  them, so write them. Override either with `@tool(name=..., description=...)`.
- `async def` handlers are awaited; sync handlers run in `asyncio.to_thread` so
  they never block the loop.
- Arguments arrive **validated and coerced**: an `int` parameter gets an `int`,
  a pydantic-model parameter gets an instance.
- Returned `dict`/`list` values are JSON-encoded; anything else is `str()`-ed.
- A decorated function is still an ordinary callable — `get_weather("Pune")`
  works in your own code and in your unit tests.

Un-schematizable signatures raise `ToolSchemaError` at decoration time, not at
runtime: `*args`, `**kwargs`, an unannotated parameter, a parameter whose name
pydantic reserves (`model_config`, anything leading-underscore), and a tool name
outside `[A-Za-z0-9_-]{1,128}` — the charset every targeted provider accepts.

## Multi-turn conversations

`Conversation` is a mutable history the loop appends to as the run progresses.

```python
from logpose import Agent, Conversation

agent = Agent("anthropic", tools=[get_weather])
chat = Conversation()

await agent.run("My name is Ada.", conversation=chat)
result = await agent.run("What is my name?", conversation=chat)
assert "Ada" in result.text

len(chat)     # number of turns so far
chat.text     # text of the most recent assistant turn
```

Because appends happen live, a run that fails partway through leaves the partial
history in place for inspection. A run without a `conversation` is single-shot
and shares nothing, so one `Agent` can serve many concurrent conversations.

## Sync usage

Two spellings, one private event loop per agent:

```python
from logpose import Agent, SyncAgent, run_sync, stream_sync

agent = Agent("anthropic", tools=[get_weather])

# wrapper object — best when you make several calls
with SyncAgent(agent) as sync_agent:
    print(sync_agent.run("What's the weather in Pune?").text)
    for event in sync_agent.stream("And in Goa?"):
        ...

# free functions — best for a one-off
result = run_sync(agent, "What's the weather in Pune?")
for event in stream_sync(agent, "And in Goa?"):
    ...
```

`stream_sync` returns a normal generator. It drives the async stream one step at
a time rather than buffering into a queue, so backpressure is preserved, `break`
propagates as a clean shutdown (in-flight tools cancelled, provider stream
closed), and exceptions surface at the `for` statement.

Each agent gets one private loop on a daemon thread, created on first sync call
and reused afterwards. That reuse is required, not an optimization: objects the
async stack builds lazily bind themselves to the first loop that touches them.
The thread is released by `SyncAgent.close()` / `close_sync(agent)`, when the
agent is garbage collected, or at interpreter exit.

Calling any sync entry point from a thread that already runs an event loop
raises `LogposeError` pointing you at the async API — blocking that thread on
another loop cannot work, so it fails loudly instead of deadlocking.

## Errors

Everything derives from `LogposeError`, so one `except` covers the surface.

| Error | Raised when |
|---|---|
| `AuthError` | No usable credential could be resolved, or a refresh failed. |
| `ProviderError` | The upstream provider failed. Carries `status_code` and `retryable`. |
| `MaxIterationsError` | The iteration cap was hit. Carries `messages` and `max_iterations`. |
| `ToolSchemaError` | A tool definition cannot be turned into a JSON Schema (decoration time). |
| `ToolExecutionError` | The tool-execution machinery itself failed. Ordinary handler failures do *not* raise — they become error tool results. |

## Providers

Resolved by name, imported lazily — `import logpose` never pulls in a provider
SDK, reads a credential, or touches the network.

```python
from logpose import known_providers, resolve

known_providers()   # ["anthropic", "docker", "docker-models", "openai-compat"]
resolve("anthropic", model_default="claude-opus-5")
```

### Local models via Docker Model Runner

Docker Desktop serves an OpenAI-compatible API on the host once the model runner
is enabled, so local models need no credential and no extra dependency:

```bash
docker desktop enable model-runner --tcp 12434
docker model pull ai/gemma4
```

```python
from logpose import Agent

agent = Agent("docker")                    # uses the first pulled model
agent = Agent("docker", model="gemma4")    # or name one
```

`Agent("docker")` leaves the model as `"auto"` and asks the runner for its first
model on the first request, so constructing a provider still never touches the
network. Point it elsewhere with `base_url=` (inside a container that is
`http://model-runner.docker.internal/engines/v1`), or with
`$DOCKER_MODEL_RUNNER_URL` and `$DOCKER_MODEL`.

Two things are worth knowing about local models specifically:

- **Reasoning arrives two different ways.** Some servers emit a
  `reasoning_content` delta; others inline `<think>…</think>` in ordinary
  content. Both are normalized to `ThinkingDelta` events and a `ThinkingBlock`.
  Pass `parse_think_tags=False` to disable the second.
- **Malformed tool arguments are reported, not raised.** Small models often emit
  invalid JSON for tool arguments. Rather than killing the run, the raw string is
  passed through as an unexpected argument, so schema validation fails and the
  model gets a tool error it can retry.

### Any OpenAI-compatible server

The same backend drives llama.cpp, vLLM, Ollama, LM Studio, OpenAI, and Kimi:

```python
agent = Agent(
    "openai-compat",
    base_url="https://api.moonshot.cn/v1",   # or $OPENAI_BASE_URL
    model="kimi-k2-0905-preview",            # or $OPENAI_MODEL
    api_key="...",                           # or $OPENAI_API_KEY
)
```

The `docker` backend deliberately does **not** read `$OPENAI_API_KEY` — an
ambient key should never be shipped to a server on localhost.

### Writing your own

Implement one method and register it:

```python
from collections.abc import AsyncIterator
from logpose import CompletionDone, CompletionRequest, ProviderEvent, register


class MyProvider:
    name = "mine"
    model_default = "my-model-1"       # used when Agent has no model= override

    async def stream(self, req: CompletionRequest) -> AsyncIterator[ProviderEvent]:
        ...
        yield CompletionDone(message=..., stop_reason="end_turn", usage=...)


register("mine", lambda **kwargs: MyProvider(**kwargs))
```

`name` and `model_default` are both part of the protocol and both checked by
`isinstance(obj, Provider)` — there is no sane model identifier to invent when
one is missing, so a provider without a usable `model_default` fails at
construction rather than sending `model=""` and getting an opaque 400. An
optional `max_tokens: int` attribute is honoured too: `Agent` falls back to it
when built without an explicit `max_tokens=`.

The stream must end with exactly one `CompletionDone` whose `message` is the
fully assembled assistant turn, blocks in wire order — thinking blocks and their
signatures included, verbatim. The loop appends it to the history unchanged,
with one exception: a turn with **no** content blocks (the shape of a pre-output
`refusal`) is reported but never stored, because providers reject an empty
content array and it would poison every later turn on the same `Conversation`.

Anything a provider returns that logpose does not model — server tool use, web
search results, code execution output — arrives as an opaque `RawBlock` and is
re-emitted verbatim, so an assistant turn always survives a `pause_turn` resend.

### Roadmap

- **v0.1 (now)** — Anthropic (subscription OAuth + BYOK), Docker Model Runner
  for local models, and a generic OpenAI-compatible backend that already covers
  OpenAI, Kimi, vLLM, Ollama, and LM Studio. No extra dependency: the
  OpenAI-compatible path is plain `httpx`, which logpose already ships.
- **Later** — Codex subscription auth (needs harness delegation; the hybrid
  decision gets revisited then), context compaction, pre/post tool-call hooks,
  MCP tool ingestion.

<a id="subscription-auth-disclaimer"></a>

## ⚠️ Subscription auth disclaimer

**Using a Claude Code subscription token against the raw Anthropic API is not an
officially supported integration.** Be clear-eyed about what that means:

- It may conflict with Anthropic's terms of service for Claude Code and for
  consumer subscriptions. logpose is not legal advice; read your agreement and
  decide for yourself.
- It relies on undocumented details — the credential-store layout, the Keychain
  service name, the OAuth client id, the token endpoint, and the
  `anthropic-beta: oauth-2025-04-20` header. Any of these can change without
  notice and break this path, possibly silently.
- Anthropic may start requiring requests on these tokens to look like Claude
  Code's. `AnthropicProvider(compat_claude_code=True)` prepends the Claude Code
  identity line to the system prompt as an escape hatch if that happens.
- Rate limits, abuse handling, and account standing are Anthropic's call, not
  ours. Using this path is at your own risk, including the risk to your account.

**BYOK (`ANTHROPIC_API_KEY`) is the supported path.** It is one environment
variable, it goes through the same provider and the same loop, and it is what
you should use in production or anywhere the consequences of breakage matter.

## Contributing

Contributions are welcome. The short version:

```bash
git clone https://github.com/xdadwal/logpose.git
cd logpose
uv sync
uv run pytest -q                          # 492 tests, no network
uv run ruff check && uv run mypy src/
```

The full guide — project layout, how to add a provider, what a good test looks
like here, and the PR checklist — is in [CONTRIBUTING.md](CONTRIBUTING.md).

Found a security issue? Please follow [SECURITY.md](SECURITY.md) rather than
opening a public issue.

## License

MIT — see [LICENSE](LICENSE).
