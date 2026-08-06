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

Backends today, one per credential rather than one per vendor: **`anthropic`** and
**`openai`** for API keys, **`claude-code`** and **`codex`** for those CLIs'
subscriptions, **`docker`** for local models, and **`openai-compat`** for anything
speaking Chat Completions — Kimi, vLLM, Ollama, LM Studio. Everything but the
Anthropic pair adds no dependency; the local path needs no credential.

If you plan to use either subscription provider, read the
[disclaimer](#subscription-auth-disclaimer) first.

---

## Contents

- [Install](#install)
- [Quickstart](#quickstart)
- [Auth](#auth)
- [Streaming](#streaming) — [gating tool calls](#gating-tool-calls)
- [Defining tools](#defining-tools)
- [Discovering tools](#discovering-tools)
- [Multi-turn conversations](#multi-turn-conversations)
- [Sync usage](#sync-usage)
- [Errors](#errors)
- [Providers](#providers) — [Anthropic & Claude Code](#anthropic-and-claude-code),
  [OpenAI & Codex](#openai-and-codex-the-responses-api),
  [local models](#local-models-via-docker-model-runner),
  [OpenAI-compatible](#any-openai-compatible-server),
  [writing your own](#writing-your-own)
- [⚠️ Subscription auth disclaimer](#subscription-auth-disclaimer)
- [Contributing](#contributing)
- [License](#license)

---

## Install

Not on PyPI yet — install from the repository:

```bash
uv add git+https://github.com/xdadwal/logpose
# or
pip install git+https://github.com/xdadwal/logpose
```

Python 3.10+. The only runtime dependencies are `anthropic`, `pydantic`, and
`httpx`; the local-model and OpenAI-compatible backends add nothing further.

> **Note:** the name `logpose` is already taken on PyPI by an unrelated logging
> library, so `pip install logpose` installs *someone else's package*. A
> distribution name is still to be chosen before this is published.

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

Four providers, two vendors, one rule each: **a provider accepts exactly one kind
of credential.** `anthropic` and `openai` take an API key; `claude-code` and
`codex` take that vendor's subscription token. Naming the provider is how you pick.

| Provider | Credential | Endpoint |
|---|--------|------|
| `anthropic` | `ANTHROPIC_API_KEY` | `api.anthropic.com` |
| `claude-code` | Claude Code subscription | same, plus the OAuth beta header |
| `openai` | `OPENAI_API_KEY` | `api.openai.com/v1/responses` |
| `codex` | ChatGPT subscription | `chatgpt.com/backend-api/codex` |

A credential of the wrong kind neither satisfies a provider nor hides a usable one
from it: `Agent("openai")` ignores a subscription token sitting in
`~/.codex/auth.json` and goes on to find `$OPENAI_API_KEY`, and if there is no API
key anywhere it fails with a message naming `codex` instead. That is the point of
splitting them — it replaces a precedence policy you had to read about with a
choice you already made.

Every path reads its credential store **read-only** and refreshes in memory only.
Blank and whitespace-only values count as absent everywhere.

### API keys

```bash
export ANTHROPIC_API_KEY=sk-ant-api03-...      # Agent("anthropic")
export OPENAI_API_KEY=sk-proj-...              # Agent("openai")
```

Or pass `api_key=` directly, which outranks the environment.

### Claude Code subscription

```bash
claude setup-token
export CLAUDE_CODE_OAUTH_TOKEN=sk-ant-oat01-...
```

If Claude Code is installed and logged in, the export is optional — logpose reads
its store: on macOS the Keychain entry `Claude Code-credentials`, elsewhere (and as
a macOS fallback) `~/.claude/.credentials.json`, or
`$CLAUDE_CONFIG_DIR/.credentials.json`. Precedence: `auth_token=`, then
`$CLAUDE_CODE_OAUTH_TOKEN`, then the store.

### Codex / ChatGPT subscription

```bash
codex login
```

Nothing to export. logpose reads `~/.codex/auth.json` (or `$CODEX_HOME/auth.json`),
takes the token out of its `tokens` block, and refreshes it when the access token's
own `exp` claim says it is due — the file records no expiry of its own. Precedence:
`auth_token=`, then the store.

`auth_mode` inside `auth.json` is ignored on purpose: logpose goes by which field
actually holds a value, because the CLI writes files where the two disagree.

If `auth.json` names no ChatGPT account, `codex` raises `AuthError` before sending
anything rather than letting the backend answer 401 with an empty body; override
with `account_id=` or `$CHATGPT_ACCOUNT_ID`.

One thing logpose deliberately does not do on either subscription path: **persist a
rotated refresh token.** A long-lived process refreshes fine, but a restart falls
back to whatever is still on disk, so if the vendor rotates the token you re-run
the CLI login. Writing it back would race the CLI for the file and break the
read-only promise.

### CLI identity on the subscription paths

Both vendors expect requests on a subscription token to look like their own CLI's,
and both subscription providers comply by default:

- **`claude-code`** prepends `You are Claude Code, Anthropic's official CLI for
  Claude.` as the first system block, with your `system=` following it as a second
  block. Without it the API answers `429 rate_limit_error` regardless of remaining
  quota — a body carrying no detail and, tellingly, none of the
  `anthropic-ratelimit-*` headers a genuine limit sends. It is a rejection, not an
  exhausted quota.
- **`codex`** opens `instructions` with `You are Codex, a coding agent based on
  GPT-5.` and sends `originator: codex_cli_rs`.

Turn either off with `compat_claude_code=False` / `compat_codex_cli=False`, and
expect the request to be rejected. The API-key providers never send these, and have
no flag for it.

The `chatgpt-account-id` and `OpenAI-Beta` headers are *not* part of that shim —
they are the Codex wire protocol, and go out regardless of `compat_codex_cli`.

### All providers

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
reads the error and adapts. Only the iteration cap, a provider failure, a
credential failure, or a gate that raises ends a run abnormally.

### Gating tool calls

`on_tool_call` sees every call the model asks for **before** any handler starts.
Return `None` to let it through, or a string to block it and hand that text back
to the model instead:

```python
def gate(call):
    if call.name in DANGEROUS and not user_approves(call.name, call.input):
        return "Denied by the user. Try another approach."
    return None

agent = Agent("anthropic", tools=[...], on_tool_call=gate)
```

Within a turn the gate runs **one call at a time, in wire order** — a gate that
asks a human cannot be asked several things at once — and entirely before the
concurrent execution phase, so a blocked call has no chance to have already run.
(An `Agent` holds no per-run state, so a gate shared across *concurrent* runs
still needs its own lock.) It may be `async def` or ordinary; a synchronous gate
runs inline on the event loop, so put anything blocking in `asyncio.to_thread`.

Blocking is reported to the model as an error by default. Return
`ToolGateResult(content=..., is_error=False)` when the block is a redirection
rather than a failure — declining a call *and* saying what to do instead reads
better to the model as a steer than as something that broke.

A gate that raises ends the run: a permission layer that breaks must not fail
open.

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

## Discovering tools

Once tools live across several modules, hand-maintaining `tools=[a, b, c]` stops
being fun — and a tool you forget to add is silently unavailable to the model.
`discover_tools` imports a module or package and collects every `@tool` it finds:

```python
from logpose import Agent, discover_tools

agent = Agent("docker", tools=discover_tools("myapp.tools"))
```

It returns an ordinary list, so it composes:

```python
tools = [*discover_tools("myapp.tools"), extra_tool]
tools = discover_tools("myapp.tools", "myapp.integrations")   # several targets
tools = discover_tools(predicate=lambda t: not t.name.startswith("debug_"))
```

With no arguments it scans the **calling** module. It reads that module's globals
directly, so it works in a script, a REPL, or `python -c` — but it only sees names
bound *above* the call, so put it at the bottom of the file.

A tool re-exported into a package's `__init__.py` — the usual
`from .weather import get_weather` — is returned **once**: results are deduplicated
by object identity, so a re-export is not mistaken for a duplicate. Two *different*
tools sharing a name is a genuine conflict and raises `LogposeError` naming both
defining modules; `predicate` is the escape hatch.

Order is **traversal order, not alphabetical**: targets as given, each module's own
namespace in definition order, then its submodules sorted by name, depth-first.
That means appending a tool leaves the preceding request bytes untouched, which
keeps automatic prefix caching intact on providers that do it — sorting by name
would shift every entry after an insertion. Sort it yourself if you want
alphabetical.

### What is and is not found

Only **module-level** names. Not found: a tool defined in a class body or inside a
function, or one held only in a list or dict. `__all__` is not consulted. A
`ToolDef` imported into a scanned module from a third-party library **is** found —
exclude it with `predicate` if you don't want it advertised.

Submodules whose name starts with `_` are skipped *without being imported*, and so
is everything beneath them (including `__main__.py`). A directory with no
`__init__.py` is a namespace package and is not descended into. Naming an
underscore module explicitly still scans it.

> **Discovery imports every module under the target, which executes its top-level
> code.** Never build a target from untrusted input, and call it at startup rather
> than inside a running event loop, since importing a package tree blocks. A
> submodule that fails to import raises rather than being skipped — a silently
> missing tool is worse than a loud error.

One sharp edge: `Agent` rejects two tools with the same *name* regardless of
identity, so an `extra_tool` passed alongside discovery must not also live inside
the scanned tree. Either keep it outside, or exclude it with `predicate`.

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

Everything *logpose* raises derives from `LogposeError`, so one `except` covers
the surface. The exception is code you supplied: whatever your `on_tool_call`
gate raises propagates unchanged, because a permission layer that breaks must
not fail open.

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

known_providers()
# ["anthropic", "claude-code", "codex", "docker", "docker-models",
#  "openai", "openai-compat"]
resolve("anthropic", model_default="claude-opus-5")
```

The four credentialed providers come in pairs — one API-key provider and one
subscription provider per vendor, sharing a wire implementation but never a
credential policy. See [Auth](#auth) for which takes what.

### Anthropic and Claude Code

```python
agent = Agent("anthropic")                            # $ANTHROPIC_API_KEY
agent = Agent("claude-code")                          # rides the local login
agent = Agent("anthropic", model_default="claude-opus-5", thinking="disabled")
```

Same Messages API, same SDK, same body. They differ only in the auth header, the
OAuth beta header, and the identity line.

### OpenAI and Codex (the Responses API)

```python
agent = Agent("openai")                               # $OPENAI_API_KEY
agent = Agent("codex")                                # rides `codex login`
agent = Agent("codex", model="gpt-5.4")               # or $CODEX_MODEL
agent = Agent("openai", reasoning_effort="high")      # low|medium|high|xhigh|max
```

**`openai` is not `openai-compat`.** They speak different APIs, and the
difference is functional, not cosmetic:

| | `openai` | `openai-compat` |
|---|---|---|
| API | Responses (`/responses`) | Chat Completions (`/chat/completions`) |
| Served by | OpenAI only | llama.cpp, vLLM, Ollama, LM Studio, Kimi, OpenAI |
| Reasoning across a tool call | preserved, as an encrypted item | **lost** — no field carries it |

Pick `openai` for OpenAI's own reasoning models; pick `openai-compat` for
anything that merely speaks OpenAI's older shape. Four things are specific to the
Responses pair:

- **Reasoning round-trips as an opaque block.** The API returns reasoning as a
  separate output item holding a summary plus an `encrypted_content` blob you
  cannot read, and it rejects a request where that item is not immediately
  followed by the item it reasoned for. logpose keeps the whole item in a
  `RawBlock` so it goes back byte-exact and in position. You still get
  `ThinkingDelta` events live from the summary, but code walking
  `result.messages` for a `ThinkingBlock` will not find one here. Whether
  reasoning is emitted at all is the *backend's* call, and the two differ: on
  `api.openai.com` a tool-calling turn carries a reasoning item, while on the
  Codex subscription backend the same turn frequently carries none, and summaries
  are often empty at low effort either way. Both observed live. Treat a turn with
  no reasoning item as normal, not as a broken round trip.
- **`max_tokens` is a no-op on `codex`.** The subscription backend answers `400
  Unsupported parameter: max_output_tokens`, so logpose omits the field there and
  the model's own budget applies. `openai` sends and honours it.
- **`store` is always `false`.** logpose owns the history, so server-side
  retention would buy nothing, and the subscription backend requires it.
- **A `Conversation` belongs to the backend that produced it.** Handing Responses
  history to Anthropic (or the reverse) is not supported: each backend's opaque
  reasoning is meaningless to the other. The Responses providers silently drop
  Anthropic thinking blocks rather than crashing, but the reasoning is gone.

`codex` defaults to `gpt-5.5`, the highest-priority model the Codex CLI lists;
`openai` defaults to `gpt-5.1`. They differ because ChatGPT-backend slugs and
public API model ids are separate namespaces, and a provider whose default moved
with the credential would be a debugging nightmare. Both defaults are verified
against their live endpoint. Override with `model=`, `$CODEX_MODEL`, or
`$OPENAI_RESPONSES_MODEL`.

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

The same backend drives llama.cpp, vLLM, Ollama, LM Studio, and Kimi. For OpenAI's
own models prefer the `openai` provider above, which speaks the newer Responses
API and keeps reasoning across tool calls:

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

- **v0.1 (now)** — Anthropic and Claude Code (Messages API, API key or
  subscription), OpenAI and Codex (Responses API, API key or subscription),
  Docker Model Runner for local models, and a generic OpenAI-compatible backend
  covering Kimi, vLLM, Ollama, and LM Studio. Only the Anthropic pair needs a
  vendor SDK; everything else is plain `httpx`, which logpose already ships.
- **Later** — context compaction, a post-tool-call hook, MCP tool ingestion, and
  cross-backend conversation portability (today a `Conversation` belongs to the
  backend that produced it).

<a id="subscription-auth-disclaimer"></a>

## ⚠️ Subscription auth disclaimer

**Neither subscription provider — `claude-code` against the Anthropic API, nor
`codex` against the Codex backend — is an officially supported integration.** Be
clear-eyed about what that means:

- It may conflict with the vendor's terms of service for their CLI and for
  consumer subscriptions. logpose is not legal advice; read your agreement and
  decide for yourself.
- It relies on undocumented details that can change without notice and break the
  path, possibly silently:
  - **`claude-code`** — the credential-store layout, the Keychain service name,
    the OAuth client id, the token endpoint, and the
    `anthropic-beta: oauth-2025-04-20` header.
  - **`codex`** — the `auth.json` layout, the OAuth client id, the
    `auth.openai.com/oauth/token` endpoint, the `chatgpt.com/backend-api/codex`
    path, the `chatgpt-account-id` / `OpenAI-Beta` / `originator` headers, and the
    identity line.
- **Both vendors require requests on these tokens to look like their CLI's.** An
  Anthropic subscription request whose system prompt does not open with the Claude
  Code identity line comes back `429 rate_limit_error` regardless of remaining
  quota; the Codex backend likewise expects a CLI-shaped `instructions` field and
  an `originator` header. Both providers send them automatically
  (`compat_claude_code` / `compat_codex_cli`). These are exactly the kind of
  undocumented requirements that can change again without notice.
- Rate limits, abuse handling, and account standing are the vendor's call, not
  ours. Using either path is at your own risk, including the risk to your account.

**The API-key providers are the supported paths** — `anthropic` with
`ANTHROPIC_API_KEY`, `openai` with `OPENAI_API_KEY`. One environment variable, the
same loop, the same events, and what you should use in production or anywhere the
consequences of breakage matter. Because they are separate providers, choosing one
is also a guarantee: no stored subscription token can divert them.

## Contributing

Contributions are welcome. The short version:

```bash
git clone https://github.com/xdadwal/logpose.git
cd logpose
uv sync
uv run pytest -q                          # 784 tests, no network
uv run ruff check && uv run mypy src/
```

The full guide — project layout, how to add a provider, what a good test looks
like here, and the PR checklist — is in [CONTRIBUTING.md](CONTRIBUTING.md).

Found a security issue? Please follow [SECURITY.md](SECURITY.md) rather than
opening a public issue.

## License

MIT — see [LICENSE](LICENSE).
