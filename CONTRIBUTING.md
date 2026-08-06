# Contributing to logpose

Thanks for taking the time. This guide covers the setup, the conventions that
matter here, and what a reviewable pull request looks like.

## Getting set up

logpose uses [uv](https://docs.astral.sh/uv/). Python 3.10+.

```bash
git clone https://github.com/xdadwal/logpose.git
cd logpose
uv sync
```

Everything below must pass before you open a PR — the same three commands CI
runs:

```bash
uv run pytest -q
uv run ruff check
uv run mypy src/
```

The suite is hermetic: every provider is exercised through an
`httpx.MockTransport` or a scripted `FakeProvider`, so it needs no credentials,
no network, and no running model server. If a change makes a test need any of
those, that is a design problem worth raising in the PR.

## Project layout

```
src/logpose/
├── agent.py          the loop — the only place tools are executed
├── messages.py       provider-neutral message model
├── events.py         public streaming events + RunResult
├── tools.py          @tool decorator, JSON Schema generation, discover_tools()
├── sync.py           blocking facade over the async core
├── errors.py         exception hierarchy
├── auth/             credential resolution (read-only)
└── providers/        backends; nothing above this directory is provider-specific
```

The main architectural invariant is that nothing above `logpose/providers/`
depends on a specific backend. A provider turns a `CompletionRequest` into a
stream of events and does not execute tools, which is why `ToolSpec` carries no
handler. Provider-specific behavior belongs behind the provider boundary.

## Adding a provider

Implement the `Provider` protocol and register it. A provider needs `name`,
`model_default`, and `stream`; an optional `max_tokens` is honoured by `Agent`
when it was built without one.

```python
from logpose import CompletionDone, CompletionRequest, ProviderEvent, register


class MyProvider:
    name = "mine"
    model_default = "my-model-1"

    async def stream(self, req: CompletionRequest) -> AsyncIterator[ProviderEvent]:
        ...
        yield CompletionDone(message=..., stop_reason="end_turn", usage=...)


register("mine", lambda **kwargs: MyProvider(**kwargs))
```

Register it lazily in `logpose/providers/__init__.py` — importing the module
inside the factory, not at the top of the file — so `import logpose` continues
to pull in no vendor SDK. There is a test that enforces this.

The stream must end with exactly one `CompletionDone` whose `message` is the
assembled assistant turn, blocks in wire order. Two existing backends document
their own divergences from logpose's model in their module docstrings, and between
them cover most of what a new one will hit:

- `logpose/providers/openai_compat.py` — a message-shaped wire format (Chat
  Completions): tool-result fan-out, string tool arguments streamed in fragments,
  reasoning with no signature to preserve.
- `logpose/providers/codex.py` — an *item*-shaped wire format (Responses), where a
  turn is a flat list of siblings rather than a message with content blocks, and
  where opaque reasoning has to survive a round trip in order. If your backend
  returns anything logpose does not model, `RawBlock` is how it goes back
  verbatim.

Anything shared between backends goes in a private helper module —
`providers/_redact.py`, `providers/_toolargs.py` — never imported from a sibling
provider, or resolving one backend would drag in the other's SDK. The same rule
applies under `auth/`: `auth/_common.py` holds the backend-neutral credential
core, and each vendor module owns only its own store, endpoint, and error strings.

## One provider per credential

A provider accepts **exactly one kind of credential**, and where a vendor supports
both an API key and a subscription token that means two registered providers over a
shared private base: `anthropic`/`claude-code` over `_anthropic_base.py`,
`openai`/`codex` over `_responses.py`.

Keep credential types explicit rather than building a provider that changes
behavior according to whichever credential it finds. The endpoint, headers,
request fields, and compatibility settings are then known at construction time,
and each provider can document a simple credential policy.

The pattern to copy: subclass the shared base, set the class variables that name
your credential kind and endpoint, and override only the hooks that genuinely
differ. `DockerModelsProvider` is the same shape applied to a policy difference
rather than a credential one.

## Conventions

- **Type annotations on everything public**, and `mypy src/` runs with
  `disallow_untyped_defs`. Prefer fixing a type error over `# type: ignore`; a
  narrow, commented ignore for a third-party stub gap is fine.
- **Google-style docstrings** on public functions and classes, with an `Args:`
  block. This is not only house style: `@tool` parses `Args:` to build
  per-parameter descriptions in the JSON Schema the model actually sees.
- **Ruff** with `E,F,I,UP,B` at line length 100.
- **Never let a credential reach a log line, a `repr`, an exception message, or
  a traceback.** Redact through `logpose.providers._redact.redact`. Note that
  `raise wrapper from exc` renders `str(exc)` in every traceback, so the chained
  cause needs scrubbing too — `scrub_exception_in_place` exists for that.

## Tests

Name the behaviour, not the function: `test_tool_results_fan_out_to_one_message_each_in_order`
beats `test_messages_to_wire`. A test that would have passed before your change
is not a regression test.

For a bug fix, add the test first and confirm it fails against the old code. For
a new provider, cover at minimum: request translation, streaming assembly,
`stop_reason` mapping, usage mapping, and error classification (which failures
are retryable).

The suite must stay hermetic, so prefer a fake over a real dependency: providers
are driven through `httpx.MockTransport` or the scripted `tests/fake_provider.py`.
The exception is `discover_tools`, whose whole job is real import machinery — it
is tested against committed fixture packages under `tests/fixtures_*`. Those are
deliberately *not* named `test_*.py` so pytest never collects them, and
`tests/fixtures_broken/` exists only to be imported and fail, so keep it out of
any happy-path scan.

## Pull requests

- One logical change per PR.
- Update `CHANGELOG.md` under `## [Unreleased]` using the
  [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) sections.
- Update `README.md` if you changed anything a user can see.
- Say what you verified and how. If you could not verify something, say that
  too — an honest gap is more useful than a confident guess.

## Reporting bugs

Include the logpose version, Python version, the provider, and a minimal
reproduction. Please scrub credentials from any traceback you paste; if you find
one that logpose failed to scrub itself, that is a security issue — see
[SECURITY.md](SECURITY.md).
