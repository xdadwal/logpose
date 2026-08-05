# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **OpenAI-compatible provider** (`openai-compat`). Drives any server exposing
  `POST /chat/completions` — llama.cpp, vLLM, Ollama, LM Studio, OpenAI, Kimi.
  Built on `httpx`, which logpose already depends on, so it adds no dependency
  and no optional extra. Always streams.
- **Docker Model Runner provider** (`docker`, alias `docker-models`) for local
  models: `Agent("docker")`. Defaults to Docker Desktop's host endpoint
  (`http://localhost:12434/engines/v1`, overridable with
  `$DOCKER_MODEL_RUNNER_URL`) and needs no credential. With no `model=`, it asks
  the runner for its first model on the first request, so constructing a
  provider still never touches the network. `$DOCKER_MODEL` pins one.
- Reasoning from local models is normalized whichever way it arrives — a
  `reasoning_content` delta, or `<think>…</think>` inlined in ordinary content —
  into `ThinkingDelta` events and a `ThinkingBlock`. The tag splitter handles
  tags straddling chunk boundaries; `parse_think_tags=False` turns it off.
- `examples/local_docker.py` — a runnable local-model example.
- Project documentation: `CONTRIBUTING.md` (setup, the provider seam, test and
  PR conventions) and `SECURITY.md` (private reporting, and what does and does
  not count as a vulnerability in a library that handles provider credentials
  and runs caller-supplied tools).
- GitHub Actions CI: tests on Python 3.10–3.13, plus ruff, mypy, and a build job
  that installs the wheel standalone and asserts `import logpose` still pulls in
  no provider SDK.

### Fixed

- `redact` moved to `logpose.providers._redact` so a backend can reuse it
  without importing a sibling provider's SDK. It is still exported from
  `logpose.providers.anthropic` for compatibility.
- Corrected the repository URL in `pyproject.toml` and the changelog link
  refs, which pointed at a repository that does not exist.
- README: the intro no longer claims v0.1 is Anthropic-only, the disclaimer
  anchor no longer depends on GitHub's emoji-anchor quirk, and there is a table
  of contents.

### Security

- The `docker` backend does not read `$OPENAI_API_KEY` (`api_key_env=None`): an
  ambient key must never be shipped to a server on localhost. The generic
  `openai-compat` backend still reads it, as callers expect.

### Notes

- Chat Completions differs from logpose's model in three ways the provider
  absorbs, so nothing above `logpose.providers` changed to add it: the single
  user message holding a turn's tool results is fanned out into one `tool`
  message per call; tool arguments stream as a JSON *string* in fragments keyed
  by `index` and are concatenated then parsed; and reasoning is not sent back,
  matching how OpenAI-style APIs work, since there is no signature to preserve.
- Tool arguments that are not valid JSON — routine from small local models — are
  passed through as an unexpected argument instead of raising, so schema
  validation fails into a tool error the model can retry rather than killing the
  run. A server that reports `finish_reason: "stop"` while still emitting tool
  calls is treated as `tool_use`, since trusting it would drop the calls.

## [0.1.0] - 2026-08-05

First release. logpose owns the agentic loop rather than wrapping a vendor
harness, so the message model, the events, and the tool API are the same
whichever backend is behind them.

### Added

- **The loop.** `Agent` drives a provider, executes tools, and terminates on
  `end_turn` / `refusal` / `max_tokens` / `stop_sequence`. Tools requested in one
  turn run concurrently and their results go back in a single user message;
  `pause_turn` is re-issued transparently; the `max_iterations` cap (default 25)
  raises `MaxIterationsError` with the partial conversation attached. A failing
  tool — or a call to a tool that does not exist — becomes an error tool result
  the model can read and adapt to, never an exception. An assistant turn with no
  content blocks (the shape of a pre-output `refusal`) is reported but never
  stored: providers reject an empty content array, so keeping it would break
  every later turn on the same `Conversation`.
- **`Agent.run` / `Agent.stream`.** `run` is implemented by draining `stream`, so
  there is exactly one loop implementation. `stream` yields `TextDelta`,
  `ThinkingDelta`, `TurnEnd`, `ToolCall`, `ToolResult`, and a terminal `RunEnd`
  carrying the same `RunResult` that `run` returns.
- **`Conversation`.** A mutable multi-turn history the loop appends to live, so a
  run that fails partway through leaves its partial history for inspection.
- **`@tool`.** Builds a `ToolDef` from an ordinary function: JSON Schema from the
  annotations, description from the docstring summary, per-parameter
  descriptions from a Google-style `Args:` block. Sync and `async` handlers both
  work (sync ones run in `asyncio.to_thread`). Arguments arrive validated and
  coerced. A decorated function stays callable from ordinary Python. Anything
  that would fail later as an opaque provider 400 raises `ToolSchemaError` at
  decoration time instead: `*args`, `**kwargs`, an unannotated parameter, a
  parameter whose name pydantic reserves (`model_config`, anything
  leading-underscore), and a tool name outside `[A-Za-z0-9_-]{1,128}`.
- **Provider-neutral message model.** `Message` with `TextBlock`,
  `ThinkingBlock`, `RedactedThinkingBlock`, `ToolUseBlock`, `ToolResultBlock`,
  and the opaque `RawBlock`; `Usage` aggregates field-wise across turns.
  Thinking-block signatures round-trip losslessly, which the Anthropic API
  requires. `RawBlock` preserves block types logpose does not model (server tool
  use, search results, code execution) verbatim, so an assistant turn always
  survives the `pause_turn` resend.
- **Provider seam.** `Provider`, `CompletionRequest`, `ToolSpec`, and the
  `ProviderEvent` union, plus a name-based registry (`register`, `resolve`,
  `known_providers`). Providers advertise tools but cannot execute them. The
  protocol requires `name` and `model_default`; an optional `max_tokens` is
  honoured when `Agent` was built without an explicit one, so a provider
  configured to cap output cost is not silently overridden. `Agent(extra=...)`
  reaches `CompletionRequest.extra`, the provider escape hatch for request
  options logpose does not model.
- **Anthropic provider.** Always streams via `AsyncAnthropic.messages.stream`.
  Adaptive thinking by default, `claude-opus-5`, `max_tokens=16000`, no sampling
  parameters unless you pass them. Maps every native stop reason into logpose's
  own, and never crashes the loop on an unknown one.
- **Claude Code subscription auth.** `CLAUDE_CODE_OAUTH_TOKEN`, `ANTHROPIC_API_KEY`,
  and read-only discovery of the local Claude Code credential store (macOS
  Keychain, `~/.claude/.credentials.json`, or `$CLAUDE_CONFIG_DIR`). Expired
  OAuth tokens are refreshed in memory with single-flight locking that also
  covers the provider's lazy first resolution, so concurrent first requests
  share one keychain read and one refresh grant rather than replaying a
  single-use refresh token N times. The store is never written to. Subscription tokens outrank a stray `ANTHROPIC_API_KEY` so
  nobody gets silently billed per token.
- **Sync facade.** `SyncAgent`, `run_sync`, `stream_sync`, and `close_sync`. Each
  agent gets one private event loop on a daemon thread, reused across calls;
  `stream_sync` returns a real generator that preserves backpressure and cleans
  up on early `break`. Calling any of them from a running event loop raises
  `LogposeError` pointing at the async API.
- **Error hierarchy.** `LogposeError` with `AuthError`, `ProviderError` (carrying
  `status_code` and `retryable`), `MaxIterationsError`, `ToolSchemaError`, and
  `ToolExecutionError`.
- `examples/weather.py` — a runnable end-to-end example with subscription and
  BYOK variants.

### Security

- Credential values never appear in a log line, a `repr`, an exception message,
  or a **traceback** — anything that must reference one redacts it to a prefix
  plus a length. A gateway that echoes the presented credential in its error
  body is scrubbed out of both the raised `ProviderError` and the SDK exception
  chained as its `__cause__`, which every traceback renders; a client injected
  via `client=` is scrubbed from its live `api_key` / `auth_token` even though
  it never went through credential resolution. Tool-argument validation errors
  report the field path and message only, never the offending value.

### Notes

- Authenticating with a Claude Code subscription token against the raw Anthropic
  API is **not an officially supported integration** and depends on
  undocumented details that can change without notice. BYOK
  (`ANTHROPIC_API_KEY`) is the supported path. See the disclaimer in the README.
- `import logpose` imports no provider SDK, reads no credential, and makes no
  network call; backends are resolved by name and import lazily.

[Unreleased]: https://github.com/xdadwal/logpose/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/xdadwal/logpose/releases/tag/v0.1.0
