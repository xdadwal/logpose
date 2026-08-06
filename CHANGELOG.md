# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Codex provider** (`codex`) — `Agent("codex")`, driving OpenAI's Responses API
  against the Codex subscription endpoint (`chatgpt.com/backend-api/codex`) on a
  credential from `codex login`. Built on `httpx`, so it adds no dependency.
  Always streams. `reasoning_effort` selects depth
  (`low`/`medium`/`high`/`xhigh`/`max`); `$CODEX_MODEL` and `$CODEX_BASE_URL`
  override the defaults.
- **OpenAI provider** (`openai`) — the same Responses API with an
  `$OPENAI_API_KEY` against `api.openai.com/v1`. Distinct from `openai-compat`,
  which speaks Chat Completions: that API has no field able to carry a reasoning
  model's chain of thought across a tool call, so multi-step tool use on a
  reasoning model loses it every turn. Defaults to `gpt-5.1` and reads
  `$OPENAI_RESPONSES_MODEL` / `$OPENAI_RESPONSES_BASE_URL`, deliberately *not*
  `openai-compat`'s `$OPENAI_MODEL` / `$OPENAI_BASE_URL`.
- **Claude Code provider** (`claude-code`) — the Anthropic Messages API on a
  Claude Code subscription token, with the identity line and the OAuth beta
  header. `anthropic` is now API-key-only.
- **Provider discovery**, so an embedding application can build a picker without
  hardcoding a table of its own. Three calls, separate because they cost very
  different amounts:
  - `provider_catalog()` / `provider_info(name)` return `ProviderInfo` — name,
    aliases, wire API, credential kind, default model, configuring env vars, and
    two flags a UI would otherwise hardcode: `officially_supported` (`False` for the
    subscription backends) and `preserves_reasoning` (`False` for Chat Completions,
    which cannot carry a reasoning model's chain of thought across a tool call).
    Pure data — no credential is read, no provider module is imported, nothing
    touches the network. The metadata is declared beside `register()` rather than on
    the provider classes precisely so asking what the options are cannot drag in a
    vendor SDK; a test pins each declaration against the class it describes.
  - `await provider_status()` returns `ProviderStatus` — whether each backend is
    usable right now, with `detail` carrying the same actionable message the provider
    would have raised. Async and opt-in because it reads credential stores, including
    a Keychain subprocess. Never raises for an absent credential.
  - `provider.list_models()` is now implemented by **every** built-in provider, over
    three mechanisms: `GET /models` for the Responses and Chat Completions backends,
    and the SDK's `models.list` for the Anthropic pair. Both `GET /models` response
    shapes are accepted (`data[].id` and the Codex backend's `models[].slug`).
- `register()` accepts an optional `info=ProviderInfo(...)`. Third-party providers
  that omit it stay fully resolvable and are simply absent from the catalog.
- `CODEX_CLIENT_VERSION` — the Codex `GET /models` endpoint requires a
  `client_version` query parameter and **gates its answer on it**; a lower version is
  served a different, sometimes larger, set. Pinned to a constant for reproducibility
  and overridable with `client_version=`.
  - Model reasoning round-trips as a verbatim `RawBlock`, not a `ThinkingBlock`.
    The Responses API returns reasoning as a sibling output item carrying an
    opaque `encrypted_content` blob that must be resent byte-exact and in
    position, and it has nowhere to live in `ThinkingBlock` — which also produces
    empty-thinking blocks at low effort, where the summary is often absent.
    `ThinkingDelta` events still stream from the summary channel.
  - `store` is always `false`: logpose owns the history, and the subscription
    backend requires it.
  - `max_tokens` is omitted on the subscription endpoint, which answers
    `400 Unsupported parameter: max_output_tokens`; it is sent and honoured on
    `api.openai.com`. Both verified against the live backends.
  - Whether a turn carries a reasoning item is a backend property, not a bug:
    `api.openai.com` emits one on a tool-calling turn, the Codex subscription
    backend frequently does not. Both observed live.
  - The turn is assembled from `response.output_item.done` events, with a
    *populated* terminal `response.output` preferred when one arrives. The
    subscription backend sends `"output": []` on `response.completed` and delivers
    everything incrementally, so an empty array is never treated as authoritative.
  - A `Conversation` is tied to the backend that produced it. Anthropic thinking
    blocks handed to Codex are dropped rather than raising, since a Conversation
    crossing backends should lose reasoning, not crash.
- **Codex / ChatGPT credential resolution** (`logpose.auth.codex`) — reads
  `~/.codex/auth.json` (or `$CODEX_HOME`) **read-only**, takes expiry from the
  access token's own `exp` claim since the file records none, and refreshes
  single-flight in memory. Subscription-first precedence, with the store
  deliberately outranking `$OPENAI_API_KEY` — the reverse of the Anthropic
  ordering, because `codex login` writes only to `auth.json` and there is no
  `CODEX_OAUTH_TOKEN` to sit above it. A rotated refresh token is **not** written
  back: that would race the Codex CLI for the file.
- **`compat_codex_cli`** — dresses subscription requests as the Codex CLI's (the
  identity line ahead of `instructions`, plus `originator: codex_cli_rs`).
  Defaults to on for OAuth and off for an API key, mirroring `compat_claude_code`.
- `examples/codex.py` — a runnable Codex example.

### Changed

- **BREAKING — one provider per credential, not per vendor.** `anthropic` accepts
  only an API key and `codex` only a subscription token; the subscription and
  API-key paths for each vendor are now separate registered providers
  (`anthropic`/`claude-code`, `openai`/`codex`) over a shared private base
  (`providers/_anthropic_base.py`, `providers/_responses.py`). `known_providers()`
  is now `["anthropic", "claude-code", "codex", "docker", "docker-models",
  "openai", "openai-compat"]`.

  `Agent("anthropic", auth_token=...)` and `Agent("anthropic",
  compat_claude_code=...)` become `Agent("claude-code", ...)`.

  The motivation was that both all-in-one providers branched on the credential
  kind — endpoint, headers, identity line, and on the Responses side whether
  `max_output_tokens` was even a legal field — and that kind is not known until
  the first request. `AnthropicProvider` had to rebuild its SDK client if the kind
  changed between turns, and the Responses provider had to *predict* its own
  `base_url` before resolving anything. Both mechanisms are deleted rather than
  moved. It also removes a precedence policy that needed three paragraphs of
  justification: a credential of the wrong kind can no longer shadow a usable one,
  because each provider only ever looks at sources of the kind it accepts.
- `resolve_credential` and `CredentialProvider.resolve` gained `require_kind`,
  which narrows the precedence chain to sources yielding that kind. Filtering
  rather than resolving-then-rejecting is the point: otherwise a subscription token
  in the store would hide a perfectly good `$OPENAI_API_KEY` and the API-key
  provider would refuse a credential it was standing next to.
- A provider handed a credential of the wrong kind now raises `AuthError` naming
  the sibling provider to use instead.
- The backend-neutral half of credential handling moved to `logpose.auth._common`
  (`Credential`, `CredentialProvider`, expiry normalisation, redaction), leaving
  `logpose.auth.claude_code` as only the Anthropic-specific half. Public names are
  unchanged and re-exported where they were; `Credential` gained an optional
  `account_id` for backends that name a tenant separately from the token.
- Tolerant tool-argument parsing moved to `logpose.providers._toolargs`, so the
  Chat Completions and Responses backends share one `UNPARSED_ARGUMENTS_KEY`.
  Consumer code special-casing that key now covers both.

- **`Agent(on_tool_call=...)`** — a gate consulted before each tool runs, so an
  embedder can put a permission prompt, a policy check, or a dry run in front of
  execution. It sees every requested call one at a time in wire order and
  entirely before the concurrent execution phase, so a gate that asks a human is
  never asked several things at once, and a call it blocks cannot have already
  run. Returning `None` allows the call; a `str` or `ToolGateResult` blocks it
  and is handed to the model in its place — use
  `ToolGateResult(..., is_error=False)` when the block is a redirection rather
  than a failure. Sync and async gates are both supported; a gate that raises
  ends the run rather than failing open.
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
- **`discover_tools()`** — collects every `@tool` in a module or package, so a
  growing tool set no longer has to be hand-listed at the call site:
  `Agent("docker", tools=discover_tools("myapp.tools"))`. `Agent` is unchanged and
  the result is a plain list, so discovery composes with hand-written tools.
  A tool re-exported into a package `__init__` is returned once (deduplicated by
  object identity); two *different* tools sharing a name raise `LogposeError`
  naming both defining modules. Results are in traversal order rather than sorted,
  so appending a tool leaves the preceding request bytes untouched and does not
  disturb automatic prefix caching. Submodules named with a leading underscore are
  skipped without being imported, and a submodule that fails to import raises
  rather than being silently dropped.
- Project documentation: `CONTRIBUTING.md` (setup, the provider seam, test and
  PR conventions) and `SECURITY.md` (private reporting, and what does and does
  not count as a vulnerability in a library that handles provider credentials
  and runs caller-supplied tools).
- GitHub Actions CI: tests on Python 3.10–3.13, plus ruff, mypy, and a build job
  that installs the wheel standalone and asserts `import logpose` still pulls in
  no provider SDK.

### Fixed

- **Every subscription (OAuth) request failed with `429 rate_limit_error`.** Not
  a quota problem: Anthropic began rejecting subscription requests whose system
  prompt does not open with the Claude Code identity line, and the rejection is
  dressed as a rate limit — an empty `"message":"Error"` body and none of the
  `anthropic-ratelimit-*` headers a real limit carries. The same token, in the
  same second, is answered `200` with the line and `429` without it.
  `AnthropicProvider(compat_claude_code=...)` therefore now defaults to `None`,
  meaning "decide per credential": the identity line is prepended for OAuth and
  omitted for an API key. Pass `True` or `False` to force either way; `True`
  keeps its old meaning, so existing callers are unaffected. Behavior on the
  API-key path is unchanged.
- `redact` moved to `logpose.providers._redact` so a backend can reuse it
  without importing a sibling provider's SDK. It is still exported from
  `logpose.providers.anthropic` for compatibility.
- Corrected the repository URL in `pyproject.toml` and the changelog link
  refs, which pointed at a repository that does not exist.
- README install instructions pointed at `pip install logpose`, but that name
  belongs to an unrelated logging library on PyPI and would have installed the
  wrong package. Now installs from git; a distribution name is still to be
  chosen before publishing.
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
