# Runtime robustness and release-readiness plan

Status: **Approved for implementation**  
Last updated: 2026-08-07  
Target: logpose `0.x` alpha releases leading to a stable `1.x` public API

This document records the agreed direction for strengthening logpose before a
broader public announcement. It is the implementation contract for the work
described below, not a list of ideas that implementation PRs may silently
reinterpret.

## Change control

- Changes to a locked decision, default, privacy boundary, or acceptance gate in
  this document require a dedicated documentation change in the relevant PR.
- The PR description must explain why the plan changed and identify the affected
  decision.
- Implementation details may evolve when tests reveal a better internal design,
  provided the documented behavior remains unchanged.
- Deferred items remain out of scope until a later decision explicitly moves
  them into a milestone.
- Completed PRs should be linked in the implementation tracker at the end of this
  document.

## Product objective

logpose targets developers building agentic applications across personal local
projects and organizational production services. Before the broader alpha
announcement, the project should provide:

- stable and predictable runtime failure behavior;
- easy integration with provider, logging, metrics, and tracing infrastructure;
- confidentiality-preserving defaults;
- reliable installation from GitHub artifacts;
- clear operational and contribution documentation.

Breaking API changes are permitted during `0.x` when documented with migration
guidance. `1.x` will establish the stable public compatibility contract.

## Locked decisions

| Area | Decision |
| --- | --- |
| Retry ownership | logpose owns provider retries; provider SDK retries are disabled. |
| Retry boundary | Transparent retries are allowed only before the first streamed delta. |
| Retry default | Three total attempts with bounded exponential backoff and jitter. |
| Authentication | Authentication failures are never retried reactively. Subscription users are instructed to log in again. |
| Tool capacity | Capacity is enforced per `Agent`; excess calls wait for a slot. |
| Default tool concurrency | Eight calls per `Agent`. |
| Default tool timeout | 300 seconds, beginning after the call acquires capacity. |
| Timed-out synchronous tools | The result is discarded; the underlying Python thread may continue. Process isolation is deferred. |
| Tool errors | Safe metadata is returned by default. Raw messages and tracebacks require explicit opt-in. |
| Tool-result size | No default limit yet; measurements will inform a later decision. |
| Anthropic dependency | The SDK moves to one optional extra named `anthropic`, shared by `anthropic` and `claude-code`. |
| Experimental auth | `claude-code` and `codex` remain visible and are marked experimental. |
| Runtime logging | logpose automatically publishes metadata-only records through `logpose.runtime`. |
| Logging ownership | logpose installs no handler, formatter, destination, or global logging configuration. |
| Observers | An `Agent` may have multiple synchronous observer callbacks; failures are isolated. |
| Observer data | Observers receive the same metadata-only runtime event used for logging. |
| Correlation | Runtime context is available to application and tool logging through `contextvars`. |
| Releases | Alpha releases are managed through GitHub. The detailed release workflow will be finalized during its implementation phase. |
| Dependency updates | Dependabot opens weekly grouped update PRs; security alerts and security updates remain enabled separately. |

## Milestone 1: provider resilience

### PR 1 — Normalize provider failures

Extend `ProviderError` with structured, optional metadata while preserving the
existing `status_code` and `retryable` API:

- provider error code;
- provider request ID;
- `Retry-After` duration;
- whether output was already emitted;
- number of attempts made.

Standardize equivalent failure classification across every built-in provider.
Disable the Anthropic SDK's internal retries so the library has one retry owner.

Retryable categories:

- connection establishment failures;
- connection resets before output;
- transport timeouts;
- HTTP 408, 409, and 429;
- HTTP 5xx;
- truncated streams before output;
- provider-declared temporary error events.

Non-retryable categories:

- authentication failures;
- invalid requests and unsupported parameters;
- unavailable or unauthorized models;
- provider protocol violations;
- tool and tool-gate failures;
- iteration limits;
- refusals and normal terminal stop reasons.

Acceptance criteria:

- Equivalent failures have equivalent metadata across providers.
- Request IDs and retry delays are extracted when available.
- Provider bodies, headers, and exception chains cannot leak credentials.
- Existing users of `ProviderError.status_code` and `.retryable` remain supported.
- A conformance test matrix covers every built-in provider.

### PR 2 — Centralize retry orchestration

Add an immutable public `RetryPolicy` and an `Agent(retry_policy=...)` option.
The default policy is:

```python
RetryPolicy(
    max_attempts=3,
    initial_delay=0.5,
    backoff_multiplier=2.0,
    max_delay=8.0,
    jitter=0.2,
    max_retry_after=60.0,
)
```

`RetryPolicy(max_attempts=1)` disables retries.

Rules:

- Retry only failures classified as retryable by the provider boundary.
- Respect `Retry-After` up to `max_retry_after`.
- Stop retrying as soon as text or reasoning has been emitted to the caller.
- Mark a post-delta failure as partial and raise it without replaying the turn.
- Never append incomplete assistant history.
- Never count usage from an incomplete attempt.
- Cancellation during a request or retry delay must end immediately.
- Backoff tests use injected clocks, sleepers, and randomness rather than real
  delays.

Acceptance criteria:

- A temporary pre-delta failure followed by success produces one clean stream.
- A post-delta failure never duplicates visible output.
- Authentication and invalid-request failures make one attempt.
- Attempt counts, retry decisions, and delays are observable.
- The provider SDKs perform no hidden additional attempts.

### PR 3 — Standardize deadlines

Use distinct transport and operation deadlines:

| Operation | Default |
| --- | ---: |
| TCP connection | 10 seconds |
| Connection-pool wait | 10 seconds |
| Request write | 60 seconds |
| Cloud stream idle/read | 600 seconds |
| Local stream idle/read | 600 seconds |
| Complete cloud provider turn | 15 minutes |
| Complete local provider turn | 30 minutes |
| Authentication refresh | 10 seconds |
| Complete agent run | Unlimited |

All limits are configurable, and `None` disables an optional operation deadline.
The complete run stays unlimited because a multi-turn run may intentionally
contain several long provider turns and tools.

Acceptance criteria:

- Every built-in provider exposes equivalent timeout configuration.
- A pre-delta turn timeout may enter retry policy.
- A post-delta turn timeout is partial and is never retried automatically.
- Cancellation closes provider streams and owned HTTP resources.
- Documentation distinguishes an idle read timeout from a complete-turn deadline.

## Milestone 2: tool resilience

### PR 4 — Bound tool concurrency and execution time

Add these `Agent` options:

```python
Agent(
    ...,
    max_concurrent_tools=8,
    tool_timeout=300.0,
)
```

Implementation contract:

- One semaphore is shared by all concurrent runs using an `Agent`.
- Calls above capacity wait for a slot rather than being dropped.
- The execution timeout begins after capacity is acquired.
- Queue and execution duration are measured separately.
- Results are returned in the model's original call order.
- Run cancellation cancels executing async tools and calls waiting for capacity.
- A timeout becomes an error tool result so the model can adapt.
- A timed-out synchronous result is discarded and never sent to the model.
- The underlying worker thread may continue because Python cannot safely kill it.

No separate queue timeout is included initially. It may be added later if
operational evidence shows that indefinite capacity waits need a distinct policy.

Acceptance criteria:

- Concurrent runs cannot exceed agent capacity.
- Calls waiting for capacity eventually run as slots become available.
- One timed-out tool does not cancel unrelated tools.
- External cancellation clears executing and queued work.
- Result ordering remains stable.
- Tests explicitly demonstrate the synchronous-thread limitation.

### PR 5 — Make tool errors safe by default

Add:

```python
ToolErrorMode = Literal["safe", "message", "traceback"]

Agent(
    ...,
    tool_error_mode="safe",
    tool_error_formatter=None,
)
```

Behavior:

| Mode | Model-facing result |
| --- | --- |
| `safe` | Tool name, exception type, and generated error ID. |
| `message` | Exception type and raw exception message. |
| `traceback` | Complete formatted traceback. |

Additional rules:

- `safe` is the production default.
- Validation errors include field locations and messages but omit rejected values.
- A custom formatter takes precedence over the selected mode.
- A formatter failure falls back to the safe representation.
- Opting into raw model-facing errors does not opt logging or observers into raw
  data.
- Documentation warns that arbitrary application exceptions cannot be reliably
  scrubbed of every possible secret.

Acceptance criteria:

- The default result contains no raw exception message or traceback.
- Every failure has a stable error ID for correlation.
- Opt-in modes expose exactly the documented amount of detail.
- Seeded secrets do not appear in default results, logs, or observer events.
- Formatter failures cannot fail the agent run.

### Deferred — tool-result limits

No default size or token limit will be added yet. Runtime observations may record
the result's byte size, but never its content. A later decision will consider:

- real result-size distributions;
- context-window and memory impact;
- JSON validity under truncation;
- rejection, truncation, storage references, and summarization strategies;
- optional application-provided token estimators.

## Milestone 3: authentication and packaging

### PR 6 — Harden authentication failures

- Keep proactive refresh for subscription credentials nearing expiry.
- Do not force-refresh or replay an unexpected authentication failure.
- Classify HTTP 401 as `AuthError` with an actionable recovery instruction.
- Treat HTTP 403 according to provider semantics; default it to a non-retryable
  `ProviderError` because it may represent entitlement rather than invalid login.
- Tell API-key users which key configuration to check.
- Tell experimental subscription users which CLI login command to run again.
- Preserve strictly read-only credential stores.
- Add an explicit provider stability field and mark `claude-code` and `codex`
  experimental. Preserve the existing metadata field as a compatibility alias
  during `0.x` if needed.

Acceptance criteria:

- Authentication failures never enter retry policy.
- Recovery guidance matches the selected provider.
- Missing, malformed, unreadable, incomplete, expired, and concurrent credential
  cases are covered.
- Credentials never reach errors, chained causes, tracebacks, logs, or observers.

### PR 7 — Make Anthropic optional

Move the SDK into one extra shared by the API-key and Claude Code providers:

```toml
[project.optional-dependencies]
anthropic = ["anthropic>=0.49"]
```

- The base installation retains HTTPX and Pydantic.
- `import logpose` and every non-Anthropic provider work without the SDK.
- Resolving `anthropic` or `claude-code` without the extra raises an actionable
  installation error.
- CI tests base installation and the Anthropic extra separately.
- CI installs and imports the exact wheel and sdist in clean environments.
- Meaningful minimum dependency versions are established through tests.
- Current-dependency and minimum-dependency jobs remain separate.

Acceptance criteria:

- Base installation supports OpenAI, Codex, Docker, and compatible providers.
- Anthropic resolution names the exact extra to install when missing.
- Wheel and sdist contain the intended typed-package metadata, license, and
  documentation.
- Tested lower bounds are declared rather than guessed.

## Milestone 4: observability

### Shared runtime event

Logging and observers use one immutable, metadata-only `RuntimeEvent`. The event
has a schema version and may contain:

- event name;
- run, turn, attempt, request, and error IDs;
- provider and model;
- monotonic durations;
- retry decision and delay;
- status code;
- tool name, queue duration, execution duration, and timeout status;
- result byte size, but not result content;
- token usage;
- exception type, but not exception message.

It never contains:

- prompts or model output;
- reasoning;
- tool arguments or results;
- exception messages or tracebacks;
- request or response headers;
- credentials, credential paths, or arbitrary provider payloads.

Initial event vocabulary:

```text
run.started
run.completed
run.cancelled
provider.attempt.started
provider.attempt.completed
provider.attempt.failed
provider.retry.scheduled
tool.queued
tool.started
tool.completed
tool.failed
tool.timed_out
tool.cancelled
```

### PR 8 — Add runtime events and context

- Add `RuntimeEvent`, `RuntimeContext`, and the observer callable type.
- Generate unique run and error IDs.
- Use `contextvars` to carry current run, turn, attempt, and tool identity.
- Expose `current_runtime_context()` for application logging correlation.
- Propagate context into async tools and `asyncio.to_thread()` tools.
- Restore previous context after completion, failure, and cancellation.

Acceptance criteria:

- Concurrent runs never share context.
- Tool code can read the correct current context.
- Context is restored after every exit path.
- Public event fields are content-free by construction.

### PR 9 — Publish through standard logging

Automatically publish runtime records using:

```python
logging.getLogger("logpose.runtime")
```

logpose will not call `basicConfig()`, add handlers, install formatters, choose a
destination, or write files. Records propagate through the application's existing
logging configuration.

Structured fields use a `logpose_` prefix and primitive values so they do not
collide with standard `LogRecord` fields or trigger unsafe `repr()` calls.

Recommended levels:

- `DEBUG`: ordinary lifecycle starts, completions, queueing, and cancellation;
- `INFO`: completed high-level runs and aggregate usage;
- `WARNING`: internally handled retries, tool failures, and tool timeouts;
- no duplicate `ERROR` record for an exception that is raised to the application.

Acceptance criteria:

- Existing Python logging configuration receives `logpose.runtime` records
  without logpose setup.
- No handler or formatter is installed by the library.
- A broken custom handler cannot fail an agent run.
- Prompts, results, exception messages, and seeded secrets never appear.
- Logging-disabled overhead is measured and negligible.

### PR 10 — Add per-Agent observers

Add:

```python
agent = Agent(
    "anthropic",
    observers=[metrics_observer, tracing_observer, audit_observer],
)
```

Observers are synchronous callables receiving the same `RuntimeEvent` used to
construct the standard log record. They may forward metadata to an application's
existing metrics, tracing, audit, or event infrastructure.

Rules:

- Observers run in registration order.
- Each observer failure is isolated; later observers still run.
- Observer failures never affect the agent run.
- `KeyboardInterrupt`, `SystemExit`, and cancellation are not swallowed.
- Networked observers should enqueue records through application-owned
  infrastructure rather than block the agent loop.
- A callback that also logs should normally use its own logger to avoid duplicate
  `logpose.runtime` records.

Acceptance criteria:

- Logging and observers receive equivalent metadata for the same operation.
- Broken observers cannot affect results or other observers.
- Queue-based observers receive events in lifecycle order.
- An observer cannot access confidential content through `RuntimeEvent`.

## Milestone 5: community and releases

### PR 11 — Add community policy and automation

Add:

```text
.github/
├── ISSUE_TEMPLATE/
│   ├── bug.yml
│   ├── feature.yml
│   └── config.yml
├── pull_request_template.md
└── dependabot.yml
CODE_OF_CONDUCT.md
SUPPORT.md
RELEASING.md
```

- Use the Contributor Covenant.
- Bug reports request logpose version, provider, Python version, redacted
  traceback, and a minimal reproduction.
- Feature requests begin with the user problem rather than a proposed API.
- The PR template covers tests, docs, changelog, security, and breaking changes.
- The support policy distinguishes the stable core from experimental providers.
- Dependabot opens weekly grouped dependency-update PRs.
- Security alerts and automatic security updates are configured separately from
  the weekly version-update schedule.
- Breaking alpha changes require changelog and migration notes.

### PR 12 — Implement the GitHub alpha release pipeline

The detailed workflow will be decided during implementation. Its required
boundary is:

- CI passes before release work begins.
- Version and tag agree.
- Wheel and sdist are built once.
- The exact artifacts are installed and tested in clean environments.
- Checksums are generated.
- Artifacts and generated notes are attached to a draft GitHub Release.
- A maintainer reviews and publishes the draft manually.
- Artifacts are never rebuilt between verification and publication.

PyPI publishing is out of scope for this plan.

## Execution protocol

Each implementation PR follows the same sequence:

1. Branch from current `master`.
2. Update or confirm the public contract first.
3. Add failing tests for the intended behavior.
4. Implement the smallest coherent change.
5. Run the full suite, Ruff, mypy, artifact checks where relevant, and focused
   fault-injection tests.
6. Update the README, changelog, and migration guidance.
7. Open a focused PR containing its design decisions and verification evidence.
8. Merge before beginning work that depends on it.

Large cross-milestone PRs should be avoided. Retry, tool scheduling,
authentication, packaging, and observability all touch important runtime
boundaries and need independent review.

## Announcement gate

Before the broader alpha announcement:

- PRs 1–10 are merged.
- Provider retry and timeout classification has fault-injection coverage.
- Tool concurrency and timeouts are verified under concurrent runs.
- No credential or application content appears in default errors, logs, or
  observer records.
- Base and Anthropic-extra installations work from built artifacts.
- Standard logging, observer, and runtime-context integration are documented.
- Community policies and issue forms are available.
- At least one alpha release has been produced through the real release process.
- A new user can install and complete the quickstart from the README alone.
- `0.x` compatibility expectations and experimental-provider status are explicit.

## Implementation tracker

| PR | Work item | Status | Link |
| --- | --- | --- | --- |
| 1 | Normalize provider failures | Completed | [#6](https://github.com/xdadwal/logpose/pull/6) |
| 2 | Central retry policy | Completed | [#7](https://github.com/xdadwal/logpose/pull/7) |
| 3 | Provider deadlines | Completed | [#8](https://github.com/xdadwal/logpose/pull/8) |
| 4 | Tool capacity and timeout | Completed | [#9](https://github.com/xdadwal/logpose/pull/9) |
| 5 | Safe tool errors | Completed | [#10](https://github.com/xdadwal/logpose/pull/10) |
| 6 | Authentication hardening | Completed | [#11](https://github.com/xdadwal/logpose/pull/11) |
| 7 | Optional Anthropic dependency | Completed | [#12](https://github.com/xdadwal/logpose/pull/12) |
| 8 | Runtime events and context | Not started | — |
| 9 | Standard logging | Not started | — |
| 10 | Per-Agent observers | Not started | — |
| 11 | Community readiness | Not started | — |
| 12 | GitHub release pipeline | Not started | — |

## Decision history

| Date | Decision |
| --- | --- |
| 2026-08-07 | Approved the five-milestone robustness and announcement-readiness program. |
| 2026-08-07 | Set provider retries to three attempts and prohibited transparent replay after the first delta. |
| 2026-08-07 | Set per-Agent tool capacity to eight and the default tool timeout to 300 seconds. |
| 2026-08-07 | Deferred process isolation and tool-result size limits. |
| 2026-08-07 | Selected safe tool errors by default with opt-in raw messages and tracebacks. |
| 2026-08-07 | Kept experimental subscription providers visible without reactive authentication retries. |
| 2026-08-07 | Selected one optional `anthropic` extra for both Anthropic providers. |
| 2026-08-07 | Selected automatic standard logging through `logpose.runtime`, per-Agent observers, and runtime context propagation. |
| 2026-08-07 | Selected weekly grouped Dependabot updates and GitHub-managed alpha releases. |
