# Per-run `extra`

Status: approved 2026-11-23 (short design; owner approved the proposed API in
conversation).

## Intent and the why

A web app shares one long-lived `Agent` across all requests (to keep the shared
httpx pool) but needs to send request-specific body fields, for example OpenAI's
`safety_identifier` (a hashed per-visitor ID) and later `prompt_cache_key`.
Today `extra` is fixed at `Agent(...)` construction. logpose stays
provider-neutral: no `safety_identifier`-specific code.

## Goals / non-goals

- Goal: an `extra=` argument on `run`, `stream`, `run_sync`, `stream_sync`,
  `SyncAgent.run` and `SyncAgent.stream`, merged over the agent's `extra` for
  that run only.
- Non-goal: changing `Agent(extra=)` semantics (backward compatible).
- Anti-goal: recording per-run values (identifiers) anywhere.

## Approaches

- **Per-run `extra=` merged over the agent's (chosen).** Smallest surface; one
  shared Agent keeps its connection pool.
- A per-request `Agent` copy (`agent.with_extra(...)`): rejected — easy to leak
  pools or diverge settings.
- A request hook (`extra_factory(context)`): rejected — more API for no gain.

## Decided

- **Merge:** `{**agent.extra, **extra}` for every model call of the run: each
  turn, structured-output correction turns, and the retry after a context-length
  rejection.
- **Validation:** `extra` must be a `Mapping` with `str` keys; it may not
  contain keys logpose builds itself: `model`, `messages`, `tools`, `stream`,
  `stream_options` → `LogposeError` when `run`/`stream` is called.
  `Agent(extra=)` keeps today's behaviour (not validated this way).
- **`response_format`:** if the merged `extra` has `response_format` and an
  `output_type` is in effect (agent or run), raise `LogposeError` at call time,
  matching the existing `Agent(extra=)` rule. Without `output_type`, a per-run
  `response_format` passes through.
- **Output ceiling:** a per-run `max_tokens` / `max_completion_tokens` replaces
  logpose's ceiling field (the provider rule already applied to request
  `extra`). Compaction still budgets with the agent's resolved `max_tokens`;
  documented.
- **Summary call:** gets the merged `extra` minus `_TURN_ONLY_EXTRA`
  (`response_format`, `tool_choice`, `parallel_tool_calls`).
- **Guardrails:** never receive `extra` (separate endpoint and settings).
- **Tracing:** the `invoke_agent` span records `logpose.request.extra_keys`, the
  sorted key names of the merged `extra` (agent + run), never values, even with
  content capture on. Omitted when empty.
- **Sessions:** per-run `extra` is not persisted.

## Left to the builder

Internal plumbing (how the merged dict reaches `_request`/`_summarize`),
test layout.
