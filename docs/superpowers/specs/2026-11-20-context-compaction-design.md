# Context compaction (built-in default policy)

Status: design for approval, 2026-11-20

## Intent and the why

Today every model request carries the full history, so cost and latency grow
every turn and a long chat eventually fails outright when it exceeds the model's
context window. logpose has all the information to manage this. It ships a
built-in compaction policy, **on by default**, that apps can tune or replace.
Stored history (the session store) is never changed; only what is sent to the
model is shaped.

## Goals

- **Tool output pointers, always:** tool outputs from *earlier* user turns
  that are longer than 1,000 characters are sent as a short pointer. Outputs
  from the current turn, and short outputs, are sent in full.
- **`recall_tool_output` tool,** added automatically while the pointer policy
  is active, so the model can fetch a pointed-to output when a follow-up needs
  it.
- **Summarization fallback** at 85% of the input budget (owner chose to keep
  messages, not trim them; "A: summarize" was Claude's recommendation, taken
  because the question went unanswered, so confirm in review). Older turns are
  replaced in the request by one model-written summary. The first user turn and
  recent turns stay verbatim.
- **Configurable and replaceable:** every number is an `Agent` setting; a custom
  `context_policy=` replaces the built-in policy; `context_policy=None` sends
  everything (today's behaviour).

## Non-goals

- Changing stored history, the `Conversation` returned to callers, or
  `RunResult.messages`.
- Exact tokenizer-based counting (no tokenizer dependency).
- Trimming or dropping messages (owner's decision).
- Semantic retrieval or memory beyond the summary.

## Anti-goals

- A request that is invalid for the server: a tool call without its result, or
  an orphaned result.
- Compaction that rewrites the start of the request every turn. That defeats
  provider prompt caching and causes thrashing.
- Losing the user's original request (pin the first user turn).

## Decided

### Settings (on `Agent`)

| Setting | Default | Meaning |
|---|---|---|
| `context_window` | `128_000` | model's total window in tokens (the app states it; logpose can't discover it reliably) |
| `context_budget` | `32_000` | web-app cost cap on input tokens per request; `None` means no cap beyond the window |
| `compaction_threshold` | `0.85` | summarize when estimated input ≥ this fraction of the input budget |
| `compaction_target` | `0.60` | summarize down to this fraction (hysteresis) |
| `pointer_min_chars` | `1_000` | tool outputs at or below this stay inline |

- `input_budget = min(context_window − max_tokens, context_budget)`, where
  `max_tokens` is the resolved per-turn output ceiling. If `input_budget ≤ 0`,
  raise `LogposeError` at construction.
- `context_policy` default is a sentinel meaning "built-in policy". An explicit
  function replaces it; `None` disables all compaction (full history, as today).

### Token estimation

- Estimate = characters of the request's messages, system and tool schemas ÷ 4.
- Calibration: after each provider turn, logpose knows the real `input_tokens`
  for the request it sent. It keeps a per-conversation correction ratio
  (`reported ÷ estimated`, smoothed) and applies it to later estimates.

### Step 1: tool output pointers (every request)

- **Current turn:** every message appended during the current run (see
  resolution 7).
- In every earlier turn, each `ToolResultBlock` whose content is longer than
  `pointer_min_chars` is replaced, in the request only, with:
  `[tool output stored: <tool_use_id>, <N> chars. Call recall_tool_output("<tool_use_id>") to read it.]`
- The block keeps its `tool_use_id` and `is_error`, so pairing stays valid.
- Deterministic: once a turn is no longer current its outputs are pointers
  permanently, so the request's start is stable from turn to turn.

### `recall_tool_output` tool

- Added to the agent's tools automatically when the built-in policy is active
  and the agent has at least one tool (see resolution 5). Name collision with an app tool raises `LogposeError` at construction.
- `recall_tool_output(call_id: str, offset: int = 0) -> str` returns up to 8,000
  characters of the full output starting at `offset`, plus a continuation hint
  if more remains. Unknown IDs give a clear error string.
- It reads from the run's full history (the conversation or session), so it
  needs no database. It works with or without sessions.

### Step 2: summarization fallback (85% of the input budget)

- After step 1, if the estimate is still ≥ `compaction_threshold × input_budget`:
  1. Choose the span to summarize: everything after the first user turn and
     before the most recent turns. Grow the span from the oldest end until the
     estimate falls to `compaction_target × input_budget`. Cut points fall only
     on turn boundaries, so tool pairs are never split.
  2. Make one extra model call (same provider and model, no tools, `max_tokens`
     2,048) with a fixed summarization instruction and the span's text
     (pointers, not full outputs). The output is the summary.
  3. The request becomes: system, the first user turn, one user-role message
     `"[Summary of earlier conversation]\n<summary>"`, then the remaining recent
     turns.
- **Stability:** the summary and its cut point are cached for the conversation
  and reused on later turns until the threshold is reached again; then the
  summary is extended (re-summarized up to a new cut point). With sessions,
  the summary is persisted (new table `logpose_summaries(session_id, upto_seq,
  text, created_at)`, schema version 2) so it survives across requests. Without
  sessions it lives in memory on the `Conversation`.
- If the request is still over the full window after summarizing, raise
  `ContextOverflowError(LogposeError)` with the estimate and budget.

### Reactive fallback

- If the provider rejects a request for exceeding the context length (HTTP 400
  whose error mentions `context_length`, `maximum context length`, or
  `context window`), force step 2 regardless of the estimate and retry the turn
  once. A second rejection propagates as the original `ProviderError`.

### Observability

- A runtime event `context.compacted` (metadata only) with
  `pointers_applied`, `summarized` (bool), `estimated_tokens_before`,
  `estimated_tokens_after`, and `input_budget`.
- `PreparedContext.diagnostics` carries the same values.

## Builder-check resolutions (Claude's calls, 2026-11-20)

1. **The built-in policy is a private path in the loop, not a `ContextPolicy`.**
   The public `ContextInput`/`PreparedContext` contract is unchanged. Passing a
   custom `context_policy=` replaces the built-in path entirely: custom policies
   get no pointers, recall tool, summary, reactive retry, or `context.compacted`
   event. The app owns everything then, as today.
2. **Per-conversation state** (calibration ratio, cached summary, cut point)
   lives in a private slot on `Conversation` (`_compaction`), not a public field.
   The cut point is stored with a fingerprint (message count up to the cut plus
   a hash of those messages). If the caller edited history so the fingerprint no
   longer matches, the cache is discarded and recomputed. Single-shot runs (no
   `Conversation`) keep this state for the run only.
3. **Order:** the session store ships first (schema version 1), and compaction
   follows in the same plan, adding schema version 2 (`logpose_summaries`).
   `upto_seq` is the stored `seq` of the last summarized message. The session
   handle keeps the `seq` of each loaded message, so `seq` maps to an in-memory
   position. Without sessions, nothing is persisted.
4. **Summary call accounting:**
   - its usage is added to `RunResult.usage` and the session `RunRecord`;
   - it is bounded by `run_timeout` and uses the retry policy;
   - it does not count toward `max_iterations`;
   - it emits `context.compacted`, not `provider.attempt.*`.

   If the summary call fails, and the uncompacted request still fits the full
   window, logpose sends it uncompacted (over budget, but valid) and the event
   records `summarized=false`. Otherwise it raises
   `ContextOverflowError` chained to the cause.
5. **The recall tool is always included** whenever the built-in policy is active
   and the agent has at least one tool. It is present from the first turn, so
   the tools part of the request never changes mid-conversation and doesn't
   break caching. Agents without tools never get it (and never have tool
   outputs).
6. **The `input_budget ≤ 0` check uses the resolved `max_tokens`:** the agent's
   value, else the provider's, else the default, at construction.
7. **"Current turn" means every message appended during the current run** (from
   the run's start index), not "since the last user text message". Loop-inserted
   user messages, such as structured-output corrections, don't start a new turn,
   and a run's own tool outputs stay full for the whole run. Step 1's rule reads:
   tool outputs from *earlier runs* longer than `pointer_min_chars` become
   pointers.

## Approaches

- **Always-on pointers plus summary fallback (chosen with the owner).** Requests
  stay small every turn and cache-stable, and nothing is dropped.
- Threshold-only trimming (keep last N messages or turns): rejected by the owner
  (no trimming); also caused thrashing and lost the user's intent.
- Fixed "last 5 tool outputs": replaced by the current-turn rule. A count is
  arbitrary when one turn can make many calls.

## Left to the builder

- The exact summarization instruction text, which must state: keep facts,
  decisions, IDs, and unresolved questions; no new information.
- The smoothing constant for the calibration ratio.
- Whether the pointer pass and the summary pass live in one module
  (`logpose/compaction.py` suggested).
