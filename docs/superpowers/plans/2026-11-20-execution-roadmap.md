# Execution roadmap: web-chat foundations (Plans 0–4)

Status: proposed 2026-11-20. This is the order and dependency map for running
the plans below with `superpowers:subagent-driven-development` (one fresh
implementer and one fresh reviewer per task, then a whole-branch review per
plan).

## Plans

| # | Plan | Spec | Tasks | Branch / PR |
|---|---|---|---|---|
| 0 | Defaults (`max_tokens` 16384, 8k tool output, `max_completion_tokens`) | `specs/2026-11-19-max-completion-tokens-design.md` | already built | `chore/defaults-tool-output-8k` → PR, merge first |
| 1 | Session store | `specs/2026-11-20-session-store-design.md` | 5 | `feat/session-store` → PR |
| 2 | Context compaction | `specs/2026-11-20-context-compaction-design.md` | 4 | `feat/context-compaction` (from master after 1 merges) → PR |
| 3 | Input guardrails | `specs/2026-11-20-input-guardrails-design.md` | 4 | `feat/input-guardrails` (after 2 merges) → PR |
| 4 | OpenTelemetry tracing | `specs/2026-11-20-otel-tracing-design.md` | 4 | `feat/otel-tracing` (after 3 merges) → PR |

## Dependency graph

```
P0 defaults ──► P1 sessions ──► P2 compaction ──► P3 guardrails ──► P4 tracing
                  │  schema v1      │  schema v2       │  outcome "blocked"   │  schema v3
                  │  RunRecord ─────┼──────────────────┼──────────────────────┤
                  │  usage.reasoning_tokens ───────────┼──────────────────────┤
                  └─ SessionStore interface ───────────┴──── write/read_spans ┘
```

**Why this order:**
- **P0 → P2:** the compaction input budget uses the resolved `max_tokens`
  (16384 after P0), and the request field is `max_completion_tokens`.
- **P1 → P2:** compaction persists summaries in the store (schema version 2)
  and maps `seq` to positions through `SessionHandle.seqs`.
- **P1 → P3:** guardrails add the `"blocked"` run outcome to `RunRecord`.
- **P1, P2, P3 → P4:** tracing stores spans in the store (schema version 3)
  and emits `compaction` and `guardrail` spans.
- **The schema versions chain** 1 → 2 → 3, so plans must land in order, or
  migrations conflict.

## Within a plan

- **Tasks run strictly in order, one implementer at a time.** Every plan's
  later tasks edit `src/logpose/agent.py`, so parallel implementers would
  conflict (and the process forbids it).
- **Each plan ends with** a whole-branch review on the most capable model, one
  fix wave, a scoped re-review, then a PR. Merge before the next plan's branch
  is cut, which keeps each review a reasonable size and each migration landing
  on top of the last.

### Task-level dependencies (for briefing implementers)

| Plan | Task | Needs |
|---|---|---|
| P1 | T1 foundations | — |
| P1 | T2 store and schema | T1 (`Conversation` JSON, `SessionBusyError`) |
| P1 | T3 lease | T2 |
| P1 | T4 agent integration | T1–T3 |
| P1 | T5 sync, docs | T4 |
| P2 | T1 pointers and recall | P1 merged |
| P2 | T2 summary fallback | T1 |
| P2 | T3 persisted summaries and reactive retry | T2, P1 store |
| P2 | T4 docs | T3 |
| P3 | T1 guard types and built-ins | P1 merged (independent of P2 code) |
| P3 | T2 blocking integration | T1, P1 `RunRecord` |
| P3 | T3 parallel mode | T2 |
| P3 | T4 exports and docs | T3 |
| P4 | T1 semconv, spans, schema v3 | P1–P3 merged |
| P4 | T2 span builder in the loop | T1; P2 compaction diagnostics; P3 guardrail results |
| P4 | T3 writer, flush, sync facade | T2 |
| P4 | T4 OTLP/JSON export and docs | T3 |

## Model tiers (per the process's model selection)

| Role | Model |
|---|---|
| Implementers for pure-module tasks (P1 T1, P1 T2, P2 T1, P3 T1, P4 T1, P4 T4) | mid tier (Sonnet) |
| Implementers for loop or concurrency tasks (P1 T3, P1 T4, P2 T2, P2 T3, P3 T2, P3 T3, P4 T2, P4 T3) | top tier (Opus) |
| Task reviewers | Sonnet, or Opus for the top-tier tasks |
| Whole-branch reviews | Opus |

## Totals

There are 17 tasks across Plans 1–4, plus the already-built Plan 0. Each plan
ends in its own PR; the whole run makes 4 PRs plus Plan 0's.

## Cross-plan corrections already applied

- The store is keyed per **(database path, event loop)**, not per process
  (found by the tracing builder check). Plan 1 Task 4 and the session-store
  spec are updated.

## Human checkpoints

- After each plan's PR: the owner reviews and merges. That is the only pause; it
  is required because merging is outward-facing.
- Open choices the owner can still overturn during PR review, each recorded as
  Claude's call in its spec:
  - compaction summarizes at 85% (option A);
  - guardrails fail closed, and parallel mode is opt-in;
  - traces are stored in SQLite and exported as OTLP/JSON.
