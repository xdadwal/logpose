# Structured model output

Status: design for approval, 2026-11-19

## Intent and the why

Web apps embedding logpose need the model's final answer as a typed object, not
free text: it can be rendered and stored without parsing, and it makes tests
consistent because they assert on fields rather than matching wording. This adds
a typed final answer to `Agent.run` / `stream`, enforced with the OpenAI
Structured Outputs mechanism.

## Goals

- `result = await agent.run("…", output_type=Invoice)` returns
  `result.output: Invoice`, a validated pydantic object.
- Enforcement uses exactly one mechanism: Chat Completions
  `response_format: {"type": "json_schema", "json_schema": {"name", "schema",
  "strict": true}}`. This is OpenAI's native Structured Outputs; vLLM, Ollama,
  llama.cpp, LM Studio and Sarvam implement it too.
- logpose validates every final answer with pydantic regardless of what the
  server enforced. On a validation failure it sends one correction turn by
  default (`output_retries=1`). `output_retries=0` fails immediately, which is
  useful in tests.
- Incompatible providers are clearly flagged, both at runtime and in the docs.

## Non-goals

- No `json_object` mode and no schema-in-prompt fallback. This is the owner's
  decision: one reliable mechanism, with incompatibility surfaced, not papered
  over.
- No strict mode for tool definitions. Tool argument validation is unchanged.
- No support for output types other than pydantic `BaseModel` subclasses in this
  version (no bare `list[...]`, `TypedDict`, dataclasses or primitives).
- No streaming of partially parsed objects. Streaming still emits raw
  `TextDelta`s of the JSON as it arrives.

## Anti-goals

- A run that returns `output` that is not a validated instance of `output_type`.
- A silent fallback to unenforced output when a server rejects strict mode.
- Any change to runs that do not pass `output_type`. Their requests and results
  must be byte-for-byte unchanged.

## Approaches

- **Strict `json_schema` `response_format`, plus validation and correction
  (chosen).** It is the most reliable method where supported (constrained
  decoding) and native to OpenAI, it keeps streamed text, and validation covers
  servers that accept the field but don't enforce it.
- Hidden "final answer" tool with `tool_choice: required`: rejected. The answer
  arrives as tool arguments (no text streaming), `required` support is uneven,
  and it needs extra rules for mixed tool calls.
- `json_object` or prompt-only modes: rejected by the owner. They are weaker
  guarantees, and the owner prefers to flag incompatible providers.

## Decided

### API

- `Agent(..., output_type: type[BaseModel] | None = None, output_retries: int = 1)`
  sets agent-level defaults.
- `run(prompt=None, *, conversation=None, output_type=..., output_retries=...)`
  overrides them per call. `stream`, `run_sync`, `stream_sync` and `SyncAgent`
  accept the same.
- `RunResult` becomes generic, `RunResult[T]`, with `output: T | None`. `output`
  is `None` when no `output_type` was in effect. `text` keeps the raw final text.
- `output_retries` must be an int ≥ 0; anything else raises `LogposeError`.
- `output_type` must be a `BaseModel` subclass; anything else raises
  `LogposeError` before any request is sent.

### Request

- When `output_type` is in effect, every provider turn of that run carries
  `response_format`. The schema `name` is the model's class name, cleaned to
  `[A-Za-z0-9_-]{1,64}`. The schema is a **strict-compatible** conversion of
  `output_type.model_json_schema()`:
  - every object lists all its properties in `required`;
  - every object has `additionalProperties: false`;
  - a field that has a default becomes a nullable union, so the model must send
    it, possibly as `null`;
  - nested models stay as `$defs` / `$ref`.
- `response_format` goes into `CompletionRequest.extra`. If the caller also set
  `response_format` in `Agent(extra=...)`, raise `LogposeError`: two sources for
  the same field is a configuration mistake. **(Claude's call)** The same applies
  to `OpenAICompatProvider(extra_body=...)`: the provider raises `LogposeError`
  when both its `extra_body` and the request set `response_format`, so neither
  silently overwrites the other.
- Tools keep working. The model can call tools on any turn, and the run's
  final answer (the turn that ends with no tool calls) is the structured output.

### Validation and correction

- **Null handling (Claude's call).** Pydantic does not turn `null` into a
  field's default; `x: int = 5` rejects `null`. So logpose parses the JSON, then
  removes every `null` value whose field's original annotation does not accept
  `None`. Each such field then takes its default. This happens recursively,
  through nested models and items of lists or dicts of models. A field whose
  annotation accepts `None` (`x: int | None = 5`) keeps the `null`: `null` means
  `None` there. Validation then runs on the cleaned object with
  `output_type.model_validate`. The cleaning is guided by the model's fields,
  not by guessing from values.
- On a `ValidationError` with retries left, logpose appends the invalid
  assistant message plus a user message stating the validation errors (field
  paths and messages only, never echoing values) and asking for a corrected
  answer. It then runs another turn. Correction turns count against
  `max_iterations` and `run_timeout`.
- When retries are exhausted, it raises `OutputValidationError(LogposeError)`
  with `raw_text`, `errors` (pydantic's error list without input values),
  `messages` (the partial history), and `output_type`.
- **History (Claude's call).** After a successful correction, the invalid
  answer(s) and correction message(s) are removed from the history (including a
  caller's `Conversation`). Only the valid final answer remains, so later runs
  never replay them. On failure, nothing is removed: `messages` and the
  `Conversation` keep the full exchange for debugging, still a valid
  user/assistant alternation.
- **Precedence (Claude's call).** A correction turn that hits `max_iterations`
  or `run_timeout` raises `MaxIterationsError` / `RunTimeoutError`. Run-wide
  limits win over output validation, consistent with how they end any other
  turn.

### Special endings (no correction turn)

- **Refusal.** The server sends a `refusal` delta or field, or
  `finish_reason == "content_filter"`. logpose raises
  `OutputRefusalError(LogposeError)` with the refusal text and `messages`.
  **Carrying it (Claude's call):** the provider collects `refusal` deltas into a
  new optional `CompletionDone.refusal: str | None`. The `Message` shape and
  stored history are unchanged, and the refusal turn is stored (or not) by the
  existing rules. Without `output_type`, the field is populated but otherwise
  unused, so non-structured runs behave exactly as today.
- **Cut off.** `finish_reason == "length"` (stop reason `max_tokens`). logpose
  raises `OutputValidationError` with `reason="max_tokens"` and a message telling
  the caller to raise `max_tokens`. No correction turn: the same limit would cut
  it off again.

### Flagging incompatible providers

- **Rejected at the server.** If a request carrying `response_format` gets an
  HTTP 400 or 422 whose body mentions `response_format`, `json_schema`, `strict` or
  `structured`, logpose raises `StructuredOutputUnsupportedError` (a
  `ProviderError` subclass, not retried). Its message says this server/model
  does not support strict `json_schema` structured output, that logpose requires
  it for `output_type`, and to use a compatible server or model.
- **Accepted but ignored.** Some servers silently drop the field. When validation
  fails on the final attempt, the `OutputValidationError` message adds a hint
  that the server may not enforce strict `json_schema`.
- **Docs.** The README gets a "Structured output" section with a compatibility
  table:

  | Provider | Support |
  |---|---|
  | OpenAI | ✅ |
  | vLLM | ✅ |
  | Ollama | ✅ |
  | llama.cpp | ✅ |
  | LM Studio | ✅ |
  | Sarvam | ✅ |
  | OpenRouter | ⚠️ depends on the model; use models listing `structured_outputs` |
  | Z.ai GLM API | ❌ documents only `json_object` |
  | DeepSeek API | ❌ documents only `json_object` |

  The table is dated, since provider support changes.

### Exports

`OutputValidationError`, `OutputRefusalError` and
`StructuredOutputUnsupportedError` are exported from `logpose`.

## Left to the builder

- The exact wording of the correction message and the error messages, within the
  rules above (no input values echoed).
- Typing mechanics for `RunResult[T]` and `run` overloads, so that
  `output_type=Invoice` type-checks as `RunResult[Invoice]`.
- Which JSON Schema keywords the strict conversion leaves alone versus rejects.
  Default to passing them through and letting the server reject them, which
  produces `StructuredOutputUnsupportedError` and a clear message.
- Where the strict-schema conversion lives: a new focused module such as
  `logpose/output.py` is suggested.
