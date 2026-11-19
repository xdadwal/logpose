# Structured Output Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `agent.run(..., output_type=Model)` returns a validated `result.output: Model`. Enforcement uses only strict `json_schema` `response_format`, with validation, a correction turn, and clear errors for incompatible providers.

**Architecture:**
- **Pure helpers** in a new `logpose/output.py`: strict-schema conversion, `response_format` construction, and null cleaning, parsing and validation.
- **Provider changes** in `openai_compat.py`: collect refusals, flag servers that reject strict mode, and catch `response_format` conflicts.
- **Agent changes:** inject `response_format`, validate the final answer, run the correction turn and history rollback, and expose `RunResult[T].output`.

**Tech Stack:** Python 3.10+, pydantic v2, httpx, pytest (asyncio auto), ruff, mypy.

**Spec:** `docs/superpowers/specs/2026-11-19-structured-output-design.md`

## Global Constraints

- One mechanism only: `response_format = {"type": "json_schema", "json_schema": {"name": <name>, "schema": <strict schema>, "strict": True}}`. No `json_object` or prompt-only mode.
- Runs without `output_type` send byte-for-byte identical requests and return identical results, except that `RunResult.output is None`.
- `output_type` must be a `pydantic.BaseModel` subclass. `output_retries` must be an int ≥ 0, default `1`. Invalid values raise `LogposeError` before any request.
- Error text never echoes model-provided input values. It contains only field paths, messages and error types.
- Gates: `uv run pytest -q`, `uv run ruff check`, `uv run mypy src/`.

## Review Focus

1. The agent sets `output_type`, and a single call passes `output_type=None`. That call must be unstructured (no `response_format`, `output is None`). Test owned by Task 3.
2. A server ignores `response_format`, and the model wraps its JSON in a markdown code fence. That gets a correction turn, and if it still fails, an `OutputValidationError` that includes the "server may not enforce strict json_schema" hint. Test owned by Task 3.
3. A `Conversation` is reused after a successful correction. It must hold the user prompt, any tool turns, and only the valid final answer, with no invalid answer or correction message. Test owned by Task 3.
4. A nested model, or a list of models, whose fields have defaults gets `null` for those fields. They take their defaults, while `X | None` fields keep `None`. Test owned by Task 1.
5. The final turn ends with empty text, or only whitespace. That counts as a validation failure and gets a correction turn, not a crash. Test owned by Task 3.

---

### Task 1: `logpose/output.py`: strict schema, `response_format`, and parsing

**Files:**
- Create: `src/logpose/output.py`
- Test: `tests/test_output.py`

**Interfaces:**
- Produces:
  - `strict_json_schema(model: type[BaseModel]) -> dict[str, Any]` returns a deep copy of `model.model_json_schema()` in which every object schema (including `$defs`) has `"additionalProperties": False` and `"required"` listing all its `properties`. A property with a default that is not already nullable becomes `{"anyOf": [<original>, {"type": "null"}]}`, with `"default"` removed. `$defs` and `$ref` are kept.
  - `schema_name(model: type[BaseModel]) -> str` takes `model.__name__` with characters outside `[A-Za-z0-9_-]` replaced by `_`, truncated to 64, or `"output"` if that comes out empty.
  - `response_format_for(model: type[BaseModel]) -> dict[str, Any]` returns the exact dict from the Global Constraints.
  - `class OutputParseFailure(Exception)` has attribute `errors: list[dict[str, Any]]`. Each item has only the keys `loc` (tuple), `msg` (str) and `type` (str).
  - `parse_output(model: type[T], text: str) -> T` does `json.loads(text)`. Invalid JSON raises `OutputParseFailure([{"loc": (), "msg": "Invalid JSON", "type": "json_invalid"}])`. It then calls `strip_disallowed_nulls(model, data)`, then `model.model_validate(...)`. A pydantic `ValidationError` becomes `OutputParseFailure` with sanitized errors (no `input`, no `ctx`, no `url`).
  - `strip_disallowed_nulls(model: type[BaseModel], data: Any) -> Any` recursively removes `None` values for fields whose annotation does not accept `None`. It recurses through nested `BaseModel` fields, `list`/`tuple` items and `dict` values whose item type is a `BaseModel`. Use `model.model_fields[name].annotation` with `typing.get_origin`/`get_args`.
  - `format_errors(errors: list[dict[str, Any]]) -> str` returns lines of `"<dotted loc or '(root)'>: <msg>"`.

- [ ] **Step 1: Write the failing tests** in `tests/test_output.py`, using these models:

  ```python
  class Item(BaseModel):
      name: str
      qty: int = 1

  class Order(BaseModel):
      customer: str
      items: list[Item]
      note: str | None = None
      priority: int = 3
  ```

  - `test_strict_schema_requires_every_property_and_forbids_extras`: for every object in the schema and in `$defs`, `required` equals the set of its property keys and `additionalProperties is False`.
  - `test_defaulted_fields_become_nullable_without_default`: `priority` and `Item.qty` are `anyOf [..., {"type": "null"}]` with no `"default"` key. `note` is nullable and appears exactly once in `required`.
  - `test_response_format_shape`: `response_format_for(Order) == {"type": "json_schema", "json_schema": {"name": "Order", "schema": strict_json_schema(Order), "strict": True}}`.
  - `test_schema_name_is_sanitized`: a model named `"My Model!"`, created with `create_model`, gives `"My_Model_"`.
  - `test_null_for_defaulted_field_takes_the_default`: `parse_output(Order, '{"customer":"a","items":[{"name":"x","qty":null}],"note":null,"priority":null}')` gives `priority == 3`, `items[0].qty == 1` and `note is None`.
  - `test_invalid_json_is_a_parse_failure`: `parse_output(Order, "not json")` raises `OutputParseFailure`, and `errors[0]["type"] == "json_invalid"`.
  - `test_validation_errors_never_echo_input`: `parse_output(Order, '{"customer": 123456789, "items": "secret-value"}')` raises. `"secret-value"` and `"123456789"` don't appear in `str(errors)` or in `format_errors(errors)`, and every error's keys are `{"loc", "msg", "type"}`.
  - `test_format_errors_lists_paths`: the output contains `"items.0.name: "` for a missing nested name.

- [ ] **Step 2: Run and confirm they fail.** Run: `uv run pytest tests/test_output.py -q`. Expected: FAIL (`ModuleNotFoundError: logpose.output`).
- [ ] **Step 3: Implement `src/logpose/output.py`** per the Interfaces.
- [ ] **Step 4: Run and confirm they pass.** Run: `uv run pytest tests/test_output.py -q && uv run mypy src/`. Expected: PASS, clean.
- [ ] **Step 5: Commit** with `git commit -m "feat(output): strict json_schema conversion and typed parsing"`.

### Task 2: Errors, refusal capture, and incompatibility detection in the provider

**Files:**
- Modify: `src/logpose/errors.py`, `src/logpose/providers/base.py` (`CompletionDone`), `src/logpose/providers/openai_compat.py`
- Test: `tests/test_provider_openai_compat.py`, `tests/test_errors.py` (create if absent)

**Interfaces:**
- Produces, in `errors.py` and in its `__all__`:
  - `class StructuredOutputUnsupportedError(ProviderError)`, always `retryable=False`.
  - `class OutputValidationError(LogposeError)` with `__init__(self, message: str, *, raw_text: str, errors: list[dict[str, Any]], messages: list[Message], output_type: type, reason: Literal["invalid", "max_tokens"] = "invalid")`, keeping each as an attribute.
  - `class OutputRefusalError(LogposeError)` with `__init__(self, message: str, *, refusal: str, messages: list[Message])`.
- Produces `CompletionDone.refusal: str | None = None`, a new last field with a default.
- In `OpenAICompatProvider.stream`:
  - concatenate `delta["refusal"]` strings into `CompletionDone.refusal`;
  - if the request body contains `response_format` and the HTTP status is 400 or 422 and the lowercased response body contains any of `"response_format"`, `"json_schema"`, `"strict"`, `"structured"`, raise `StructuredOutputUnsupportedError`. The message must contain `"does not support strict json_schema structured output"` and the base URL, and must not contain the response body;
  - in `_build_body`, if `"response_format"` is in both `self._extra_body` and `req.extra`, raise `LogposeError` matching `"response_format"`.

- [ ] **Step 1: Write the failing tests** in `tests/test_provider_openai_compat.py`:
  - `test_refusal_deltas_are_collected`: the stream has `delta_chunk(refusal="I can't ")`, then `delta_chunk(refusal="help.")`, then `finish_chunk("stop")`. The resulting `done.refusal == "I can't help."`.
  - `test_no_refusal_by_default`: a plain text stream gives `done.refusal is None`.
  - `test_rejected_response_format_is_flagged` (parametrized over status 400 and 422): the handler returns `{"error": {"message": "response_format json_schema is not supported"}}`, and the request has `extra={"response_format": {...}}`. It raises `StructuredOutputUnsupportedError` with `retryable is False` and the message containing `"does not support strict json_schema structured output"`.
  - `test_400_without_response_format_is_a_plain_provider_error`: the same body, but the request has no `response_format`. The raised type is exactly `ProviderError`.
  - `test_extra_body_and_request_response_format_conflict`: `make_provider(handler, extra_body={"response_format": {"type": "text"}})` and a request with `extra={"response_format": {...}}` raise `LogposeError` matching `"response_format"`.
  - In `tests/test_errors.py`, `test_output_errors_carry_their_fields`: construct each new error and check its attributes. `issubclass(StructuredOutputUnsupportedError, ProviderError)`.
- [ ] **Step 2: Run and confirm they fail.** Run: `uv run pytest tests/test_provider_openai_compat.py tests/test_errors.py -q`. Expected: the new tests FAIL.
- [ ] **Step 3: Implement** per the Interfaces. Detect unsupported servers at the existing call site of `_status_error` inside `stream()`, where the request body is known.
- [ ] **Step 4: Run and confirm they pass.** Run: `uv run pytest -q && uv run mypy src/`. Expected: PASS, clean.
- [ ] **Step 5: Commit** with `git commit -m "feat(providers): refusal capture and strict structured-output incompatibility errors"`.

### Task 3: Agent integration

**Files:**
- Modify: `src/logpose/agent.py`, `src/logpose/events.py`
- Test: `tests/test_agent.py`, plus a new `tests/test_structured_output.py` for the end-to-end cases

**Interfaces:**
- Consumes: Task 1's `response_format_for`, `parse_output`, `OutputParseFailure` and `format_errors`; Task 2's errors and `CompletionDone.refusal`.
- Produces:
  - `RunResult` becomes `Generic[T]`, with a new last field `output: T | None = None`.
  - `Agent(..., output_type: type[BaseModel] | None = None, output_retries: int = 1)`.
  - `Agent.run(prompt=None, *, conversation=None, output_type=_UNSET, output_retries=_UNSET)`, with the same keywords on `stream`. `_UNSET` is a private sentinel that means "use the agent default", and an explicit `None` disables. Add typing overloads so that `run(..., output_type=M)` gives `RunResult[M]`.
  - `_prepare_request` adds `extra["response_format"] = response_format_for(output_type)` when an output type is in effect. If `self.extra` already has `response_format`, raise `LogposeError` matching `"response_format"` at the start of the run.
  - At the final turn (no tool calls), with an output type in effect:
    - if `done.refusal` is set, or `stop_reason == "refusal"`, raise `OutputRefusalError`;
    - if `stop_reason == "max_tokens"`, raise `OutputValidationError(reason="max_tokens")` with a message containing `"raise max_tokens"`;
    - otherwise run `parse_output`. If it succeeds, set `result.output`. If it fails with retries left, append a user message (role `user`, one `TextBlock`) whose text is `"Your previous answer did not match the required JSON schema:\n" + format_errors(errors) + "\nReply again with only the corrected JSON."`, then `continue` the loop. If it fails with no retries left, raise `OutputValidationError` whose message includes `"the server may not enforce strict json_schema"`;
    - after a successful correction, remove from `history` every invalid assistant answer and correction message added during this run, keeping the final valid answer.

- [ ] **Step 1: Write the failing tests** in `tests/test_structured_output.py`, using `FakeProvider` with `class Answer(BaseModel): city: str; temp_c: float`:
  - `test_valid_answer_returns_typed_output`: the scripted text is `'{"city":"Pune","temp_c":31}'`. The result is `isinstance(result.output, Answer)`, `result.output.city == "Pune"`, and the request carries `extra["response_format"] == response_format_for(Answer)`.
  - `test_no_output_type_is_unchanged`: no `response_format` in the request extra, and `result.output is None`.
  - `test_per_call_none_disables_agent_default`: `Agent(..., output_type=Answer)`, then `run("x", output_type=None)`. No `response_format` is sent and `output is None`.
  - `test_correction_turn_fixes_invalid_answer`: the turns are `'{"city":"Pune"}'` then `'{"city":"Pune","temp_c":31}'`. `output.temp_c == 31`, the provider was called twice, and the second request's last message text contains `"temp_c"` and `"did not match the required JSON schema"`.
  - `test_successful_correction_is_removed_from_conversation`: run the case above with `conversation=conv`. `[m.role for m in conv.messages] == ["user", "assistant"]` and `conv.messages[-1].text == '{"city":"Pune","temp_c":31}'`.
  - `test_output_retries_zero_fails_immediately`: an invalid first answer with `output_retries=0` raises `OutputValidationError`, the provider was called once, `"temp_c" in str(exc.errors)`, and the message contains `"the server may not enforce strict json_schema"`.
  - `test_fenced_json_gets_a_correction_then_fails`: both turns are ```` "```json\n{\"city\":\"Pune\",\"temp_c\":31}\n```" ````. It raises `OutputValidationError` after 2 provider calls.
  - `test_empty_final_text_is_corrected`: the turns are `""` then valid JSON. Output is set and the provider was called twice.
  - `test_refusal_raises_without_correction`: a turn whose `CompletionDone.refusal="no"` raises `OutputRefusalError` with `exc.refusal == "no"`, after one call. Extend `ScriptedTurn` with an optional `refusal`.
  - `test_cut_off_answer_raises_max_tokens`: `stop_reason="max_tokens"` raises `OutputValidationError` with `reason == "max_tokens"`, after one call.
  - `test_tools_then_structured_answer`: a turn with a tool call, then valid JSON. Output is set and both requests carried `response_format`.
  - `test_correction_hitting_max_iterations_raises_max_iterations`: `max_iterations=1` with an invalid first answer raises `MaxIterationsError`.
  - `test_extra_response_format_conflict`: `Agent(..., extra={"response_format": {}}, output_type=Answer)`, and `run` raises `LogposeError` matching `"response_format"`.
  - `test_invalid_output_type_and_retries_rejected`: `output_type=dict` and `output_retries=-1` each raise `LogposeError`.
  - `test_stream_run_end_carries_output`: the `RunEnd.result.output` from `stream()` is an `Answer`.
- [ ] **Step 2: Run and confirm they fail.** Run: `uv run pytest tests/test_structured_output.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** per the Interfaces. Validate `output_type` and `output_retries` in `__init__` and again per call.
- [ ] **Step 4: Run and confirm they pass.** Run: `uv run pytest -q && uv run ruff check && uv run mypy src/`. Expected: all clean.
- [ ] **Step 5: Commit** with `git commit -m "feat(agent): typed structured output with strict json_schema and correction turn"`.

### Task 4: Sync facade, exports, docs

**Files:**
- Modify: `src/logpose/sync.py`, `src/logpose/__init__.py`, `README.md`, `CHANGELOG.md`
- Test: `tests/test_sync.py`, `tests/test_public_api.py`

**Interfaces:**
- Produces:
  - `run_sync`, `stream_sync`, `SyncAgent.run` and `SyncAgent.stream` accept `output_type` and `output_retries` and forward them only when passed, so the agent default still applies.
  - `logpose` exports `OutputValidationError`, `OutputRefusalError` and `StructuredOutputUnsupportedError`.

- [ ] **Step 1: Write the failing tests.**
  - `test_run_sync_forwards_output_type` in `tests/test_sync.py`: a scripted valid JSON answer gives `run_sync(agent, "x", output_type=Answer).output.city == "Pune"`.
  - In `tests/test_public_api.py`, add the three errors to the expected exports and the module map.
- [ ] **Step 2: Run and confirm they fail.** Run: `uv run pytest tests/test_sync.py tests/test_public_api.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** the forwarding and exports.
- [ ] **Step 4: Update the docs.**
  - **README:** add a "Structured output" section with the `Invoice` example, the `output_retries` semantics, the three errors, and the "per-call `output_type=None` disables" rule. Add the dated compatibility table from the spec, headed "as of 2026-11". Add the three errors to the errors table.
  - **CHANGELOG `[Unreleased]` `### Added`:** structured output, the three errors, `CompletionDone.refusal`, and the generic `RunResult[T].output`.
- [ ] **Step 5: Verify.** Run: `uv run pytest -q && uv run ruff check && uv run mypy src/`. Expected: clean.
- [ ] **Step 6: Commit** with `git commit -m "feat: structured output in sync facade, exports, and docs"`.
