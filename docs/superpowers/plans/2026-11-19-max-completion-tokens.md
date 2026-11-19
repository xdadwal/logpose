# max_completion_tokens Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Send the output ceiling as `max_completion_tokens` by default, with `max_tokens` selectable per provider.

**Architecture:** Add one constructor option on `OpenAICompatProvider` that names the wire field in `_build_body`.

**Tech Stack:** Python 3.10+, httpx, pytest.

**Spec:** `docs/superpowers/specs/2026-11-19-max-completion-tokens-design.md`

## Global Constraints

- Default `max_tokens_field="max_completion_tokens"`. Allowed values are exactly `"max_completion_tokens"` and `"max_tokens"`.
- Never send both fields from logpose's own value.
- `extra_body` and request `extra` merge order is unchanged.
- Gates: `uv run pytest -q`, `uv run ruff check`, `uv run mypy src/`.

## Review Focus

1. A caller passes `extra_body={"max_tokens": 50}` while the default field is `max_completion_tokens`. Both fields reach the server, because logpose sends one and the app the other; that's the app's explicit choice and is documented. Test: the body contains both, unchanged.

---

### Task 1: Configurable output-limit field

**Files:**
- Modify: `src/logpose/providers/openai_compat.py` (constructor L340-395, `_build_body` L574)
- Test: `tests/test_provider_openai_compat.py`
- Modify: `README.md`, `CHANGELOG.md`

**Interfaces:**
- Produces: `OpenAICompatProvider.__init__(..., max_tokens_field: Literal["max_completion_tokens", "max_tokens"] = "max_completion_tokens")`, stored as `self.max_tokens_field`.

- [ ] **Step 1: Write the failing tests.**
  - `test_default_sends_max_completion_tokens`: a body built with `request(max_tokens=128)` has `"max_completion_tokens": 128` and no `"max_tokens"`.
  - `test_max_tokens_field_can_select_legacy`: with `make_provider(handler, max_tokens_field="max_tokens")`, the body has `"max_tokens": 128` and no `"max_completion_tokens"`.
  - `test_invalid_max_tokens_field_rejected`: `max_tokens_field="max_output_tokens"` raises `LogposeError` matching `"max_tokens_field"`.
  - `test_extra_body_legacy_field_passes_through`: `extra_body={"max_tokens": 50}` with the default field gives a body with `max_tokens == 50` and `max_completion_tokens == 128`.
  - Update existing tests that assert `body["max_tokens"]`.
- [ ] **Step 2: Run and confirm they fail.** Run: `uv run pytest tests/test_provider_openai_compat.py -q`. Expected: the new tests FAIL.
- [ ] **Step 3: Implement** the option, the validation, and the field name in `_build_body`. Update the docstrings.
- [ ] **Step 4: Run the gates.** Run: `uv run pytest -q && uv run ruff check && uv run mypy src/`. Expected: clean.
- [ ] **Step 5: Docs.**
  - **README "Output limit" section:** logpose sends `max_completion_tokens`; use `max_tokens_field="max_tokens"` for servers that only accept the old field (older Ollama or LM Studio builds); a server that ignores the field applies no cap.
  - **CHANGELOG:** add a Breaking bullet under Changed.
- [ ] **Step 6: Commit** with `git commit -m "feat(providers): send max_completion_tokens by default; max_tokens_field option"`.
