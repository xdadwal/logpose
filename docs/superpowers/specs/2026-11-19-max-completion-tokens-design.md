# Output-limit wire field (`max_completion_tokens`)

Status: approved 2026-11-19 (small change, short design)

**Intent:** OpenAI deprecated the Chat Completions `max_tokens` field, and its
reasoning models (o-series, GPT-5 class, likely the owner's target "luna")
reject it in favour of `max_completion_tokens`. vLLM and llama.cpp also
deprecated `max_tokens`. logpose always sends `max_tokens`, so OpenAI reasoning
models fail.

**Decided (approach A):**
- New setting `OpenAICompatProvider(max_tokens_field: Literal["max_completion_tokens",
  "max_tokens"] = "max_completion_tokens")`. Any other value raises `LogposeError`.
- The request body carries the turn's output ceiling (`req.max_tokens`) under
  that field name and never under both.
- `Agent(max_tokens=...)`, the 16384 default, and their precedence are unchanged;
  only the wire field name changes.
- A value in `extra_body` or `Agent(extra=...)` for either field still overrides
  as today. Merge order is unchanged.

**Approaches considered:** (B) the same setting defaulting to `max_tokens`
breaks OpenAI reasoning models by default. (C) auto-selecting by host
reintroduces provider-specific guessing and misses OpenAI-compatible proxies.

**Non-goals:** auto-detection, sending both fields, or a fallback retry when a
server rejects the field.

**Docs:** the README says which field is sent and how to switch. It flags that
older Ollama and LM Studio builds may only accept `max_tokens`, and that a
server which silently ignores the field applies no output cap (time limits still
bound the turn). The CHANGELOG gets a breaking entry: the field sent by default
changes.
