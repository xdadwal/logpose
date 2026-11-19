# Neutral provider defaults

Status: approved 2026-11-19

## Intent

logpose should not guess anything about a model server. The integrating
application supplies provider values, as arguments or through neutral
`LOGPOSE_*` environment variables. The only built-in model-facing value is an
output-token ceiling sized for current reasoning models.

## Goals

- `openai-compat` is the only built-in provider; `docker` is removed.
- Amended 2026-11-19: model made mandatory on `Agent` (see below).
- Provider configuration reads only `LOGPOSE_BASE_URL` and `LOGPOSE_API_KEY`
  (no `OPENAI_*` / `DOCKER_*`).
- A missing base URL raises `LogposeError` naming the argument and the env var.
  A missing API key means no auth header (local servers).
- `model=` is a required keyword argument of `Agent`; an empty or blank string
  raises `LogposeError`. The model lives only on the Agent: the provider `model=`
  argument, `LOGPOSE_MODEL`, and the `Provider.model_default` protocol attribute
  are removed. Direct provider callers already pass `CompletionRequest.model`.
- `model="auto"` remains as an explicit opt-in to "use the first listed model".
- One output ceiling: `Agent` default `max_tokens=32000`. Precedence is unchanged:
  `Agent(max_tokens=)`, then provider `max_tokens=` (now `None` by default), then
  32000.
- Timeouts are optional and off by default: `Agent(provider_turn_timeout=None)`
  (the `"default"` sentinel and provider recommendations are removed) and
  `OpenAICompatProvider(timeout=None)` means no HTTP timeouts.

## Non-goals

- `max_completion_tokens` support for OpenAI reasoning models (next change).
- Toolkit defaults (shell/web/workspace) and loop settings (`max_iterations`,
  `RetryPolicy`, tool concurrency/timeouts, error mode) are unchanged.
- `parse_think_tags=True` and `preserve_reasoning=False` are behavior switches and
  stay.

## Anti-goals

- No silent fallback to a provider-specific endpoint, model, or credential.

## Approaches

- Required arguments with no env reads: rejected by the user; env configuration
  is wanted but must be provider-neutral.
- Keep `docker` with required values: rejected; without defaults it duplicates
  `openai-compat`.
- Required limits/timeouts: rejected; limits are optional, with a sensible token
  default.
- Chosen: neutral `LOGPOSE_*` env fallbacks, `openai-compat` only, 32000 token
  default, timeouts off unless set.

## Builder notes

- `ProviderInfo` for `openai-compat`: `env_vars=("LOGPOSE_BASE_URL",
  "LOGPOSE_API_KEY")`, `base_url_env="LOGPOSE_BASE_URL"`.
- Remove `DEFAULT_CLOUD_TURN_TIMEOUT`, `DEFAULT_LOCAL_TURN_TIMEOUT`,
  `LOCAL_TURN_TIMEOUT_MULTIPLIER`, `DEFAULT_PROVIDER_TURN_TIMEOUT`, provider
  `turn_timeout` recommendations, and `DEFAULT_DOCKER_BASE_URL`.
- `examples/local_docker.py` becomes an `openai-compat` example passing the Docker
  Model Runner URL explicitly. Docs, CHANGELOG (breaking), and CI updated.
