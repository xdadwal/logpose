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

## Amendment: web-app safe limits (approved 2026-11-19)

Target client: a small web app where a user waits on one request. Every value
that could run unbounded gets a safe default; `None` stays available as an
explicit "unlimited". Tool output shares one budget of ~4k tokens per call so a
result fits even a 32k-context model.

| Setting | Default |
|---|---|
| `Agent(run_timeout=)` (new) | 320 s wall clock for a whole run; `None` = unlimited |
| `Agent(provider_turn_timeout=)` | 120 s |
| `Agent(max_tokens=)` fallback | 8192 (was 32000) |
| `Agent(max_iterations=)` | 10 (was 25) |
| `Agent(tool_timeout=)` | 30 s (was 300) |
| `OpenAICompatProvider(timeout=)` | connect 5 s, read 60 s, write 10 s, pool 5 s; `None` = no HTTP timeouts |
| `RetryPolicy` | `max_attempts=2`, `max_delay=4.0`, `max_retry_after=10.0` |
| shell | timeout 30 s, capture 1 MB, output to model 16,000 chars |
| `read_file` | no line cap (`max_lines=None` reads to the page limit); page 16,000 chars |
| `list_dir` | no entry cap (`max_entries=None`); output 16,000 chars with continuation |
| grep | output 16,000 chars |
| `fetch_readable_text` | timeout 10 s, 1 MB, output 16,000 chars |
| `search_duckduckgo` | timeout 10 s |
| `make_search_tool` | output 8,000 chars |

Run deadline behavior: measured from run start, including time the caller
spends between streamed events. Each provider turn is bounded by the smaller of
`provider_turn_timeout` and the remaining run time; tool execution is bounded
by the remaining run time. Expiry raises `RunTimeoutError(LogposeError)`
carrying the partial `messages` and `run_timeout`, emits a `run.timed_out`
runtime event, and is never retried.
