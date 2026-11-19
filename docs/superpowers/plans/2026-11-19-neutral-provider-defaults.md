# Neutral Provider Defaults Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `openai-compat` the only provider, configured from arguments or neutral `LOGPOSE_*` env vars with no provider-specific fallbacks, a single 32000-token default, and timeouts off unless set.

**Architecture:** Configuration resolution stays in `OpenAICompatProvider.__init__`; request-time model validation stays in `Agent._resolve_model` and `OpenAICompatProvider._resolve_model`; deadline and token precedence stay in `Agent._turn_timeout` / `Agent._max_tokens`. Readiness metadata stays declarative in `ProviderInfo`.

**Tech Stack:** Python 3.10+, httpx, pydantic, pytest (asyncio auto mode), ruff, mypy.

**Spec:** `docs/superpowers/specs/2026-11-19-neutral-provider-defaults-design.md`

## Global Constraints

- Env vars read: `LOGPOSE_BASE_URL`, `LOGPOSE_MODEL`, `LOGPOSE_API_KEY` only. No `OPENAI_*`, no `DOCKER_*` anywhere in `src/`.
- `Agent` default `max_tokens` is `32000` (`DEFAULT_MAX_TOKENS = 32000` in `agent.py`).
- `Agent(provider_turn_timeout=None)` default; the `"default"` sentinel is removed.
- `OpenAICompatProvider(timeout=None)` means `httpx.Timeout(None)` (no limits).
- `model="auto"` (`AUTO_MODEL`) is the only trigger for model discovery.
- Breaking changes are listed under `## [Unreleased] / ### Removed` (or `### Changed`) in `CHANGELOG.md`.
- Gates for every task: `uv run pytest -q`, `uv run ruff check`, `uv run mypy src/` all clean.

## Review Focus

1. Whitespace-only env values (`LOGPOSE_BASE_URL="  "`, `LOGPOSE_MODEL=" "`) must count as missing, not be sent. Test owned by Task 1.
2. `Agent("openai-compat", model="m", base_url=...)` with no provider model and no `LOGPOSE_MODEL` must work (Agent model wins at request time). Test owned by Task 2.
3. A request with an empty model sent straight to the provider (no Agent) must raise `LogposeError`, not silently discover. Test owned by Task 1.
4. `Agent(provider_turn_timeout="default")` from old code must fail loudly with a message naming the accepted values. Test owned by Task 2.
5. An ambient `OPENAI_API_KEY` must never be sent now that the provider reads only `LOGPOSE_API_KEY`. Test owned by Task 1.

---

### Task 1: Provider configuration without provider-specific defaults

**Files:**
- Modify: `src/logpose/providers/openai_compat.py` (constructor ~L319-390, `_resolve_model` ~L438, module constants L60-90, remove `DockerModelsProvider` ~L721-end, module docstring)
- Modify: `src/logpose/providers/base.py` (remove `DEFAULT_CLOUD_TURN_TIMEOUT`, `LOCAL_TURN_TIMEOUT_MULTIPLIER`, `DEFAULT_LOCAL_TURN_TIMEOUT` and their `__all__` entries)
- Test: `tests/test_provider_openai_compat.py`

**Interfaces:**
- Produces:
  - `OpenAICompatProvider.__init__(*, base_url: str | None = None, model: str | None = None, api_key: str | None = None, max_tokens: int | None = None, timeout: float | httpx.Timeout | None = None, parse_think_tags: bool = True, default_headers: dict[str, str] | None = None, client: httpx.AsyncClient | None = None, extra_body: dict[str, Any] | None = None, preserve_reasoning: bool = False)`. The `api_key_env` parameter is removed.
  - Module constants `ENV_BASE_URL = "LOGPOSE_BASE_URL"`, `ENV_MODEL = "LOGPOSE_MODEL"`, `ENV_API_KEY = "LOGPOSE_API_KEY"` (exported in `__all__`).
  - `provider.model_default: str`, which is `""` when neither `model=` nor `LOGPOSE_MODEL` is set.
  - `provider.max_tokens: int | None`, `None` by default.
  - No `turn_timeout` attribute on the class.
  - `DockerModelsProvider`, `DEFAULT_DOCKER_BASE_URL`, `DEFAULT_MAX_TOKENS` (provider-level) removed from the module and `__all__`.

- [ ] **Step 1: Rewrite the configuration tests.** In `tests/test_provider_openai_compat.py`:
  - Delete every Docker-specific test and the `DockerModelsProvider` / `DEFAULT_DOCKER_BASE_URL` imports.
  - Change remaining `OPENAI_*` env usage to `LOGPOSE_*`.
  - Add these tests, using `monkeypatch` to clear `LOGPOSE_*` and `OPENAI_*` first:
    - `test_base_url_falls_back_to_logpose_env`: set `LOGPOSE_BASE_URL=http://env.test/v1/`; `OpenAICompatProvider().base_url == "http://env.test/v1"`.
    - `test_missing_base_url_names_argument_and_env`: `pytest.raises(LogposeError, match=r"base_url=.*LOGPOSE_BASE_URL")`.
    - `test_whitespace_env_values_count_as_missing`: `LOGPOSE_BASE_URL="  "` raises as above; with `base_url=BASE`, `LOGPOSE_MODEL=" "` gives `model_default == ""`; `LOGPOSE_API_KEY=" "` sends no `Authorization` header.
    - `test_model_falls_back_to_logpose_env`: `LOGPOSE_MODEL=m-env` gives `model_default == "m-env"`.
    - `test_openai_env_vars_are_ignored`: set `OPENAI_BASE_URL`, `OPENAI_MODEL`, `OPENAI_API_KEY=sk-ambient`. Construction without `base_url` raises. With `base_url=BASE`, `model_default == ""`, and a streamed request carries no `Authorization` header.
    - `test_api_key_falls_back_to_logpose_env`: `LOGPOSE_API_KEY=k1` gives `Authorization: Bearer k1` on the request.
    - `test_max_tokens_defaults_to_none`: `OpenAICompatProvider(base_url=BASE).max_tokens is None`.
    - `test_timeout_none_disables_http_timeouts`: `provider._http().timeout == httpx.Timeout(None)`.
    - `test_empty_request_model_raises_instead_of_discovering`: a `Recorder` with models; `drain(make_provider(handler, model=None), request(model=""))` raises `LogposeError` matching `"model"`, and `handler.model_calls == 0`.
    - `test_auto_model_still_discovers`: `request(model="auto")` sends `"m-first"` and `handler.model_calls == 1`.

- [ ] **Step 2: Run them and confirm they fail.**
  Run: `uv run pytest tests/test_provider_openai_compat.py -q`
  Expected: the new tests FAIL (env names, timeout, max_tokens, empty-model discovery).

- [ ] **Step 3: Implement the constructor and model rules in `openai_compat.py`.**
  - Resolve each value with a private helper `_env(name: str) -> str | None` that returns the stripped value, or `None` if it's empty.
  - The `base_url` error message must contain both `base_url=` and `LOGPOSE_BASE_URL`.
  - `timeout=None` becomes `httpx.Timeout(None)`.
  - In `_resolve_model`, an empty `requested` raises `LogposeError("No model for the OpenAI-compatible provider. Pass model=... or set LOGPOSE_MODEL; use model='auto' to pick the server's first model.")`. Only `AUTO_MODEL` triggers discovery.
  - Remove `DockerModelsProvider`, its constants, the `turn_timeout` class attributes, and the base.py turn-timeout constants. Update docstrings that mention `OPENAI_*`, Docker, or the 4096 default.

- [ ] **Step 4: Run the provider tests and confirm they pass.**
  Run: `uv run pytest tests/test_provider_openai_compat.py -q`
  Expected: PASS. Other test files may fail until Tasks 2-3.

- [ ] **Step 5: Commit.**
  ```bash
  git add src/logpose/providers/openai_compat.py src/logpose/providers/base.py tests/test_provider_openai_compat.py
  git commit -m "refactor(providers)!: neutral LOGPOSE_* config, no provider defaults, drop docker"
  ```

### Task 2: Agent token and deadline defaults

**Files:**
- Modify: `src/logpose/agent.py` (constants L137-163, `__init__` signature/validation L360-460, `_turn_timeout` L1054, `_resolve_model` L1139 message, docstrings)
- Test: `tests/test_agent.py`

**Interfaces:**
- Consumes: Task 1's `provider.max_tokens: int | None`, plus a provider that has no `turn_timeout`.
- Produces:
  - `DEFAULT_MAX_TOKENS = 32000`.
  - `Agent(provider_turn_timeout: float | None = None)`.
  - `DEFAULT_PROVIDER_TURN_TIMEOUT` is removed from the module and `__all__`.
  - `_turn_timeout()` returns `self.provider_turn_timeout` unchanged.

- [ ] **Step 1: Update and add tests in `tests/test_agent.py`.**
  - Delete `test_built_in_provider_deadlines_use_shared_defaults` and any test asserting 900/1800 or the `"default"` sentinel.
  - Add or replace with:
    - `test_max_tokens_defaults_to_32000`: `Agent(FakeProvider())` sends a request whose `max_tokens == 32000` (FakeProvider has no `max_tokens`).
    - `test_provider_max_tokens_still_wins_over_default`: a provider with `max_tokens=2048` gives a request with 2048.
    - `test_no_turn_deadline_by_default`: `Agent(FakeProvider())._turn_timeout() is None`.
    - `test_turn_deadline_is_used_when_set`: `Agent(FakeProvider(), provider_turn_timeout=5)._turn_timeout() == 5`.
    - `test_default_sentinel_is_rejected`: `pytest.raises(LogposeError, match="positive number or None")` for `provider_turn_timeout="default"`.
    - `test_agent_model_overrides_missing_provider_model`: with `LOGPOSE_MODEL` cleared, `Agent("openai-compat", model="m", base_url="http://local.test/v1")._resolve_model() == "m"`.
    - `test_missing_model_error_names_logpose_model`: `Agent("openai-compat", base_url="http://local.test/v1")` with `LOGPOSE_MODEL` cleared; `run("hi")` raises `LogposeError` matching `"LOGPOSE_MODEL"`.

- [ ] **Step 2: Run them and confirm they fail.**
  Run: `uv run pytest tests/test_agent.py -q`
  Expected: the new tests FAIL.

- [ ] **Step 3: Implement in `agent.py`.**
  - The validation message becomes `"provider_turn_timeout must be a positive number or None; got {value!r}."`.
  - The `_resolve_model` error must mention `model=` and `LOGPOSE_MODEL`.
  - Update the docstrings that describe 16000, `"default"`, and provider recommendations.

- [ ] **Step 4: Run the agent tests and confirm they pass.**
  Run: `uv run pytest tests/test_agent.py -q`
  Expected: PASS.

- [ ] **Step 5: Commit.**
  ```bash
  git add src/logpose/agent.py tests/test_agent.py
  git commit -m "refactor(agent)!: 32000-token default, no implicit turn deadline"
  ```

### Task 3: Registry, catalog, and readiness

**Files:**
- Modify: `src/logpose/providers/__init__.py` (remove `_docker_factory`, `_DOCKER_INFO`, both docker registrations; update `_OPENAI_COMPAT_INFO`; `_probe`)
- Modify: `src/logpose/providers/catalog.py` (add `model_env` field and doc)
- Modify: `src/logpose/__init__.py` (docstring example using `"docker"`), `src/logpose/sync.py` (docstring examples using `"docker"`)
- Test: `tests/test_provider_catalog.py`, `tests/test_public_api.py`, `tests/test_sync.py`

**Interfaces:**
- Consumes: Task 1's `ENV_BASE_URL`, `ENV_MODEL`, `ENV_API_KEY` values (declare them as string literals in `__init__.py` so the catalog doesn't import the provider module).
- Produces:
  - `ProviderInfo.model_env: str | None = None`.
  - `_OPENAI_COMPAT_INFO` has `env_vars=("LOGPOSE_BASE_URL", "LOGPOSE_MODEL", "LOGPOSE_API_KEY")`, `base_url_env="LOGPOSE_BASE_URL"`, `model_env="LOGPOSE_MODEL"`, `credential="optional"`, `default_model=None`.
  - `known_providers() == ["openai-compat"]`.

- [ ] **Step 1: Update the tests.**
  - `tests/test_provider_catalog.py`:
    - `BUILT_INS = {"openai-compat"}`.
    - The autouse fixture clears `LOGPOSE_BASE_URL`, `LOGPOSE_MODEL`, `LOGPOSE_API_KEY`, `KEYED_TEST_KEY`.
    - Delete the alias and docker tests.
    - `test_the_credential_env_var_comes_first` asserts `provider_info("openai-compat").env_vars == ("LOGPOSE_BASE_URL", "LOGPOSE_MODEL", "LOGPOSE_API_KEY")`.
    - Add `test_a_missing_model_is_reported`: with only `LOGPOSE_BASE_URL` set, the status is not ready and `"LOGPOSE_MODEL"` is in `detail`.
    - Add `test_endpoint_and_model_make_it_ready`: with both set, the status is ready.
    - The end-to-end picker flow sets both env vars and expects `ready == {"openai-compat"}`.
    - Replace the every-provider-has-`list_models` loop with only `OpenAICompatProvider`.
  - `tests/test_public_api.py`: in the import-purity snippets, drop the `OPENAI_API_KEY` pop and keep the `openai_compat` / `httpx` checks.
  - `tests/test_sync.py`: no change is expected. Fix it if it references docker.

- [ ] **Step 2: Run them and confirm they fail.**
  Run: `uv run pytest tests/test_provider_catalog.py tests/test_public_api.py -q`
  Expected: FAIL (docker still registered, env names, no model check).

- [ ] **Step 3: Implement.** In `_probe`, after the `base_url_env` check, if `info.model_env` is set and its value is missing or blank, return not-ready with `detail=f"No model configured. Pass model=... or set ${info.model_env}."`. Change docstring examples from `Agent("docker", ...)` to `Agent("openai-compat", ...)`.

- [ ] **Step 4: Run the full suite and confirm it passes.**
  Run: `uv run pytest -q && uv run ruff check && uv run mypy src/`
  Expected: all clean.

- [ ] **Step 5: Commit.**
  ```bash
  git add src/logpose tests
  git commit -m "refactor(catalog)!: openai-compat only, LOGPOSE_* readiness"
  ```

### Task 4: Examples, docs, CI, changelog

**Files:**
- Modify: `examples/local_docker.py`, `examples/weather.py`, `README.md`, `CONTRIBUTING.md`, `CHANGELOG.md`, `.github/workflows/ci.yml`

- [ ] **Step 1: Update the examples.**
  - `examples/local_docker.py` uses `Agent("openai-compat", base_url="http://localhost:12434/engines/v1", model="auto", max_tokens=4096, ...)`, with a comment explaining why `max_tokens` is set (small local context).
  - `examples/weather.py` docstring uses the `LOGPOSE_BASE_URL` / `LOGPOSE_MODEL` / `LOGPOSE_API_KEY` exports, and the auth hint says `Check LOGPOSE_API_KEY.`

- [ ] **Step 2: Update README.md.**
  - Quickstart exports become `LOGPOSE_*`.
  - The provider table lists only `openai-compat`.
  - The Docker section becomes an `openai-compat` example with the explicit URL.
  - "Provider deadlines" says there's no deadline by default and shows `provider_turn_timeout=600`.
  - The configuration reference is a single `LOGPOSE_*` row.
  - Mention the 32000 default and the advice to lower it for small-context local models.
  - Remove `docker` and `OPENAI_*` everywhere.

- [ ] **Step 3: Update CONTRIBUTING.md and CI.**
  - CONTRIBUTING: remove the `DockerModelsProvider` and `docker` examples from "One provider per credential" and from the subclass pattern paragraph.
  - CI "resolves" step: `logpose.resolve('openai-compat', base_url='http://localhost:8000/v1')` only.

- [ ] **Step 4: Update CHANGELOG `[Unreleased]`.** Add **Breaking** bullets:
  - the `docker` provider and its alias are removed;
  - `OPENAI_*` / `DOCKER_*` env vars are replaced by `LOGPOSE_BASE_URL`, `LOGPOSE_MODEL`, `LOGPOSE_API_KEY`;
  - discovery only happens with `model="auto"`;
  - `Agent` defaults to `max_tokens=32000` and the provider's 4096 is removed;
  - no turn deadline or HTTP timeouts by default, and the `"default"` sentinel is removed;
  - `api_key_env` is removed;
  - `ProviderInfo.model_env` is added.

- [ ] **Step 5: Verify.**
  Run: `git grep -n -E 'OPENAI_(BASE_URL|MODEL|API_KEY)|DOCKER_|DockerModels|"docker"|docker-models|TURN_TIMEOUT' -- ':!CHANGELOG.md' ':!docs/superpowers'`
  Expected: no output.
  Run: `uv run pytest -q && uv run ruff check && uv run mypy src/ && env -u LOGPOSE_BASE_URL uv run python examples/weather.py`
  Expected: checks clean, and the example exits 1 with a message containing `LOGPOSE_BASE_URL`.

- [ ] **Step 6: Commit.**
  ```bash
  git add examples README.md CONTRIBUTING.md CHANGELOG.md .github/workflows/ci.yml
  git commit -m "docs: document neutral LOGPOSE_* configuration"
  ```
