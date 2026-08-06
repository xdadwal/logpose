"""Tests for the OpenAI provider — the Responses API on an API key.

Only what is specific to this backend: the endpoint, the plain bearer header, the
absence of any Codex CLI dress-up, ``max_output_tokens`` actually being sent, and
the refusal to accept a subscription token. The wire format and streaming
machinery it shares with the ``codex`` provider are covered in
``test_responses_base.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from logpose import Agent, known_providers, resolve, tool
from logpose.auth import codex as codex_auth
from logpose.errors import AuthError
from logpose.providers.openai import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    OPENAI_BASE_URL,
    OpenAIProvider,
)
from logpose.tools import ToolDef
from tests.jwt_helpers import make_jwt
from tests.responses_helpers import (
    API_KEY,
    OAUTH_TOKEN,
    Recorder,
    completed,
    drain,
    function_call_item,
    message_item,
    mock_client,
    request,
    sse,
    text_delta,
)

BASE = "http://openai.test/v1"


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Keep the developer's credentials and environment out of every test.

    Returns:
        An empty Codex home, which tests may populate to exercise discovery.
    """
    home = tmp_path / "codex-home"
    home.mkdir()
    for name in (
        codex_auth.ENV_API_KEY,
        codex_auth.ENV_ACCOUNT_ID,
        "OPENAI_RESPONSES_MODEL",
        "OPENAI_RESPONSES_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(codex_auth.ENV_CODEX_HOME, str(home))
    return home


def make_provider(handler: Any, **kwargs: Any) -> OpenAIProvider:
    """Build a provider wired to a MockTransport with an explicit API key."""
    kwargs.setdefault("base_url", BASE)
    kwargs.setdefault("api_key", API_KEY)
    return OpenAIProvider(client=mock_client(handler), **kwargs)


async def run_turn(handler: Any, req: Any = None, **kwargs: Any) -> list[Any]:
    """Build a provider, stream one turn, and return its events."""
    return await drain(make_provider(handler, **kwargs), req)


# ---------------------------------------------------------------------------
# headers and endpoint
# ---------------------------------------------------------------------------


async def test_requests_send_only_a_bearer_token() -> None:
    handler = Recorder()
    await run_turn(handler)
    headers = handler.last_headers
    assert headers["authorization"] == f"Bearer {API_KEY}"
    assert "chatgpt-account-id" not in headers
    assert "openai-beta" not in headers
    assert "originator" not in headers


async def test_no_identity_line_is_prepended() -> None:
    """The public API is not the Codex CLI and must not pretend to be."""
    handler = Recorder()
    await run_turn(handler, request(system="Be concise."))
    assert handler.last_body["instructions"] == "Be concise."


async def test_instructions_are_omitted_entirely_when_there_is_no_prompt() -> None:
    handler = Recorder()
    await run_turn(handler)
    assert "instructions" not in handler.last_body


async def test_the_instructions_argument_precedes_the_system_prompt() -> None:
    handler = Recorder()
    await run_turn(handler, request(system="Be concise."), instructions="House rules.")
    assert handler.last_body["instructions"] == "House rules.\n\nBe concise."


def test_the_endpoint_is_the_public_api() -> None:
    assert OpenAIProvider().base_url == OPENAI_BASE_URL


async def test_an_explicit_base_url_overrides_the_default() -> None:
    handler = Recorder()
    await run_turn(handler)
    assert handler.last_url == f"{BASE}/responses"


async def test_the_base_url_env_var_is_honoured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_RESPONSES_BASE_URL", "http://from-env.test/v1/")
    handler = Recorder()
    provider = OpenAIProvider(client=mock_client(handler), api_key=API_KEY)
    await drain(provider, request())
    assert handler.last_url == "http://from-env.test/v1/responses"


# ---------------------------------------------------------------------------
# max_output_tokens
# ---------------------------------------------------------------------------


async def test_max_tokens_becomes_max_output_tokens() -> None:
    """Unlike the subscription endpoint, this one accepts the field."""
    handler = Recorder()
    await run_turn(handler, request(max_tokens=999))
    assert handler.last_body["max_output_tokens"] == 999


async def test_the_provider_default_applies_when_the_request_omits_one() -> None:
    handler = Recorder()
    await run_turn(handler, request(max_tokens=0), max_tokens=4321)
    assert handler.last_body["max_output_tokens"] == 4321


def test_the_provider_declares_that_it_sends_the_field() -> None:
    assert OpenAIProvider.SENDS_MAX_OUTPUT_TOKENS is True


# ---------------------------------------------------------------------------
# credential policy
# ---------------------------------------------------------------------------


async def test_a_stored_subscription_token_is_refused_and_names_the_codex_provider(
    isolated_env: Path,
) -> None:
    """The whole point of the split: a subscription cannot satisfy this provider."""
    (isolated_env / "auth.json").write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "tokens": {
                    "access_token": make_jwt({"exp": 4_000_000_000}),
                    "account_id": "acc_stored",
                },
            }
        )
    )
    provider = OpenAIProvider(client=mock_client(Recorder()), base_url=BASE)
    with pytest.raises(AuthError) as excinfo:
        await drain(provider, request())
    message = str(excinfo.value)
    assert 'Agent("codex")' in message
    assert codex_auth.ENV_API_KEY in message


async def test_a_subscription_token_in_the_store_does_not_shadow_the_env_key(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reason require_kind filters rather than resolve-then-reject.

    A subscription token sitting in the store must not hide a perfectly good
    ``$OPENAI_API_KEY`` and get this provider refused next to a usable credential.
    """
    (isolated_env / "auth.json").write_text(
        json.dumps({"tokens": {"access_token": "subscription-token", "account_id": "acc"}})
    )
    monkeypatch.setenv(codex_auth.ENV_API_KEY, API_KEY)
    handler = Recorder()
    provider = OpenAIProvider(client=mock_client(handler), base_url=BASE)
    await drain(provider, request())
    assert handler.last_headers["authorization"] == f"Bearer {API_KEY}"


async def test_the_env_api_key_is_used_when_no_explicit_one_is_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(codex_auth.ENV_API_KEY, API_KEY)
    handler = Recorder()
    provider = OpenAIProvider(client=mock_client(handler), base_url=BASE)
    await drain(provider, request())
    assert handler.last_headers["authorization"] == f"Bearer {API_KEY}"


async def test_an_api_key_written_into_auth_json_is_found(isolated_env: Path) -> None:
    """`codex login` in api-key mode writes it there, so it is a real source."""
    (isolated_env / "auth.json").write_text(json.dumps({"OPENAI_API_KEY": API_KEY}))
    handler = Recorder()
    provider = OpenAIProvider(client=mock_client(handler), base_url=BASE)
    await drain(provider, request())
    assert handler.last_headers["authorization"] == f"Bearer {API_KEY}"


async def test_nothing_available_raises_naming_the_env_var() -> None:
    provider = OpenAIProvider(client=mock_client(Recorder()), base_url=BASE)
    with pytest.raises(AuthError) as excinfo:
        await drain(provider, request())
    assert codex_auth.ENV_API_KEY in str(excinfo.value)


def test_construction_never_raises_auth_error_and_reads_no_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def explode(*args: object, **kwargs: object) -> None:
        raise AssertionError("construction must not resolve a credential")

    monkeypatch.setattr(codex_auth.CredentialProvider, "resolve", explode)
    monkeypatch.setattr(codex_auth, "load_stored_credential", explode)
    provider = OpenAIProvider()
    assert provider.model_default == DEFAULT_MODEL
    assert provider.max_tokens == DEFAULT_MAX_TOKENS


async def test_a_subscription_token_passed_as_an_api_key_is_still_only_a_bearer() -> None:
    """No auth_token= parameter exists here, so there is no slot to confuse."""
    assert "auth_token" not in OpenAIProvider.__init__.__code__.co_varnames


# ---------------------------------------------------------------------------
# defaults, registration, and the loop
# ---------------------------------------------------------------------------


def test_defaults() -> None:
    provider = OpenAIProvider()
    assert provider.name == "openai"
    assert provider.model_default == DEFAULT_MODEL
    assert provider.max_tokens == DEFAULT_MAX_TOKENS


def test_the_model_default_differs_from_the_codex_one() -> None:
    """ChatGPT-backend slugs and public API model ids are separate namespaces."""
    from logpose.providers.codex import DEFAULT_MODEL as CODEX_MODEL

    assert DEFAULT_MODEL != CODEX_MODEL


def test_the_model_env_var_is_honoured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_RESPONSES_MODEL", "gpt-5.1-mini")
    assert OpenAIProvider().model_default == "gpt-5.1-mini"


def test_the_chat_completions_env_vars_are_not_read(monkeypatch: pytest.MonkeyPatch) -> None:
    """$OPENAI_MODEL belongs to openai-compat; the two must not collide."""
    monkeypatch.setenv("OPENAI_MODEL", "should-be-ignored")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://should-be-ignored.test")
    provider = OpenAIProvider()
    assert provider.model_default == DEFAULT_MODEL
    assert provider.base_url == OPENAI_BASE_URL


def test_openai_is_registered_and_resolvable() -> None:
    assert "openai" in known_providers()
    assert isinstance(resolve("openai", api_key=API_KEY), OpenAIProvider)


def test_openai_and_openai_compat_are_different_providers() -> None:
    from logpose.providers.openai_compat import OpenAICompatProvider

    compat = resolve("openai-compat", base_url="http://x.test/v1", model="m")
    assert isinstance(compat, OpenAICompatProvider)
    assert not isinstance(compat, OpenAIProvider)


def test_importing_logpose_does_not_import_the_openai_provider() -> None:
    import subprocess
    import sys

    code = (
        "import sys, logpose; "
        "assert 'logpose.providers.openai' not in sys.modules; "
        "assert 'httpx' not in sys.modules; "
        "print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False, timeout=60
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


@tool
def get_weather(city: str) -> str:
    """Get the current weather.

    Args:
        city: City name.
    """
    return f"18C and raining in {city}"


async def test_a_full_tool_round_trip_through_the_agent() -> None:
    handler = Recorder(
        sse(completed(output=[function_call_item()])),
        sse(
            text_delta("It is raining."),
            completed(output=[message_item("It is raining.")]),
        ),
    )
    assert isinstance(get_weather, ToolDef)
    agent = Agent(make_provider(handler), tools=[get_weather])
    result = await agent.run("Weather in Pune?")
    assert result.text == "It is raining."
    assert result.iterations == 2
    assert handler.body(1)["input"][-1] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": "18C and raining in Pune",
    }


async def test_the_unused_oauth_token_constant_is_not_accepted() -> None:
    """Sanity: OAUTH_TOKEN is a subscription value and has no place in this suite."""
    handler = Recorder()
    provider = OpenAIProvider(client=mock_client(handler), base_url=BASE, api_key=OAUTH_TOKEN)
    await drain(provider, request())
    # It is sent as a bearer token like any other string; the provider does not
    # inspect it. The point is that no subscription *machinery* engages.
    assert handler.last_headers["authorization"] == f"Bearer {OAUTH_TOKEN}"
    assert "chatgpt-account-id" not in handler.last_headers
