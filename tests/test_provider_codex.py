"""Tests for the Codex provider — the ChatGPT subscription path.

Only what is specific to this backend: the endpoint, the subscription headers, the
Codex CLI identity shim, the account-id requirement, the ``max_output_tokens``
quirk, and the refusal to accept an API key. The wire format and streaming
machinery it shares with the ``openai`` provider are covered in
``test_responses_base.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from logpose import Agent, RawBlock, known_providers, resolve, tool
from logpose.auth import codex as codex_auth
from logpose.errors import AuthError
from logpose.providers.codex import (
    CHATGPT_BASE_URL,
    CODEX_CLI_IDENTITY,
    CODEX_ORIGINATOR,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    RESPONSES_BETA_HEADER,
    CodexProvider,
)
from logpose.tools import ToolDef
from tests.jwt_helpers import make_jwt
from tests.responses_helpers import (
    ACCOUNT_ID,
    API_KEY,
    OAUTH_TOKEN,
    Recorder,
    completed,
    drain,
    function_call_item,
    message_item,
    mock_client,
    reasoning_item,
    request,
    sse,
    summary_delta,
    text_delta,
)

BASE = "http://codex.test/v1"


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Keep the developer's Codex login and environment out of every test.

    Returns:
        An empty Codex home, which tests may populate to exercise discovery.
    """
    home = tmp_path / "codex-home"
    home.mkdir()
    for name in (
        codex_auth.ENV_API_KEY,
        codex_auth.ENV_ACCOUNT_ID,
        "CODEX_MODEL",
        "CODEX_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(codex_auth.ENV_CODEX_HOME, str(home))
    return home


def make_provider(handler: Any, **kwargs: Any) -> CodexProvider:
    """Build a provider wired to a MockTransport with an explicit subscription token."""
    kwargs.setdefault("base_url", BASE)
    kwargs.setdefault("auth_token", OAUTH_TOKEN)
    kwargs.setdefault("account_id", ACCOUNT_ID)
    return CodexProvider(client=mock_client(handler), **kwargs)


async def run_turn(handler: Any, req: Any = None, **kwargs: Any) -> list[Any]:
    """Build a provider, stream one turn, and return its events."""
    return await drain(make_provider(handler, **kwargs), req)


# ---------------------------------------------------------------------------
# subscription headers and endpoint
# ---------------------------------------------------------------------------


async def test_subscription_requests_send_bearer_account_id_and_the_beta_header() -> None:
    handler = Recorder()
    await run_turn(handler)
    headers = handler.last_headers
    assert headers["authorization"] == f"Bearer {OAUTH_TOKEN}"
    assert headers["chatgpt-account-id"] == ACCOUNT_ID
    assert headers["openai-beta"] == RESPONSES_BETA_HEADER


async def test_the_account_id_and_beta_headers_are_not_gated_on_the_shim() -> None:
    """They are the subscription wire protocol, not the identity shim."""
    handler = Recorder()
    await run_turn(handler, compat_codex_cli=False)
    assert handler.last_headers["chatgpt-account-id"] == ACCOUNT_ID
    assert handler.last_headers["openai-beta"] == RESPONSES_BETA_HEADER
    assert "originator" not in handler.last_headers


async def test_default_headers_are_merged_in() -> None:
    handler = Recorder()
    await run_turn(handler, default_headers={"session_id": "abc"})
    assert handler.last_headers["session_id"] == "abc"


def test_the_endpoint_is_the_subscription_backend() -> None:
    assert CodexProvider().base_url == CHATGPT_BASE_URL


async def test_an_explicit_base_url_overrides_the_default() -> None:
    handler = Recorder()
    await run_turn(handler)
    assert handler.last_url == f"{BASE}/responses"


async def test_the_base_url_env_var_is_honoured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODEX_BASE_URL", "http://from-env.test/v1/")
    handler = Recorder()
    provider = CodexProvider(
        client=mock_client(handler), auth_token=OAUTH_TOKEN, account_id=ACCOUNT_ID
    )
    await drain(provider, request())
    assert handler.last_url == "http://from-env.test/v1/responses"


# ---------------------------------------------------------------------------
# instructions and the CLI compatibility shim
# ---------------------------------------------------------------------------


async def test_the_identity_line_precedes_the_system_prompt() -> None:
    handler = Recorder()
    await run_turn(handler, request(system="Be concise."))
    assert handler.last_body["instructions"] == f"{CODEX_CLI_IDENTITY}\n\nBe concise."


async def test_the_originator_header_is_sent() -> None:
    handler = Recorder()
    await run_turn(handler)
    assert handler.last_headers["originator"] == CODEX_ORIGINATOR


async def test_the_shim_can_be_turned_off() -> None:
    handler = Recorder()
    await run_turn(handler, request(system="Be concise."), compat_codex_cli=False)
    assert handler.last_body["instructions"] == "Be concise."
    assert "originator" not in handler.last_headers


async def test_instructions_are_never_empty() -> None:
    """The subscription backend rejects an empty instructions field."""
    handler = Recorder()
    await run_turn(handler, compat_codex_cli=False)
    assert handler.last_body["instructions"]


async def test_the_instructions_argument_sits_between_identity_and_system() -> None:
    handler = Recorder()
    await run_turn(handler, request(system="Be concise."), instructions="House rules.")
    assert handler.last_body["instructions"] == (
        f"{CODEX_CLI_IDENTITY}\n\nHouse rules.\n\nBe concise."
    )


# ---------------------------------------------------------------------------
# max_output_tokens
# ---------------------------------------------------------------------------


async def test_max_output_tokens_is_never_sent() -> None:
    """Verified live: the backend answers 400 Unsupported parameter for this field."""
    handler = Recorder()
    await run_turn(handler, request(max_tokens=999))
    assert "max_output_tokens" not in handler.last_body


def test_the_provider_declares_that_it_omits_the_field() -> None:
    assert CodexProvider.SENDS_MAX_OUTPUT_TOKENS is False


# ---------------------------------------------------------------------------
# credential policy
# ---------------------------------------------------------------------------


async def test_an_api_key_is_refused_and_names_the_openai_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole point of the split: this provider cannot be fed an API key."""
    monkeypatch.setenv(codex_auth.ENV_API_KEY, API_KEY)
    provider = CodexProvider(client=mock_client(Recorder()), base_url=BASE)
    with pytest.raises(AuthError) as excinfo:
        await drain(provider, request())
    message = str(excinfo.value)
    assert 'Agent("openai")' in message
    assert "codex login" in message
    assert API_KEY not in message


async def test_a_stored_subscription_token_is_discovered(isolated_env: Path) -> None:
    token = make_jwt({"exp": 4_000_000_000})
    (isolated_env / "auth.json").write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "tokens": {"access_token": token, "account_id": "acc_stored"},
            }
        )
    )
    handler = Recorder()
    provider = CodexProvider(client=mock_client(handler), base_url=BASE)
    await drain(provider, request())
    assert handler.last_headers["authorization"] == f"Bearer {token}"
    assert handler.last_headers["chatgpt-account-id"] == "acc_stored"


async def test_a_subscription_credential_without_an_account_id_fails_before_sending() -> None:
    handler = Recorder()
    provider = CodexProvider(client=mock_client(handler), base_url=BASE, auth_token=OAUTH_TOKEN)
    with pytest.raises(AuthError) as excinfo:
        await drain(provider, request())
    assert "codex login" in str(excinfo.value)
    assert handler.requests == []


async def test_the_account_id_argument_overrides_the_store(isolated_env: Path) -> None:
    (isolated_env / "auth.json").write_text(
        json.dumps({"tokens": {"access_token": "tok", "account_id": "acc_stored"}})
    )
    handler = Recorder()
    provider = CodexProvider(client=mock_client(handler), base_url=BASE, account_id="acc_explicit")
    await drain(provider, request())
    assert handler.last_headers["chatgpt-account-id"] == "acc_explicit"


def test_construction_never_raises_auth_error_and_reads_no_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def explode(*args: object, **kwargs: object) -> None:
        raise AssertionError("construction must not resolve a credential")

    monkeypatch.setattr(codex_auth.CredentialProvider, "resolve", explode)
    monkeypatch.setattr(codex_auth, "load_stored_credential", explode)
    provider = CodexProvider()
    assert provider.model_default == DEFAULT_MODEL
    assert provider.max_tokens == DEFAULT_MAX_TOKENS


async def test_the_auth_token_never_appears_in_an_error_body() -> None:
    from logpose.errors import ProviderError

    with pytest.raises(ProviderError) as excinfo:
        await run_turn(Recorder(status=401, text=f"bad token {OAUTH_TOKEN}"))
    assert OAUTH_TOKEN not in str(excinfo.value)
    assert "<redacted" in str(excinfo.value)


async def test_the_repr_redacts_every_credential() -> None:
    provider = make_provider(Recorder())
    await drain(provider, request())
    text = repr(provider)
    assert OAUTH_TOKEN not in text
    assert ACCOUNT_ID not in text


# ---------------------------------------------------------------------------
# defaults, registration, and the loop
# ---------------------------------------------------------------------------


def test_defaults() -> None:
    provider = CodexProvider()
    assert provider.name == "codex"
    assert provider.model_default == DEFAULT_MODEL
    assert provider.max_tokens == DEFAULT_MAX_TOKENS
    assert provider.compat_codex_cli is True


def test_the_model_env_var_is_honoured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODEX_MODEL", "gpt-5.4")
    assert CodexProvider().model_default == "gpt-5.4"


def test_an_explicit_model_beats_the_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODEX_MODEL", "gpt-5.4")
    assert CodexProvider(model="gpt-5.4-mini").model_default == "gpt-5.4-mini"


def test_codex_is_registered_and_resolvable() -> None:
    assert "codex" in known_providers()
    assert isinstance(resolve("codex", auth_token=OAUTH_TOKEN), CodexProvider)


def test_importing_logpose_does_not_import_the_codex_provider() -> None:
    import subprocess
    import sys

    code = (
        "import sys, logpose; "
        "assert 'logpose.providers.codex' not in sys.modules; "
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
    return f"22C and sunny in {city}"


async def test_a_full_tool_round_trip_through_the_agent() -> None:
    """The turn that matters: reasoning, a call, then a second turn resending both."""
    handler = Recorder(
        sse(
            summary_delta("Need the weather."),
            completed(output=[reasoning_item(), function_call_item()]),
        ),
        sse(
            text_delta("It is sunny."),
            completed(
                output=[message_item("It is sunny.")],
                usage={"input_tokens": 20, "output_tokens": 5},
            ),
        ),
    )
    assert isinstance(get_weather, ToolDef)
    agent = Agent(make_provider(handler), tools=[get_weather], system="Be concise.")
    result = await agent.run("What's the weather in Pune?")

    assert result.text == "It is sunny."
    assert result.iterations == 2
    assert result.stop_reason == "end_turn"

    # The second request must carry the reasoning item verbatim, immediately
    # before the function_call it reasoned for, then the tool output.
    second = handler.body(1)["input"]
    assert [item["type"] for item in second] == [
        "message",
        "reasoning",
        "function_call",
        "function_call_output",
    ]
    assert second[1] == reasoning_item()
    assert second[3] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": "22C and sunny in Pune",
    }
    blocks = [block for message in result.messages for block in message.content]
    assert any(isinstance(block, RawBlock) for block in blocks)


async def test_the_agent_streams_thinking_and_text_events() -> None:
    from logpose import TextDelta, ThinkingDelta

    handler = Recorder(
        sse(
            summary_delta("Hmm."),
            text_delta("Hi."),
            completed(output=[reasoning_item(), message_item("Hi.")]),
        )
    )
    agent = Agent(make_provider(handler))
    events = [event async for event in agent.stream("hello")]
    assert any(isinstance(e, ThinkingDelta) and e.text == "Hmm." for e in events)
    assert any(isinstance(e, TextDelta) and e.text == "Hi." for e in events)
