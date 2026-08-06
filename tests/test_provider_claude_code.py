"""Tests for the Claude Code provider — the Anthropic subscription path.

Only what is specific to this backend: the OAuth bearer header and beta header,
the identity line, subscription-token refresh, and the refusal to accept an API
key. The wire format and streaming machinery it shares with the ``anthropic``
provider are covered in ``test_provider_anthropic.py``.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest
from anthropic._models import FinalRequestOptions

from logpose.auth import claude_code
from logpose.errors import AuthError, ProviderError
from logpose.providers import resolve
from logpose.providers.claude_code import (
    CLAUDE_CODE_IDENTITY,
    OAUTH_BETA_HEADER,
    ClaudeCodeProvider,
)
from tests.anthropic_helpers import FakeClient, drain, sdk_message, simple_request

OAUTH_TOKEN = "sk-ant-oat01-subscription-token"
API_KEY = "sk-ant-api03-byok-key"


@pytest.fixture(autouse=True)
def _isolate_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep ambient credentials out of every test by default."""
    for name in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "ANTHROPIC_PROFILE",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture()
def no_stored_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hide the developer's real Claude Code credential store from the resolver."""
    monkeypatch.setattr(claude_code, "load_stored_credential", lambda: None)


def make_provider(client: FakeClient, **kwargs: Any) -> ClaudeCodeProvider:
    """Build a subscription provider around a fake SDK client."""
    return ClaudeCodeProvider(client=client, **kwargs)  # type: ignore[arg-type]


def built_headers(client: Any) -> dict[str, str]:
    """Headers the SDK would actually put on a /v1/messages request."""
    request = client._build_request(
        FinalRequestOptions(method="post", url="/v1/messages", json_data={})
    )
    return {key.lower(): value for key, value in request.headers.items()}


# ---------------------------------------------------------------------------
# protocol and registration
# ---------------------------------------------------------------------------


def test_registry_resolves_to_this_class() -> None:
    assert isinstance(resolve("claude-code", auth_token=OAUTH_TOKEN), ClaudeCodeProvider)


def test_the_provider_is_named_for_the_cli_it_impersonates() -> None:
    assert ClaudeCodeProvider.name == "claude-code"


def test_repr_never_carries_a_credential() -> None:
    provider = ClaudeCodeProvider(auth_token=OAUTH_TOKEN)
    text = repr(provider)
    assert OAUTH_TOKEN not in text
    assert "compat_claude_code=True" in text


# ---------------------------------------------------------------------------
# the identity line
# ---------------------------------------------------------------------------


async def test_the_identity_line_precedes_the_caller_system_prompt() -> None:
    # A bare subscription request (no identity line) is answered with HTTP 429
    # rate_limit_error even when the account has quota left, so the line is not
    # optional in practice.
    client = FakeClient(final=sdk_message([]))
    await drain(make_provider(client), simple_request(system="You are terse."))
    assert client.messages.calls[0]["system"] == [
        {"type": "text", "text": CLAUDE_CODE_IDENTITY},
        {"type": "text", "text": "You are terse."},
    ]


async def test_the_identity_is_sent_without_a_caller_system_prompt() -> None:
    client = FakeClient(final=sdk_message([]))
    await drain(make_provider(client), simple_request())
    assert client.messages.calls[0]["system"] == [{"type": "text", "text": CLAUDE_CODE_IDENTITY}]


async def test_compat_claude_code_false_suppresses_the_identity() -> None:
    client = FakeClient(final=sdk_message([]))
    await drain(
        make_provider(client, compat_claude_code=False),
        simple_request(system="You are terse."),
    )
    assert client.messages.calls[0]["system"] == "You are terse."


async def test_compat_claude_code_false_with_no_prompt_sends_no_system() -> None:
    client = FakeClient(final=sdk_message([]))
    await drain(make_provider(client, compat_claude_code=False), simple_request())
    assert "system" not in client.messages.calls[0]


async def test_the_identity_does_not_depend_on_inspecting_the_client() -> None:
    """The old provider guessed from client.auth_token; this one simply knows."""
    client = FakeClient(final=sdk_message([]), api_key="looks-like-an-api-key")
    await drain(make_provider(client), simple_request())
    assert client.messages.calls[0]["system"] == [{"type": "text", "text": CLAUDE_CODE_IDENTITY}]


# ---------------------------------------------------------------------------
# auth headers
# ---------------------------------------------------------------------------


async def test_only_a_bearer_is_sent_even_with_ANTHROPIC_API_KEY_set(
    monkeypatch: pytest.MonkeyPatch,
    no_stored_credential: None,
) -> None:
    # The SDK reads ANTHROPIC_API_KEY when api_key is not supplied; if that
    # leaked through, both auth headers would go out and the API would 401.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api-env-key-must-not-leak")
    provider = ClaudeCodeProvider(auth_token=OAUTH_TOKEN)

    client = await provider.get_client()
    headers = built_headers(client)

    assert headers["authorization"] == f"Bearer {OAUTH_TOKEN}"
    assert "x-api-key" not in headers
    assert headers["anthropic-beta"] == OAUTH_BETA_HEADER
    assert client.api_key is None
    assert set(client.auth_headers) == {"Authorization"}
    await provider.aclose()


async def test_the_client_is_cached_across_turns(no_stored_credential: None) -> None:
    provider = ClaudeCodeProvider(auth_token=OAUTH_TOKEN)
    first = await provider.get_client()
    second = await provider.get_client()
    assert first is second
    await provider.aclose()


# ---------------------------------------------------------------------------
# credential policy
# ---------------------------------------------------------------------------


async def test_an_api_key_is_refused_and_names_the_anthropic_provider(
    monkeypatch: pytest.MonkeyPatch, no_stored_credential: None
) -> None:
    """The whole point of the split: this provider cannot be fed an API key."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", API_KEY)
    provider = ClaudeCodeProvider()
    with pytest.raises(AuthError) as excinfo:
        await provider.get_client()
    message = str(excinfo.value)
    assert 'Agent("anthropic")' in message
    assert "claude setup-token" in message
    assert API_KEY not in message


async def test_the_env_token_is_used_when_nothing_explicit(
    monkeypatch: pytest.MonkeyPatch, no_stored_credential: None
) -> None:
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-from-env")
    provider = ClaudeCodeProvider()
    client = await provider.get_client()
    assert client.auth_token == "sk-ant-oat01-from-env"
    assert client.api_key is None
    await provider.aclose()


async def test_an_ambient_api_key_neither_satisfies_nor_shadows_the_env_token(
    monkeypatch: pytest.MonkeyPatch, no_stored_credential: None
) -> None:
    """An ANTHROPIC_API_KEY left over from other tooling must be invisible here."""
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-from-env")
    monkeypatch.setenv("ANTHROPIC_API_KEY", API_KEY)
    provider = ClaudeCodeProvider()
    client = await provider.get_client()
    assert client.auth_token == "sk-ant-oat01-from-env"
    assert client.api_key is None
    await provider.aclose()


async def test_auth_error_when_no_subscription_anywhere(no_stored_credential: None) -> None:
    provider = ClaudeCodeProvider()
    with pytest.raises(AuthError, match="No Claude Code subscription credential found"):
        await provider.get_client()


async def test_constructing_the_provider_never_raises_auth_error(
    no_stored_credential: None,
) -> None:
    ClaudeCodeProvider()  # lazy: resolution is deferred to the first request


async def test_a_refresh_reaches_the_wire_without_a_new_connection_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An expiring subscription token is refreshed and swapped in, in place."""
    stale = claude_code.Credential(
        kind="oauth",
        value="sk-ant-oat01-stale",
        expires_at=time.time() - 1,
        refresh_token="rt_1",
    )
    fresh = claude_code.Credential(
        kind="oauth",
        value="sk-ant-oat01-refreshed",
        expires_at=time.time() + 3600,
    )
    refreshes = 0

    async def refresher(credential: claude_code.Credential) -> claude_code.Credential:
        nonlocal refreshes
        refreshes += 1
        return fresh

    credentials = claude_code.CredentialProvider(stale, refresher=refresher)
    monkeypatch.setattr(
        claude_code.CredentialProvider,
        "resolve",
        classmethod(lambda cls, *args, **kwargs: credentials),
    )
    provider = ClaudeCodeProvider()

    first = await provider.get_client()
    second = await provider.get_client()

    assert refreshes == 1
    assert first is second, "a refreshed token must not cost a new connection pool"
    assert second.auth_token == "sk-ant-oat01-refreshed"
    assert second.api_key is None
    await provider.aclose()


async def test_concurrent_first_requests_resolve_and_refresh_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: the lazy build of the CredentialProvider was unguarded.

    N concurrent first requests each saw ``self._credentials is None``, each
    built its own provider — and therefore its own lock — so the single-flight
    re-check inside ``CredentialProvider.get`` never saw the others. On an
    already-expired subscription token that meant N keychain subprocesses and N
    refresh grants replaying the same single-use refresh token, so all but one
    came back ``invalid_grant``.
    """
    resolves = 0
    refreshes = 0

    async def refresher(credential: claude_code.Credential) -> claude_code.Credential:
        nonlocal refreshes
        refreshes += 1
        await asyncio.sleep(0.01)
        return claude_code.Credential(
            kind="oauth",
            value="sk-ant-oat01-refreshed",
            expires_at=time.time() + 3600,
            refresh_token="rt_2",
        )

    def resolve_stub(
        cls: object, *args: object, **kwargs: object
    ) -> claude_code.CredentialProvider:
        nonlocal resolves
        resolves += 1
        time.sleep(0.02)  # the keychain subprocess, which is why resolve() is threaded
        return claude_code.CredentialProvider(
            claude_code.Credential(
                kind="oauth",
                value="sk-ant-oat01-expired",
                expires_at=time.time() - 1,
                refresh_token="rt_1",
            ),
            refresher=refresher,
        )

    monkeypatch.setattr(
        claude_code.CredentialProvider, "resolve", classmethod(resolve_stub)
    )
    provider = ClaudeCodeProvider()

    clients = await asyncio.gather(*(provider.get_client() for _ in range(10)))

    assert resolves == 1
    assert refreshes == 1
    assert len({id(client) for client in clients}) == 1
    assert clients[0].auth_token == "sk-ant-oat01-refreshed"
    await provider.aclose()


async def test_the_subscription_token_never_appears_in_an_error(
    no_stored_credential: None,
) -> None:
    """A gateway that echoes the token back must not leak it through the wrapper."""
    import anthropic
    import httpx

    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(401, request=request, text=f"bad token {OAUTH_TOKEN}")
    error = anthropic.AuthenticationError(
        f"bad token {OAUTH_TOKEN}", response=response, body=None
    )
    client = FakeClient(error=error, auth_token=OAUTH_TOKEN)
    provider = make_provider(client)

    with pytest.raises(ProviderError) as excinfo:
        await drain(provider, simple_request())

    message = str(excinfo.value)
    assert OAUTH_TOKEN not in message
    assert "<redacted" in message
    # The chained cause is rendered in every traceback, so it must be scrubbed too.
    assert OAUTH_TOKEN not in str(excinfo.value.__cause__)


async def test_a_401_names_the_sibling_provider(no_stored_credential: None) -> None:
    import anthropic
    import httpx

    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(401, request=request, text="unauthorized")
    error = anthropic.AuthenticationError("unauthorized", response=response, body=None)
    provider = make_provider(FakeClient(error=error, auth_token=OAUTH_TOKEN))

    with pytest.raises(ProviderError) as excinfo:
        await drain(provider, simple_request())
    assert 'Agent("anthropic")' in str(excinfo.value)
