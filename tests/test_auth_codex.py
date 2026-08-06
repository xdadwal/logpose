"""Tests for Codex / ChatGPT credential discovery, expiry, and refresh.

Nothing here touches the network, the real ``~/.codex``, or the developer's own
credentials: ``$CODEX_HOME`` is redirected at a temp directory for every test and
every HTTP call goes through ``httpx.MockTransport``.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from logpose.auth import codex
from logpose.auth._common import Credential
from logpose.errors import AuthError
from tests.jwt_helpers import make_jwt

# Long, obviously-secret values. The redaction prefix is 13 chars, so the marker
# text never legitimately appears in a repr.
FUTURE_EXP = 4_000_000_000
ACCESS_TOKEN = make_jwt({"exp": FUTURE_EXP, "sub": "S3CR3T" * 20})
REFRESH_TOKEN = "rt-" + "R3FR3SH" * 20
API_KEY = "sk-proj-" + "K3YV4L" * 20
ACCOUNT_ID = "acc_from_tokens"
ID_TOKEN_ACCOUNT_ID = "acc_from_id_token"
ID_TOKEN = make_jwt(
    {codex.ACCOUNT_ID_CLAIM_NAMESPACE: {"chatgpt_account_id": ID_TOKEN_ACCOUNT_ID}}
)


@pytest.fixture(autouse=True)
def isolated_codex_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Isolate credential discovery from the machine running the tests.

    Points ``$CODEX_HOME`` at an empty temp directory and clears every credential
    environment variable, so a developer's real Codex login can never satisfy a
    test that is supposed to find nothing.

    Returns:
        The temp Codex home.
    """
    home = tmp_path / "codex-home"
    home.mkdir()
    monkeypatch.setenv(codex.ENV_CODEX_HOME, str(home))
    monkeypatch.delenv(codex.ENV_API_KEY, raising=False)
    monkeypatch.delenv(codex.ENV_ACCOUNT_ID, raising=False)
    return home


def write_auth(home: Path, payload: object) -> Path:
    """Write an ``auth.json`` for the tests to discover.

    Args:
        home: The isolated Codex home.
        payload: Either a str (written verbatim) or an object to JSON-encode.

    Returns:
        The path written.
    """
    path = home / "auth.json"
    text = payload if isinstance(payload, str) else json.dumps(payload)
    path.write_text(text, encoding="utf-8")
    return path


def auth_payload(
    *,
    access_token: str | None = ACCESS_TOKEN,
    refresh_token: str | None = REFRESH_TOKEN,
    id_token: str | None = ID_TOKEN,
    account_id: str | None = ACCOUNT_ID,
    api_key: str | None = None,
    auth_mode: str = "chatgpt",
) -> dict[str, Any]:
    """Build an ``auth.json`` payload, omitting whatever is passed as ``None``."""
    tokens: dict[str, Any] = {}
    if access_token is not None:
        tokens["access_token"] = access_token
    if refresh_token is not None:
        tokens["refresh_token"] = refresh_token
    if id_token is not None:
        tokens["id_token"] = id_token
    if account_id is not None:
        tokens["account_id"] = account_id
    payload: dict[str, Any] = {"auth_mode": auth_mode, "last_refresh": "2026-08-06T09:49:00Z"}
    payload[codex.ENV_API_KEY] = api_key
    if tokens:
        payload["tokens"] = tokens
    return payload


def mock_client(handler: Any) -> httpx.AsyncClient:
    """Build an AsyncClient whose transport never touches the network."""
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# ---------------------------------------------------------------------------
# redaction
# ---------------------------------------------------------------------------


def test_repr_redacts_the_access_token() -> None:
    credential = Credential(kind="oauth", value=ACCESS_TOKEN, account_id=ACCOUNT_ID)
    text = repr(credential)
    assert ACCESS_TOKEN not in text
    assert "S3CR3T" not in text
    assert f"len={len(ACCESS_TOKEN)}" in text


def test_repr_shape_is_unchanged_by_the_account_id_field() -> None:
    """Adding account_id must not alter the repr — it identifies an account."""
    without = repr(Credential(kind="oauth", value=ACCESS_TOKEN))
    with_id = repr(Credential(kind="oauth", value=ACCESS_TOKEN, account_id=ACCOUNT_ID))
    assert without == with_id
    assert ACCOUNT_ID not in with_id


def test_credential_equality_ignores_an_absent_account_id() -> None:
    """The four-field constructions in the Anthropic suite must keep comparing equal."""
    assert Credential(kind="oauth", value="v") == Credential(
        kind="oauth", value="v", account_id=None
    )


def test_provider_repr_redacts_and_names_the_skew() -> None:
    provider = codex.CredentialProvider(Credential(kind="oauth", value=ACCESS_TOKEN))
    text = repr(provider)
    assert ACCESS_TOKEN not in text
    assert "skew=" in text


# ---------------------------------------------------------------------------
# JWT parsing
# ---------------------------------------------------------------------------


def test_expiry_comes_from_the_access_token_exp_claim(isolated_codex_env: Path) -> None:
    write_auth(isolated_codex_env, auth_payload())
    credential = codex.load_stored_credential()
    assert credential is not None
    assert credential.expires_at == float(FUTURE_EXP)


def test_unpadded_base64url_payload_decodes(isolated_codex_env: Path) -> None:
    """Real JWTs strip '=' padding; make_jwt does too, so this is the live path."""
    token = make_jwt({"exp": FUTURE_EXP, "pad": "a"})
    assert "=" not in token
    write_auth(isolated_codex_env, auth_payload(access_token=token))
    credential = codex.load_stored_credential()
    assert credential is not None
    assert credential.expires_at == float(FUTURE_EXP)


def test_account_id_falls_back_to_the_id_token_claim(isolated_codex_env: Path) -> None:
    write_auth(isolated_codex_env, auth_payload(account_id=None))
    credential = codex.load_stored_credential()
    assert credential is not None
    assert credential.account_id == ID_TOKEN_ACCOUNT_ID


def test_the_stored_account_id_outranks_the_id_token_claim(
    isolated_codex_env: Path,
) -> None:
    write_auth(isolated_codex_env, auth_payload())
    credential = codex.load_stored_credential()
    assert credential is not None
    assert credential.account_id == ACCOUNT_ID


def test_the_account_id_env_var_is_the_last_resort(
    isolated_codex_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(codex.ENV_ACCOUNT_ID, "acc_from_env")
    write_auth(isolated_codex_env, auth_payload(account_id=None, id_token=None))
    credential = codex.resolve_credential()
    assert credential.account_id == "acc_from_env"


def test_an_explicit_account_id_outranks_everything(isolated_codex_env: Path) -> None:
    write_auth(isolated_codex_env, auth_payload())
    credential = codex.resolve_credential(explicit_account_id="acc_explicit")
    assert credential.account_id == "acc_explicit"


@pytest.mark.parametrize(
    "token",
    [
        "not-a-jwt",
        "a.b",
        "a.b.c.d",
        "a.!!!not-base64!!!.c",
        "a..c",
        make_jwt({"no_exp": True}),
        make_jwt({"exp": "not a number"}),
        make_jwt({"exp": None}),
    ],
)
def test_a_malformed_or_expiryless_token_degrades_to_no_expiry(
    isolated_codex_env: Path, token: str
) -> None:
    write_auth(isolated_codex_env, auth_payload(access_token=token))
    credential = codex.load_stored_credential()
    assert credential is not None
    assert credential.value == token
    assert credential.expires_at is None
    # No expiry known means "usable", not "expired" — a token we cannot date must
    # still be tried rather than pre-emptively refused.
    assert credential.is_expired() is False


def test_a_non_object_jwt_payload_degrades_to_no_expiry(isolated_codex_env: Path) -> None:
    payload = base64.urlsafe_b64encode(b"[1, 2, 3]").decode().rstrip("=")
    write_auth(isolated_codex_env, auth_payload(access_token=f"h.{payload}.s"))
    credential = codex.load_stored_credential()
    assert credential is not None
    assert credential.expires_at is None


def test_jwt_parsing_never_leaks_the_token_into_an_exception() -> None:
    """A garbage token must not raise at all, let alone raise carrying itself."""
    token = "garbage." + "S3CR3T" * 5 + ".sig"
    assert codex._jwt_expiry(token) is None
    assert codex._jwt_claims(token) == {}


def test_a_millisecond_exp_is_normalized(isolated_codex_env: Path) -> None:
    write_auth(
        isolated_codex_env, auth_payload(access_token=make_jwt({"exp": FUTURE_EXP * 1000}))
    )
    credential = codex.load_stored_credential()
    assert credential is not None
    assert credential.expires_at == float(FUTURE_EXP)


def test_a_malformed_id_token_yields_no_account_id(isolated_codex_env: Path) -> None:
    write_auth(isolated_codex_env, auth_payload(account_id=None, id_token="not-a-jwt"))
    credential = codex.load_stored_credential()
    assert credential is not None
    assert credential.account_id is None


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------


def test_auth_file_path_honors_codex_home(isolated_codex_env: Path) -> None:
    assert codex.auth_file_path() == isolated_codex_env / "auth.json"
    assert codex.codex_home() == isolated_codex_env


def test_auth_file_path_defaults_to_home_dot_codex(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(codex.ENV_CODEX_HOME, raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: Path("/fake/home")))
    assert codex.auth_file_path() == Path("/fake/home/.codex/auth.json")


def test_a_missing_auth_file_is_not_found() -> None:
    assert codex.load_stored_credential() is None


def test_an_unreadable_auth_file_is_not_found(
    isolated_codex_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real file is mode 0600, so PermissionError is the realistic failure."""
    write_auth(isolated_codex_env, auth_payload())

    def explode(*args: object, **kwargs: object) -> str:
        raise PermissionError("nope")

    monkeypatch.setattr(Path, "read_text", explode)
    assert codex.load_stored_credential() is None


@pytest.mark.parametrize(
    "payload",
    [
        "{not json",
        "",
        '"bare string"',
        "[]",
        "null",
        "42",
        {},
        {"tokens": None},
        {"tokens": "nope"},
        {"tokens": []},
        {"tokens": {}},
        {"tokens": {"access_token": None}},
        {"tokens": {"access_token": ""}},
        {"tokens": {"access_token": "   "}},
        {"tokens": {"access_token": 42}},
        {"tokens": {"refresh_token": REFRESH_TOKEN}},
        {"OPENAI_API_KEY": None},
        {"OPENAI_API_KEY": ""},
        {"OPENAI_API_KEY": 42},
    ],
)
def test_malformed_or_partial_auth_json_degrades_to_not_found(
    isolated_codex_env: Path, payload: object
) -> None:
    write_auth(isolated_codex_env, payload)
    assert codex.load_stored_credential() is None


def test_a_token_with_no_refresh_token_is_still_usable(isolated_codex_env: Path) -> None:
    write_auth(isolated_codex_env, auth_payload(refresh_token=None))
    credential = codex.load_stored_credential()
    assert credential is not None
    assert credential.refresh_token is None


# ---------------------------------------------------------------------------
# precedence
# ---------------------------------------------------------------------------


def test_rule1_an_explicit_auth_token_wins_over_everything(
    isolated_codex_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(codex.ENV_API_KEY, API_KEY)
    write_auth(isolated_codex_env, auth_payload())
    credential = codex.resolve_credential(
        explicit_api_key=API_KEY, explicit_auth_token="explicit-token"
    )
    assert credential.kind == "oauth"
    assert credential.value == "explicit-token"


def test_rule2_an_explicit_api_key_wins_over_the_store(isolated_codex_env: Path) -> None:
    write_auth(isolated_codex_env, auth_payload())
    credential = codex.resolve_credential(explicit_api_key=API_KEY)
    assert credential.kind == "api_key"
    assert credential.value == API_KEY


def test_rule3_the_stored_subscription_beats_openai_api_key(
    isolated_codex_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Subscription-first. Deliberate, and the reverse of the Anthropic ordering.

    Codex has no OAuth environment variable, so if a stray ``$OPENAI_API_KEY``
    outranked ``codex login`` every user with one exported for unrelated tooling
    would be billed per-token without noticing.
    """
    monkeypatch.setenv(codex.ENV_API_KEY, API_KEY)
    write_auth(isolated_codex_env, auth_payload())
    credential = codex.resolve_credential()
    assert credential.kind == "oauth"
    assert credential.value == ACCESS_TOKEN


def test_rule4_the_env_api_key_is_used_when_the_store_has_no_tokens(
    isolated_codex_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(codex.ENV_API_KEY, API_KEY)
    write_auth(isolated_codex_env, auth_payload(access_token=None))
    credential = codex.resolve_credential()
    assert credential.kind == "api_key"
    assert credential.value == API_KEY


def test_rule5_an_api_key_inside_auth_json_is_the_last_source(
    isolated_codex_env: Path,
) -> None:
    write_auth(isolated_codex_env, auth_payload(access_token=None, api_key=API_KEY))
    credential = codex.resolve_credential()
    assert credential.kind == "api_key"
    assert credential.value == API_KEY


def test_the_env_api_key_outranks_the_one_inside_auth_json(
    isolated_codex_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(codex.ENV_API_KEY, "sk-from-env")
    write_auth(isolated_codex_env, auth_payload(access_token=None, api_key=API_KEY))
    assert codex.resolve_credential().value == "sk-from-env"


def test_auth_mode_is_not_dispatched_on(isolated_codex_env: Path) -> None:
    """A file claiming apikey mode while holding live tokens still yields oauth."""
    write_auth(isolated_codex_env, auth_payload(auth_mode="apikey", api_key=API_KEY))
    credential = codex.resolve_credential()
    assert credential.kind == "oauth"


def test_nothing_found_raises_an_actionable_auth_error() -> None:
    with pytest.raises(AuthError) as excinfo:
        codex.resolve_credential()
    message = str(excinfo.value)
    assert "codex login" in message
    assert codex.ENV_API_KEY in message


@pytest.mark.parametrize("blank", ["", "   ", "\n\t "])
def test_blank_values_are_treated_as_absent(
    isolated_codex_env: Path, monkeypatch: pytest.MonkeyPatch, blank: str
) -> None:
    monkeypatch.setenv(codex.ENV_API_KEY, blank)
    with pytest.raises(AuthError):
        codex.resolve_credential(explicit_api_key=blank, explicit_auth_token=blank)


def test_resolution_never_writes_to_codex_home(isolated_codex_env: Path) -> None:
    write_auth(isolated_codex_env, auth_payload())

    def snapshot() -> set[tuple[str, float, int]]:
        return {
            (p.name, p.stat().st_mtime, p.stat().st_size)
            for p in isolated_codex_env.iterdir()
        }

    before = snapshot()
    codex.resolve_credential()
    codex.load_stored_credential()
    assert snapshot() == before


# ---------------------------------------------------------------------------
# require_account_id
# ---------------------------------------------------------------------------


def test_require_account_id_returns_the_id() -> None:
    credential = Credential(kind="oauth", value=ACCESS_TOKEN, account_id=ACCOUNT_ID)
    assert codex.require_account_id(credential) == ACCOUNT_ID


def test_require_account_id_raises_naming_codex_login() -> None:
    credential = Credential(kind="oauth", value=ACCESS_TOKEN)
    with pytest.raises(AuthError) as excinfo:
        codex.require_account_id(credential)
    message = str(excinfo.value)
    assert "codex login" in message
    assert codex.ENV_ACCOUNT_ID in message
    assert ACCESS_TOKEN not in message


# ---------------------------------------------------------------------------
# refresh
# ---------------------------------------------------------------------------


def expiring_credential(
    *, expires_at: float | None = 0.0, refresh_token: str | None = REFRESH_TOKEN
) -> Credential:
    """A subscription credential that is already due for refresh."""
    return Credential(
        kind="oauth",
        value=ACCESS_TOKEN,
        expires_at=expires_at,
        refresh_token=refresh_token,
        account_id=ACCOUNT_ID,
    )


async def test_refresh_posts_the_expected_grant() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"access_token": "fresh", "expires_in": 3600})

    async with mock_client(handler) as client:
        refreshed = await codex.refresh(expiring_credential(), client=client)

    assert seen["url"] == codex.OAUTH_TOKEN_URL
    assert seen["body"] == {
        "grant_type": "refresh_token",
        "refresh_token": REFRESH_TOKEN,
        "client_id": codex.OAUTH_CLIENT_ID,
        "scope": codex.OAUTH_SCOPE,
    }
    assert refreshed.value == "fresh"
    assert refreshed.expires_at is not None and refreshed.expires_at > time.time()


async def test_refresh_keeps_the_old_refresh_token_when_none_is_returned() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"access_token": "fresh"})

    async with mock_client(handler) as client:
        refreshed = await codex.refresh(expiring_credential(), client=client)
    assert refreshed.refresh_token == REFRESH_TOKEN


async def test_refresh_adopts_a_rotated_refresh_token() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"access_token": "fresh", "refresh_token": "rt-new"})

    async with mock_client(handler) as client:
        refreshed = await codex.refresh(expiring_credential(), client=client)
    assert refreshed.refresh_token == "rt-new"


async def test_refresh_falls_back_to_the_new_tokens_exp_when_expires_in_is_absent() -> None:
    """auth.json records no expiry, so without this a refreshed token is expiry-blind."""
    fresh = make_jwt({"exp": FUTURE_EXP})

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"access_token": fresh})

    async with mock_client(handler) as client:
        refreshed = await codex.refresh(expiring_credential(), client=client)
    assert refreshed.expires_at == float(FUTURE_EXP)


async def test_refresh_carries_the_account_id_forward() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"access_token": "fresh"})

    async with mock_client(handler) as client:
        refreshed = await codex.refresh(expiring_credential(), client=client)
    assert refreshed.account_id == ACCOUNT_ID


async def test_refresh_upgrades_the_account_id_from_a_returned_id_token() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"access_token": "fresh", "id_token": ID_TOKEN})

    async with mock_client(handler) as client:
        refreshed = await codex.refresh(expiring_credential(), client=client)
    assert refreshed.account_id == ID_TOKEN_ACCOUNT_ID


async def test_refreshing_an_api_key_is_refused() -> None:
    with pytest.raises(AuthError) as excinfo:
        await codex.refresh(Credential(kind="api_key", value=API_KEY))
    assert API_KEY not in str(excinfo.value)


async def test_refresh_without_a_refresh_token_raises() -> None:
    with pytest.raises(AuthError) as excinfo:
        await codex.refresh(expiring_credential(refresh_token=None))
    assert "codex login" in str(excinfo.value)


@pytest.mark.parametrize("status", [400, 401, 403, 429, 500, 503])
async def test_refresh_http_failure_reports_the_status_but_never_the_body(
    status: int,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text=f"leaked {ACCESS_TOKEN} {REFRESH_TOKEN}")

    async with mock_client(handler) as client:
        with pytest.raises(AuthError) as excinfo:
            await codex.refresh(expiring_credential(), client=client)

    message = str(excinfo.value)
    assert f"HTTP {status}" in message
    assert ACCESS_TOKEN not in message
    assert REFRESH_TOKEN not in message


async def test_refresh_transport_failure_raises_auth_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route")

    async with mock_client(handler) as client:
        with pytest.raises(AuthError) as excinfo:
            await codex.refresh(expiring_credential(), client=client)
    assert "codex login" in str(excinfo.value)


async def test_refresh_closes_a_client_it_created(monkeypatch: pytest.MonkeyPatch) -> None:
    closed: list[bool] = []

    class RecordingClient(httpx.AsyncClient):
        async def post(self, *args: Any, **kwargs: Any) -> httpx.Response:
            raise httpx.ConnectError("no route")

        async def aclose(self) -> None:
            closed.append(True)

    monkeypatch.setattr(codex.httpx, "AsyncClient", RecordingClient)
    with pytest.raises(AuthError):
        await codex.refresh(expiring_credential())
    assert closed == [True]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"access_token": None},
        {"access_token": ""},
        {"access_token": "   "},
        {"access_token": 42},
        [],
        "bare string",
    ],
)
async def test_refresh_unusable_body_raises_auth_error(payload: object) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    async with mock_client(handler) as client:
        with pytest.raises(AuthError) as excinfo:
            await codex.refresh(expiring_credential(), client=client)
    assert "codex login" in str(excinfo.value)


async def test_refresh_non_json_body_raises_auth_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>nope</html>")

    async with mock_client(handler) as client:
        with pytest.raises(AuthError) as excinfo:
            await codex.refresh(expiring_credential(), client=client)
    assert "non-JSON" in str(excinfo.value)


# ---------------------------------------------------------------------------
# CredentialProvider
# ---------------------------------------------------------------------------


class CountingRefresher:
    """Records how many times a refresh was actually performed."""

    def __init__(self, *, delay: float = 0.01, expires_in: float = 3600.0) -> None:
        self.calls = 0
        self.delay = delay
        self.expires_in = expires_in

    async def __call__(self, credential: Credential) -> Credential:
        """Pretend to refresh, counting invocations."""
        self.calls += 1
        await asyncio.sleep(self.delay)
        return Credential(
            kind="oauth",
            value=f"refreshed-{self.calls}",
            expires_at=time.time() + self.expires_in,
            refresh_token=credential.refresh_token,
            account_id=credential.account_id,
        )


async def test_get_refreshes_inside_the_skew_window() -> None:
    refresher = CountingRefresher()
    credential = expiring_credential(expires_at=time.time() + 5.0)
    provider = codex.CredentialProvider(credential, skew=60.0, refresher=refresher)
    fresh = await provider.get()
    assert refresher.calls == 1
    assert fresh.value == "refreshed-1"
    assert fresh.account_id == ACCOUNT_ID


async def test_get_leaves_a_healthy_credential_alone() -> None:
    refresher = CountingRefresher()
    credential = expiring_credential(expires_at=time.time() + 3600.0)
    provider = codex.CredentialProvider(credential, refresher=refresher)
    assert (await provider.get()).value == ACCESS_TOKEN
    assert refresher.calls == 0


async def test_concurrent_gets_trigger_exactly_one_refresh() -> None:
    refresher = CountingRefresher(delay=0.02)
    provider = codex.CredentialProvider(expiring_credential(), refresher=refresher)
    results = await asyncio.gather(*(provider.get() for _ in range(25)))
    assert refresher.calls == 1
    assert {credential.value for credential in results} == {"refreshed-1"}


async def test_expired_without_a_refresh_token_raises_naming_codex_login() -> None:
    provider = codex.CredentialProvider(expiring_credential(refresh_token=None))
    with pytest.raises(AuthError) as excinfo:
        await provider.get()
    assert "codex login" in str(excinfo.value)


async def test_an_api_key_is_never_refreshed() -> None:
    refresher = CountingRefresher()
    provider = codex.CredentialProvider(
        Credential(kind="api_key", value=API_KEY), refresher=refresher
    )
    assert (await provider.get()).value == API_KEY
    assert refresher.calls == 0


async def test_resolve_propagates_the_auth_error() -> None:
    with pytest.raises(AuthError):
        codex.CredentialProvider.resolve()


async def test_resolve_carries_the_explicit_account_id(isolated_codex_env: Path) -> None:
    write_auth(isolated_codex_env, auth_payload())
    provider = codex.CredentialProvider.resolve(None, None, "acc_explicit")
    assert provider.current.account_id == "acc_explicit"
