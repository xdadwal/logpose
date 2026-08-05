"""Tests for Claude Code subscription credential resolution and refresh.

Everything here is hermetic: no network, no real Keychain, no reads of the
developer's real ``~/.claude`` directory. The ``isolated_env`` fixture is
autouse and points credential discovery at an empty temp directory.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from logpose.auth import claude_code
from logpose.auth.claude_code import (
    Credential,
    CredentialProvider,
    credentials_file_path,
    load_stored_credential,
    refresh,
    resolve_credential,
)
from logpose.errors import AuthError, LogposeError

# A long, obviously-secret token. The redaction prefix is 13 chars
# ("sk-ant-oat01-"), so "S3CR3T" never legitimately appears in a repr.
OAUTH_TOKEN = "sk-ant-oat01-" + "S3CR3T" * 20
API_KEY = "sk-ant-api03-" + "K3YV4L" * 20
REFRESH_TOKEN = "sk-ant-ort01-" + "R3FR3SH" * 10


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Isolate credential discovery from the machine running the tests.

    Clears every credential environment variable, points ``CLAUDE_CONFIG_DIR``
    at an empty temp directory, and pretends we are not on macOS so the
    Keychain is never consulted unless a test opts in.

    Returns:
        The temp config directory used for the credential file.
    """
    config_dir = tmp_path / "claude-config"
    config_dir.mkdir()
    monkeypatch.delenv(claude_code.ENV_OAUTH_TOKEN, raising=False)
    monkeypatch.delenv(claude_code.ENV_API_KEY, raising=False)
    monkeypatch.setenv(claude_code.ENV_CONFIG_DIR, str(config_dir))
    monkeypatch.setattr(claude_code, "_is_macos", lambda: False)
    return config_dir


def write_store(config_dir: Path, payload: object) -> Path:
    """Write a credential-store file for the tests to discover.

    Args:
        config_dir: The isolated config directory.
        payload: Either a str (written verbatim) or an object to JSON-encode.

    Returns:
        The path written.
    """
    path = config_dir / ".credentials.json"
    text = payload if isinstance(payload, str) else json.dumps(payload)
    path.write_text(text, encoding="utf-8")
    return path


def store_payload(
    *,
    access_token: str = OAUTH_TOKEN,
    refresh_token: str | None = REFRESH_TOKEN,
    expires_at: object = 1770000000000,
) -> dict[str, Any]:
    """Build a payload shaped like Claude Code's credential store.

    Args:
        access_token: Value for ``claudeAiOauth.accessToken``.
        refresh_token: Value for ``claudeAiOauth.refreshToken``; omitted if None.
        expires_at: Value for ``claudeAiOauth.expiresAt``; omitted if None.

    Returns:
        A dict ready to JSON-encode.
    """
    oauth: dict[str, Any] = {
        "accessToken": access_token,
        "scopes": ["user:inference"],
        "subscriptionType": "max",
    }
    if refresh_token is not None:
        oauth["refreshToken"] = refresh_token
    if expires_at is not None:
        oauth["expiresAt"] = expires_at
    return {"claudeAiOauth": oauth}


# ---------------------------------------------------------------------------
# Credential: redaction
# ---------------------------------------------------------------------------


def test_repr_redacts_oauth_token() -> None:
    credential = Credential(kind="oauth", value=OAUTH_TOKEN)
    text = repr(credential)
    assert OAUTH_TOKEN not in text
    assert "S3CR3T" not in text
    assert text == f"Credential(kind='oauth', value='sk-ant-oat01-…', len={len(OAUTH_TOKEN)})"


def test_str_and_format_also_redact() -> None:
    credential = Credential(kind="api_key", value=API_KEY, refresh_token=REFRESH_TOKEN)
    for rendering in (str(credential), f"{credential}", f"{credential!r}", format(credential)):
        assert API_KEY not in rendering
        assert REFRESH_TOKEN not in rendering
        assert "K3YV4L" not in rendering


def test_repr_never_leaks_short_secrets() -> None:
    assert "abc" not in repr(Credential(kind="api_key", value="abc"))
    assert "x" not in repr(Credential(kind="api_key", value="x"))
    assert repr(Credential(kind="api_key", value="")) == (
        "Credential(kind='api_key', value='…', len=0)"
    )


def test_provider_repr_redacts() -> None:
    provider = CredentialProvider(Credential(kind="oauth", value=OAUTH_TOKEN))
    assert OAUTH_TOKEN not in repr(provider)
    assert "S3CR3T" not in repr(provider)


def test_is_expired_semantics() -> None:
    now = 1_000_000.0
    assert Credential(kind="oauth", value="v").is_expired() is False
    assert Credential(kind="oauth", value="v", expires_at=now + 10).is_expired(now=now) is False
    assert Credential(kind="oauth", value="v", expires_at=now - 1).is_expired(now=now) is True
    stale = Credential(kind="oauth", value="v", expires_at=now + 30)
    assert stale.is_expired(now=now, skew=60) is True


# ---------------------------------------------------------------------------
# resolve_credential: precedence
# ---------------------------------------------------------------------------


def test_rule1_explicit_auth_token_wins_over_everything(
    monkeypatch: pytest.MonkeyPatch, isolated_env: Path
) -> None:
    monkeypatch.setenv(claude_code.ENV_OAUTH_TOKEN, "env-oauth")
    monkeypatch.setenv(claude_code.ENV_API_KEY, "env-key")
    write_store(isolated_env, store_payload(access_token="stored"))

    credential = resolve_credential(explicit_api_key="arg-key", explicit_auth_token=OAUTH_TOKEN)

    assert credential == Credential(kind="oauth", value=OAUTH_TOKEN)


def test_rule1_explicit_api_key_wins_over_env(
    monkeypatch: pytest.MonkeyPatch, isolated_env: Path
) -> None:
    monkeypatch.setenv(claude_code.ENV_OAUTH_TOKEN, "env-oauth")
    monkeypatch.setenv(claude_code.ENV_API_KEY, "env-key")
    write_store(isolated_env, store_payload(access_token="stored"))

    credential = resolve_credential(explicit_api_key=API_KEY)

    assert credential == Credential(kind="api_key", value=API_KEY)


def test_rule2_oauth_env_beats_api_key_env(
    monkeypatch: pytest.MonkeyPatch, isolated_env: Path
) -> None:
    """Subscription-first: CLAUDE_CODE_OAUTH_TOKEN outranks ANTHROPIC_API_KEY."""
    monkeypatch.setenv(claude_code.ENV_OAUTH_TOKEN, OAUTH_TOKEN)
    monkeypatch.setenv(claude_code.ENV_API_KEY, API_KEY)
    write_store(isolated_env, store_payload(access_token="stored"))

    credential = resolve_credential()

    assert credential.kind == "oauth"
    assert credential.value == OAUTH_TOKEN


def test_rule3_api_key_env_used_when_no_oauth_env(
    monkeypatch: pytest.MonkeyPatch, isolated_env: Path
) -> None:
    monkeypatch.setenv(claude_code.ENV_API_KEY, API_KEY)
    write_store(isolated_env, store_payload(access_token="stored"))

    credential = resolve_credential()

    assert credential == Credential(kind="api_key", value=API_KEY)


def test_rule4_store_used_when_env_is_empty(isolated_env: Path) -> None:
    write_store(isolated_env, store_payload())

    credential = resolve_credential()

    assert credential.kind == "oauth"
    assert credential.value == OAUTH_TOKEN
    assert credential.refresh_token == REFRESH_TOKEN
    assert credential.expires_at == 1770000000.0


def test_rule5_nothing_found_raises_actionable_auth_error() -> None:
    with pytest.raises(AuthError) as excinfo:
        resolve_credential()

    message = str(excinfo.value)
    assert isinstance(excinfo.value, LogposeError)
    assert "claude setup-token" in message
    assert claude_code.ENV_OAUTH_TOKEN in message
    assert claude_code.ENV_API_KEY in message


def test_blank_and_whitespace_values_are_treated_as_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(claude_code.ENV_OAUTH_TOKEN, "   ")
    monkeypatch.setenv(claude_code.ENV_API_KEY, "")

    with pytest.raises(AuthError):
        resolve_credential(explicit_api_key="  ", explicit_auth_token="\n\t")


def test_values_are_stripped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(claude_code.ENV_OAUTH_TOKEN, f"  {OAUTH_TOKEN}\n")

    assert resolve_credential().value == OAUTH_TOKEN


# ---------------------------------------------------------------------------
# Credential store: file discovery and defensive parsing
# ---------------------------------------------------------------------------


def test_credentials_file_path_honors_config_dir(isolated_env: Path) -> None:
    assert credentials_file_path() == isolated_env / ".credentials.json"


def test_credentials_file_path_defaults_to_home(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(claude_code.ENV_CONFIG_DIR, raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: Path("/nowhere")))

    assert credentials_file_path() == Path("/nowhere/.claude/.credentials.json")


def test_missing_store_file_is_not_found(isolated_env: Path) -> None:
    assert load_stored_credential() is None


def test_expires_at_milliseconds_are_normalized_to_seconds(isolated_env: Path) -> None:
    write_store(isolated_env, store_payload(expires_at=1770000000000))

    credential = load_stored_credential()

    assert credential is not None
    assert credential.expires_at == 1770000000.0


def test_expires_at_already_in_seconds_passes_through(isolated_env: Path) -> None:
    write_store(isolated_env, store_payload(expires_at=1770000000))

    credential = load_stored_credential()

    assert credential is not None
    assert credential.expires_at == 1770000000.0


def test_expires_at_numeric_string_is_normalized(isolated_env: Path) -> None:
    write_store(isolated_env, store_payload(expires_at="1770000000000"))

    credential = load_stored_credential()

    assert credential is not None
    assert credential.expires_at == 1770000000.0


@pytest.mark.parametrize("bogus", ["soon", None, True, {"nested": 1}, [1, 2]])
def test_unusable_expires_at_degrades_to_none(isolated_env: Path, bogus: object) -> None:
    payload = store_payload()
    payload["claudeAiOauth"]["expiresAt"] = bogus
    write_store(isolated_env, payload)

    credential = load_stored_credential()

    assert credential is not None
    assert credential.expires_at is None


def test_absent_expires_at_and_refresh_token_are_tolerated(isolated_env: Path) -> None:
    write_store(isolated_env, store_payload(refresh_token=None, expires_at=None))

    credential = load_stored_credential()

    assert credential is not None
    assert credential.value == OAUTH_TOKEN
    assert credential.expires_at is None
    assert credential.refresh_token is None


@pytest.mark.parametrize(
    "payload",
    [
        "{not json at all",
        "",
        "   ",
        '"a bare string"',
        "[1, 2, 3]",
        "null",
        json.dumps({}),
        json.dumps({"claudeAiOauth": None}),
        json.dumps({"claudeAiOauth": "renamed-to-a-string"}),
        json.dumps({"claudeAiOauth": {}}),
        json.dumps({"claudeAiOauth": {"accessToken": None}}),
        json.dumps({"claudeAiOauth": {"accessToken": 12345}}),
        json.dumps({"claudeAiOauth": {"accessToken": "   "}}),
        json.dumps({"oauthAccount": {"accessToken": "renamed-outer-key"}}),
    ],
)
def test_malformed_or_partial_store_degrades_to_not_found(isolated_env: Path, payload: str) -> None:
    write_store(isolated_env, payload)

    assert load_stored_credential() is None
    with pytest.raises(AuthError):
        resolve_credential()


def test_non_string_refresh_token_degrades_to_none(isolated_env: Path) -> None:
    payload = store_payload()
    payload["claudeAiOauth"]["refreshToken"] = 42
    write_store(isolated_env, payload)

    credential = load_stored_credential()

    assert credential is not None
    assert credential.refresh_token is None


def test_unreadable_store_file_is_not_found(
    monkeypatch: pytest.MonkeyPatch, isolated_env: Path
) -> None:
    write_store(isolated_env, store_payload())

    def boom(*args: object, **kwargs: object) -> str:
        raise PermissionError("nope")

    monkeypatch.setattr(Path, "read_text", boom)

    assert load_stored_credential() is None


# ---------------------------------------------------------------------------
# Credential store: macOS Keychain (subprocess mocked)
# ---------------------------------------------------------------------------


def fake_security(
    monkeypatch: pytest.MonkeyPatch,
    *,
    returncode: int = 0,
    stdout: str = "",
    raises: BaseException | None = None,
    recorder: list[list[str]] | None = None,
) -> None:
    """Replace ``subprocess.run`` for the Keychain lookup.

    Args:
        monkeypatch: Fixture used to install the fake.
        returncode: Exit status the fake ``security`` reports.
        stdout: What the fake ``security`` prints.
        raises: Exception to raise instead of running.
        recorder: If given, each invocation's argv is appended here.
    """

    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if recorder is not None:
            recorder.append(argv)
        if raises is not None:
            raise raises
        return subprocess.CompletedProcess(argv, returncode, stdout=stdout, stderr="")

    monkeypatch.setattr(claude_code, "_is_macos", lambda: True)
    monkeypatch.setattr(claude_code.subprocess, "run", run)


def test_keychain_success(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder: list[list[str]] = []
    fake_security(monkeypatch, stdout=json.dumps(store_payload()) + "\n", recorder=recorder)

    credential = load_stored_credential()

    assert credential is not None
    assert credential.kind == "oauth"
    assert credential.value == OAUTH_TOKEN
    assert credential.refresh_token == REFRESH_TOKEN
    assert credential.expires_at == 1770000000.0
    assert recorder == [
        ["security", "find-generic-password", "-s", claude_code.KEYCHAIN_SERVICE, "-w"]
    ]


def test_keychain_resolves_through_resolve_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_security(monkeypatch, stdout=json.dumps(store_payload()))

    assert resolve_credential().value == OAUTH_TOKEN


@pytest.mark.parametrize(
    ("returncode", "stdout", "raises"),
    [
        (1, "", None),
        (44, "The specified item could not be found in the keychain.", None),
        (0, "", None),
        (0, "not json", None),
        (0, "", FileNotFoundError("security")),
        (0, "", PermissionError("denied")),
        (0, "", subprocess.TimeoutExpired(cmd="security", timeout=5.0)),
        (0, "", subprocess.SubprocessError("boom")),
    ],
)
def test_keychain_failures_are_not_found_not_fatal(
    monkeypatch: pytest.MonkeyPatch,
    returncode: int,
    stdout: str,
    raises: BaseException | None,
) -> None:
    fake_security(monkeypatch, returncode=returncode, stdout=stdout, raises=raises)

    assert load_stored_credential() is None
    with pytest.raises(AuthError) as excinfo:
        resolve_credential()
    assert "claude setup-token" in str(excinfo.value)


def test_keychain_miss_falls_back_to_credentials_file(
    monkeypatch: pytest.MonkeyPatch, isolated_env: Path
) -> None:
    fake_security(monkeypatch, returncode=1)
    write_store(isolated_env, store_payload(access_token="file-token", refresh_token=None))

    credential = load_stored_credential()

    assert credential is not None
    assert credential.value == "file-token"


def test_keychain_is_not_consulted_off_macos(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(*args: object, **kwargs: object) -> None:
        raise AssertionError("subprocess must not run when not on macOS")

    monkeypatch.setattr(claude_code.subprocess, "run", explode)

    assert load_stored_credential() is None


# ---------------------------------------------------------------------------
# refresh()
# ---------------------------------------------------------------------------


def expiring_credential(
    *,
    expires_at: float | None = None,
    refresh_token: str | None = REFRESH_TOKEN,
) -> Credential:
    """Build an oauth credential for refresh tests.

    Args:
        expires_at: Absolute expiry in epoch seconds.
        refresh_token: Refresh token to carry.

    Returns:
        The credential.
    """
    return Credential(
        kind="oauth",
        value=OAUTH_TOKEN,
        expires_at=expires_at,
        refresh_token=refresh_token,
    )


def mock_client(handler: Any) -> httpx.AsyncClient:
    """Build an AsyncClient whose transport never touches the network.

    Args:
        handler: Callable taking an ``httpx.Request`` and returning a Response.

    Returns:
        A client backed by ``httpx.MockTransport``.
    """
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_refresh_success_posts_the_expected_grant() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "access_token": "sk-ant-oat01-FRESH",
                "refresh_token": "sk-ant-ort01-NEXT",
                "expires_in": 3600,
                "token_type": "Bearer",
            },
        )

    before = time.time()
    async with mock_client(handler) as client:
        result = await refresh(expiring_credential(expires_at=1.0), client=client)

    assert seen["url"] == claude_code.OAUTH_TOKEN_URL
    assert seen["body"] == {
        "grant_type": "refresh_token",
        "refresh_token": REFRESH_TOKEN,
        "client_id": claude_code.OAUTH_CLIENT_ID,
    }
    assert result.kind == "oauth"
    assert result.value == "sk-ant-oat01-FRESH"
    assert result.refresh_token == "sk-ant-ort01-NEXT"
    assert result.expires_at is not None
    assert before + 3600 <= result.expires_at <= time.time() + 3600


async def test_refresh_keeps_old_refresh_token_when_none_returned() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"access_token": "fresh"})

    async with mock_client(handler) as client:
        result = await refresh(expiring_credential(), client=client)

    assert result.refresh_token == REFRESH_TOKEN
    assert result.expires_at is None


async def test_refresh_accepts_absolute_expiry_in_milliseconds() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"access_token": "fresh", "expires_at": 1770000000000})

    async with mock_client(handler) as client:
        result = await refresh(expiring_credential(), client=client)

    assert result.expires_at == 1770000000.0


async def test_refresh_creates_and_closes_its_own_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[httpx.AsyncClient] = []

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"access_token": "fresh"})

    class RecordingClient(httpx.AsyncClient):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(transport=httpx.MockTransport(handler))
            created.append(self)

    monkeypatch.setattr(claude_code.httpx, "AsyncClient", RecordingClient)

    result = await refresh(expiring_credential())

    assert result.value == "fresh"
    assert len(created) == 1
    assert created[0].is_closed


@pytest.mark.parametrize("status", [400, 401, 403, 429, 500, 503])
async def test_refresh_http_error_raises_autherror_with_status_but_no_body(
    status: int,
) -> None:
    leaky_body = {"error": "invalid_grant", "access_token": OAUTH_TOKEN, "hint": "S3CR3T"}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=leaky_body)

    async with mock_client(handler) as client:
        with pytest.raises(AuthError) as excinfo:
            await refresh(expiring_credential(), client=client)

    message = str(excinfo.value)
    assert str(status) in message
    assert "claude setup-token" in message
    assert OAUTH_TOKEN not in message
    assert REFRESH_TOKEN not in message
    assert "S3CR3T" not in message
    assert "invalid_grant" not in message


async def test_refresh_network_error_raises_autherror() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host", request=request)

    async with mock_client(handler) as client:
        with pytest.raises(AuthError) as excinfo:
            await refresh(expiring_credential(), client=client)

    assert "claude setup-token" in str(excinfo.value)
    assert OAUTH_TOKEN not in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, httpx.ConnectError)


async def test_refresh_closes_own_client_even_on_transport_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[httpx.AsyncClient] = []

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    class RecordingClient(httpx.AsyncClient):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(transport=httpx.MockTransport(handler))
            created.append(self)

    monkeypatch.setattr(claude_code.httpx, "AsyncClient", RecordingClient)

    with pytest.raises(AuthError):
        await refresh(expiring_credential())

    assert created[0].is_closed


@pytest.mark.parametrize(
    ("status", "kwargs"),
    [
        (200, {"text": "<html>not json</html>"}),
        (200, {"json": ["not", "a", "dict"]}),
        (200, {"json": {}}),
        (200, {"json": {"access_token": None}}),
        (200, {"json": {"access_token": 12345}}),
        (200, {"json": {"access_token": "   "}}),
        (200, {"json": {"accessToken": "wrong-case-key"}}),
    ],
)
async def test_refresh_unusable_body_raises_autherror(status: int, kwargs: Any) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, **kwargs)

    async with mock_client(handler) as client:
        with pytest.raises(AuthError) as excinfo:
            await refresh(expiring_credential(), client=client)

    assert "claude setup-token" in str(excinfo.value)


async def test_refresh_without_refresh_token_raises_and_redacts() -> None:
    with pytest.raises(AuthError) as excinfo:
        await refresh(expiring_credential(refresh_token=None))

    message = str(excinfo.value)
    assert "claude setup-token" in message
    assert OAUTH_TOKEN not in message
    assert "S3CR3T" not in message


async def test_refresh_rejects_api_key_credentials() -> None:
    with pytest.raises(AuthError) as excinfo:
        await refresh(Credential(kind="api_key", value=API_KEY, refresh_token="r"))

    assert API_KEY not in str(excinfo.value)
    assert "K3YV4L" not in str(excinfo.value)


# ---------------------------------------------------------------------------
# CredentialProvider
# ---------------------------------------------------------------------------


class CountingRefresher:
    """Records how many times a refresh was actually performed."""

    def __init__(self, *, delay: float = 0.01, expires_in: float = 3600.0) -> None:
        """Initialize the fake refresher.

        Args:
            delay: How long each refresh awaits, to force real interleaving.
            expires_in: Lifetime of the credential handed back.
        """
        self.calls = 0
        self.delay = delay
        self.expires_in = expires_in

    async def __call__(self, credential: Credential) -> Credential:
        """Pretend to refresh, counting invocations.

        Args:
            credential: The stale credential.

        Returns:
            A fresh credential.
        """
        self.calls += 1
        await asyncio.sleep(self.delay)
        return Credential(
            kind="oauth",
            value=f"refreshed-{self.calls}",
            expires_at=time.time() + self.expires_in,
            refresh_token=credential.refresh_token,
        )


async def test_get_returns_credential_when_no_expiry_is_known() -> None:
    refresher = CountingRefresher()
    provider = CredentialProvider(expiring_credential(), refresher=refresher)

    assert (await provider.get()).value == OAUTH_TOKEN
    assert refresher.calls == 0


async def test_get_does_not_refresh_a_fresh_credential() -> None:
    refresher = CountingRefresher()
    provider = CredentialProvider(
        expiring_credential(expires_at=time.time() + 3600), refresher=refresher
    )

    assert (await provider.get()).value == OAUTH_TOKEN
    assert refresher.calls == 0


async def test_get_refreshes_inside_the_skew_window() -> None:
    refresher = CountingRefresher()
    provider = CredentialProvider(
        expiring_credential(expires_at=time.time() + 30), refresher=refresher
    )

    credential = await provider.get()

    assert refresher.calls == 1
    assert credential.value == "refreshed-1"
    assert provider.current.value == "refreshed-1"


async def test_get_refreshes_an_already_expired_credential() -> None:
    refresher = CountingRefresher()
    provider = CredentialProvider(
        expiring_credential(expires_at=time.time() - 5), refresher=refresher
    )

    assert (await provider.get()).value == "refreshed-1"
    assert refresher.calls == 1


async def test_second_get_reuses_the_refreshed_credential() -> None:
    refresher = CountingRefresher()
    provider = CredentialProvider(
        expiring_credential(expires_at=time.time() - 5), refresher=refresher
    )

    first = await provider.get()
    second = await provider.get()

    assert refresher.calls == 1
    assert first is second


async def test_skew_is_configurable() -> None:
    refresher = CountingRefresher()
    provider = CredentialProvider(
        expiring_credential(expires_at=time.time() + 30), refresher=refresher, skew=5
    )

    assert (await provider.get()).value == OAUTH_TOKEN
    assert refresher.calls == 0


async def test_concurrent_gets_trigger_exactly_one_refresh() -> None:
    refresher = CountingRefresher(delay=0.02)
    provider = CredentialProvider(
        expiring_credential(expires_at=time.time() - 1), refresher=refresher
    )

    results = await asyncio.gather(*(provider.get() for _ in range(25)))

    assert refresher.calls == 1
    assert {credential.value for credential in results} == {"refreshed-1"}


async def test_expired_without_refresh_token_raises_autherror() -> None:
    provider = CredentialProvider(
        expiring_credential(expires_at=time.time() - 1, refresh_token=None)
    )

    with pytest.raises(AuthError) as excinfo:
        await provider.get()

    message = str(excinfo.value)
    assert "claude setup-token" in message
    assert claude_code.ENV_OAUTH_TOKEN in message
    assert OAUTH_TOKEN not in message
    assert "S3CR3T" not in message


async def test_near_expiry_without_refresh_token_is_still_usable() -> None:
    """Inside the skew window but not yet expired: hand it back rather than fail."""
    provider = CredentialProvider(
        expiring_credential(expires_at=time.time() + 30, refresh_token=None)
    )

    assert (await provider.get()).value == OAUTH_TOKEN


async def test_refresh_failure_propagates_and_leaves_credential_untouched() -> None:
    async def failing(credential: Credential) -> Credential:
        raise AuthError("refresh rejected; run `claude setup-token`")

    original = expiring_credential(expires_at=time.time() - 1)
    provider = CredentialProvider(original, refresher=failing)

    with pytest.raises(AuthError):
        await provider.get()

    assert provider.current is original


async def test_provider_resolve_builds_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(claude_code.ENV_OAUTH_TOKEN, OAUTH_TOKEN)

    provider = CredentialProvider.resolve()

    assert (await provider.get()).value == OAUTH_TOKEN


async def test_provider_resolve_propagates_auth_error() -> None:
    with pytest.raises(AuthError):
        CredentialProvider.resolve()


async def test_api_key_credentials_are_never_refreshed() -> None:
    refresher = CountingRefresher()
    provider = CredentialProvider(Credential(kind="api_key", value=API_KEY), refresher=refresher)

    assert (await provider.get()).value == API_KEY
    assert refresher.calls == 0
