"""Credential discovery and refresh for Anthropic access.

logpose prefers a **Claude Code subscription** OAuth token, falling back to a
plain ``ANTHROPIC_API_KEY`` (BYOK). Both paths end at the same
:class:`Credential`; the Anthropic provider decides how to present it on the
wire (``Authorization: Bearer`` for ``oauth``, ``x-api-key`` for ``api_key``).

Everything backend-neutral — the credential dataclass, expiry handling, the
single-flight refresh loop — lives in :mod:`logpose.auth._common`. This module is
only the Anthropic-specific half: where the store is, what the payload looks
like, which endpoint refreshes it, and what to tell the user when it fails.

The Claude Code subscription path is experimental. It depends on unstable CLI
authentication details, including the credential-store layout and token refresh
flow, and may stop working without notice.

Discovery of the local Claude Code credential store is strictly **read-only**:
logpose never writes to ``~/.claude/.credentials.json`` and never writes to the
macOS Keychain. Refreshed tokens are held in memory for the life of the
process only.

Security
--------
No function in this module puts a credential value into a log line, an
exception message, or a ``repr``. :class:`Credential` redacts itself to a short
prefix plus a length; error paths report HTTP status codes but never response
bodies, which can carry tokens.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Literal

import httpx

from logpose.auth import _common
from logpose.auth._common import (
    EXPIRY_SKEW_SECONDS,
    Credential,
    clean,
    expiry_from_token_response,
    normalize_epoch,
)
from logpose.errors import AuthError

__all__ = [
    "Credential",
    "CredentialProvider",
    "resolve_credential",
    "refresh",
    "load_stored_credential",
    "credentials_file_path",
    "EXPIRY_SKEW_SECONDS",
    "OAUTH_CLIENT_ID",
    "OAUTH_TOKEN_URL",
    "KEYCHAIN_SERVICE",
    "ENV_OAUTH_TOKEN",
    "ENV_API_KEY",
    "ENV_CONFIG_DIR",
]

ENV_OAUTH_TOKEN = "CLAUDE_CODE_OAUTH_TOKEN"
"""Environment variable holding a Claude Code subscription OAuth token."""

ENV_API_KEY = "ANTHROPIC_API_KEY"
"""Environment variable holding a plain Anthropic API key (BYOK)."""

ENV_CONFIG_DIR = "CLAUDE_CONFIG_DIR"
"""Environment variable overriding Claude Code's config directory."""

KEYCHAIN_SERVICE = "Claude Code-credentials"
"""macOS Keychain generic-password service name used by Claude Code."""

OAUTH_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
"""Claude Code's public OAuth client id (undocumented; may change)."""

OAUTH_TOKEN_URL = "https://console.anthropic.com/v1/oauth/token"
"""Endpoint the refresh-token grant is posted to (undocumented; may change)."""

_KEYCHAIN_TIMEOUT_SECONDS = 5.0

_SETUP_HINT = (
    "Run `claude setup-token` and export the result as "
    f"{ENV_OAUTH_TOKEN}, or export {ENV_API_KEY} to use an API key instead."
)


def credentials_file_path() -> Path:
    """Locate Claude Code's on-disk credential file.

    Honors ``CLAUDE_CONFIG_DIR`` and otherwise uses ``~/.claude``. The file is
    only ever read.

    Returns:
        Path to ``.credentials.json`` (which may not exist).
    """
    config_dir = clean(os.environ.get(ENV_CONFIG_DIR))
    base = Path(config_dir).expanduser() if config_dir else Path.home() / ".claude"
    return base / ".credentials.json"


def _is_macos() -> bool:
    """Report whether the Keychain lookup is applicable.

    Returns:
        ``True`` on macOS.
    """
    return sys.platform == "darwin"


def _read_keychain() -> str | None:
    """Read the Claude Code credential blob out of the macOS Keychain.

    Shells out to ``security find-generic-password``. A missing binary, a
    non-zero exit (item absent, user denied access), or a timeout all mean
    "not found" — never a fatal error. Subprocess output is never logged.

    Returns:
        The raw JSON payload, or ``None`` when unavailable.
    """
    try:
        # Fixed argv, no shell: nothing here is interpolated from user input.
        completed = subprocess.run(
            ["security", "find-generic-password", "-s", KEYCHAIN_SERVICE, "-w"],
            capture_output=True,
            text=True,
            timeout=_KEYCHAIN_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return clean(completed.stdout)


def _read_credentials_file(path: Path) -> str | None:
    """Read the credential file if it is present and readable.

    Args:
        path: Location of ``.credentials.json``.

    Returns:
        The file's contents, or ``None`` when missing/unreadable/empty.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    return clean(raw)


def _parse_credentials_payload(raw: str) -> Credential | None:
    """Parse Claude Code's undocumented credential JSON.

    The expected shape is::

        {"claudeAiOauth": {"accessToken": "...", "refreshToken": "...",
                           "expiresAt": 1770000000000, "scopes": [...]}}

    Every field is treated as optional and untrusted: a missing key, a renamed
    key, or a wrong type degrades to ``None`` instead of raising.

    Args:
        raw: The JSON text read from the Keychain or the credential file.

    Returns:
        An ``oauth`` :class:`Credential`, or ``None`` when nothing usable was
        found.
    """
    try:
        data: Any = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    section = data.get("claudeAiOauth")
    if not isinstance(section, dict):
        return None
    access_token = section.get("accessToken")
    if not isinstance(access_token, str):
        return None
    value = clean(access_token)
    if value is None:
        return None
    raw_refresh = section.get("refreshToken")
    refresh_token = clean(raw_refresh) if isinstance(raw_refresh, str) else None
    return Credential(
        kind="oauth",
        value=value,
        expires_at=normalize_epoch(section.get("expiresAt")),
        refresh_token=refresh_token,
    )


def load_stored_credential() -> Credential | None:
    """Discover a credential from the local Claude Code store (read-only).

    On macOS the Keychain is consulted first, then ``~/.claude/.credentials.json``
    as a fallback; elsewhere only the file is read. Nothing is ever written.

    Returns:
        The stored ``oauth`` credential, or ``None`` when none could be read.
    """
    if _is_macos():
        raw = _read_keychain()
        if raw is not None:
            credential = _parse_credentials_payload(raw)
            if credential is not None:
                return credential
    raw = _read_credentials_file(credentials_file_path())
    if raw is None:
        return None
    return _parse_credentials_payload(raw)


def resolve_credential(
    explicit_api_key: str | None = None,
    explicit_auth_token: str | None = None,
    *,
    require_kind: Literal["api_key", "oauth"] | None = None,
) -> Credential:
    """Resolve the credential logpose should authenticate with.

    Precedence, first match wins:

    1. ``explicit_auth_token`` → ``oauth``
    2. ``explicit_api_key`` → ``api_key``
    3. ``CLAUDE_CODE_OAUTH_TOKEN`` → ``oauth``
    4. ``ANTHROPIC_API_KEY`` → ``api_key``
    5. The local Claude Code credential store → ``oauth``

    Note on 2 vs 3 (deliberate, do not reorder): **subscription-first**.
    ``CLAUDE_CODE_OAUTH_TOKEN`` outranks ``ANTHROPIC_API_KEY`` because logpose's
    v0.1 story is "ride your Claude Code subscription", and an
    ``ANTHROPIC_API_KEY`` is very often present in a developer's environment for
    unrelated tooling. If that key silently won, users who deliberately set up a
    subscription token would be billed per-token without noticing. A caller that
    wants the API key regardless passes ``explicit_api_key``, which outranks both
    environment variables. Within the explicit arguments the same ordering holds:
    ``explicit_auth_token`` beats ``explicit_api_key`` when both are given.

    ``require_kind`` narrows that list to the sources that can produce the
    requested kind, and is how the providers use this function: ``anthropic``
    accepts only an API key and ``claude-code`` only a subscription token.
    Filtering rather than resolving and then rejecting matters — otherwise a
    subscription token in the environment would shadow a perfectly good
    ``$ANTHROPIC_API_KEY`` and the API-key provider would refuse a credential it
    was standing next to.

    Blank and whitespace-only values are treated as absent at every level.

    Args:
        explicit_api_key: API key supplied directly by the caller.
        explicit_auth_token: Subscription OAuth token supplied directly by the
            caller.
        require_kind: Consider only sources yielding this kind of credential.

    Returns:
        The highest-precedence credential found.

    Raises:
        AuthError: If no credential of the requested kind could be resolved. The
            message tells the user exactly which command to run and which
            variable to export, and contains no credential material.
    """
    want_oauth = require_kind != "api_key"
    want_key = require_kind != "oauth"

    if want_oauth:
        auth_token = clean(explicit_auth_token)
        if auth_token is not None:
            return Credential(kind="oauth", value=auth_token)

    if want_key:
        api_key = clean(explicit_api_key)
        if api_key is not None:
            return Credential(kind="api_key", value=api_key)

    if want_oauth:
        env_token = clean(os.environ.get(ENV_OAUTH_TOKEN))
        if env_token is not None:
            return Credential(kind="oauth", value=env_token)

    if want_key:
        env_key = clean(os.environ.get(ENV_API_KEY))
        if env_key is not None:
            return Credential(kind="api_key", value=env_key)

    if want_oauth:
        stored = load_stored_credential()
        if stored is not None:
            return stored

    raise AuthError(_missing_credential_message(require_kind))


def _missing_credential_message(require_kind: str | None) -> str:
    """Explain what was looked for and how to supply it.

    Args:
        require_kind: The kind that was required, if any.

    Returns:
        An actionable message containing no credential material.
    """
    if require_kind == "oauth":
        return (
            "No Claude Code subscription credential found. Run `claude setup-token` and "
            f"export the token as {ENV_OAUTH_TOKEN}, or pass auth_token=... explicitly."
        )
    if require_kind == "api_key":
        return (
            f"No Anthropic API key found. Export {ENV_API_KEY}, or pass api_key=... "
            "explicitly."
        )
    return (
        "No Anthropic credential found. To use a Claude Code subscription, run "
        f"`claude setup-token` and export the token as {ENV_OAUTH_TOKEN}. To use "
        f"an API key instead, export {ENV_API_KEY}. You can also pass "
        "api_key=... or auth_token=... explicitly."
    )


async def refresh(
    cred: Credential,
    *,
    client: httpx.AsyncClient | None = None,
) -> Credential:
    """Exchange a refresh token for a fresh subscription access token.

    Posts the standard ``refresh_token`` grant with Claude Code's public client
    id. The result is returned as a new in-memory :class:`Credential`; nothing is
    written back to disk or to the Keychain.

    Args:
        cred: The (probably expired) ``oauth`` credential to refresh.
        client: Optional HTTP client to reuse. When omitted a short-lived one is
            created and closed.

    Returns:
        A new ``oauth`` credential carrying the refreshed access token, its
        expiry in epoch seconds, and the refresh token to use next time.

    Raises:
        AuthError: On any failure — wrong credential kind, missing refresh
            token, transport error, non-2xx status, or an unusable response
            body. The message includes the HTTP status when there was one but
            never the response body, which can contain tokens.
    """
    if cred.kind != "oauth":
        raise AuthError(
            f"Only subscription credentials can be refreshed (got kind={cred.kind!r}). "
            + _SETUP_HINT
        )
    if cred.refresh_token is None:
        raise AuthError(
            "Cannot refresh the Claude Code subscription token: no refresh token is "
            "available. Re-authenticate with `claude setup-token`."
        )

    payload = {
        "grant_type": "refresh_token",
        "refresh_token": cred.refresh_token,
        "client_id": OAUTH_CLIENT_ID,
    }

    owns_client = client is None
    timeout = _common.REFRESH_TIMEOUT_SECONDS
    http = client if client is not None else httpx.AsyncClient(timeout=timeout)
    try:
        try:
            response = await http.post(OAUTH_TOKEN_URL, json=payload)
        except httpx.HTTPError as exc:
            raise AuthError(
                "Could not reach the Anthropic OAuth token endpoint to refresh the "
                "Claude Code subscription token. Check connectivity, then "
                "re-authenticate with `claude setup-token`."
            ) from exc
    finally:
        if owns_client:
            await http.aclose()

    status = response.status_code
    if status >= 400:
        raise AuthError(
            f"Refreshing the Claude Code subscription token failed with HTTP {status}. "
            "Re-authenticate with `claude setup-token`."
        )

    try:
        body: Any = response.json()
    except ValueError as exc:
        raise AuthError(
            f"The Anthropic OAuth token endpoint returned a non-JSON response (HTTP "
            f"{status}). Re-authenticate with `claude setup-token`."
        ) from exc
    if not isinstance(body, dict):
        raise AuthError(
            f"The Anthropic OAuth token endpoint returned an unexpected payload (HTTP "
            f"{status}). Re-authenticate with `claude setup-token`."
        )

    raw_access = body.get("access_token")
    access_token = clean(raw_access) if isinstance(raw_access, str) else None
    if access_token is None:
        raise AuthError(
            f"The Anthropic OAuth token endpoint returned no access token (HTTP "
            f"{status}). Re-authenticate with `claude setup-token`."
        )

    raw_refresh = body.get("refresh_token")
    next_refresh = clean(raw_refresh) if isinstance(raw_refresh, str) else None

    expires_at = expiry_from_token_response(body)

    return Credential(
        kind="oauth",
        value=access_token,
        expires_at=expires_at,
        refresh_token=next_refresh or cred.refresh_token,
    )


class CredentialProvider(_common.CredentialProvider):
    """Holds an Anthropic credential and refreshes it on demand, single-flight.

    Wrap the credential the provider was built with, then call :meth:`get`
    before every request. The refresh loop itself lives in
    :class:`logpose.auth._common.CredentialProvider`; this subclass supplies the
    Anthropic refresh call and the Claude-Code-specific failure message.

    Attributes:
        skew: How many seconds before expiry a refresh is triggered.
    """

    EXPIRED_MESSAGE = (
        "The Claude Code subscription token has expired and no refresh "
        "token is available. Re-authenticate with `claude setup-token` "
        f"and export the new token as {ENV_OAUTH_TOKEN}."
    )

    async def _refresh(self, credential: Credential) -> Credential:
        """Refresh via the Claude Code OAuth token endpoint.

        Args:
            credential: The expiring subscription credential.

        Returns:
            A refreshed credential.

        Raises:
            AuthError: On any refresh failure — see :func:`refresh`.
        """
        return await refresh(credential)

    @classmethod
    def resolve(
        cls,
        explicit_api_key: str | None = None,
        explicit_auth_token: str | None = None,
        *,
        require_kind: Literal["api_key", "oauth"] | None = None,
        skew: float = EXPIRY_SKEW_SECONDS,
        refresher: Callable[[Credential], Awaitable[Credential]] | None = None,
    ) -> CredentialProvider:
        """Build a provider around :func:`resolve_credential`.

        Args:
            explicit_api_key: API key supplied directly by the caller.
            explicit_auth_token: Subscription OAuth token supplied directly.
            require_kind: Consider only sources yielding this kind of credential.
            skew: Refresh this many seconds before the recorded expiry.
            refresher: Coroutine used to perform the refresh.

        Returns:
            A provider holding the resolved credential.

        Raises:
            AuthError: If no credential could be resolved.
        """
        credential = resolve_credential(
            explicit_api_key, explicit_auth_token, require_kind=require_kind
        )
        return cls(credential, skew=skew, refresher=refresher)
