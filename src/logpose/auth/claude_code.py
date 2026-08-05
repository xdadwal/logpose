"""Credential discovery and refresh for Anthropic access.

logpose prefers a **Claude Code subscription** OAuth token, falling back to a
plain ``ANTHROPIC_API_KEY`` (BYOK). Both paths end at the same
:class:`Credential`; the Anthropic provider decides how to present it on the
wire (``Authorization: Bearer`` for ``oauth``, ``x-api-key`` for ``api_key``).

.. warning::

   **Gray area — read this.** Using a Claude Code *subscription* token against
   the raw Anthropic API is **not an officially supported integration path**.
   Anthropic publishes no contract for it. Doing so may violate the Claude
   Code / Anthropic consumer terms of service, and it may stop working at any
   time without notice (the credential store layout, the OAuth client id, and
   the token endpoint used here are all undocumented internals). If you need a
   supported, stable integration, use an API key (``ANTHROPIC_API_KEY``) and
   accept the metered billing that comes with it. You are responsible for
   deciding whether the subscription path is acceptable for your use.

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

import asyncio
import json
import os
import subprocess
import sys
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import httpx

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

EXPIRY_SKEW_SECONDS = 60.0
"""Refresh this many seconds before a token's actual expiry."""

_KEYCHAIN_TIMEOUT_SECONDS = 5.0
_REFRESH_TIMEOUT_SECONDS = 30.0
_REDACT_PREFIX = 13
_MILLISECOND_THRESHOLD = 1e11

_SETUP_HINT = (
    "Run `claude setup-token` and export the result as "
    f"{ENV_OAUTH_TOKEN}, or export {ENV_API_KEY} to use an API key instead."
)


def _redact(value: str) -> str:
    """Render a secret as a short quoted prefix plus an ellipsis.

    Never returns more than half of the input, so short values do not leak.

    Args:
        value: The secret to redact.

    Returns:
        A quoted, truncated form safe to place in a ``repr``.
    """
    keep = min(_REDACT_PREFIX, len(value) // 2)
    if keep <= 0:
        return "'…'"
    return f"'{value[:keep]}…'"


@dataclass(frozen=True, repr=False)
class Credential:
    """A resolved Anthropic credential and how it must be presented.

    Attributes:
        kind: ``"oauth"`` for a Claude Code subscription token (sent as a
            bearer token), ``"api_key"`` for a plain API key.
        value: The secret itself. Never log, print, or format this into an
            error message; use :meth:`__repr__`, which redacts.
        expires_at: Expiry as epoch **seconds**, or ``None`` when unknown.
            Sources reporting milliseconds are normalized on the way in.
        refresh_token: Token usable with :func:`refresh`, when one is known.
    """

    kind: Literal["api_key", "oauth"]
    value: str
    expires_at: float | None = None
    refresh_token: str | None = None

    def __repr__(self) -> str:
        """Return a representation with the secret redacted.

        Returns:
            For example ``Credential(kind='oauth', value='sk-ant-oat01-…', len=108)``.
        """
        return (
            f"{type(self).__name__}(kind={self.kind!r}, "
            f"value={_redact(self.value)}, len={len(self.value)})"
        )

    def is_expired(self, *, now: float | None = None, skew: float = 0.0) -> bool:
        """Report whether the credential is past (or nearly past) its expiry.

        Args:
            now: Epoch seconds to compare against; defaults to the current time.
            skew: Treat the credential as expired this many seconds early.

        Returns:
            ``False`` when no expiry is known, otherwise whether
            ``now + skew`` has reached ``expires_at``.
        """
        if self.expires_at is None:
            return False
        current = time.time() if now is None else now
        return current + skew >= self.expires_at


def _clean(value: str | None) -> str | None:
    """Strip a value and collapse blanks to ``None``.

    Args:
        value: Raw string, possibly ``None`` or whitespace-only.

    Returns:
        The stripped value, or ``None`` when it carried no content.
    """
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _normalize_expiry(raw: object) -> float | None:
    """Coerce an undocumented ``expiresAt`` field into epoch seconds.

    Claude Code writes milliseconds. Values that are already plausibly in
    seconds are passed through, so the function is safe if that ever changes.
    Anything non-numeric degrades to ``None`` rather than raising.

    Args:
        raw: The value found under ``claudeAiOauth.expiresAt``, if any.

    Returns:
        Epoch seconds, or ``None`` when the value is missing or unusable.
    """
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        number = float(raw)
    elif isinstance(raw, str):
        try:
            number = float(raw.strip())
        except ValueError:
            return None
    else:
        return None
    if number != number or number in (float("inf"), float("-inf")):  # NaN / inf
        return None
    if abs(number) >= _MILLISECOND_THRESHOLD:
        return number / 1000.0
    return number


def credentials_file_path() -> Path:
    """Locate Claude Code's on-disk credential file.

    Honors ``CLAUDE_CONFIG_DIR`` and otherwise uses ``~/.claude``. The file is
    only ever read.

    Returns:
        Path to ``.credentials.json`` (which may not exist).
    """
    config_dir = _clean(os.environ.get(ENV_CONFIG_DIR))
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
    return _clean(completed.stdout)


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
    return _clean(raw)


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
    value = _clean(access_token)
    if value is None:
        return None
    raw_refresh = section.get("refreshToken")
    refresh_token = _clean(raw_refresh) if isinstance(raw_refresh, str) else None
    return Credential(
        kind="oauth",
        value=value,
        expires_at=_normalize_expiry(section.get("expiresAt")),
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

    Blank and whitespace-only values are treated as absent at every level.

    Args:
        explicit_api_key: API key supplied directly by the caller.
        explicit_auth_token: Subscription OAuth token supplied directly by the
            caller.

    Returns:
        The highest-precedence credential found.

    Raises:
        AuthError: If no credential could be resolved. The message tells the
            user exactly which command to run and which variable to export, and
            contains no credential material.
    """
    auth_token = _clean(explicit_auth_token)
    if auth_token is not None:
        return Credential(kind="oauth", value=auth_token)

    api_key = _clean(explicit_api_key)
    if api_key is not None:
        return Credential(kind="api_key", value=api_key)

    env_token = _clean(os.environ.get(ENV_OAUTH_TOKEN))
    if env_token is not None:
        return Credential(kind="oauth", value=env_token)

    env_key = _clean(os.environ.get(ENV_API_KEY))
    if env_key is not None:
        return Credential(kind="api_key", value=env_key)

    stored = load_stored_credential()
    if stored is not None:
        return stored

    raise AuthError(
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
    http = client if client is not None else httpx.AsyncClient(timeout=_REFRESH_TIMEOUT_SECONDS)
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
    access_token = _clean(raw_access) if isinstance(raw_access, str) else None
    if access_token is None:
        raise AuthError(
            f"The Anthropic OAuth token endpoint returned no access token (HTTP "
            f"{status}). Re-authenticate with `claude setup-token`."
        )

    raw_refresh = body.get("refresh_token")
    next_refresh = _clean(raw_refresh) if isinstance(raw_refresh, str) else None

    expires_at = _expiry_from_token_response(body)

    return Credential(
        kind="oauth",
        value=access_token,
        expires_at=expires_at,
        refresh_token=next_refresh or cred.refresh_token,
    )


def _expiry_from_token_response(body: dict[str, Any]) -> float | None:
    """Derive an absolute expiry from a token-endpoint payload.

    Prefers the OAuth-standard ``expires_in`` (relative seconds) and falls back
    to an absolute ``expires_at`` / ``expiresAt``.

    Args:
        body: The parsed JSON response.

    Returns:
        Epoch seconds, or ``None`` when the payload said nothing usable.
    """
    expires_in = body.get("expires_in")
    if isinstance(expires_in, (int, float)) and not isinstance(expires_in, bool):
        return time.time() + float(expires_in)
    if isinstance(expires_in, str):
        try:
            return time.time() + float(expires_in.strip())
        except ValueError:
            pass
    for key in ("expires_at", "expiresAt"):
        if key in body:
            normalized = _normalize_expiry(body[key])
            if normalized is not None:
                return normalized
    return None


class CredentialProvider:
    """Holds a credential and refreshes it on demand, single-flight.

    Wrap the credential the provider was built with, then call :meth:`get`
    before every request. Concurrent callers cannot stampede the refresh
    endpoint: the first one through takes the lock and does the work, and the
    rest re-check expiry after acquiring it and reuse the result.

    Attributes:
        skew: How many seconds before expiry a refresh is triggered.
    """

    def __init__(
        self,
        credential: Credential,
        *,
        skew: float = EXPIRY_SKEW_SECONDS,
        refresher: Callable[[Credential], Awaitable[Credential]] | None = None,
    ) -> None:
        """Initialize the provider.

        Args:
            credential: The starting credential, e.g. from
                :func:`resolve_credential`.
            skew: Refresh this many seconds before the recorded expiry.
            refresher: Coroutine used to perform the refresh. Defaults to
                :func:`refresh`; override it in tests or to share an HTTP client.
        """
        self._credential = credential
        self.skew = skew
        self._refresher: Callable[[Credential], Awaitable[Credential]] = (
            refresher if refresher is not None else refresh
        )
        self._lock = asyncio.Lock()

    @classmethod
    def resolve(
        cls,
        explicit_api_key: str | None = None,
        explicit_auth_token: str | None = None,
        *,
        skew: float = EXPIRY_SKEW_SECONDS,
        refresher: Callable[[Credential], Awaitable[Credential]] | None = None,
    ) -> CredentialProvider:
        """Build a provider around :func:`resolve_credential`.

        Args:
            explicit_api_key: API key supplied directly by the caller.
            explicit_auth_token: Subscription OAuth token supplied directly.
            skew: Refresh this many seconds before the recorded expiry.
            refresher: Coroutine used to perform the refresh.

        Returns:
            A provider holding the resolved credential.

        Raises:
            AuthError: If no credential could be resolved.
        """
        credential = resolve_credential(explicit_api_key, explicit_auth_token)
        return cls(credential, skew=skew, refresher=refresher)

    @property
    def current(self) -> Credential:
        """The credential held right now, without refreshing.

        Returns:
            The most recently resolved or refreshed credential.
        """
        return self._credential

    def __repr__(self) -> str:
        """Return a representation that redacts the held credential.

        Returns:
            A ``repr`` safe to log.
        """
        return f"{type(self).__name__}(credential={self._credential!r}, skew={self.skew!r})"

    async def get(self) -> Credential:
        """Return a usable credential, refreshing it first if it is due.

        A credential is due when its ``expires_at`` is within ``skew`` seconds.
        The refresh happens under a lock and is re-checked after acquisition, so
        N concurrent callers trigger exactly one refresh.

        Returns:
            A credential that is not (yet) expired.

        Raises:
            AuthError: If the credential has already expired and cannot be
                refreshed, or if the refresh itself fails.
        """
        credential = self._credential
        if not self._is_due(credential):
            return credential

        async with self._lock:
            credential = self._credential
            if not self._is_due(credential):
                return credential
            if credential.refresh_token is None:
                if credential.is_expired():
                    raise AuthError(
                        "The Claude Code subscription token has expired and no refresh "
                        "token is available. Re-authenticate with `claude setup-token` "
                        f"and export the new token as {ENV_OAUTH_TOKEN}."
                    )
                return credential
            refreshed = await self._refresher(credential)
            self._credential = refreshed
            return refreshed

    def _is_due(self, credential: Credential) -> bool:
        """Report whether a credential is inside the refresh window.

        Args:
            credential: The credential to test.

        Returns:
            ``True`` when a refresh should be attempted.
        """
        return credential.is_expired(skew=self.skew)
