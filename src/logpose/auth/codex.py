"""Credential discovery and refresh for OpenAI Codex / ChatGPT access.

logpose prefers a **ChatGPT subscription** OAuth token written by ``codex
login``, falling back to a plain ``OPENAI_API_KEY`` (BYOK). Both paths end at
the same :class:`~logpose.auth._common.Credential`; the Codex provider decides
how to present it on the wire and which endpoint to send it to (the ChatGPT
backend for ``oauth``, ``api.openai.com`` for ``api_key``).

.. warning::

   **Gray area — read this.** Using a ChatGPT *subscription* token against the
   Codex backend is **not an officially supported integration path**. OpenAI
   publishes no contract for it. Doing so may violate the Codex / ChatGPT
   consumer terms of service, and it may stop working at any time without notice
   (the ``auth.json`` layout, the OAuth client id, the token endpoint, and the
   ``backend-api/codex`` path used here are all undocumented internals). If you
   need a supported, stable integration, use an API key (``OPENAI_API_KEY``) and
   accept the metered billing that comes with it. You are responsible for
   deciding whether the subscription path is acceptable for your use.

Discovery of the local Codex credential store is strictly **read-only**: logpose
never writes to ``~/.codex/auth.json``. Refreshed tokens are held in memory for
the life of the process only — see :func:`refresh` for what that costs.

Security
--------
No function in this module puts a credential value into a log line, an exception
message, or a ``repr``. The access token is a JWT and is parsed for its ``exp``
claim, but a malformed token degrades to "no expiry known" rather than raising
with the token in the message; error paths report HTTP status codes but never
response bodies, which can carry tokens.
"""

from __future__ import annotations

import base64
import json
import os
from collections.abc import Awaitable, Callable
from dataclasses import replace
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
    "ACCOUNT_ID_CLAIM_NAMESPACE",
    "ENV_ACCOUNT_ID",
    "ENV_API_KEY",
    "ENV_CODEX_HOME",
    "EXPIRY_SKEW_SECONDS",
    "OAUTH_CLIENT_ID",
    "OAUTH_SCOPE",
    "OAUTH_TOKEN_URL",
    "Credential",
    "CredentialProvider",
    "auth_file_path",
    "codex_home",
    "load_stored_credential",
    "refresh",
    "require_account_id",
    "resolve_credential",
]

ENV_CODEX_HOME = "CODEX_HOME"
"""Environment variable overriding Codex's config directory."""

ENV_API_KEY = "OPENAI_API_KEY"
"""Environment variable holding a plain OpenAI API key (BYOK)."""

ENV_ACCOUNT_ID = "CHATGPT_ACCOUNT_ID"
"""Environment variable naming the ChatGPT account to bill a subscription request to.

An escape hatch. The account id normally comes out of ``auth.json``; this exists
for the case where the store has none and the backend therefore rejects every
request.
"""

OAUTH_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
"""Codex's public OAuth client id (undocumented; may change)."""

OAUTH_TOKEN_URL = "https://auth.openai.com/oauth/token"
"""Endpoint the refresh-token grant is posted to (undocumented; may change)."""

OAUTH_SCOPE = "openid profile email"
"""Scope sent with the refresh-token grant."""

ACCOUNT_ID_CLAIM_NAMESPACE = "https://api.openai.com/auth"
"""Namespaced ``id_token`` claim carrying ``chatgpt_account_id``."""

_SETUP_HINT = (
    "Run `codex login` to authenticate with a ChatGPT subscription, or export "
    f"{ENV_API_KEY} to use an API key instead."
)


def codex_home() -> Path:
    """Locate Codex's config directory.

    Honors ``CODEX_HOME`` and otherwise uses ``~/.codex``.

    Returns:
        The directory path (which may not exist).
    """
    configured = clean(os.environ.get(ENV_CODEX_HOME))
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".codex"


def auth_file_path() -> Path:
    """Locate Codex's on-disk credential file.

    Returns:
        Path to ``auth.json`` inside :func:`codex_home` (which may not exist).
        The file is only ever read.
    """
    return codex_home() / "auth.json"


def _read_auth_file(path: Path) -> str | None:
    """Read the credential file if it is present and readable.

    Args:
        path: Location of ``auth.json``.

    Returns:
        The file's contents, or ``None`` when missing/unreadable/empty. The real
        file is mode ``0600``, so ``PermissionError`` is a realistic outcome when
        another user's home directory is inspected — it means "not found", not
        "crash".
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    return clean(raw)


def _jwt_claims(token: str) -> dict[str, Any]:
    """Decode a JWT's payload segment without verifying its signature.

    Verification would be pointless here: the token came off the local
    filesystem, and the only things read out of it are an expiry hint and a
    tenant id — both of which the server re-checks anyway. Never raises, and
    never puts any part of the token into an exception.

    Args:
        token: The compact-serialization JWT.

    Returns:
        The decoded claims, or ``{}`` for anything that is not a JWT carrying a
        JSON object payload.
    """
    parts = token.split(".")
    if len(parts) != 3:
        return {}
    segment = parts[1]
    try:
        # binascii.Error and UnicodeDecodeError both subclass ValueError.
        raw = base64.urlsafe_b64decode(segment + "=" * (-len(segment) % 4))
        claims: Any = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    return claims if isinstance(claims, dict) else {}


def _jwt_expiry(token: str) -> float | None:
    """Read a JWT's expiry.

    ``auth.json`` records no ``expires_at`` of its own, so the access token's own
    ``exp`` claim is the only expiry signal available before the first refresh.

    Args:
        token: The access token.

    Returns:
        Epoch seconds, or ``None`` when the token carries no usable ``exp``.
    """
    return normalize_epoch(_jwt_claims(token).get("exp"))


def _account_id_from_id_token(id_token: object) -> str | None:
    """Extract the ChatGPT account id from an ``id_token``.

    Args:
        id_token: The ``id_token`` value, which may be missing or the wrong type.

    Returns:
        The ``chatgpt_account_id`` claim, or ``None`` when it is absent.
    """
    if not isinstance(id_token, str):
        return None
    section = _jwt_claims(id_token).get(ACCOUNT_ID_CLAIM_NAMESPACE)
    if not isinstance(section, dict):
        return None
    account_id = section.get("chatgpt_account_id")
    return clean(account_id) if isinstance(account_id, str) else None


def _parse_auth_payload(raw: str) -> Credential | None:
    """Parse Codex's undocumented credential JSON.

    The expected shape is::

        {"auth_mode": "chatgpt", "OPENAI_API_KEY": null,
         "tokens": {"id_token": "...", "access_token": "...",
                    "refresh_token": "...", "account_id": "..."},
         "last_refresh": "2026-08-06T09:49:00Z"}

    Every field is treated as optional and untrusted: a missing key, a renamed
    key, or a wrong type degrades to ``None`` instead of raising.

    ``auth_mode`` is deliberately **not** dispatched on. The CLI produces files
    where it disagrees with the populated fields — during a login transition, or
    when a user switches modes — so going by which field actually holds a value
    is strictly more robust and needs no extra branch. Do not "fix" this.

    Args:
        raw: The JSON text read from ``auth.json``.

    Returns:
        The best credential in the file (``oauth`` if a subscription token is
        present, otherwise the file's own API key), or ``None`` when nothing
        usable was found.
    """
    try:
        data: Any = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None

    tokens = data.get("tokens")
    if isinstance(tokens, dict):
        raw_access = tokens.get("access_token")
        access_token = clean(raw_access) if isinstance(raw_access, str) else None
        if access_token is not None:
            raw_refresh = tokens.get("refresh_token")
            raw_account = tokens.get("account_id")
            account_id = clean(raw_account) if isinstance(raw_account, str) else None
            return Credential(
                kind="oauth",
                value=access_token,
                expires_at=_jwt_expiry(access_token),
                refresh_token=clean(raw_refresh) if isinstance(raw_refresh, str) else None,
                account_id=account_id or _account_id_from_id_token(tokens.get("id_token")),
            )

    raw_key = data.get(ENV_API_KEY)
    api_key = clean(raw_key) if isinstance(raw_key, str) else None
    if api_key is not None:
        return Credential(kind="api_key", value=api_key)
    return None


def load_stored_credential() -> Credential | None:
    """Discover a credential from the local Codex store (read-only).

    Reads ``$CODEX_HOME/auth.json``, or ``~/.codex/auth.json``. Nothing is ever
    written. Unlike Claude Code there is no Keychain entry to consult — Codex
    keeps everything in the file.

    Returns:
        The stored credential, or ``None`` when none could be read.
    """
    raw = _read_auth_file(auth_file_path())
    if raw is None:
        return None
    return _parse_auth_payload(raw)


def _with_account_id(credential: Credential, explicit: str | None) -> Credential:
    """Settle which ChatGPT account a subscription credential acts for.

    Precedence: the caller's argument, then whatever the store supplied, then
    ``$CHATGPT_ACCOUNT_ID``. API keys carry no account id — the header does not
    apply to them — so they pass through untouched.

    Args:
        credential: The credential to annotate.
        explicit: Account id supplied directly by the caller.

    Returns:
        The credential, with ``account_id`` set when one could be determined.
    """
    if credential.kind != "oauth":
        return credential
    resolved = (
        clean(explicit) or credential.account_id or clean(os.environ.get(ENV_ACCOUNT_ID))
    )
    if resolved == credential.account_id:
        return credential
    return replace(credential, account_id=resolved)


def resolve_credential(
    explicit_api_key: str | None = None,
    explicit_auth_token: str | None = None,
    explicit_account_id: str | None = None,
    *,
    require_kind: Literal["api_key", "oauth"] | None = None,
) -> Credential:
    """Resolve the credential logpose should authenticate OpenAI with.

    Precedence, first match wins:

    1. ``explicit_auth_token`` → ``oauth``
    2. ``explicit_api_key`` → ``api_key``
    3. The ``tokens`` block in ``$CODEX_HOME/auth.json`` → ``oauth``
    4. ``OPENAI_API_KEY`` → ``api_key``
    5. ``OPENAI_API_KEY`` *inside* ``auth.json`` → ``api_key``

    ``require_kind`` narrows that list to the sources that can produce the
    requested kind, and is how the providers use this function: each accepts
    exactly one kind, so ``CodexProvider`` sees rules 1 and 3 and
    ``OpenAIProvider`` sees rules 2, 4 and 5. Filtering rather than resolving and
    then rejecting matters — otherwise a subscription token in the store would
    shadow a perfectly good ``$OPENAI_API_KEY`` and the API-key provider would
    refuse a credential it was standing next to.

    With no ``require_kind`` the full chain applies, subscription-first. That
    ordering is only reachable by calling this function directly; note the two
    divergences from :func:`logpose.auth.claude_code.resolve_credential` if you
    do. Rule 3 outranks rule 4 because ``codex login`` writes only to
    ``auth.json`` — there is no ``CODEX_OAUTH_TOKEN`` above it — so demoting the
    store would defeat subscription-first for anyone with a stray key exported for
    unrelated tooling. And there is no OAuth environment variable at all, because
    nothing writes one; ``explicit_auth_token`` covers the programmatic case.

    Blank and whitespace-only values are treated as absent at every level.

    Args:
        explicit_api_key: API key supplied directly by the caller.
        explicit_auth_token: Subscription OAuth token supplied directly by the
            caller.
        explicit_account_id: ChatGPT account id to bill subscription requests
            to, overriding whatever the store or environment says.
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
            return _with_account_id(
                Credential(kind="oauth", value=auth_token), explicit_account_id
            )

    if want_key:
        api_key = clean(explicit_api_key)
        if api_key is not None:
            return Credential(kind="api_key", value=api_key)

    stored = load_stored_credential()
    if want_oauth and stored is not None and stored.kind == "oauth":
        return _with_account_id(stored, explicit_account_id)

    if want_key:
        env_key = clean(os.environ.get(ENV_API_KEY))
        if env_key is not None:
            return Credential(kind="api_key", value=env_key)
        if stored is not None and stored.kind == "api_key":
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
            "No ChatGPT subscription credential found. Run `codex login`, or pass "
            "auth_token=... explicitly."
        )
    if require_kind == "api_key":
        return (
            f"No OpenAI API key found. Export {ENV_API_KEY}, or pass api_key=... "
            "explicitly."
        )
    return (
        "No OpenAI credential found. To use a ChatGPT subscription, run `codex login`. "
        f"To use an API key instead, export {ENV_API_KEY}. You can also pass "
        "api_key=... or auth_token=... explicitly."
    )


def require_account_id(credential: Credential) -> str:
    """Return the ChatGPT account id a subscription credential must carry.

    The Codex backend rejects a subscription request that does not name an
    account, so failing here with an actionable message beats sending a request
    that comes back 401 with an opaque body.

    Args:
        credential: An ``oauth`` credential.

    Returns:
        The account id.

    Raises:
        AuthError: If the credential carries no account id.
    """
    if credential.account_id:
        return credential.account_id
    raise AuthError(
        "The ChatGPT subscription credential names no account, which the Codex backend "
        "requires. Re-authenticate with `codex login`, or pass account_id=... (or export "
        f"{ENV_ACCOUNT_ID})."
    )


async def refresh(
    cred: Credential,
    *,
    client: httpx.AsyncClient | None = None,
) -> Credential:
    """Exchange a refresh token for a fresh subscription access token.

    Posts the standard ``refresh_token`` grant with Codex's public client id. The
    result is returned as a new in-memory :class:`Credential`; nothing is written
    back to ``auth.json``.

    That last point has a consequence worth knowing: if OpenAI rotates the
    refresh token, a long-lived process keeps working, but a *restart* falls back
    to the now-stale token still on disk and the user has to re-run ``codex
    login``. Writing the new token back would fix that and is deliberately not
    done — it would race the Codex CLI for the file and break the read-only
    promise this module makes, which is the worse trade.

    Args:
        cred: The (probably expired) ``oauth`` credential to refresh.
        client: Optional HTTP client to reuse. When omitted a short-lived one is
            created and closed.

    Returns:
        A new ``oauth`` credential carrying the refreshed access token, its
        expiry in epoch seconds, the refresh token to use next time, and the
        account id.

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
            "Cannot refresh the ChatGPT subscription token: no refresh token is "
            "available. Re-authenticate with `codex login`."
        )

    payload = {
        "grant_type": "refresh_token",
        "refresh_token": cred.refresh_token,
        "client_id": OAUTH_CLIENT_ID,
        "scope": OAUTH_SCOPE,
    }

    owns_client = client is None
    timeout = _common.REFRESH_TIMEOUT_SECONDS
    http = client if client is not None else httpx.AsyncClient(timeout=timeout)
    try:
        try:
            response = await http.post(OAUTH_TOKEN_URL, json=payload)
        except httpx.HTTPError as exc:
            raise AuthError(
                "Could not reach the OpenAI OAuth token endpoint to refresh the ChatGPT "
                "subscription token. Check connectivity, then re-authenticate with "
                "`codex login`."
            ) from exc
    finally:
        if owns_client:
            await http.aclose()

    status = response.status_code
    if status >= 400:
        raise AuthError(
            f"Refreshing the ChatGPT subscription token failed with HTTP {status}. "
            "Re-authenticate with `codex login`."
        )

    try:
        body: Any = response.json()
    except ValueError as exc:
        raise AuthError(
            f"The OpenAI OAuth token endpoint returned a non-JSON response (HTTP "
            f"{status}). Re-authenticate with `codex login`."
        ) from exc
    if not isinstance(body, dict):
        raise AuthError(
            f"The OpenAI OAuth token endpoint returned an unexpected payload (HTTP "
            f"{status}). Re-authenticate with `codex login`."
        )

    raw_access = body.get("access_token")
    access_token = clean(raw_access) if isinstance(raw_access, str) else None
    if access_token is None:
        raise AuthError(
            f"The OpenAI OAuth token endpoint returned no access token (HTTP "
            f"{status}). Re-authenticate with `codex login`."
        )

    raw_refresh = body.get("refresh_token")
    next_refresh = clean(raw_refresh) if isinstance(raw_refresh, str) else None

    # The endpoint's own expires_in is preferred, but the access token always
    # carries an `exp`, so a refreshed credential is never expiry-blind.
    expires_at = expiry_from_token_response(body) or _jwt_expiry(access_token)

    return Credential(
        kind="oauth",
        value=access_token,
        expires_at=expires_at,
        refresh_token=next_refresh or cred.refresh_token,
        account_id=_account_id_from_id_token(body.get("id_token")) or cred.account_id,
    )


class CredentialProvider(_common.CredentialProvider):
    """Holds a Codex credential and refreshes it on demand, single-flight.

    Wrap the credential the provider was built with, then call :meth:`get`
    before every request. The refresh loop itself lives in
    :class:`logpose.auth._common.CredentialProvider`; this subclass supplies the
    Codex refresh call and the ChatGPT-specific failure message.

    Attributes:
        skew: How many seconds before expiry a refresh is triggered.
    """

    EXPIRED_MESSAGE = (
        "The ChatGPT subscription token has expired and no refresh token is "
        "available. Re-authenticate with `codex login`."
    )

    async def _refresh(self, credential: Credential) -> Credential:
        """Refresh via the OpenAI OAuth token endpoint.

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
        explicit_account_id: str | None = None,
        *,
        require_kind: Literal["api_key", "oauth"] | None = None,
        skew: float = EXPIRY_SKEW_SECONDS,
        refresher: Callable[[Credential], Awaitable[Credential]] | None = None,
    ) -> CredentialProvider:
        """Build a provider around :func:`resolve_credential`.

        Args:
            explicit_api_key: API key supplied directly by the caller.
            explicit_auth_token: Subscription OAuth token supplied directly.
            explicit_account_id: ChatGPT account id override.
            require_kind: Consider only sources yielding this kind of credential.
            skew: Refresh this many seconds before the recorded expiry.
            refresher: Coroutine used to perform the refresh.

        Returns:
            A provider holding the resolved credential.

        Raises:
            AuthError: If no credential of the requested kind could be resolved.
        """
        credential = resolve_credential(
            explicit_api_key,
            explicit_auth_token,
            explicit_account_id,
            require_kind=require_kind,
        )
        return cls(credential, skew=skew, refresher=refresher)
