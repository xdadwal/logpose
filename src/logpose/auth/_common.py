"""Backend-neutral credential plumbing shared by the auth modules.

Every backend logpose can authenticate against has the same shape of problem:
resolve a secret from somewhere, know when it expires, refresh it before it does,
and never let it reach a log line. That is what lives here. Nothing in this
module may name a vendor, a CLI, an endpoint, or an environment variable —
:mod:`logpose.auth.claude_code` and :mod:`logpose.auth.codex` own all of that.

Two redactions, deliberately not unified
----------------------------------------
:func:`redact_secret` renders ``'sk-ant-oat01-…'`` and is for a ``repr``.
:func:`logpose.providers._redact.redact` renders
``<redacted sk-ant… len=64>`` and is for an error message, where the length is
diagnostic. Both formats are pinned by tests. They look like duplicates and are
not; leave them alone.

Security
--------
No function here puts a credential value into a log line, an exception message,
or a ``repr``. :class:`Credential` redacts itself to a short prefix plus a
length.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

from logpose.errors import AuthError

__all__ = [
    "EXPIRY_SKEW_SECONDS",
    "MILLISECOND_THRESHOLD",
    "REDACT_PREFIX",
    "REFRESH_TIMEOUT_SECONDS",
    "Credential",
    "CredentialProvider",
    "clean",
    "expiry_from_token_response",
    "normalize_epoch",
    "redact_secret",
]

EXPIRY_SKEW_SECONDS = 60.0
"""Refresh this many seconds before a token's actual expiry."""

REDACT_PREFIX = 13
"""Longest prefix of a secret :func:`redact_secret` will ever reveal."""

MILLISECOND_THRESHOLD = 1e11
"""Above this, an epoch timestamp is assumed to be in milliseconds."""

REFRESH_TIMEOUT_SECONDS = 30.0
"""Timeout for a token-refresh request when the caller supplies no client."""


def redact_secret(value: str) -> str:
    """Render a secret as a short quoted prefix plus an ellipsis.

    Never returns more than half of the input, so short values do not leak.

    Args:
        value: The secret to redact.

    Returns:
        A quoted, truncated form safe to place in a ``repr``.
    """
    keep = min(REDACT_PREFIX, len(value) // 2)
    if keep <= 0:
        return "'…'"
    return f"'{value[:keep]}…'"


def clean(value: str | None) -> str | None:
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


def normalize_epoch(raw: object) -> float | None:
    """Coerce an untrusted timestamp field into epoch seconds.

    Credential stores are inconsistent about units: Claude Code writes
    milliseconds, a JWT ``exp`` claim is seconds, and either can arrive as a
    numeric string. Values that are already plausibly in seconds pass through.
    Anything non-numeric degrades to ``None`` rather than raising, and NaN/inf
    are rejected so they cannot poison :meth:`Credential.is_expired`.

    Args:
        raw: The value found in the store or token payload, if any.

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
    if abs(number) >= MILLISECOND_THRESHOLD:
        return number / 1000.0
    return number


def expiry_from_token_response(body: dict[str, Any]) -> float | None:
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
            normalized = normalize_epoch(body[key])
            if normalized is not None:
                return normalized
    return None


@dataclass(frozen=True, repr=False)
class Credential:
    """A resolved credential and how it must be presented.

    Attributes:
        kind: ``"oauth"`` for a subscription token (sent as a bearer token),
            ``"api_key"`` for a plain API key.
        value: The secret itself. Never log, print, or format this into an
            error message; use :meth:`__repr__`, which redacts.
        expires_at: Expiry as epoch **seconds**, or ``None`` when unknown.
            Sources reporting milliseconds are normalized on the way in.
        refresh_token: Token usable to obtain a new ``value``, when one is known.
        account_id: Tenant the credential acts for, when the backend needs one
            named separately from the token — OpenAI's ChatGPT backend requires
            it as a ``chatgpt-account-id`` header. Not a secret, but not
            printed either, because it identifies an account.
    """

    kind: Literal["api_key", "oauth"]
    value: str
    expires_at: float | None = None
    refresh_token: str | None = None
    account_id: str | None = None

    def __repr__(self) -> str:
        """Return a representation with the secret redacted.

        Deliberately does *not* enumerate the dataclass fields: a future field
        holding something sensitive would otherwise start leaking the moment it
        was added.

        Returns:
            For example ``Credential(kind='oauth', value='sk-ant-oat01-…', len=108)``.
        """
        return (
            f"{type(self).__name__}(kind={self.kind!r}, "
            f"value={redact_secret(self.value)}, len={len(self.value)})"
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


class CredentialProvider:
    """Holds a credential and refreshes it on demand, single-flight.

    Wrap the credential a provider was built with, then call :meth:`get` before
    every request. Concurrent callers cannot stampede the refresh endpoint: the
    first one through takes the lock and does the work, and the rest re-check
    expiry after acquiring it and reuse the result.

    Subclasses supply the backend-specific parts by overriding
    :attr:`EXPIRED_MESSAGE` and :meth:`_refresh`. This base class refuses to
    refresh anything, so a backend that forgets to override :meth:`_refresh`
    fails loudly instead of silently reusing a dead token.

    Attributes:
        skew: How many seconds before expiry a refresh is triggered.
    """

    EXPIRED_MESSAGE: str = (
        "The subscription token has expired and no refresh token is available. "
        "Re-authenticate with the vendor's CLI."
    )
    """Message raised when an expired credential cannot be refreshed."""

    def __init__(
        self,
        credential: Credential,
        *,
        skew: float = EXPIRY_SKEW_SECONDS,
        refresher: Callable[[Credential], Awaitable[Credential]] | None = None,
    ) -> None:
        """Initialize the provider.

        Args:
            credential: The starting credential.
            skew: Refresh this many seconds before the recorded expiry.
            refresher: Coroutine used to perform the refresh. Defaults to this
                class's :meth:`_refresh`; override it in tests or to share an
                HTTP client.
        """
        self._credential = credential
        self.skew = skew
        self._refresher: Callable[[Credential], Awaitable[Credential]] = (
            refresher if refresher is not None else self._refresh
        )
        self._lock = asyncio.Lock()

    async def _refresh(self, credential: Credential) -> Credential:
        """Exchange an expiring credential for a fresh one.

        Args:
            credential: The credential due for renewal.

        Returns:
            A refreshed credential.

        Raises:
            AuthError: Always, in this base class. Backends that support
                refresh override this method.
        """
        raise AuthError(self.EXPIRED_MESSAGE)

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
                    raise AuthError(self.EXPIRED_MESSAGE)
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
