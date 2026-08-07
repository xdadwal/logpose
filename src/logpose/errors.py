"""Exception hierarchy for logpose.

Every error raised by logpose derives from :class:`LogposeError`, so callers can
catch the whole surface with a single ``except LogposeError``.

Security note: never place a credential (API key, OAuth access/refresh token) in
an exception message. If an error must reference a token, redact it to a short
prefix plus its length.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:  # pragma: no cover - import cycle avoidance only
    from logpose.messages import Message

__all__ = [
    "LogposeError",
    "AuthError",
    "ProviderError",
    "MaxIterationsError",
    "ToolSchemaError",
    "ToolExecutionError",
]


class LogposeError(Exception):
    """Base class for every error raised by logpose."""


class AuthError(LogposeError):
    """Raised when no usable credential could be resolved, or a token is rejected.

    Never include the credential value itself in the message; redact to a prefix
    and a length instead.
    """


class ProviderError(LogposeError):
    """Raised when an upstream model provider fails a request.

    Wraps a provider SDK exception while preserving the HTTP status and whether
    the failure is worth retrying. Chain the original exception with
    ``raise ProviderError(...) from exc`` so ``__cause__`` is populated.

    Attributes:
        message: Human-readable description of the failure.
        status_code: HTTP status returned by the provider, if any.
        retryable: Whether retrying the same request may succeed.
        error_code: Provider-specific machine-readable failure code, if known.
        request_id: Provider request identifier, if supplied.
        retry_after: Suggested delay in seconds before retrying, if supplied.
        partial: Whether provider output was already delivered before failure.
        attempts: Number of attempts made for this logical provider turn.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retryable: bool = False,
        error_code: str | None = None,
        request_id: str | None = None,
        retry_after: float | None = None,
        partial: bool = False,
        attempts: int = 1,
    ) -> None:
        """Initialize the error.

        Args:
            message: Human-readable description of the failure.
            status_code: HTTP status returned by the provider, if any.
            retryable: Whether retrying the same request may succeed.
            error_code: Provider-specific machine-readable failure code, if known.
            request_id: Provider request identifier, if supplied.
            retry_after: Suggested delay in seconds before retrying, if supplied.
            partial: Whether provider output was already delivered before failure.
            attempts: Number of attempts made for this logical provider turn.
        """
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.retryable = retryable
        self.error_code = error_code
        self.request_id = request_id
        self.retry_after = retry_after
        self.partial = partial
        self.attempts = attempts

    def __repr__(self) -> str:
        """Return an unambiguous representation of the error."""
        return (
            f"{type(self).__name__}(message={self.message!r}, "
            f"status_code={self.status_code!r}, retryable={self.retryable!r}, "
            f"error_code={self.error_code!r}, request_id={self.request_id!r}, "
            f"retry_after={self.retry_after!r}, partial={self.partial!r}, "
            f"attempts={self.attempts!r})"
        )


class _ProviderHeaderMetadata(TypedDict, total=False):
    """Normalized, safe metadata that may accompany a provider failure."""

    request_id: str
    retry_after: float


def _provider_metadata_from_headers(
    headers: Mapping[str, str] | None,
) -> _ProviderHeaderMetadata:
    """Extract safe retry metadata from provider response headers.

    Private because providers should expose the normalized values through
    :class:`ProviderError`, not make applications depend on header names.
    """
    if headers is None:
        return {}
    normalized = {str(key).lower(): str(value) for key, value in headers.items()}
    request_id = next(
        (
            normalized[name]
            for name in ("x-request-id", "request-id", "anthropic-request-id")
            if normalized.get(name)
        ),
        None,
    )
    retry_after = _parse_retry_after(normalized.get("retry-after"))
    metadata: _ProviderHeaderMetadata = {}
    if request_id is not None:
        metadata["request_id"] = request_id
    if retry_after is not None:
        metadata["retry_after"] = retry_after
    return metadata


def _parse_retry_after(value: str | None) -> float | None:
    """Parse an HTTP ``Retry-After`` value into a non-negative duration."""
    if value is None:
        return None
    try:
        return max(float(value), 0.0)
    except ValueError:
        pass
    try:
        target = parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError, OverflowError):
        return None
    if target.tzinfo is None:
        target = target.replace(tzinfo=timezone.utc)
    return max((target - datetime.now(timezone.utc)).total_seconds(), 0.0)


def _provider_error_code_from_body(body: str) -> str | None:
    """Extract a provider error code from a JSON response without retaining its body."""
    try:
        payload = json.loads(body)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    error = payload.get("error")
    source = error if isinstance(error, dict) else payload
    for key in ("code", "type"):
        value = source.get(key)
        if isinstance(value, str) and value:
            return value
    return None


class MaxIterationsError(LogposeError):
    """Raised when the agentic loop exceeds its iteration cap.

    Carries the partial conversation so callers can inspect what happened or
    resume the run with a higher cap.

    Attributes:
        message: Human-readable description of the failure.
        messages: The full conversation accumulated so far, including the input.
        max_iterations: The cap that was exceeded, if known.
    """

    def __init__(
        self,
        message: str,
        messages: list[Message],
        max_iterations: int | None = None,
    ) -> None:
        """Initialize the error.

        Args:
            message: Human-readable description of the failure.
            messages: The full conversation accumulated so far, including the input.
            max_iterations: The cap that was exceeded, if known.
        """
        super().__init__(message)
        self.message = message
        self.messages = messages
        self.max_iterations = max_iterations


class ToolSchemaError(LogposeError):
    """Raised when a tool definition cannot be turned into a valid JSON schema."""


class ToolExecutionError(LogposeError):
    """Raised when a tool cannot be invoked or its handler raises.

    ``safe_to_expose`` distinguishes validation feedback that is useful to send
    back to a model from an arbitrary handler exception, whose message may
    contain application data or credentials. The agent always renders the
    latter safely by default.
    """

    def __init__(self, message: str, *, safe_to_expose: bool = False) -> None:
        """Initialize the error.

        Args:
            message: Internal failure description.
            safe_to_expose: Whether ``message`` contains argument-validation
                feedback that can safely be returned to the model.
        """
        super().__init__(message)
        self.safe_to_expose = safe_to_expose
