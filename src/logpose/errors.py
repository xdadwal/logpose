"""Exception hierarchy for logpose.

Every error raised by logpose derives from :class:`LogposeError`, so callers can
catch the whole surface with a single ``except LogposeError``.

Security note: never place a credential (API key, OAuth access/refresh token) in
an exception message. If an error must reference a token, redact it to a short
prefix plus its length.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retryable: bool = False,
    ) -> None:
        """Initialize the error.

        Args:
            message: Human-readable description of the failure.
            status_code: HTTP status returned by the provider, if any.
            retryable: Whether retrying the same request may succeed.
        """
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.retryable = retryable

    def __repr__(self) -> str:
        """Return an unambiguous representation of the error."""
        return (
            f"{type(self).__name__}(message={self.message!r}, "
            f"status_code={self.status_code!r}, retryable={self.retryable!r})"
        )


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
    """Raised when a tool handler fails in a way the loop cannot report to the model.

    Note: ordinary handler exceptions are converted into an error
    ``ToolResultBlock`` so the model can adapt. This error is for failures of the
    execution machinery itself (for example, an unknown tool name).
    """
