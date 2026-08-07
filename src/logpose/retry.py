"""Provider retry policy for the agent loop.

Retries belong to logpose rather than individual providers so every backend has
the same attempt accounting and streaming guarantee. A turn is retried only
before any assistant-visible delta has been emitted; replaying a partially
streamed turn would duplicate output for the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from random import random

from logpose.errors import LogposeError

__all__ = ["DEFAULT_RETRY_POLICY", "RetryPolicy"]


@dataclass(frozen=True)
class RetryPolicy:
    """Controls retry timing for one provider turn.

    Attributes:
        max_attempts: Total attempts, including the initial request. Set to one
            to disable retries.
        initial_delay: Delay before the first retry, in seconds.
        backoff_multiplier: Factor applied to each later retry delay.
        max_delay: Largest locally calculated delay, in seconds.
        jitter: Symmetric random proportion applied to a local delay. ``0`` is
            deterministic; ``0.2`` varies it by up to 20 percent.
        max_retry_after: Largest server-requested retry delay honoured, in
            seconds.
    """

    max_attempts: int = 3
    initial_delay: float = 0.5
    backoff_multiplier: float = 2.0
    max_delay: float = 8.0
    jitter: float = 0.2
    max_retry_after: float = 60.0

    def __post_init__(self) -> None:
        """Validate policy values at construction time."""
        if isinstance(self.max_attempts, bool) or self.max_attempts < 1:
            raise LogposeError(f"max_attempts must be at least 1, got {self.max_attempts!r}.")
        for name, value, minimum in (
            ("initial_delay", self.initial_delay, 0.0),
            ("backoff_multiplier", self.backoff_multiplier, 1.0),
            ("max_delay", self.max_delay, 0.0),
            ("max_retry_after", self.max_retry_after, 0.0),
        ):
            if value < minimum:
                raise LogposeError(f"{name} must be at least {minimum}, got {value!r}.")
        if not 0.0 <= self.jitter <= 1.0:
            raise LogposeError(f"jitter must be between 0 and 1, got {self.jitter!r}.")
        if self.max_delay < self.initial_delay:
            raise LogposeError(
                f"max_delay ({self.max_delay!r}) must be at least initial_delay "
                f"({self.initial_delay!r})."
            )

    def delay(self, failed_attempt: int, *, retry_after: float | None = None) -> float:
        """Return the delay after a failed attempt.

        Args:
            failed_attempt: One-based number of the attempt that just failed.
            retry_after: Server-requested delay in seconds, if known.

        Returns:
            A non-negative delay. A valid server request takes precedence over a
            shorter locally calculated delay, up to :attr:`max_retry_after`.

        Raises:
            LogposeError: If ``failed_attempt`` is not positive.
        """
        if failed_attempt < 1:
            raise LogposeError(f"failed_attempt must be at least 1, got {failed_attempt!r}.")
        local = min(
            self.initial_delay * self.backoff_multiplier ** (failed_attempt - 1),
            self.max_delay,
        )
        if self.jitter:
            local *= 1.0 + self.jitter * (2.0 * random() - 1.0)
        server = 0.0
        if retry_after is not None:
            server = min(max(retry_after, 0.0), self.max_retry_after)
        return max(local, server)


DEFAULT_RETRY_POLICY = RetryPolicy()
"""Default policy: three attempts with bounded exponential backoff and jitter."""
