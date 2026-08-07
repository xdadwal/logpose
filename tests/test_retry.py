"""Tests for the public provider retry policy."""

from __future__ import annotations

import pytest

from logpose import LogposeError, RetryPolicy


def test_default_delay_is_bounded_exponential_without_jitter() -> None:
    policy = RetryPolicy(initial_delay=0.5, backoff_multiplier=2, max_delay=3, jitter=0)
    assert policy.delay(1) == 0.5
    assert policy.delay(2) == 1.0
    assert policy.delay(3) == 2.0
    assert policy.delay(4) == 3.0


def test_server_retry_after_overrides_a_shorter_local_delay() -> None:
    policy = RetryPolicy(initial_delay=0.5, max_delay=8, max_retry_after=60, jitter=0)
    assert policy.delay(1, retry_after=12) == 12
    assert policy.delay(1, retry_after=120) == 60


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_attempts": 0},
        {"initial_delay": -1},
        {"backoff_multiplier": 0.5},
        {"max_delay": -1},
        {"jitter": -0.1},
        {"jitter": 1.1},
        {"max_retry_after": -1},
        {"initial_delay": 2, "max_delay": 1},
    ],
)
def test_invalid_policy_values_fail_at_construction(kwargs: dict[str, float | int]) -> None:
    with pytest.raises(LogposeError):
        RetryPolicy(**kwargs)


def test_delay_rejects_an_invalid_attempt_number() -> None:
    with pytest.raises(LogposeError, match="failed_attempt"):
        RetryPolicy().delay(0)
