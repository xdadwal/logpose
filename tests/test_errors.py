"""Tests for the structured-output error types."""

from __future__ import annotations

from logpose import Message
from logpose.errors import (
    LogposeError,
    OutputRefusalError,
    OutputValidationError,
    ProviderError,
    StructuredOutputUnsupportedError,
)


def test_output_errors_carry_their_fields() -> None:
    messages = [Message.user("hi")]
    errors = [{"loc": ["x"], "type": "missing"}]
    invalid = OutputValidationError(
        "bad", raw_text="{}", errors=errors, messages=messages, output_type=int
    )
    assert invalid.raw_text == "{}"
    assert invalid.errors == errors
    assert invalid.messages is messages
    assert invalid.output_type is int
    assert invalid.reason == "invalid"
    assert isinstance(invalid, LogposeError)

    truncated = OutputValidationError(
        "cut", raw_text="{", errors=[], messages=messages, output_type=int, reason="max_tokens"
    )
    assert truncated.reason == "max_tokens"

    refusal = OutputRefusalError("no", refusal="I can't", messages=messages)
    assert refusal.refusal == "I can't"
    assert refusal.messages is messages
    assert isinstance(refusal, LogposeError)

    assert issubclass(StructuredOutputUnsupportedError, ProviderError)
    unsupported = StructuredOutputUnsupportedError("nope", status_code=400, retryable=True)
    assert unsupported.retryable is False
