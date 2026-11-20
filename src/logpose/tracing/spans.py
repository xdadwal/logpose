"""Span records as stored and exported. No database or OpenTelemetry imports."""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import Any

__all__ = ["SpanRecord", "new_span_id", "new_trace_id"]


@dataclass(frozen=True)
class SpanRecord:
    """One finished span. ``kind`` and ``status_code`` use the OTLP enum values."""

    trace_id: str
    span_id: str
    parent_span_id: str | None
    run_id: str
    session_id: str | None
    name: str
    kind: int
    start_ns: int
    end_ns: int
    status_code: int
    status_message: str | None
    attributes: dict[str, Any]
    events: list[dict[str, Any]]


def new_trace_id() -> str:
    """Return a random 32-hex-character trace ID."""
    return secrets.token_hex(16)


def new_span_id() -> str:
    """Return a random 16-hex-character span ID."""
    return secrets.token_hex(8)
