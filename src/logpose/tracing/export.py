"""Export stored spans as OTLP/JSON, one ``ExportTraceServiceRequest`` per line.

The output is the file format read by the OpenTelemetry Collector's
``otlpjsonfile`` receiver. Importing this module does not import aiosqlite; the
store is opened through the store manager when :func:`export_traces` runs.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from logpose.errors import LogposeError
from logpose.tracing import semconv as sc
from logpose.tracing.spans import SpanRecord

__all__ = ["ExportResult", "export_traces"]

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
_US = timedelta(microseconds=1)
_INT64 = 2**63
SERVICE_NAME = "logpose"


@dataclass(frozen=True)
class ExportResult:
    """Outcome of :func:`export_traces`.

    Attributes:
        traces: Number of traces written.
        through: Start time (UTC, microsecond precision) of the latest root span
            exported, or ``None`` when nothing was exported. Pass it as ``since``
            to continue an incremental export.
    """

    traces: int
    through: datetime | None


def _any_value(value: Any) -> dict[str, Any] | None:
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        if -_INT64 <= value < _INT64:
            return {"intValue": str(value)}
        return {"stringValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value} if math.isfinite(value) else {"stringValue": str(value)}
    if isinstance(value, str):
        return {"stringValue": value}
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        items = [v for v in (_any_value(x) for x in value) if v is not None]
        return {"arrayValue": {"values": items}}
    return {"stringValue": json.dumps(value, ensure_ascii=False, default=str)}


def _attributes(attrs: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for key, value in attrs.items():
        converted = _any_value(value)
        if converted is not None:
            out.append({"key": key, "value": converted})
    return out


def _span_json(span: SpanRecord) -> dict[str, Any]:
    out: dict[str, Any] = {
        "traceId": span.trace_id,
        "spanId": span.span_id,
        "name": span.name,
        "kind": span.kind,
        "startTimeUnixNano": str(span.start_ns),
        "endTimeUnixNano": str(span.end_ns),
        "attributes": _attributes(span.attributes),
    }
    if span.parent_span_id:
        out["parentSpanId"] = span.parent_span_id
    status: dict[str, Any] = {}
    if span.status_code:
        status["code"] = span.status_code
    if span.status_message:
        status["message"] = span.status_message
    out["status"] = status
    if span.events:
        out["events"] = [
            {
                "name": str(ev.get("name", "")),
                "timeUnixNano": str(ev.get("time_unix_nano", span.start_ns)),
                "attributes": _attributes(ev.get("attributes") or {}),
            }
            for ev in span.events
        ]
    return out


def _request(spans: list[SpanRecord], version: str, service_name: str) -> dict[str, Any]:
    resource = {
        "service.name": service_name,
        "telemetry.sdk.name": "logpose",
        "telemetry.sdk.version": version,
        "logpose.semconv.version": sc.SEMCONV_VERSION,
    }
    return {
        "resourceSpans": [
            {
                "resource": {"attributes": _attributes(resource)},
                "scopeSpans": [
                    {
                        "scope": {"name": "logpose", "version": version},
                        "schemaUrl": sc.SCHEMA_URL,
                        "spans": [_span_json(s) for s in spans],
                    }
                ],
                "schemaUrl": sc.SCHEMA_URL,
            }
        ]
    }


def _to_us(moment: datetime, name: str) -> int:
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise LogposeError(f"{name} must be a timezone-aware datetime.")
    return (moment - _EPOCH) // _US


async def export_traces(
    path: str | os.PathLike[str],
    *,
    since: datetime | None = None,
    until: datetime | None = None,
    session_id: str | None = None,
    service_name: str = SERVICE_NAME,
) -> ExportResult:
    """Write finished traces from the span store to ``path`` as OTLP/JSON lines.

    Only traces whose root ``invoke_agent`` span has ended are exported. The
    window matches the root span's start time at microsecond resolution:
    ``since`` is exclusive and ``until`` inclusive, so
    ``export_traces(path, since=previous.through)`` never repeats a trace. The file
    is replaced atomically (written to a temporary file, then renamed).

    Args:
        path: Output file; any existing file is replaced.
        since: Export roots that started after this timezone-aware time.
        until: Export roots that started at or before this timezone-aware time.
        session_id: Only export traces of this session.
        service_name: The ``service.name`` resource attribute. Spans do not store
            it, so it is chosen at export time.

    Raises:
        LogposeError: A naive ``since`` or ``until``.
    """
    import logpose
    from logpose.store import manager

    since_ns = None if since is None else (_to_us(since, "since") + 1) * 1000
    until_ns = None if until is None else (_to_us(until, "until") + 1) * 1000 - 1
    version = logpose.__version__
    target = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    target.parent.mkdir(parents=True, exist_ok=True)
    store = await manager.get_store()
    count = 0
    last_start: int | None = None
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            async for root in store.read_spans(
                since_ns=since_ns, until_ns=until_ns, session_id=session_id
            ):
                if root.parent_span_id is not None:
                    continue
                spans = [s async for s in store.read_spans(trace_id=root.trace_id)]
                line = json.dumps(
                    _request(spans, version, service_name),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                handle.write(line + "\n")
                count += 1
                last_start = root.start_ns
        os.replace(tmp_name, target)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise
    through = None if last_start is None else _EPOCH + timedelta(microseconds=last_start // 1000)
    return ExportResult(traces=count, through=through)
