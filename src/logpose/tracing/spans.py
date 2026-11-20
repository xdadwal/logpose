"""Span records, the tracing settings, and the per-run span builder.

No database or OpenTelemetry imports. The agent loop drives one private
:class:`_SpanBuilder` per traced run; finished spans go to a :class:`_SpanSink`,
in production the store's background
:class:`~logpose.tracing.writer.SpanWriter`.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import logging
import math
import os
import secrets
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Concatenate, ParamSpec, Protocol, TypeVar

from logpose.errors import LogposeError
from logpose.messages import (
    Message,
    RedactedThinkingBlock,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
)
from logpose.providers._redact import redact
from logpose.tracing import semconv as sc

if TYPE_CHECKING:  # pragma: no cover
    from logpose.providers.base import CompletionDone, CompletionRequest, ToolSpec

__all__ = ["SpanRecord", "Tracing", "new_span_id", "new_trace_id"]

_LOGGER = logging.getLogger("logpose.trace")

TRACE_ENV = "LOGPOSE_TRACE"
TRACE_CONTENT_ENV = "LOGPOSE_TRACE_CONTENT"
_TRUTHY = frozenset({"1", "true", "yes"})

# OTLP enum values.
KIND_INTERNAL = 1
KIND_CLIENT = 3
STATUS_UNSET = 0
STATUS_ERROR = 2

AGENT_NAME = "logpose"
SUMMARY_PURPOSE = "summary"

# logpose stop reasons mapped to the GenAI finish-reason vocabulary.
_FINISH_REASONS = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "tool_use": "tool_call",
    "max_tokens": "length",
    "refusal": "content_filter",
}


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


@dataclass(frozen=True)
class Tracing:
    """Opt-in OpenTelemetry GenAI tracing settings.

    Attributes:
        capture_content: Also record prompts, outputs, reasoning, tool
            arguments and results, and exception messages. Configured API keys
            are scrubbed; headers are never recorded.
        service_name: The ``service.name`` resource attribute on export.
        content_max_chars: Cap per content attribute. Longer values keep their
            end, prefixed with ``…[truncated N chars]``.
        flush_timeout: Seconds a run waits at its end for its spans to be written.
    """

    capture_content: bool = False
    service_name: str = "logpose"
    content_max_chars: int = 64_000
    flush_timeout: float = 1.0

    def __post_init__(self) -> None:
        """Validate the settings.

        Raises:
            LogposeError: If a field has the wrong type or range.
        """
        if not isinstance(self.capture_content, bool):
            raise LogposeError(f"capture_content must be a bool, got {self.capture_content!r}.")
        if not isinstance(self.service_name, str) or not self.service_name.strip():
            raise LogposeError(f"service_name must be a non-empty str, got {self.service_name!r}.")
        if (
            isinstance(self.content_max_chars, bool)
            or not isinstance(self.content_max_chars, int)
            or self.content_max_chars < 1
        ):
            raise LogposeError(
                f"content_max_chars must be a positive int, got {self.content_max_chars!r}."
            )
        if (
            isinstance(self.flush_timeout, bool)
            or not isinstance(self.flush_timeout, (int, float))
            or not math.isfinite(self.flush_timeout)
            or self.flush_timeout < 0
        ):
            raise LogposeError(f"flush_timeout must be a number >= 0, got {self.flush_timeout!r}.")


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUTHY


def tracing_from_env() -> Tracing | None:
    """``Tracing`` from ``LOGPOSE_TRACE`` / ``LOGPOSE_TRACE_CONTENT``, or ``None``.

    ``LOGPOSE_TRACE_CONTENT`` only matters when ``LOGPOSE_TRACE`` is on.
    """
    if not _env_flag(TRACE_ENV):
        return None
    return Tracing(capture_content=_env_flag(TRACE_CONTENT_ENV))


# -- sinks ---------------------------------------------------------------------


class _SpanSink(Protocol):
    """Where finished spans go. ``submit`` must never block or raise into a run."""

    def submit(self, span: SpanRecord) -> None: ...

    async def run_id_exists(self, run_id: str) -> bool: ...

    async def flush(self, run_id: str | None, timeout: float) -> bool: ...

    def pending_for(self, run_id: str) -> int: ...


# -- content helpers -----------------------------------------------------------


def _part(block: object) -> dict[str, Any]:
    if isinstance(block, TextBlock):
        return {"type": "text", "content": block.text}
    if isinstance(block, ThinkingBlock):
        return {"type": "reasoning", "content": block.thinking}
    if isinstance(block, RedactedThinkingBlock):
        return {"type": "reasoning", "content": ""}
    if isinstance(block, ToolUseBlock):
        return {"type": "tool_call", "id": block.id, "name": block.name, "arguments": block.input}
    if isinstance(block, ToolResultBlock):
        return {"type": "tool_call_response", "id": block.tool_use_id, "response": block.content}
    return {"type": str(getattr(block, "type", type(block).__name__))}


def _message_dict(message: Message) -> dict[str, Any]:
    role: str = message.role
    if message.content and all(isinstance(b, ToolResultBlock) for b in message.content):
        role = "tool"
    return {"role": role, "parts": [_part(block) for block in message.content]}


def _messages_json(messages: Sequence[Message]) -> str:
    """GenAI ``gen_ai.input.messages`` JSON for ``messages``."""
    return json.dumps([_message_dict(m) for m in messages], ensure_ascii=False, default=str)


def _usage_attributes(usage: Usage) -> dict[str, Any]:
    return {
        sc.GEN_AI_USAGE_INPUT_TOKENS: usage.input_tokens + usage.cache_read_input_tokens,
        sc.GEN_AI_USAGE_OUTPUT_TOKENS: usage.output_tokens,
        sc.LOGPOSE_USAGE_INPUT_TOKENS: usage.input_tokens,
        sc.LOGPOSE_USAGE_REASONING_TOKENS: usage.reasoning_tokens,
        sc.LOGPOSE_USAGE_CACHE_READ_INPUT_TOKENS: usage.cache_read_input_tokens,
        sc.LOGPOSE_USAGE_CACHE_CREATION_INPUT_TOKENS: usage.cache_creation_input_tokens,
    }


def _error_type(error: BaseException) -> str:
    # Internal signals name the error the caller will actually see.
    named = getattr(error, "trace_error_type", None)
    return named if isinstance(named, str) else type(error).__name__


def _is_cancellation(error: BaseException) -> bool:
    return isinstance(error, (asyncio.CancelledError, GeneratorExit))


# -- the builder ---------------------------------------------------------------


@dataclass(eq=False)
class _OpenSpan:
    name: str
    kind: int
    span_id: str
    parent_span_id: str | None
    start_ns: int
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    status_code: int = STATUS_UNSET
    status_message: str | None = None
    ended: bool = False
    discarded: bool = False


_P = ParamSpec("_P")
_R = TypeVar("_R")


def _guarded(
    method: Callable[Concatenate[_SpanBuilder, _P], _R],
) -> Callable[Concatenate[_SpanBuilder, _P], _R | None]:
    """Never raise into the run: a builder failure disables tracing for the run."""

    @functools.wraps(method)
    def wrapper(self: _SpanBuilder, *args: _P.args, **kwargs: _P.kwargs) -> _R | None:
        if self._disabled:
            return None
        try:
            return method(self, *args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - tracing must never fail a run
            self._disabled = True
            _LOGGER.warning(
                "tracing disabled for run %s: the span builder raised %s",
                self.run_id,
                type(exc).__name__,
            )
            return None

    return wrapper


class _SpanBuilder:
    """Builds one run's spans from the loop's lifecycle points.

    Every public method is guarded: an internal failure is logged once to the
    ``logpose.trace`` logger and turns the rest of the run's tracing off. All
    methods accept ``None`` span handles (from a disabled builder) and no-op.
    """

    def __init__(
        self,
        *,
        tracing: Tracing,
        sink: _SpanSink,
        run_id: str,
        session_id: str | None,
        provider: str,
        model: str,
        metadata: Mapping[str, str | int | float | bool] | None,
        config: Mapping[str, Any],
        system: str | None,
        tool_specs: Sequence[ToolSpec],
        secrets: Sequence[str] = (),
    ) -> None:
        self._disabled = False
        self.run_id = run_id
        self._tracing = tracing
        self._capture = tracing.capture_content
        self._sink = sink
        self._session_id = session_id
        self._provider = provider
        self._model = model
        self._metadata = dict(metadata or {})
        self._config = dict(config)
        self._system = system
        self._tool_specs = tuple(tool_specs)
        self._secrets = tuple(s for s in secrets if s)
        self._trace_id = new_trace_id()
        self._root: _OpenSpan | None = None
        self._open: set[_OpenSpan] = set()
        self._held: list[SpanRecord] | None = None
        self._context: dict[str, Any] = {}

    @classmethod
    def disabled(cls) -> _SpanBuilder:
        """A builder that records nothing (tracing off)."""
        builder = cls.__new__(cls)
        builder._disabled = True
        builder.run_id = ""
        return builder

    @property
    def active(self) -> bool:
        return not self._disabled

    # -- internals -----------------------------------------------------------

    def _scrub(self, text: str) -> str:
        for secret in self._secrets:
            text = text.replace(secret, redact(secret))
        return text

    def _capped(self, span: _OpenSpan, text: str) -> str:
        """Scrub ``text`` and cap it at ``content_max_chars``, keeping the end.

        The one truncation rule for every content value: the result starts with
        ``…[truncated N chars]`` when shortened (N counts every character left
        out), marks the span ``logpose.content.truncated``, and never exceeds
        the cap, even when the cap is shorter than the marker itself.
        """
        text = self._scrub(text)
        cap = self._tracing.content_max_chars
        if len(text) <= cap:
            return text
        span.attributes[sc.LOGPOSE_CONTENT_TRUNCATED] = True
        keep = 0
        for _ in range(4):  # the marker's own length depends on N
            marker = f"…[truncated {len(text) - keep} chars]"
            keep = max(cap - len(marker), 0)
        marker = f"…[truncated {len(text) - keep} chars]"
        if keep == 0:
            return marker[:cap]
        return marker + text[-keep:]

    def _content(self, span: _OpenSpan, key: str, text: str) -> None:
        """Set a content attribute through :meth:`_capped`."""
        span.attributes[key] = self._capped(span, text)

    def _open_span(
        self, name: str, kind: int, *, start_ns: int | None = None, parent: _OpenSpan | None = None
    ) -> _OpenSpan:
        owner = parent if parent is not None else self._root
        span = _OpenSpan(
            name=name,
            kind=kind,
            span_id=new_span_id(),
            parent_span_id=owner.span_id if owner is not None else None,
            start_ns=start_ns if start_ns is not None else time.time_ns(),
        )
        if span is not self._root:
            self._open.add(span)
        return span

    def _fail(self, span: _OpenSpan, error: BaseException) -> None:
        if _is_cancellation(error):
            return
        span.status_code = STATUS_ERROR
        span.attributes[sc.ERROR_TYPE] = _error_type(error)
        if self._capture:
            span.status_message = self._capped(span, str(error))

    def _event(self, span: _OpenSpan, name: str, attributes: Mapping[str, Any]) -> None:
        span.events.append(
            {"name": name, "time_unix_nano": time.time_ns(), "attributes": dict(attributes)}
        )

    def _finish(self, span: _OpenSpan, *, end_ns: int | None = None, guard: bool = False) -> None:
        if span.ended:
            return
        span.ended = True
        self._open.discard(span)
        if span.discarded:
            self._event(span, sc.EVENT_DISCARDED, {})
        record = SpanRecord(
            trace_id=self._trace_id,
            span_id=span.span_id,
            parent_span_id=span.parent_span_id,
            run_id=self.run_id,
            session_id=self._session_id,
            name=span.name,
            kind=span.kind,
            start_ns=span.start_ns,
            end_ns=max(end_ns if end_ns is not None else time.time_ns(), span.start_ns),
            status_code=span.status_code,
            status_message=span.status_message,
            attributes=span.attributes,
            events=span.events,
        )
        if self._held is not None and not guard and span is not self._root:
            self._held.append(record)
        else:
            self._sink.submit(record)

    def _chat_attributes(
        self, model: str, max_tokens: int, attempt: int, purpose: str | None
    ) -> dict[str, Any]:
        attrs: dict[str, Any] = {
            sc.GEN_AI_OPERATION_NAME: sc.OPERATION_CHAT,
            sc.GEN_AI_PROVIDER_NAME: self._provider,
            sc.GEN_AI_REQUEST_MODEL: model,
            sc.GEN_AI_REQUEST_MAX_TOKENS: max_tokens,
            sc.LOGPOSE_ATTEMPT_NUMBER: attempt,
        }
        if purpose is not None:
            attrs[sc.LOGPOSE_CHAT_PURPOSE] = purpose
        else:
            for key, value in self._context.items():
                attrs[sc.LOGPOSE_CONTEXT_PREFIX + key] = value
        return attrs

    # -- run -----------------------------------------------------------------

    @_guarded
    def start_run(self) -> None:
        root = self._open_span(f"{sc.OPERATION_INVOKE_AGENT} {AGENT_NAME}", KIND_INTERNAL)
        self._open.discard(root)
        self._root = root
        attrs = root.attributes
        attrs[sc.GEN_AI_OPERATION_NAME] = sc.OPERATION_INVOKE_AGENT
        attrs[sc.GEN_AI_PROVIDER_NAME] = self._provider
        attrs[sc.GEN_AI_REQUEST_MODEL] = self._model
        attrs[sc.GEN_AI_AGENT_NAME] = AGENT_NAME
        attrs[sc.LOGPOSE_RUN_ID] = self.run_id
        attrs[sc.LOGPOSE_SERVICE_NAME] = self._tracing.service_name
        if self._session_id is not None:
            attrs[sc.GEN_AI_CONVERSATION_ID] = self._session_id
        for key, value in self._metadata.items():
            attrs[sc.LOGPOSE_METADATA_PREFIX + key] = self._scrub(str(value))
        for key, value in self._config.items():
            if value is not None:
                attrs[sc.LOGPOSE_CONFIG_PREFIX + key] = value
        attrs[sc.LOGPOSE_CONFIG_PREFIX + "system_prompt_sha256"] = hashlib.sha256(
            (self._system or "").encode()
        ).hexdigest()
        if self._capture:
            if self._system is not None:
                self._content(
                    root,
                    sc.GEN_AI_SYSTEM_INSTRUCTIONS,
                    json.dumps([{"type": "text", "content": self._system}], ensure_ascii=False),
                )
            if self._tool_specs:
                definitions = [
                    {
                        "type": "function",
                        "name": spec.name,
                        "description": spec.description,
                        "parameters": spec.input_schema,
                    }
                    for spec in self._tool_specs
                ]
                self._content(
                    root,
                    sc.GEN_AI_TOOL_DEFINITIONS,
                    json.dumps(definitions, ensure_ascii=False, default=str),
                )

    @_guarded
    def end_run(
        self,
        *,
        outcome: str,
        iterations: int,
        usage: Usage,
        error: BaseException | None = None,
    ) -> None:
        root = self._root
        if root is None or root.ended:
            return
        self.release()
        for span in list(self._open):  # anything the loop never closed
            self._finish(span)
        root.attributes[sc.LOGPOSE_RUN_OUTCOME] = outcome
        root.attributes[sc.LOGPOSE_RUN_ITERATIONS] = iterations
        root.attributes.update(_usage_attributes(usage))
        if error is not None and outcome not in ("blocked", "cancelled"):
            self._fail(root, error)
        self._finish(root)

    # -- parallel guardrail mode ---------------------------------------------

    @_guarded
    def hold(self) -> None:
        """Buffer finished non-guardrail spans until :meth:`release` or :meth:`discard`."""
        if self._held is None:
            self._held = []

    @_guarded
    def release(self) -> None:
        held, self._held = self._held, None
        for record in held or ():
            self._sink.submit(record)

    @_guarded
    def discard(self) -> None:
        """Mark the held work as discarded (parallel-mode block); never drop it."""
        held, self._held = self._held, None
        if held is None:
            return
        now = time.time_ns()
        for record in held:
            event = {"name": sc.EVENT_DISCARDED, "time_unix_nano": now, "attributes": {}}
            self._sink.submit(replace(record, events=[*record.events, event]))
        for span in self._open:
            if span.name != sc.SPAN_GUARDRAIL:
                span.discarded = True

    # -- chat ----------------------------------------------------------------

    @_guarded
    def set_context(self, diagnostics: Mapping[str, Any]) -> None:
        self._context = dict(diagnostics)

    @_guarded
    def start_chat(
        self,
        request: CompletionRequest,
        attempt: int,
        *,
        first: _OpenSpan | None = None,
        purpose: str | None = None,
    ) -> _OpenSpan:
        span = self._open_span(f"{sc.OPERATION_CHAT} {request.model}", KIND_CLIENT)
        span.attributes.update(
            self._chat_attributes(request.model, request.max_tokens, attempt, purpose)
        )
        if attempt > 1 and first is not None:
            span.attributes[sc.LOGPOSE_INPUT_SAME_AS_SPAN] = first.span_id
        elif self._capture:
            self._content(span, sc.GEN_AI_INPUT_MESSAGES, _messages_json(request.messages))
        return span

    @_guarded
    def end_chat(
        self,
        span: _OpenSpan | None,
        *,
        done: CompletionDone | None = None,
        error: BaseException | None = None,
    ) -> None:
        if span is None or span.ended:
            return
        if done is not None:
            reason = _FINISH_REASONS.get(done.stop_reason, done.stop_reason)
            span.attributes[sc.GEN_AI_RESPONSE_FINISH_REASONS] = [reason]
            span.attributes.update(_usage_attributes(done.usage))
            if self._capture:
                output = _message_dict(done.message)
                output["finish_reason"] = reason
                self._content(
                    span,
                    sc.GEN_AI_OUTPUT_MESSAGES,
                    json.dumps([output], ensure_ascii=False, default=str),
                )
        if error is not None:
            retryable = getattr(error, "retryable", None)
            if isinstance(retryable, bool):
                span.attributes[sc.LOGPOSE_ATTEMPT_RETRYABLE] = retryable
            self._fail(span, error)
        self._finish(span)

    @_guarded
    def retry_scheduled(self, span: _OpenSpan | None, delay: float) -> None:
        if span is not None:
            self._event(span, sc.EVENT_RETRY_SCHEDULED, {sc.LOGPOSE_RETRY_DELAY_SECONDS: delay})

    # -- tools ---------------------------------------------------------------

    def _tool_span(self, call: ToolUseBlock) -> _OpenSpan:
        span = self._open_span(f"{sc.OPERATION_EXECUTE_TOOL} {call.name}", KIND_INTERNAL)
        span.attributes.update(
            {
                sc.GEN_AI_OPERATION_NAME: sc.OPERATION_EXECUTE_TOOL,
                sc.GEN_AI_TOOL_NAME: call.name,
                sc.GEN_AI_TOOL_CALL_ID: call.id,
            }
        )
        if self._capture:
            self._content(
                span,
                sc.GEN_AI_TOOL_CALL_ARGUMENTS,
                json.dumps(call.input, ensure_ascii=False, default=str),
            )
        return span

    @_guarded
    def start_tool(self, call: ToolUseBlock) -> _OpenSpan:
        return self._tool_span(call)

    @_guarded
    def end_tool(
        self,
        span: _OpenSpan | None,
        result: ToolResultBlock | None,
        *,
        queue_seconds: float | None = None,
        timed_out: bool = False,
        error_type: str | None = None,
        error: BaseException | None = None,
    ) -> None:
        if span is None or span.ended:
            return
        if queue_seconds is not None:
            span.attributes[sc.LOGPOSE_TOOL_QUEUE_SECONDS] = queue_seconds
        span.attributes[sc.LOGPOSE_TOOL_TIMED_OUT] = timed_out
        if result is not None:
            span.attributes[sc.LOGPOSE_TOOL_IS_ERROR] = result.is_error
            if self._capture:
                self._content(span, sc.GEN_AI_TOOL_CALL_RESULT, result.content)
        if error is not None:
            self._fail(span, error)
        elif error_type is not None:
            span.status_code = STATUS_ERROR
            span.attributes[sc.ERROR_TYPE] = error_type
        self._finish(span)

    @_guarded
    def gate_blocked(self, call: ToolUseBlock, result: ToolResultBlock) -> None:
        span = self._tool_span(call)
        attributes: dict[str, Any] = {sc.LOGPOSE_GATE_IS_ERROR: result.is_error}
        if self._capture:
            attributes[sc.LOGPOSE_GATE_REPLACEMENT] = self._capped(span, result.content)
        self._event(span, sc.EVENT_GATE_BLOCKED, attributes)
        span.attributes[sc.LOGPOSE_TOOL_IS_ERROR] = result.is_error
        span.attributes[sc.LOGPOSE_TOOL_TIMED_OUT] = False
        if self._capture:
            self._content(span, sc.GEN_AI_TOOL_CALL_RESULT, result.content)
        self._finish(span)

    # -- guardrails and compaction -------------------------------------------

    @_guarded
    def guardrail(
        self,
        *,
        name: str,
        allowed: bool,
        category: str | None,
        scores_json: str,
        reason: str | None,
        usage: object,
        error_type: str | None,
        duration: float,
        text: Callable[[], str],
    ) -> None:
        end_ns = time.time_ns()
        span = self._open_span(
            sc.SPAN_GUARDRAIL, KIND_INTERNAL, start_ns=end_ns - int(max(duration, 0.0) * 1e9)
        )
        attrs = span.attributes
        attrs[sc.LOGPOSE_GUARDRAIL_NAME] = name
        attrs[sc.LOGPOSE_GUARDRAIL_ALLOWED] = allowed
        if category is not None:
            attrs[sc.LOGPOSE_GUARDRAIL_CATEGORY] = category
        attrs[sc.LOGPOSE_GUARDRAIL_SCORES] = scores_json
        if isinstance(usage, Usage):
            attrs.update(_usage_attributes(usage))
        if error_type is not None:
            span.status_code = STATUS_ERROR
            attrs[sc.ERROR_TYPE] = error_type
        if self._capture:
            self._content(span, sc.LOGPOSE_GUARDRAIL_INPUT, text())
            if reason:
                self._content(span, sc.LOGPOSE_GUARDRAIL_REASON, reason)
        self._finish(span, end_ns=end_ns, guard=True)

    @_guarded
    def compaction(
        self,
        *,
        start_ns: int,
        pointers: int,
        summarized: bool,
        before: int,
        after: int,
        budget: int,
        summary: str | None,
    ) -> None:
        span = self._open_span(sc.SPAN_COMPACTION, KIND_INTERNAL, start_ns=start_ns)
        span.attributes.update(
            {
                sc.LOGPOSE_COMPACTION_POINTERS: pointers,
                sc.LOGPOSE_COMPACTION_SUMMARIZED: summarized,
                sc.LOGPOSE_COMPACTION_ESTIMATE_BEFORE: before,
                sc.LOGPOSE_COMPACTION_ESTIMATE_AFTER: after,
                sc.LOGPOSE_COMPACTION_BUDGET: budget,
            }
        )
        if self._capture and summary is not None:
            self._content(span, sc.LOGPOSE_COMPACTION_SUMMARY, summary)
        self._finish(span)


_NO_TRACE = _SpanBuilder.disabled()
"""Shared do-nothing builder for runs without tracing."""
