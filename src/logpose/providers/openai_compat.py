"""OpenAI-compatible Chat Completions provider.

This is the built-in implementation of :class:`~logpose.providers.base.Provider`.
It depends only on the neutral message model, and nothing above
``logpose.providers`` is specific to it.

Why raw ``httpx`` and not the ``openai`` SDK
--------------------------------------------
``httpx`` is already a logpose dependency, so a local-model backend adds none.
It is also the more forgiving choice: self-hosted servers are only *mostly*
OpenAI-compatible, and hand-rolled SSE parsing lets us tolerate the differences
(missing ``finish_reason``, absent tool-call ``id``, unparseable tool arguments)
instead of raising on them.

The three real impedance mismatches with logpose's model
--------------------------------------------------------
1. **Tool results.** logpose batches every result of a turn into one ``user``
   message; Chat Completions wants one ``{"role": "tool"}`` message per call.
   :func:`_messages_to_wire` fans them out, preserving order.
2. **Tool arguments.** logpose stores parsed JSON; Chat Completions streams a
   *string* in fragments that must be concatenated by ``index`` and parsed at
   the end. Small local models routinely emit invalid JSON here, so a parse
   failure is reported to the model rather than raised (see
   :data:`UNPARSED_ARGUMENTS_KEY`).
3. **Reasoning.** There is no signature to preserve, so thinking is *not* sent
   back — matching how OpenAI-style APIs work. Reasoning arrives either as a
   ``reasoning_content`` delta or as ``<think>`` tags inside ordinary content;
   both are normalised to :class:`~logpose.messages.ThinkingBlock`.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator, Mapping
from typing import Any, Literal

import httpx

from logpose.errors import (
    AuthError,
    LogposeError,
    ProviderError,
    StructuredOutputUnsupportedError,
    _provider_error_code_from_body,
    _provider_metadata_from_headers,
)
from logpose.messages import (
    ContentBlock,
    Message,
    StopReason,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
)
from logpose.providers._redact import redact, scrub_exception_in_place
from logpose.providers._toolargs import UNPARSED_ARGUMENTS_KEY, parse_tool_arguments
from logpose.providers.base import (
    CompletionDone,
    CompletionRequest,
    ProviderEvent,
    ProviderTextDelta,
    ProviderThinkingDelta,
    ToolSpec,
)

__all__ = [
    "AUTO_MODEL",
    "DEFAULT_HTTP_TIMEOUT",
    "ENV_API_KEY",
    "ENV_BASE_URL",
    "UNPARSED_ARGUMENTS_KEY",
    "OpenAICompatProvider",
]

ENV_BASE_URL = "LOGPOSE_BASE_URL"
"""Environment variable read when ``base_url=`` is not passed."""

ENV_API_KEY = "LOGPOSE_API_KEY"
"""Environment variable read when ``api_key=`` is not passed."""

DEFAULT_HTTP_TIMEOUT = httpx.Timeout(connect=5.0, read=60.0, write=10.0, pool=5.0)
"""Web-app safe HTTP timeouts. ``read`` is the longest silence allowed between
streamed chunks, not a cap on the whole turn."""

AUTO_MODEL = "auto"
"""Sentinel model id meaning "ask the server what it has and use the first one".

Discovery happens only when a request names this model explicitly.

Resolution happens on the first request, not in ``__init__``, so constructing a
provider never touches the network.
"""

_RETRYABLE_STATUS = frozenset({408, 409, 429})

_FINISH_REASONS: dict[str, StopReason] = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "length": "max_tokens",
    "content_filter": "refusal",
}

_STRUCTURED_REJECTION_MARKERS = ("response_format", "json_schema", "strict", "structured")

_PROVIDER_REASON_LIMIT = 500


def _is_strict_json_schema_request(body: Mapping[str, Any]) -> bool:
    """Return whether ``body`` asks for ``json_schema`` structured output."""
    response_format = body.get("response_format")
    return isinstance(response_format, dict) and response_format.get("type") == "json_schema"


def _provider_error_message_from_body(body: str) -> str | None:
    """Return ``error.message`` from a JSON error body, or ``None``."""
    try:
        data = json.loads(body)
    except ValueError:
        return None
    error = data.get("error") if isinstance(data, dict) else None
    message = error.get("message") if isinstance(error, dict) else None
    return message.strip() or None if isinstance(message, str) else None

_CONNECT_HINT = "Could not reach the model server at {base_url}. Check that it is running."


# ---------------------------------------------------------------------------
# <think> tag handling
# ---------------------------------------------------------------------------


class _ThinkTagSplitter:
    """Split ``<think>`` reasoning out of a streamed content channel.

    Many local reasoning models (DeepSeek-R1, QwQ, Qwen3 and their GGUF ports)
    have no ``reasoning_content`` field and inline reasoning into ordinary
    content instead. Tags routinely straddle chunk boundaries, so this holds
    back any trailing text that could still turn out to be a tag.
    """

    OPEN = "<think>"
    CLOSE = "</think>"

    def __init__(self) -> None:
        self._buf = ""
        self._inside = False

    def feed(self, chunk: str) -> tuple[str, str]:
        """Consume a content fragment.

        Args:
            chunk: The newly arrived content text.

        Returns:
            ``(visible_text, thinking_text)`` for this fragment. Either may be
            empty; text that might still be part of a split tag is withheld
            until a later ``feed`` or :meth:`flush` resolves it.
        """
        self._buf += chunk
        visible: list[str] = []
        thinking: list[str] = []

        while True:
            target = self.CLOSE if self._inside else self.OPEN
            index = self._buf.find(target)
            if index == -1:
                break
            head, self._buf = self._buf[:index], self._buf[index + len(target) :]
            (thinking if self._inside else visible).append(head)
            self._inside = not self._inside

        hold = self._partial_tag_len()
        if hold:
            emit, self._buf = self._buf[: len(self._buf) - hold], self._buf[len(self._buf) - hold :]
        else:
            emit, self._buf = self._buf, ""
        (thinking if self._inside else visible).append(emit)
        return "".join(visible), "".join(thinking)

    def flush(self) -> tuple[str, str]:
        """Release any withheld text at end of stream.

        Returns:
            ``(visible_text, thinking_text)`` for whatever is left over — an
            unterminated ``<think>`` keeps its text as reasoning.
        """
        rest, self._buf = self._buf, ""
        return ("", rest) if self._inside else (rest, "")

    def _partial_tag_len(self) -> int:
        """Length of the trailing buffer that could still become a tag."""
        target = self.CLOSE if self._inside else self.OPEN
        for size in range(min(len(target) - 1, len(self._buf)), 0, -1):
            if self._buf.endswith(target[:size]):
                return size
        return 0


# ---------------------------------------------------------------------------
# wire translation
# ---------------------------------------------------------------------------


def _messages_to_wire(
    messages: list[Message], system: str | None, *, preserve_reasoning: bool = False
) -> list[dict[str, Any]]:
    """Translate logpose messages into Chat Completions messages.

    Args:
        messages: Conversation history in logpose's neutral model.
        system: Optional system prompt, emitted as the leading message.
        preserve_reasoning: Replay unmodified thinking for servers requiring it.

    Returns:
        Chat Completions message dicts. One logpose message may become several:
        a user turn carrying N tool results becomes N ``tool`` messages.
    """
    wire: list[dict[str, Any]] = []
    if system:
        wire.append({"role": "system", "content": system})

    for message in messages:
        tool_results = [b for b in message.content if isinstance(b, ToolResultBlock)]
        text = "".join(b.text for b in message.content if isinstance(b, TextBlock))
        tool_uses = [b for b in message.content if isinstance(b, ToolUseBlock)]

        # Tool results are their own role and must precede any further user text.
        for result in tool_results:
            wire.append(
                {
                    "role": "tool",
                    "tool_call_id": result.tool_use_id,
                    "content": _error_prefixed(result),
                }
            )

        if message.role == "assistant":
            # Generic servers may reject reasoning_content. Replay is opt-in
            # for providers whose tool protocol requires the original reasoning.
            reasoning = "".join(b.thinking for b in message.content if isinstance(b, ThinkingBlock))
            if not text and not tool_uses and not (preserve_reasoning and reasoning):
                continue
            entry: dict[str, Any] = {"role": "assistant", "content": text}
            if preserve_reasoning:
                entry["reasoning_content"] = reasoning
            if tool_uses:
                entry["tool_calls"] = [
                    {
                        "id": use.id,
                        "type": "function",
                        "function": {"name": use.name, "arguments": json.dumps(use.input)},
                    }
                    for use in tool_uses
                ]
            wire.append(entry)
        elif text:
            wire.append({"role": "user", "content": text})

    return wire


def _error_prefixed(result: ToolResultBlock) -> str:
    """Render a tool result, making failure visible without an ``is_error`` field.

    Chat Completions has no error flag on tool messages, so the signal has to
    live in the text or the model cannot tell success from failure.
    """
    return f"ERROR: {result.content}" if result.is_error else result.content


def _tool_to_wire(spec: ToolSpec) -> dict[str, Any]:
    """Translate a :class:`~logpose.providers.base.ToolSpec` to a function tool."""
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.input_schema,
        },
    }


def _map_usage(raw: Any) -> Usage:
    """Translate a Chat Completions ``usage`` object into logpose's model.

    ``prompt_tokens`` counts cached tokens too, whereas logpose's
    ``input_tokens`` means *uncached* prompt tokens, so the cached portion is
    subtracted out rather than double-counted.
    """
    if not isinstance(raw, dict):
        return Usage()
    prompt = int(raw.get("prompt_tokens") or 0)
    details = raw.get("prompt_tokens_details")
    cached = int(details.get("cached_tokens") or 0) if isinstance(details, dict) else 0
    cached = min(cached, prompt)
    completion_details = raw.get("completion_tokens_details")
    reasoning = (
        int(completion_details.get("reasoning_tokens") or 0)
        if isinstance(completion_details, dict)
        else 0
    )
    return Usage(
        input_tokens=max(prompt - cached, 0),
        output_tokens=int(raw.get("completion_tokens") or 0),
        cache_read_input_tokens=cached,
        reasoning_tokens=reasoning,
    )


def _status_is_retryable(status_code: int | None) -> bool:
    """Whether a failed request is worth retrying unchanged."""
    if status_code is None:
        return True
    return status_code in _RETRYABLE_STATUS or status_code >= 500


# ---------------------------------------------------------------------------
# provider
# ---------------------------------------------------------------------------


def _env(name: str) -> str | None:
    """Read an environment variable, treating blank values as unset."""
    value = os.environ.get(name, "").strip()
    return value or None


class OpenAICompatProvider:
    """A :class:`~logpose.providers.base.Provider` for any Chat Completions API.

    Works against llama.cpp, vLLM, Ollama, LM Studio, OpenAI itself, Kimi, and
    anything else exposing ``POST /chat/completions``. Always streams.

    The model is chosen per request (``CompletionRequest.model``); the provider
    holds no model of its own.

    Attributes:
        name: Provider identifier.
        max_tokens: Output ceiling honoured by an ``Agent`` built without one,
            or ``None`` to leave the choice to the agent.
    """

    name = "openai-compat"

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        max_tokens: int | None = None,
        max_tokens_field: Literal["max_completion_tokens", "max_tokens"] = (
            "max_completion_tokens"
        ),
        timeout: float | httpx.Timeout | None = DEFAULT_HTTP_TIMEOUT,
        parse_think_tags: bool = True,
        default_headers: dict[str, str] | None = None,
        client: httpx.AsyncClient | None = None,
        extra_body: dict[str, Any] | None = None,
        preserve_reasoning: bool = False,
    ) -> None:
        """Configure the backend.

        Args:
            base_url: Root of the OpenAI-compatible API, e.g.
                ``"https://api.openai.com/v1"``. Falls back to
                ``$LOGPOSE_BASE_URL``.
            api_key: Bearer token. Optional — local servers usually need none.
                Falls back to ``$LOGPOSE_API_KEY`` only when omitted; pass
                ``api_key=""`` to send no ``Authorization`` header even when the
                env var is set.
            max_tokens: Output-token ceiling for a turn, or ``None`` to let the
                agent decide.
            max_tokens_field: Request field that carries the output ceiling.
                ``"max_completion_tokens"`` (default) is what OpenAI, vLLM and
                llama.cpp expect, and OpenAI reasoning models require it. Use
                ``"max_tokens"`` for servers that only accept the legacy field.
            timeout: httpx timeout. Defaults to :data:`DEFAULT_HTTP_TIMEOUT`
                (connect 5 s, read 60 s, write 10 s, pool 5 s); ``None`` disables
                every HTTP timeout.
            parse_think_tags: Split ``<think>`` blocks out of content into
                reasoning. Harmless for models that do not use them.
            default_headers: Extra headers sent with every request.
            client: Inject a pre-built ``httpx.AsyncClient`` (used by tests).
            extra_body: Fields merged into every request body, e.g.
                ``{"temperature": 0.2}``.
            preserve_reasoning: Replay thinking blocks as reasoning_content.
                Enable only for a server that explicitly supports this field.

        Raises:
            LogposeError: If no ``base_url`` is available.
        """
        resolved_base = (base_url or "").strip() or _env(ENV_BASE_URL)
        if not resolved_base:
            raise LogposeError(
                "No base_url for the OpenAI-compatible provider. Pass base_url=... or set "
                f"{ENV_BASE_URL}."
            )
        if timeout is None:
            # Explicitly unlimited: httpx's own 5s default would otherwise apply.
            timeout = httpx.Timeout(None)

        self.base_url = resolved_base.rstrip("/")
        if max_tokens_field not in ("max_completion_tokens", "max_tokens"):
            raise LogposeError(
                "max_tokens_field must be 'max_completion_tokens' or 'max_tokens'; "
                f"got {max_tokens_field!r}."
            )
        self.max_tokens = max_tokens
        self.max_tokens_field = max_tokens_field
        self.parse_think_tags = parse_think_tags
        self.preserve_reasoning = preserve_reasoning
        # An explicit api_key always wins, and a blank one means "no auth": it is
        # the only way to keep an ambient LOGPOSE_API_KEY away from this server.
        self._api_key = (api_key.strip() or None) if api_key is not None else _env(ENV_API_KEY)
        self._default_headers = dict(default_headers or {})
        self._extra_body = dict(extra_body or {})
        self._client = client
        self._owns_client = client is None
        self._timeout = timeout
        self._discovered_model: str | None = None

    def __repr__(self) -> str:
        """Describe the provider without revealing the API key."""
        key = redact(self._api_key) if self._api_key else None
        return (
            f"{type(self).__name__}(base_url={self.base_url!r}, "
            f"api_key={key!r})"
        )

    # -- plumbing -----------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        """Build request headers, including bearer auth when configured."""
        headers = {"Content-Type": "application/json", **self._default_headers}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _http(self) -> httpx.AsyncClient:
        """Return the shared client, creating it on first use."""
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=self._timeout)
        return self._client

    async def aclose(self) -> None:
        """Close the underlying HTTP client if this provider created it."""
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    async def list_models(self) -> list[str]:
        """Fetch the model ids the server currently serves.

        Returns:
            Model identifiers, in the order the server reported them.

        Raises:
            ProviderError: If the server cannot be reached or rejects the call.
        """
        try:
            response = await self._http().get(f"{self.base_url}/models", headers=self._headers())
        except httpx.HTTPError as exc:
            raise self._transport_error(exc) from exc
        if response.status_code >= 400:
            raise self._status_error(response.status_code, response.text, response.headers)
        payload = response.json()
        entries = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(entries, list):
            return []
        return [e["id"] for e in entries if isinstance(e, dict) and isinstance(e.get("id"), str)]

    async def _resolve_model(self, requested: str) -> str:
        """Turn a requested model id into a concrete one.

        Args:
            requested: The request's model, or :data:`AUTO_MODEL`.

        Returns:
            A concrete model id.

        Raises:
            LogposeError: If the request names no model.
            ProviderError: If discovery is needed and the server serves nothing.
        """
        if not requested.strip():
            raise LogposeError(
                "No model in the request. Pass model=... to Agent, or model='auto' to pick "
                "the server's first model."
            )
        if requested != AUTO_MODEL:
            return requested
        if self._discovered_model:
            return self._discovered_model
        models = await self.list_models()
        if not models:
            raise ProviderError(
                f"{self.base_url} reports no available models. Load one on the server, "
                "or pass an explicit model=... to Agent.",
                status_code=None,
                retryable=False,
            )
        self._discovered_model = models[0]
        return self._discovered_model

    # -- errors -------------------------------------------------------------

    def _secrets(self) -> tuple[str, ...]:
        """Credential values that must never appear in an error."""
        return (self._api_key,) if self._api_key else ()

    def _scrub(self, text: str) -> str:
        """Replace any credential that leaked into ``text`` with a redaction."""
        for secret in self._secrets():
            text = text.replace(secret, redact(secret))
        return text

    def _status_error(
        self,
        status: int,
        body: str,
        headers: Mapping[str, str] | None = None,
    ) -> ProviderError | AuthError:
        """Build a safe error from an HTTP failure."""
        if status == 401:
            # This backend may be local and unauthenticated, so the guidance is
            # intentionally conditional. Never include the response body: a
            # compatible server can reflect the bearer token in it.
            return AuthError(
                f"{self.name} authentication failed. Check api_key=... or the "
                f"${ENV_API_KEY} environment variable if this server requires authentication."
            )
        return ProviderError(
            f"{self.name} request failed with status {status}: {self._scrub(body[:2000])}",
            status_code=status,
            retryable=_status_is_retryable(status),
            error_code=_provider_error_code_from_body(body),
            **_provider_metadata_from_headers(headers),
        )

    def _structured_unsupported_error(
        self, status: int, body: str, headers: Mapping[str, str]
    ) -> StructuredOutputUnsupportedError:
        """Build the error for a server rejecting a strict ``json_schema`` request.

        Only the provider's parsed ``error.message`` is included (scrubbed and
        truncated), never the raw body.
        """
        message = (
            f"{self.base_url} does not support strict json_schema structured output for "
            "this model (the server rejected the response_format). logpose requires it "
            "for output_type; use a compatible server or model."
        )
        reason = _provider_error_message_from_body(body)
        if reason:
            reason = self._scrub(reason)
            if len(reason) > _PROVIDER_REASON_LIMIT:
                reason = reason[:_PROVIDER_REASON_LIMIT] + "…"
            message += f" Provider said: {reason}"
        return StructuredOutputUnsupportedError(
            message,
            status_code=status,
            error_code=_provider_error_code_from_body(body),
            **_provider_metadata_from_headers(headers),
        )

    def _transport_error(self, exc: httpx.HTTPError) -> ProviderError:
        """Build a :class:`~logpose.errors.ProviderError` from a transport failure."""
        scrub_exception_in_place(exc, self._secrets())
        if isinstance(exc, httpx.ConnectError):
            return ProviderError(
                _CONNECT_HINT.format(base_url=self.base_url),
                status_code=None,
                retryable=True,
            )
        return ProviderError(
            f"Could not reach {self.base_url}: {self._scrub(str(exc))}",
            status_code=None,
            retryable=True,
        )

    # -- request ------------------------------------------------------------

    def _build_body(self, req: CompletionRequest, model: str) -> dict[str, Any]:
        """Assemble the Chat Completions request body."""
        body: dict[str, Any] = {
            "model": model,
            "messages": _messages_to_wire(
                req.messages, req.system, preserve_reasoning=self.preserve_reasoning
            ),
            self.max_tokens_field: req.max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if req.tools:
            body["tools"] = [_tool_to_wire(spec) for spec in req.tools]
        if "response_format" in self._extra_body and "response_format" in req.extra:
            raise LogposeError(
                "response_format is set both in the provider's extra_body and on the request. "
                "Remove one; structured output owns response_format."
            )
        body.update(self._extra_body)
        body.update(req.extra)
        return body

    # -- streaming ----------------------------------------------------------

    async def stream(self, req: CompletionRequest) -> AsyncIterator[ProviderEvent]:
        """Run one turn and stream its events.

        Args:
            req: The turn to run.

        Yields:
            :data:`~logpose.providers.base.ProviderEvent` values, ending with
            exactly one :class:`~logpose.providers.base.CompletionDone`.

        Raises:
            ProviderError: If the server is unreachable or returns an error.
        """
        model = await self._resolve_model(req.model)
        body = self._build_body(req, model)

        text_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_frags: dict[int, dict[str, Any]] = {}
        splitter = _ThinkTagSplitter() if self.parse_think_tags else None
        finish_reason: str | None = None
        usage = Usage()
        refusal_parts: list[str] = []

        try:
            async with self._http().stream(
                "POST", f"{self.base_url}/chat/completions", json=body, headers=self._headers()
            ) as response:
                if response.status_code >= 400:
                    raw = await response.aread()
                    text = raw.decode("utf-8", "replace")
                    if (
                        _is_strict_json_schema_request(body)
                        and response.status_code in (400, 422)
                        and any(m in text.lower() for m in _STRUCTURED_REJECTION_MARKERS)
                    ):
                        raise self._structured_unsupported_error(
                            response.status_code, text, response.headers
                        )
                    raise self._status_error(response.status_code, text, response.headers)
                async for line in response.aiter_lines():
                    chunk = _parse_sse_line(line)
                    if chunk is None:
                        continue

                    if isinstance(chunk.get("usage"), dict):
                        usage = _map_usage(chunk["usage"])

                    for choice in chunk.get("choices") or []:
                        if not isinstance(choice, dict):
                            continue
                        if choice.get("finish_reason"):
                            finish_reason = choice["finish_reason"]
                        delta = choice.get("delta")
                        if not isinstance(delta, dict):
                            continue

                        refusal_delta = delta.get("refusal")
                        if isinstance(refusal_delta, str) and refusal_delta:
                            refusal_parts.append(refusal_delta)

                        reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                        if isinstance(reasoning, str) and reasoning:
                            thinking_parts.append(reasoning)
                            yield ProviderThinkingDelta(text=reasoning)

                        content = delta.get("content")
                        if isinstance(content, str) and content:
                            if splitter is not None:
                                visible, thought = splitter.feed(content)
                            else:
                                visible, thought = content, ""
                            if thought:
                                thinking_parts.append(thought)
                                yield ProviderThinkingDelta(text=thought)
                            if visible:
                                text_parts.append(visible)
                                yield ProviderTextDelta(text=visible)

                        _accumulate_tool_calls(delta.get("tool_calls"), tool_frags)
        except httpx.HTTPError as exc:
            raise self._transport_error(exc) from exc

        if splitter is not None:
            visible, thought = splitter.flush()
            if thought:
                thinking_parts.append(thought)
                yield ProviderThinkingDelta(text=thought)
            if visible:
                text_parts.append(visible)
                yield ProviderTextDelta(text=visible)

        content_blocks: list[ContentBlock] = []
        if thinking_parts:
            content_blocks.append(ThinkingBlock(thinking="".join(thinking_parts)))
        joined = "".join(text_parts)
        if joined:
            content_blocks.append(TextBlock(text=joined))
        for index in sorted(tool_frags):
            frag = tool_frags[index]
            content_blocks.append(
                ToolUseBlock(
                    id=frag["id"] or f"call_{index}",
                    name=frag["name"] or "",
                    input=parse_tool_arguments("".join(frag["args"])),
                )
            )

        yield CompletionDone(
            message=Message(role="assistant", content=content_blocks),
            stop_reason=_map_finish_reason(finish_reason, bool(tool_frags)),
            usage=usage,
            refusal="".join(refusal_parts) or None,
        )


def _parse_sse_line(line: str) -> dict[str, Any] | None:
    """Decode one SSE line into a chunk object.

    Args:
        line: A raw line from the response stream.

    Returns:
        The decoded chunk, or ``None`` for keep-alives, comments, non-data
        fields, the ``[DONE]`` sentinel, and anything that is not valid JSON —
        partially-compliant servers emit all of these.
    """
    line = line.strip()
    if not line or not line.startswith("data:"):
        return None
    payload = line[len("data:") :].strip()
    if not payload or payload == "[DONE]":
        return None
    try:
        chunk = json.loads(payload)
    except (json.JSONDecodeError, ValueError):
        return None
    return chunk if isinstance(chunk, dict) else None


def _accumulate_tool_calls(raw: Any, frags: dict[int, dict[str, Any]]) -> None:
    """Merge a streamed ``tool_calls`` delta into the per-index accumulator.

    Chat Completions streams one tool call across many chunks: the first carries
    ``id`` and ``function.name``, later ones only append ``function.arguments``
    fragments. They are keyed by ``index``, which is the only field guaranteed
    to be present on every fragment.

    Args:
        raw: The ``delta.tool_calls`` value, if any.
        frags: Accumulator mutated in place, keyed by tool-call index.
    """
    if not isinstance(raw, list):
        return
    for position, call in enumerate(raw):
        if not isinstance(call, dict):
            continue
        index = call.get("index")
        if not isinstance(index, int):
            index = position
        slot = frags.setdefault(index, {"id": None, "name": None, "args": []})
        if isinstance(call.get("id"), str) and call["id"]:
            slot["id"] = call["id"]
        function = call.get("function")
        if not isinstance(function, dict):
            continue
        if isinstance(function.get("name"), str) and function["name"]:
            slot["name"] = function["name"]
        arguments = function.get("arguments")
        if isinstance(arguments, str) and arguments:
            slot["args"].append(arguments)


def _map_finish_reason(raw: str | None, saw_tool_calls: bool) -> StopReason:
    """Translate ``finish_reason`` into a logpose stop reason.

    Args:
        raw: The server's ``finish_reason``, which some servers omit entirely.
        saw_tool_calls: Whether any tool call was assembled this turn.

    Returns:
        The logpose stop reason. Tool calls win when the field is missing or
        unrecognised, because ending the run there would silently drop them.
    """
    if raw and raw in _FINISH_REASONS:
        mapped = _FINISH_REASONS[raw]
        # A server can report "stop" while still emitting tool calls; trusting
        # it would end the run with the calls unexecuted.
        if mapped == "end_turn" and saw_tool_calls:
            return "tool_use"
        return mapped
    return "tool_use" if saw_tool_calls else "end_turn"
