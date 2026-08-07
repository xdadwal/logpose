"""OpenAI-compatible provider, and the Docker Model Runner backend built on it.

This is the second implementation of :class:`~logpose.providers.base.Provider`,
and the one that proves the seam: nothing here is shared with the Anthropic
backend except the neutral message model, and nothing above
``logpose.providers`` changes to use it.

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
2. **Tool arguments.** Anthropic returns parsed JSON; Chat Completions streams a
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
from typing import Any

import httpx

from logpose.errors import (
    AuthError,
    LogposeError,
    ProviderError,
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
    "DEFAULT_DOCKER_BASE_URL",
    "DEFAULT_MAX_TOKENS",
    "UNPARSED_ARGUMENTS_KEY",
    "DockerModelsProvider",
    "OpenAICompatProvider",
]

DEFAULT_DOCKER_BASE_URL = "http://localhost:12434/engines/v1"
"""Docker Model Runner's host-side endpoint, as exposed by Docker Desktop."""

DEFAULT_MAX_TOKENS = 4096
"""Output ceiling for local models.

Deliberately roomy: a reasoning model spends this budget on ``reasoning_content``
before it emits a tool call, so a small ceiling truncates mid-thought and yields
``finish_reason="length"`` with nothing usable.
"""

AUTO_MODEL = "auto"
"""Sentinel model id meaning "ask the server what it has and use the first one".

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

_CONNECT_HINT = (
    "Could not reach the model server at {base_url}. If this is Docker Model Runner, check that "
    "it is enabled and listening on the host: `docker desktop enable model-runner --tcp 12434`, "
    "then `docker model ls` to confirm a model is pulled."
)


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


def _messages_to_wire(messages: list[Message], system: str | None) -> list[dict[str, Any]]:
    """Translate logpose messages into Chat Completions messages.

    Args:
        messages: Conversation history in logpose's neutral model.
        system: Optional system prompt, emitted as the leading message.

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
            # Thinking is intentionally dropped: unlike Anthropic there is no
            # signature to preserve, and echoing `reasoning_content` back is
            # rejected by several OpenAI-compatible servers.
            if not text and not tool_uses:
                continue
            entry: dict[str, Any] = {"role": "assistant", "content": text}
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
    return Usage(
        input_tokens=max(prompt - cached, 0),
        output_tokens=int(raw.get("completion_tokens") or 0),
        cache_read_input_tokens=cached,
    )


def _status_is_retryable(status_code: int | None) -> bool:
    """Whether a failed request is worth retrying unchanged."""
    if status_code is None:
        return True
    return status_code in _RETRYABLE_STATUS or status_code >= 500


# ---------------------------------------------------------------------------
# provider
# ---------------------------------------------------------------------------


class OpenAICompatProvider:
    """A :class:`~logpose.providers.base.Provider` for any Chat Completions API.

    Works against llama.cpp, vLLM, Ollama, LM Studio, OpenAI itself, Kimi, and
    anything else exposing ``POST /chat/completions``. Always streams.

    Attributes:
        name: Provider identifier.
        model_default: Model used when a request omits one. :data:`AUTO_MODEL`
            asks the server for its first model on the first request.
        max_tokens: Output ceiling honoured by an ``Agent`` built without one.
    """

    name = "openai-compat"
    turn_timeout = 900.0
    """Recommended complete-turn deadline in seconds for generic servers."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout: float | httpx.Timeout | None = None,
        parse_think_tags: bool = True,
        default_headers: dict[str, str] | None = None,
        client: httpx.AsyncClient | None = None,
        extra_body: dict[str, Any] | None = None,
        api_key_env: str | None = "OPENAI_API_KEY",
    ) -> None:
        """Configure the backend.

        Args:
            base_url: Root of the OpenAI-compatible API, e.g.
                ``"http://localhost:12434/engines/v1"``. Falls back to
                ``$OPENAI_BASE_URL``.
            model: Default model id, or :data:`AUTO_MODEL` to discover one.
                Falls back to ``$OPENAI_MODEL``.
            api_key: Bearer token. Optional — local servers usually need none.
                Falls back to ``$OPENAI_API_KEY``.
            max_tokens: Output-token ceiling for a turn.
            timeout: httpx timeout. The default is deliberately generous
                because a quantized local model can take minutes.
            parse_think_tags: Split ``<think>`` blocks out of content into
                reasoning. Harmless for models that do not use them.
            default_headers: Extra headers sent with every request.
            client: Inject a pre-built ``httpx.AsyncClient`` (used by tests).
            extra_body: Fields merged into every request body, e.g.
                ``{"temperature": 0.2}``.
            api_key_env: Environment variable consulted when ``api_key`` is not
                given. ``None`` disables the fallback entirely, which is what a
                local backend wants: a stray ``$OPENAI_API_KEY`` should never be
                shipped to a server on localhost.

        Raises:
            LogposeError: If no ``base_url`` is available.
        """
        resolved_base = base_url or os.environ.get("OPENAI_BASE_URL") or ""
        if not resolved_base:
            raise LogposeError(
                "No base_url for the OpenAI-compatible provider. Pass base_url=... or set "
                "OPENAI_BASE_URL."
            )
        if timeout is None:
            # Generous by default: a quantized local model can spend minutes on
            # one turn, and httpx's 5s default would abort every request.
            timeout = httpx.Timeout(connect=10.0, read=600.0, write=60.0, pool=10.0)

        env_key = os.environ.get(api_key_env) if api_key_env else None
        self.base_url = resolved_base.rstrip("/")
        self.model_default = model or os.environ.get("OPENAI_MODEL") or ""
        self.max_tokens = max_tokens
        self.parse_think_tags = parse_think_tags
        self._api_key = api_key or env_key or None
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
            f"model_default={self.model_default!r}, api_key={key!r})"
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
            requested: The request's model, possibly empty or :data:`AUTO_MODEL`.

        Returns:
            A concrete model id.

        Raises:
            ProviderError: If discovery is needed and the server serves nothing.
        """
        if requested and requested != AUTO_MODEL:
            return requested
        if self._discovered_model:
            return self._discovered_model
        models = await self.list_models()
        if not models:
            raise ProviderError(
                f"{self.base_url} reports no available models. Pull one first "
                "(for Docker Model Runner: `docker model pull <name>`), or pass an explicit "
                "model=... to the provider.",
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
                "$OPENAI_API_KEY environment variable if this server requires authentication."
            )
        return ProviderError(
            f"{self.name} request failed with status {status}: {self._scrub(body[:2000])}",
            status_code=status,
            retryable=_status_is_retryable(status),
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
            "messages": _messages_to_wire(req.messages, req.system),
            "max_tokens": req.max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if req.tools:
            body["tools"] = [_tool_to_wire(spec) for spec in req.tools]
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

        try:
            async with self._http().stream(
                "POST", f"{self.base_url}/chat/completions", json=body, headers=self._headers()
            ) as response:
                if response.status_code >= 400:
                    raw = await response.aread()
                    raise self._status_error(
                        response.status_code,
                        raw.decode("utf-8", "replace"),
                        response.headers,
                    )
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


class DockerModelsProvider(OpenAICompatProvider):
    """Local models served by Docker Model Runner.

    Docker Desktop exposes an OpenAI-compatible API on the host once the model
    runner is enabled, so this is :class:`OpenAICompatProvider` with local
    defaults: Docker's endpoint, no credential, a roomy token ceiling, and
    model auto-discovery.

    .. code-block:: python

        from logpose import Agent

        agent = Agent("docker")                     # uses the first pulled model
        agent = Agent("docker", model="gemma4")     # or name one

    Attributes:
        name: ``"docker"``.
    """

    name = "docker"
    turn_timeout = 1800.0
    """Recommended complete-turn deadline in seconds for local inference."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Configure the Docker Model Runner backend.

        Args:
            base_url: Override the endpoint. Falls back to
                ``$DOCKER_MODEL_RUNNER_URL`` and then
                :data:`DEFAULT_DOCKER_BASE_URL`. Inside a container use
                ``http://model-runner.docker.internal/engines/v1``.
            model: Model id, e.g. ``"gemma4"``. Falls back to ``$DOCKER_MODEL``
                and then :data:`AUTO_MODEL`, which picks the first model the
                runner serves.
            **kwargs: Forwarded to :class:`OpenAICompatProvider`.
        """
        # api_key_env is disabled: Docker Model Runner needs no credential, and
        # a developer's ambient $OPENAI_API_KEY must not be sent to localhost.
        kwargs.setdefault("api_key_env", None)
        super().__init__(
            base_url=(
                base_url or os.environ.get("DOCKER_MODEL_RUNNER_URL") or DEFAULT_DOCKER_BASE_URL
            ),
            model=model or os.environ.get("DOCKER_MODEL") or AUTO_MODEL,
            **kwargs,
        )
