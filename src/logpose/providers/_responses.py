"""Shared machinery for OpenAI's Responses API.

Two public providers sit on this module — :mod:`logpose.providers.openai` for an
API key against ``api.openai.com``, and :mod:`logpose.providers.codex` for a
ChatGPT subscription against the Codex backend. They speak the same wire format
and differ only in credential policy, endpoint, headers, and how much they dress
up to look like the Codex CLI, so all of that is expressed as subclass overrides
rather than as runtime branches on "which kind of credential did we end up with".

That split is deliberate. An earlier single provider decided its endpoint, its
headers, its instructions, and whether ``max_output_tokens`` was even legal from
one latent fact — the credential kind — which is not known until the first
request. It made ``base_url`` a value the provider had to *predict* before it
could be known. Naming the two paths separately makes each one's behaviour
constant from construction.

Why raw ``httpx`` and not the ``openai`` SDK
--------------------------------------------
``httpx`` is already a logpose dependency, so these backends add none, and
``import logpose`` keeps its promise of pulling in no vendor SDK. The subscription
endpoint is also not a shape the official SDK is built to talk to.

The four impedance mismatches with logpose's model
--------------------------------------------------
1. **A turn is a list of *items*, not a message.** logpose has one
   :class:`~logpose.messages.Message` per turn holding N content blocks; the
   Responses API has a flat ``input`` array where an assistant turn's reasoning,
   text, and two tool calls are four sibling items. :func:`messages_to_input`
   fans blocks out one-to-one, and a user turn carrying three tool results
   becomes three ``function_call_output`` items.
2. **Reasoning is opaque, and ordering is load-bearing.** The API rejects a
   request in which a ``reasoning`` item is not immediately followed by the item
   it reasoned for (``400 Item 'rs_...' of type 'reasoning' was provided without
   its required following item``). Reasoning also carries an
   ``encrypted_content`` blob that must come back byte-exact. Both constraints
   are met by keeping the whole item in a :class:`~logpose.messages.RawBlock` and
   emitting blocks in ``Message.content`` order — never as a
   :class:`~logpose.messages.ThinkingBlock`, which has nowhere to put the item id
   and would drop the encrypted payload.
3. **Reasoning arrives on one channel and is stored from another.** The live text
   comes from ``response.reasoning_summary_text.delta``; the bytes that must be
   resent come from the terminal ``reasoning`` item. They are different things
   and both are needed, so a turn yields
   :class:`~logpose.providers.base.ProviderThinkingDelta` events *and* a
   ``RawBlock``.
4. **Tool arguments are a JSON string.** As in Chat Completions — but delivered
   whole on ``response.output_item.done``, so no fragment accumulator is needed.
   Malformed JSON is still reported to the model rather than raised; see
   :mod:`logpose.providers._toolargs`.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator, Mapping
from typing import Any, ClassVar, Literal

import httpx

from logpose.auth.codex import Credential, CredentialProvider
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
    RawBlock,
    StopReason,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
)
from logpose.providers._redact import redact, scrub_exception_in_place
from logpose.providers._toolargs import parse_tool_arguments
from logpose.providers.base import (
    CompletionDone,
    CompletionRequest,
    ProviderEvent,
    ProviderTextDelta,
    ProviderThinkingDelta,
    ToolSpec,
)

__all__ = [
    "DEFAULT_REASONING_EFFORT",
    "DEFAULT_REASONING_SUMMARY",
    "REASONING_EFFORTS",
    "ResponsesProvider",
    "item_to_blocks",
    "map_stop_reason",
    "map_usage",
    "messages_to_input",
    "parse_sse_data",
    "tool_to_wire",
]

DEFAULT_REASONING_EFFORT = "medium"
"""Reasoning depth requested by default — the Codex CLI's own default."""

DEFAULT_REASONING_SUMMARY = "auto"
"""Reasoning-summary mode requested by default.

Load-bearing: with no ``summary`` asked for, the API emits no
``response.reasoning_summary_text.delta`` events at all and
:class:`~logpose.providers.base.ProviderThinkingDelta` silently never fires. Pass
``reasoning_summary=None`` only if you actively do not want reasoning text.
"""

REASONING_EFFORTS = frozenset({"low", "medium", "high", "xhigh", "max"})
"""Accepted ``reasoning_effort`` values.

Validated at construction to turn a typo into a clear error rather than an opaque
400. ``max`` is included because at least one listed model advertises it.
"""

_RETRYABLE_STATUS = frozenset({408, 409, 429})

_RETRYABLE_ERROR_CODES = frozenset(
    {"rate_limit_exceeded", "server_error", "internal_error", "overloaded"}
)

_REASONING_DELTA_EVENTS = frozenset(
    {"response.reasoning_summary_text.delta", "response.reasoning_text.delta"}
)

_INCOMPLETE_REASONS: dict[str, StopReason] = {
    "max_output_tokens": "max_tokens",
    "content_filter": "refusal",
}


# ---------------------------------------------------------------------------
# wire translation
# ---------------------------------------------------------------------------


def messages_to_input(messages: list[Message]) -> list[dict[str, Any]]:
    """Translate logpose history into Responses ``input`` items.

    Ordering is load-bearing: the API rejects a request in which a ``reasoning``
    item is not immediately followed by the item it reasoned for. Blocks are
    emitted in ``Message.content`` order — which is the order the provider
    assembled them in — so the invariant is preserved by construction rather than
    reconstructed here.

    Args:
        messages: Conversation history in logpose's neutral model.

    Returns:
        Responses input items. One logpose message becomes several: an assistant
        turn with reasoning, text and two tool calls is four items.
    """
    items: list[dict[str, Any]] = []
    for message in messages:
        for block in message.content:
            item = _block_to_input(block, message.role)
            if item is not None:
                items.append(item)
    return items


def _block_to_input(block: ContentBlock, role: str) -> dict[str, Any] | None:
    """Translate one content block into a Responses input item.

    A rebuilt ``function_call`` carries ``call_id`` and no ``id``. The ``fc_...``
    id the API returns alongside it is dropped: with ``store: false`` the request
    is stateless, ``call_id`` is the documented correlation key, and
    :class:`~logpose.messages.ToolUseBlock` has one id field which must hold
    ``call_id`` because the loop matches results against it. Verified against the
    live subscription backend across a full two-turn tool round trip.

    Args:
        block: The block to translate.
        role: The role of the message the block came from.

    Returns:
        The item, or ``None`` for a block this API cannot represent.
    """
    if isinstance(block, TextBlock):
        part = "output_text" if role == "assistant" else "input_text"
        return {
            "type": "message",
            "role": role,
            "content": [{"type": part, "text": block.text}],
        }
    if isinstance(block, ToolUseBlock):
        return {
            "type": "function_call",
            "call_id": block.id,
            "name": block.name,
            "arguments": json.dumps(block.input),
        }
    if isinstance(block, ToolResultBlock):
        return {
            "type": "function_call_output",
            "call_id": block.tool_use_id,
            "output": _error_prefixed(block),
        }
    if isinstance(block, RawBlock):
        # Verbatim: this is how a `reasoning` item keeps its id and its
        # encrypted_content across turns.
        return dict(block.data)
    # ThinkingBlock and RedactedThinkingBlock are dropped rather than raising.
    # They are exactly what an Anthropic run leaves behind in a Conversation, an
    # Anthropic thinking signature means nothing here, and a user moving history
    # between backends should lose the reasoning, not crash.
    return None


def _error_prefixed(result: ToolResultBlock) -> str:
    """Render a tool result, making failure visible without an ``is_error`` field.

    ``function_call_output`` has no error flag, so the signal has to live in the
    text or the model cannot tell success from failure.

    Args:
        result: The tool result to render.

    Returns:
        The result content, prefixed when it represents a failure.
    """
    return f"ERROR: {result.content}" if result.is_error else result.content


def tool_to_wire(spec: ToolSpec) -> dict[str, Any]:
    """Translate a :class:`~logpose.providers.base.ToolSpec` to a function tool.

    Flat, unlike Chat Completions: no nested ``"function"`` object. ``strict`` is
    deliberately not set — logpose's generated schemas are not guaranteed to meet
    strict mode's constraints, and a rejection there is an opaque 400.

    Args:
        spec: The tool to advertise.

    Returns:
        The wire form of the tool.
    """
    return {
        "type": "function",
        "name": spec.name,
        "description": spec.description,
        "parameters": spec.input_schema,
    }


def map_usage(raw: Any) -> Usage:
    """Translate a Responses ``usage`` object into logpose's model.

    ``input_tokens`` counts cached tokens too, whereas logpose's ``input_tokens``
    means *uncached* prompt tokens, so the cached portion is subtracted rather
    than double-counted.

    ``output_tokens_details.reasoning_tokens`` is deliberately added to nothing:
    it is already inside ``output_tokens``, and logpose's :class:`Usage` has no
    field for it, so adding it would inflate the run's reported cost.

    Args:
        raw: The ``usage`` object from the terminal response, if any.

    Returns:
        The mapped usage, all zeros when nothing usable was reported.
    """
    if not isinstance(raw, dict):
        return Usage()
    total_input = int(raw.get("input_tokens") or 0)
    details = raw.get("input_tokens_details")
    if not isinstance(details, dict):
        details = {}
    cached = min(int(details.get("cached_tokens") or 0), total_input)
    return Usage(
        input_tokens=max(total_input - cached, 0),
        output_tokens=int(raw.get("output_tokens") or 0),
        cache_read_input_tokens=cached,
        # The subscription backend reports cache writes; the public API does not.
        cache_creation_input_tokens=int(details.get("cache_write_tokens") or 0),
    )


def map_stop_reason(
    final: dict[str, Any] | None,
    *,
    saw_tool_calls: bool,
    saw_refusal: bool,
) -> StopReason:
    """Translate a terminal response into a logpose stop reason.

    Precedence, and why:

    1. ``status == "incomplete"`` wins over everything, including tool calls. A
       truncated turn can carry a ``function_call`` whose ``arguments`` string was
       cut mid-JSON, and running that would execute a tool with wrong arguments.
       Reporting ``max_tokens`` ends the run, which is correct.
    2. A refusal content part.
    3. Tool calls beat a ``"completed"`` status, for the same reason
       ``openai_compat`` lets them beat ``"stop"``: ending the run here would
       leave the calls unexecuted.
    4. Otherwise ``end_turn`` — including for an unknown or missing status, so a
       new server state never crashes the loop.

    ``pause_turn`` and ``stop_sequence`` are never produced: this API has no
    analogue of either.

    Args:
        final: The terminal ``response`` object, if one arrived.
        saw_tool_calls: Whether any function call was assembled this turn.
        saw_refusal: Whether a refusal content part arrived.

    Returns:
        The logpose stop reason.
    """
    if isinstance(final, dict) and final.get("status") == "incomplete":
        incomplete = final.get("incomplete_details")
        reason = incomplete.get("reason") if isinstance(incomplete, dict) else None
        if isinstance(reason, str):
            return _INCOMPLETE_REASONS.get(reason, "max_tokens")
        return "max_tokens"
    if saw_refusal:
        return "refusal"
    if saw_tool_calls:
        return "tool_use"
    return "end_turn"


def parse_sse_data(line: str) -> dict[str, Any] | None:
    """Decode one SSE ``data:`` line.

    Only ``data:`` is read. The event type is taken from the payload's own
    ``type`` field rather than the ``event:`` line, because the payload always
    carries it and it survives a proxy that strips event names. Keep-alives,
    comments, ``[DONE]`` and unparseable payloads all return ``None``.

    Args:
        line: A raw line from the response stream.

    Returns:
        The decoded event, or ``None`` when the line carries none.
    """
    line = line.strip()
    if not line or not line.startswith("data:"):
        return None
    payload = line[len("data:") :].strip()
    if not payload or payload == "[DONE]":
        return None
    try:
        event = json.loads(payload)
    except (json.JSONDecodeError, ValueError):
        return None
    return event if isinstance(event, dict) else None


def item_to_blocks(item: dict[str, Any]) -> tuple[list[ContentBlock], bool]:
    """Translate one output item into content blocks.

    Args:
        item: An output item from ``response.output_item.done`` or the terminal
            response's ``output`` array.

    Returns:
        ``(blocks, saw_refusal)``. Anything these providers do not model —
        reasoning, server-side tool use, web-search results — becomes a verbatim
        :class:`~logpose.messages.RawBlock` so it survives a resend.
    """
    kind = item.get("type")

    if kind == "message":
        blocks: list[ContentBlock] = []
        saw_refusal = False
        for part in item.get("content") or []:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "refusal":
                text = part.get("refusal")
                saw_refusal = True
            else:
                text = part.get("text")
            if isinstance(text, str) and text:
                blocks.append(TextBlock(text=text))
        return blocks, saw_refusal

    if kind == "function_call":
        raw_arguments = item.get("arguments")
        call_id = item.get("call_id") or item.get("id") or ""
        return (
            [
                ToolUseBlock(
                    id=str(call_id),
                    name=str(item.get("name") or ""),
                    input=parse_tool_arguments(
                        raw_arguments if isinstance(raw_arguments, str) else ""
                    ),
                )
            ],
            False,
        )

    return [RawBlock(data=dict(item))], False


def _status_is_retryable(status_code: int | None) -> bool:
    """Whether a failed request is worth retrying unchanged.

    Args:
        status_code: The HTTP status, or ``None`` for a transport failure.

    Returns:
        ``True`` when a retry could plausibly succeed.
    """
    if status_code is None:
        return True
    return status_code in _RETRYABLE_STATUS or status_code >= 500


def _model_ids(payload: Any) -> list[str]:
    """Pull model identifiers out of a ``GET /models`` response.

    Accepts both shapes the two endpoints use: ``{"data": [{"id": ...}]}`` on the
    public API and ``{"models": [{"slug": ...}]}`` on the Codex backend. Entries
    without a usable identifier are skipped rather than raising, so one unfamiliar
    row cannot break discovery.

    Args:
        payload: The decoded response body.

    Returns:
        Identifiers in the order the backend listed them.
    """
    if not isinstance(payload, dict):
        return []
    for key, id_field in (("data", "id"), ("models", "slug")):
        entries = payload.get(key)
        if not isinstance(entries, list):
            continue
        found = [
            entry[id_field]
            for entry in entries
            if isinstance(entry, dict) and isinstance(entry.get(id_field), str)
        ]
        if found:
            return found
    return []


# ---------------------------------------------------------------------------
# provider base
# ---------------------------------------------------------------------------


class ResponsesProvider:
    """Base for the two Responses API backends.

    Subclasses set the class variables below and may override :meth:`_headers`,
    :meth:`_build_instructions`, and :meth:`_resolve_kwargs`. Everything else —
    the request body, the streaming state machine, error classification,
    redaction, lazy credential resolution — is shared.

    Attributes:
        model_default: Model used when a request omits one.
        max_tokens: Output ceiling honoured by an ``Agent`` built without one.
        base_url: Endpoint root in use. Fixed at construction, because this
            provider accepts exactly one kind of credential.
    """

    # Not a ClassVar: the Provider protocol declares `name` as an instance
    # variable, and a ClassVar cannot satisfy that.
    name: str
    """Registered provider name."""

    REQUIRED_KIND: ClassVar[Literal["api_key", "oauth"]]
    """The one credential kind this provider accepts."""

    BASE_URL: ClassVar[str]
    """Endpoint root this provider talks to."""

    DEFAULT_MODEL: ClassVar[str]
    """Model used when neither the request, the caller, nor the environment names one."""

    DEFAULT_MAX_TOKENS: ClassVar[int]
    """Output ceiling when the caller names none."""

    MODEL_ENV: ClassVar[str]
    """Environment variable naming a default model."""

    BASE_URL_ENV: ClassVar[str]
    """Environment variable overriding :attr:`BASE_URL`."""

    SENDS_MAX_OUTPUT_TOKENS: ClassVar[bool] = True
    """Whether the endpoint accepts ``max_output_tokens`` at all."""

    SIBLING_HINT: ClassVar[str] = ""
    """One line appended to a credential failure, naming the provider to use instead."""

    def __init__(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        max_tokens: int | None = None,
        reasoning_effort: str | None = DEFAULT_REASONING_EFFORT,
        reasoning_summary: str | None = DEFAULT_REASONING_SUMMARY,
        instructions: str | None = None,
        prompt_cache_key: str | None = None,
        timeout: float | httpx.Timeout | None = None,
        default_headers: dict[str, str] | None = None,
        client: httpx.AsyncClient | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        """Configure the backend.

        Nothing here reads a file, resolves a credential, or touches the network.
        The first request does all of it.

        Args:
            model: Default model id. Falls back to the subclass's
                :attr:`MODEL_ENV` and then :attr:`DEFAULT_MODEL`.
            base_url: Endpoint root override. Falls back to :attr:`BASE_URL_ENV`
                and then :attr:`BASE_URL`.
            api_key: Credential supplied directly by the caller.
            max_tokens: Output-token ceiling for a turn. Defaults to the
                subclass's :attr:`DEFAULT_MAX_TOKENS`.
            reasoning_effort: One of :data:`REASONING_EFFORTS`, or ``None`` to
                omit the ``reasoning`` field entirely.
            reasoning_summary: ``"auto"``, ``"concise"``, ``"detailed"``, or
                ``None`` to ask for no summary — which also means no thinking
                events. See :data:`DEFAULT_REASONING_SUMMARY`.
            instructions: Text placed ahead of the caller's ``system`` prompt.
            prompt_cache_key: Stable key improving prompt-cache hit rates across
                turns of one conversation.
            timeout: httpx timeout. The default is generous because a
                high-reasoning turn can take minutes.
            default_headers: Extra headers sent with every request.
            client: Inject a pre-built ``httpx.AsyncClient`` (used by tests).
            extra_body: Fields merged into every request body.

        Raises:
            LogposeError: If ``max_tokens`` is not positive or
                ``reasoning_effort`` is not a recognised value.
        """
        resolved_max_tokens = self.DEFAULT_MAX_TOKENS if max_tokens is None else max_tokens
        if resolved_max_tokens <= 0:
            raise LogposeError(f"max_tokens must be positive, got {resolved_max_tokens!r}.")
        if reasoning_effort is not None and reasoning_effort not in REASONING_EFFORTS:
            known = ", ".join(sorted(REASONING_EFFORTS))
            raise LogposeError(
                f"Unknown reasoning_effort {reasoning_effort!r}. Known values: {known}."
            )
        if timeout is None:
            # Generous by default: a high-effort reasoning turn can spend minutes
            # before its first token, and httpx's 5s default would abort it.
            timeout = httpx.Timeout(connect=10.0, read=600.0, write=60.0, pool=10.0)

        self.model_default = model or os.environ.get(self.MODEL_ENV) or self.DEFAULT_MODEL
        self.max_tokens = resolved_max_tokens
        self.base_url = (base_url or os.environ.get(self.BASE_URL_ENV) or self.BASE_URL).rstrip(
            "/"
        )

        self._explicit_api_key = api_key
        self._reasoning_effort = reasoning_effort
        self._reasoning_summary = reasoning_summary
        self._instructions = instructions
        self._prompt_cache_key = prompt_cache_key
        self._default_headers = dict(default_headers or {})
        self._extra_body = dict(extra_body or {})
        self._timeout = timeout
        self._client = client
        self._owns_client = client is None

        self._credentials: CredentialProvider | None = None
        # Guards the lazy CredentialProvider build only. Without it, N concurrent
        # first requests each build their own provider — and therefore their own
        # lock — so the single-flight refresh inside CredentialProvider.get never
        # sees the others, firing N grants against the same refresh token.
        self._credentials_lock = asyncio.Lock()

    def __repr__(self) -> str:
        """Return an unambiguous representation. Never includes a credential."""
        return (
            f"{type(self).__name__}(base_url={self.base_url!r}, "
            f"model_default={self.model_default!r}, max_tokens={self.max_tokens!r})"
        )

    # -- credentials --------------------------------------------------------

    def _resolve_kwargs(self) -> dict[str, Any]:
        """Name the caller's explicit credential material for this provider.

        Returns:
            Keyword arguments for ``CredentialProvider.resolve``.
        """
        return {"explicit_api_key": self._explicit_api_key}

    async def _current_credential(self) -> Credential:
        """Return a usable credential for this turn.

        The first call builds a
        :class:`~logpose.auth.codex.CredentialProvider` (which may read
        ``auth.json``, hence the thread — on a network home directory that is not
        a cheap read); later calls just ask it for a credential, and it refreshes
        an expiring subscription token single-flight.

        Only sources yielding :attr:`REQUIRED_KIND` are considered, so a
        credential of the other kind sitting in the environment or the store
        neither satisfies nor shadows this provider.

        Returns:
            A credential of this provider's :attr:`REQUIRED_KIND`.

        Raises:
            AuthError: If no credential of that kind could be resolved, or a
                refresh failed. The message names the sibling provider that does
                accept the other kind.
        """
        credentials = self._credentials
        if credentials is None:
            async with self._credentials_lock:
                credentials = self._credentials
                if credentials is None:
                    kwargs = self._resolve_kwargs()
                    try:
                        credentials = await asyncio.to_thread(
                            lambda: CredentialProvider.resolve(
                                require_kind=self.REQUIRED_KIND, **kwargs
                            )
                        )
                    except AuthError as exc:
                        raise AuthError(f"{exc}{self.SIBLING_HINT}") from exc
                    self._credentials = credentials
        return await credentials.get()

    def _http(self) -> httpx.AsyncClient:
        """Return the shared client, creating it on first use."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    def _models_params(self) -> dict[str, str]:
        """Query parameters for the model-list request.

        Returns:
            Nothing by default; the Codex backend needs a client version.
        """
        return {}

    async def list_models(self) -> list[str]:
        """Ask the backend which models it serves.

        Both endpoints answer ``GET /models`` but in different shapes — the public
        API returns ``{"data": [{"id": ...}]}`` and the Codex backend returns
        ``{"models": [{"slug": ...}]}`` — so both are accepted rather than split
        across two overrides.

        Returns:
            Model identifiers in the order the backend reported them, which on the
            Codex backend is its own priority order.

        Raises:
            AuthError: If no credential could be resolved.
            ProviderError: If the backend is unreachable or rejects the request.
        """
        credential = await self._current_credential()
        try:
            response = await self._http().get(
                f"{self.base_url}/models",
                params=self._models_params(),
                headers=self._headers(credential),
            )
        except httpx.HTTPError as exc:
            raise self._transport_error(exc) from exc
        if response.status_code >= 400:
            raise self._status_error(response.status_code, response.text, response.headers)
        return _model_ids(response.json())

    async def aclose(self) -> None:
        """Close the underlying HTTP client if this provider created it."""
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    # -- request ------------------------------------------------------------

    def _headers(self, credential: Credential) -> dict[str, str]:
        """Build request headers.

        Args:
            credential: The credential for this turn.

        Returns:
            The headers to send.
        """
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {credential.value}",
            **self._default_headers,
        }

    def _build_instructions(self, system: str | None) -> str | None:
        """Build the ``instructions`` field.

        Args:
            system: The caller's system prompt, if any.

        Returns:
            The instructions to send, or ``None`` to omit the field.
        """
        parts = [part for part in (self._instructions, system) if part]
        return "\n\n".join(parts) or None

    def _build_body(self, req: CompletionRequest) -> dict[str, Any]:
        """Assemble the Responses request body.

        ``extra_body`` and then ``req.extra`` are merged last, so a caller can
        override anything here — including setting ``store: True`` and getting a
        400 from the subscription backend. That is the escape hatch working as
        designed.

        Args:
            req: The turn to run.

        Returns:
            The request body.
        """
        body: dict[str, Any] = {
            "model": req.model or self.model_default,
            "input": messages_to_input(req.messages),
            # store=false because logpose keeps the history, so server-side
            # retention would buy nothing. The subscription backend requires it.
            "store": False,
            "stream": True,
            "include": ["reasoning.encrypted_content"],
        }
        instructions = self._build_instructions(req.system)
        if instructions is not None:
            body["instructions"] = instructions
        if self.SENDS_MAX_OUTPUT_TOKENS:
            body["max_output_tokens"] = req.max_tokens or self.max_tokens
        if self._reasoning_effort is not None:
            reasoning: dict[str, Any] = {"effort": self._reasoning_effort}
            if self._reasoning_summary is not None:
                reasoning["summary"] = self._reasoning_summary
            body["reasoning"] = reasoning
        if req.tools:
            body["tools"] = [tool_to_wire(spec) for spec in req.tools]
            body["tool_choice"] = "auto"
            body["parallel_tool_calls"] = True
        if self._prompt_cache_key:
            body["prompt_cache_key"] = self._prompt_cache_key
        body.update(self._extra_body)
        body.update(req.extra)
        return body

    # -- errors -------------------------------------------------------------

    def _secrets(self) -> tuple[str, ...]:
        """Credential values that must never appear in an error."""
        values: list[str] = []
        credentials = self._credentials
        if credentials is not None:
            credential = credentials.current
            values.append(credential.value)
            if credential.refresh_token:
                values.append(credential.refresh_token)
        if self._explicit_api_key:
            values.append(self._explicit_api_key)
        return tuple(values)

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
    ) -> ProviderError:
        """Build a :class:`~logpose.errors.ProviderError` from an HTTP failure.

        Args:
            status: The HTTP status code.
            body: The response body, which is scrubbed and truncated.

        Returns:
            The error to raise.
        """
        hint = self.SIBLING_HINT if status in (401, 403) else ""
        return ProviderError(
            f"{self.name} request failed with status {status}: "
            f"{self._scrub(body[:2000])}{hint}",
            status_code=status,
            retryable=_status_is_retryable(status),
            error_code=_provider_error_code_from_body(body),
            **_provider_metadata_from_headers(headers),
        )

    def _transport_error(self, exc: httpx.HTTPError) -> ProviderError:
        """Build a :class:`~logpose.errors.ProviderError` from a transport failure.

        Args:
            exc: The httpx failure. Scrubbed in place, because
                ``raise ... from exc`` renders ``str(exc)`` in every traceback.

        Returns:
            The error to raise.
        """
        scrub_exception_in_place(exc, self._secrets())
        if isinstance(exc, httpx.ConnectError):
            return ProviderError(
                f"Could not reach {self.base_url}. Check connectivity and, on a "
                "subscription credential, that the CLI login is still current.",
                status_code=None,
                retryable=True,
            )
        return ProviderError(
            f"Could not reach {self.base_url}: {self._scrub(str(exc))}",
            status_code=None,
            retryable=True,
        )

    def _event_error(self, event: dict[str, Any]) -> ProviderError:
        """Build a :class:`~logpose.errors.ProviderError` from a failure event.

        A stream can fail after a 200: ``response.failed`` carries the error
        inside its ``response`` object, a bare ``error`` event carries it at the
        top level.

        Args:
            event: The failure event.

        Returns:
            The error to raise.
        """
        payload = event.get("response")
        error = payload.get("error") if isinstance(payload, dict) else None
        if not isinstance(error, dict):
            error = event
        code = error.get("code")
        message = error.get("message")
        detail = message if isinstance(message, str) and message else "no message"
        suffix = f" (code {code!r})" if isinstance(code, str) and code else ""
        return ProviderError(
            f"{self.name} stream failed: {self._scrub(detail)}{suffix}",
            status_code=None,
            retryable=isinstance(code, str) and code in _RETRYABLE_ERROR_CODES,
            error_code=code if isinstance(code, str) and code else None,
        )

    # -- streaming ----------------------------------------------------------

    async def stream(self, req: CompletionRequest) -> AsyncIterator[ProviderEvent]:
        """Run one turn and stream its events.

        Text and reasoning are yielded from delta events as they arrive, but the
        assistant turn is assembled from complete output items — from
        ``response.output_item.done`` as they land, with a *populated* terminal
        ``response.output`` preferred when one arrives. A partially-populated
        ``response.output_item.added`` never contributes.

        Args:
            req: The turn to run.

        Yields:
            :data:`~logpose.providers.base.ProviderEvent` values, ending with
            exactly one :class:`~logpose.providers.base.CompletionDone`.

        Raises:
            AuthError: If no credential could be resolved.
            ProviderError: If the backend is unreachable, rejects the request, or
                fails mid-stream.
        """
        credential = await self._current_credential()
        headers = self._headers(credential)
        body = self._build_body(req)

        items: dict[int, dict[str, Any]] = {}
        final: dict[str, Any] | None = None
        failure: dict[str, Any] | None = None

        try:
            async with self._http().stream(
                "POST", f"{self.base_url}/responses", json=body, headers=headers
            ) as response:
                if response.status_code >= 400:
                    raw = await response.aread()
                    raise self._status_error(
                        response.status_code,
                        raw.decode("utf-8", "replace"),
                        response.headers,
                    )
                async for line in response.aiter_lines():
                    event = parse_sse_data(line)
                    if event is None:
                        continue
                    kind = event.get("type")

                    if kind == "response.output_text.delta":
                        text = event.get("delta")
                        if isinstance(text, str) and text:
                            yield ProviderTextDelta(text=text)

                    elif kind in _REASONING_DELTA_EVENTS:
                        text = event.get("delta")
                        if isinstance(text, str) and text:
                            yield ProviderThinkingDelta(text=text)

                    elif kind == "response.output_item.done":
                        item = event.get("item")
                        if isinstance(item, dict):
                            index = event.get("output_index")
                            items[index if isinstance(index, int) else len(items)] = item

                    elif kind in ("response.completed", "response.incomplete"):
                        payload = event.get("response")
                        if isinstance(payload, dict):
                            final = payload

                    elif kind in ("response.failed", "error"):
                        failure = event
                        break
        except httpx.HTTPError as exc:
            raise self._transport_error(exc) from exc

        if failure is not None:
            raise self._event_error(failure)

        output = final.get("output") if isinstance(final, dict) else None
        if isinstance(output, list) and output:
            # A populated terminal response is authoritative: it carries every
            # item, in order, however the incremental events happened to arrive.
            # It is only a preference and not the rule, because the subscription
            # backend sends `"output": []` on `response.completed` and delivers
            # everything through `response.output_item.done` instead — verified
            # against the live endpoint. Trusting an empty array there would
            # silently discard the whole turn.
            assembled = [item for item in output if isinstance(item, dict)]
        elif items:
            assembled = [items[index] for index in sorted(items)]
        elif final is None:
            raise ProviderError(
                f"{self.name} stream ended without a terminal event. The connection was "
                "probably cut mid-turn.",
                status_code=None,
                retryable=True,
            )
        else:
            assembled = []

        content: list[ContentBlock] = []
        saw_refusal = False
        for item in assembled:
            blocks, refused = item_to_blocks(item)
            content.extend(blocks)
            saw_refusal = saw_refusal or refused

        saw_tool_calls = any(isinstance(block, ToolUseBlock) for block in content)

        yield CompletionDone(
            message=Message(role="assistant", content=content),
            stop_reason=map_stop_reason(
                final, saw_tool_calls=saw_tool_calls, saw_refusal=saw_refusal
            ),
            usage=map_usage(final.get("usage") if isinstance(final, dict) else None),
        )
