"""Shared machinery for the two Anthropic backends.

Two public providers sit on this module — :mod:`logpose.providers.anthropic` for
an API key, and :mod:`logpose.providers.claude_code` for a Claude Code
subscription. Both drive the official ``anthropic`` SDK through
``AsyncAnthropic.messages.stream(...)`` — never raw HTTP, and never the
non-streaming endpoint (streaming avoids request timeouts on long turns and is
what feeds delta events).

They differ in exactly three ways, all expressed as subclass overrides rather than
runtime branches: which credential kind they accept, which auth header the SDK
emits, and whether the Claude Code identity line is prepended to the system
prompt.

Naming them separately is what lets that be static. A single provider serving both
kinds had to rebuild its SDK client whenever the credential *kind* changed between
turns — because the OAuth beta header is baked into ``default_headers`` — and had
to guess whether to send the identity line before any credential was resolved.
Neither situation can arise now.

Authentication
--------------
The two credential shapes are **mutually exclusive on the wire**:

* an API key, sent as ``x-api-key``;
* a Claude Code subscription OAuth token, sent as ``Authorization: Bearer``
  together with the ``anthropic-beta: oauth-2025-04-20`` header.

The SDK falls back to the ``ANTHROPIC_API_KEY`` environment variable when no
``api_key`` argument is given, so an OAuth-mode client could silently end up
sending *both* headers — which the API rejects. :meth:`AnthropicBaseProvider._apply_credential`
pins one slot and clears the other after construction, so exactly one auth header
is emitted regardless of the environment or the installed SDK version.

Security: credential values never appear in log output, ``repr``, or exception
messages — **including tracebacks**. A gateway that echoes the presented
credential back in an error body would otherwise leak it twice: once through the
wrapper message, and once through the chained SDK exception that every traceback
renders. Both are scrubbed; anything that must reference a token is passed
through :func:`~logpose.providers._redact.redact`.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from typing import Any, ClassVar, Literal

import anthropic
from anthropic import AsyncAnthropic

from logpose.auth.claude_code import Credential, CredentialProvider
from logpose.errors import AuthError, LogposeError, ProviderError, _provider_metadata_from_headers
from logpose.messages import (
    ContentBlock,
    Message,
    RawBlock,
    RedactedThinkingBlock,
    StopReason,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
)
from logpose.providers._redact import redact
from logpose.providers.base import (
    DEFAULT_CLOUD_TURN_TIMEOUT,
    CompletionDone,
    CompletionRequest,
    ProviderEvent,
    ProviderTextDelta,
    ProviderThinkingDelta,
    ToolSpec,
)

__all__ = [
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_MODEL",
    "AnthropicBaseProvider",
    "block_from_wire",
    "map_stop_reason",
    "map_usage",
    "message_to_wire",
    "normalize_thinking",
    "tool_to_wire",
]

DEFAULT_MODEL = "claude-opus-5"
"""Model used when a :class:`~logpose.providers.base.CompletionRequest` omits one."""

DEFAULT_MAX_TOKENS = 16000
"""Output-token ceiling used when a request omits one."""

_ADAPTIVE_THINKING: dict[str, Any] = {"type": "adaptive", "display": "summarized"}
# ``display="summarized"`` is required: the API default is "omitted", which
# streams thinking blocks whose text is empty.

_DISABLED_THINKING: dict[str, Any] = {"type": "disabled"}

_MODEL_PAGE_SIZE = 1000
# One page is plenty — Anthropic lists tens of models, not thousands — so
# list_models avoids pagination entirely.

_RETRYABLE_STATUS = frozenset({408, 409, 429})

_STOP_REASONS: dict[str, StopReason] = {
    "end_turn": "end_turn",
    "tool_use": "tool_use",
    "max_tokens": "max_tokens",
    "stop_sequence": "stop_sequence",
    "refusal": "refusal",
    "pause_turn": "pause_turn",
    # Not in logpose's StopReason literal; the turn was cut short by a limit,
    # which is what "max_tokens" means to the loop.
    "model_context_window_exceeded": "max_tokens",
}


# ---------------------------------------------------------------------------
# request translation
# ---------------------------------------------------------------------------


def _content_block_to_wire(block: ContentBlock) -> dict[str, Any]:
    """Translate one logpose content block into its Anthropic wire form.

    Thinking blocks keep their ``signature`` / ``data`` verbatim: the API
    rejects a turn whose thinking blocks were altered.

    Args:
        block: The block to translate.

    Returns:
        The Anthropic content-block dict.

    Raises:
        LogposeError: If the block type has no Anthropic representation.
    """
    if isinstance(block, TextBlock):
        return {"type": "text", "text": block.text}
    if isinstance(block, ThinkingBlock):
        wire: dict[str, Any] = {"type": "thinking", "thinking": block.thinking}
        if block.signature is not None:
            wire["signature"] = block.signature
        return wire
    if isinstance(block, RedactedThinkingBlock):
        return {"type": "redacted_thinking", "data": block.data}
    if isinstance(block, ToolUseBlock):
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    if isinstance(block, ToolResultBlock):
        return {
            "type": "tool_result",
            "tool_use_id": block.tool_use_id,
            "content": block.content,
            "is_error": block.is_error,
        }
    if isinstance(block, RawBlock):
        # Opaque round trip: whatever the provider sent, sent back unchanged.
        return dict(block.data)
    raise LogposeError(
        f"Cannot send content block of type {type(block).__name__!r} to Anthropic."
    )


def message_to_wire(message: Message) -> dict[str, Any]:
    """Translate one logpose message into its Anthropic wire form."""
    return {
        "role": message.role,
        "content": [_content_block_to_wire(block) for block in message.content],
    }


def tool_to_wire(spec: ToolSpec) -> dict[str, Any]:
    """Translate a :class:`~logpose.providers.base.ToolSpec` into a tool definition."""
    return {
        "name": spec.name,
        "description": spec.description,
        "input_schema": spec.input_schema,
    }


# ---------------------------------------------------------------------------
# response translation
# ---------------------------------------------------------------------------


def _raw_from_wire(block: Any) -> RawBlock | None:
    """Capture a block logpose has no model for, verbatim.

    Dropping such a block would make the assistant turn un-resendable: the API
    reads a trailing assistant turn with no server-tool block as a prefill and
    rejects it, which is exactly the turn the loop re-sends on ``pause_turn``.

    Args:
        block: A block from ``Message.content`` as returned by the SDK.

    Returns:
        A :class:`~logpose.messages.RawBlock` holding the block's JSON form, or
        ``None`` if it could not be serialized (nothing usable to send back).
    """
    dump = getattr(block, "model_dump", None)
    if callable(dump):
        try:
            data: Any = dump(mode="json", exclude_none=True)
        except Exception:  # noqa: BLE001 - an unserializable block is simply dropped
            return None
    elif isinstance(block, dict):
        data = dict(block)
    else:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("type"), str):
        return None
    return RawBlock(data=data)


def block_from_wire(block: Any) -> ContentBlock | None:
    """Translate one Anthropic response block back into a logpose block.

    Args:
        block: A block from ``Message.content`` as returned by the SDK.

    Returns:
        The logpose block, or ``None`` when the block cannot be represented in a
        form that could be sent back. Block types logpose does not model are
        preserved as a :class:`~logpose.messages.RawBlock` rather than dropped.
    """
    block_type = getattr(block, "type", None)
    if block_type == "text":
        return TextBlock(text=block.text)
    if block_type == "thinking":
        signature = getattr(block, "signature", None)
        if not signature:
            # A thinking block truncated by max_tokens never receives its
            # signature_delta, so the SDK leaves `signature` at its "" seed.
            # Echoing that back presents an unsigned block as a complete one and
            # the API rejects the turn for an invalid signature — and the block
            # is worthless anyway, since a signature cannot be reconstructed.
            # Its text already reached the consumer as ThinkingDelta events.
            return None
        return ThinkingBlock(thinking=block.thinking, signature=signature)
    if block_type == "redacted_thinking":
        return RedactedThinkingBlock(data=block.data)
    if block_type == "tool_use":
        # block.input is already parsed by the SDK — never re-parse or
        # string-match the serialized form.
        raw_input = block.input
        return ToolUseBlock(
            id=block.id,
            name=block.name,
            input=raw_input if isinstance(raw_input, dict) else {},
        )
    return _raw_from_wire(block)


def map_stop_reason(raw: Any) -> StopReason:
    """Map an Anthropic stop reason onto logpose's :data:`~logpose.messages.StopReason`.

    Args:
        raw: The provider's ``stop_reason`` (may be ``None`` mid-stream).

    Returns:
        The logpose stop reason; unknown or missing values fall back to
        ``"end_turn"``.
    """
    if isinstance(raw, str):
        return _STOP_REASONS.get(raw, "end_turn")
    return "end_turn"


def map_usage(raw: Any) -> Usage:
    """Map Anthropic token accounting onto :class:`~logpose.messages.Usage`."""
    if raw is None:
        return Usage()
    return Usage(
        input_tokens=getattr(raw, "input_tokens", 0) or 0,
        output_tokens=getattr(raw, "output_tokens", 0) or 0,
        cache_read_input_tokens=getattr(raw, "cache_read_input_tokens", 0) or 0,
        cache_creation_input_tokens=getattr(raw, "cache_creation_input_tokens", 0) or 0,
    )


def normalize_thinking(thinking: str | dict[str, Any] | None) -> dict[str, Any] | None:
    """Resolve the ``thinking`` constructor argument into a wire value.

    Args:
        thinking: ``"adaptive"``, ``"disabled"``, ``None``, or a dict passed
            through verbatim.

    Returns:
        The ``thinking`` request parameter, or ``None`` to omit it entirely.

    Raises:
        LogposeError: If ``thinking`` is an unrecognized string.
    """
    if thinking is None:
        return None
    if isinstance(thinking, dict):
        return dict(thinking)
    normalized = thinking.strip().lower()
    if normalized == "adaptive":
        return dict(_ADAPTIVE_THINKING)
    if normalized == "disabled":
        return dict(_DISABLED_THINKING)
    raise LogposeError(
        f"Unknown thinking mode {thinking!r}. Use 'adaptive', 'disabled', None, or a dict."
    )


def _status_is_retryable(status_code: int | None) -> bool:
    """Whether an HTTP status is worth retrying (429/408/409 and 5xx)."""
    if status_code is None:
        return False
    return status_code in _RETRYABLE_STATUS or status_code >= 500


# ---------------------------------------------------------------------------
# provider base
# ---------------------------------------------------------------------------


class AnthropicBaseProvider:
    """Base for the two Anthropic backends. Drives one turn per :meth:`stream` call.

    Executes no tools: it advertises them to the model and reports back what the
    model asked for. Only the loop above it may cause side effects.

    Attributes:
        model_default: Model used when a request omits one.
        max_tokens: Output-token ceiling used when a request omits one.
    """

    # Not a ClassVar: the Provider protocol declares `name` as an instance
    # variable, and a ClassVar cannot satisfy that.
    name: str
    """Registered provider name."""

    REQUIRED_KIND: ClassVar[Literal["api_key", "oauth"]]
    """The one credential kind this provider accepts."""

    AUTH_ATTR: ClassVar[str]
    """SDK client attribute carrying this provider's credential."""

    UNUSED_AUTH_ATTR: ClassVar[str]
    """SDK client attribute that must be cleared so only one auth header is sent."""

    SDK_HEADERS: ClassVar[dict[str, str]] = {}
    """Headers baked into the SDK client at construction."""

    SIBLING_HINT: ClassVar[str] = ""
    """One line appended to a credential failure, naming the provider to use instead."""

    turn_timeout = DEFAULT_CLOUD_TURN_TIMEOUT
    """Recommended complete-turn deadline in seconds for cloud inference."""

    def __init__(
        self,
        *,
        model_default: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        thinking: str | dict[str, Any] | None = "adaptive",
        timeout: float | None = 600.0,
        client: AsyncAnthropic | None = None,
    ) -> None:
        """Configure the provider.

        Credentials come from
        :meth:`logpose.auth.claude_code.CredentialProvider.resolve`, restricted to
        this provider's :attr:`REQUIRED_KIND`. Resolution is deferred to the first
        request, so constructing the provider never raises
        :class:`~logpose.errors.AuthError` and never touches the keychain.

        Args:
            model_default: Model used when a request omits one.
            max_tokens: Output-token ceiling used when a request omits one.
            thinking: ``"adaptive"`` (default) requests adaptive thinking with
                summarized display, which is required for thinking deltas to
                carry any text. ``"disabled"`` sends ``{"type": "disabled"}``;
                ``None`` omits the parameter; a dict is sent verbatim.
            timeout: SDK transport timeout in seconds. ``None`` leaves the SDK
                default in place; the agent still enforces its complete-turn
                deadline separately.
            client: A pre-built ``AsyncAnthropic``. When given, no credential
                resolution happens and the caller owns the client's lifetime.

        Raises:
            LogposeError: If ``thinking``, ``max_tokens``, or ``timeout`` is invalid.
        """
        if max_tokens <= 0:
            raise LogposeError(f"max_tokens must be positive, got {max_tokens!r}.")
        if timeout is not None and timeout <= 0:
            raise LogposeError(f"timeout must be positive or None, got {timeout!r}.")

        self.model_default = model_default
        self.max_tokens = max_tokens

        self._thinking = normalize_thinking(thinking)
        self._timeout = timeout
        self._client: Any = client
        self._owns_client = client is None
        self._credentials: CredentialProvider | None = None
        self._credential_value: str | None = None
        # Guards the *lazy build* of the credential provider. Without it, N
        # concurrent first requests each build their own CredentialProvider —
        # and therefore their own lock — so the single-flight refresh inside
        # CredentialProvider.get never sees the others. Bound to a loop on
        # first use, like every other lock in the package.
        self._credentials_lock = asyncio.Lock()

    def __repr__(self) -> str:
        """Return an unambiguous representation. Never includes a credential."""
        return (
            f"{type(self).__name__}(model_default={self.model_default!r}, "
            f"max_tokens={self.max_tokens!r})"
        )

    # -- credentials --------------------------------------------------------

    def _resolve_kwargs(self) -> dict[str, Any]:
        """Name the caller's explicit credential material for this provider.

        Returns:
            Keyword arguments for ``CredentialProvider.resolve``.
        """
        return {}

    def _apply_credential(self, client: Any, credential: Credential) -> None:
        """Pin the client's auth so exactly one auth header is ever sent.

        The SDK emits ``x-api-key`` whenever ``client.api_key`` is set and
        ``Authorization`` whenever ``client.auth_token`` is set — independently. An
        ``ANTHROPIC_API_KEY`` picked up from the environment therefore rides along
        with an OAuth token unless the unused slot is explicitly cleared.

        Args:
            client: The ``AsyncAnthropic`` instance to pin.
            credential: The credential to install.
        """
        setattr(client, self.UNUSED_AUTH_ATTR, None)
        setattr(client, self.AUTH_ATTR, credential.value)

    def _build_client(self, credential: Credential) -> AsyncAnthropic:
        """Construct an ``AsyncAnthropic`` wired for this provider's credential kind.

        Args:
            credential: The credential to install.

        Returns:
            A configured SDK client.
        """
        kwargs: dict[str, Any] = {
            self.AUTH_ATTR: credential.value,
            self.UNUSED_AUTH_ATTR: None,
        }
        if self.SDK_HEADERS:
            kwargs["default_headers"] = dict(self.SDK_HEADERS)
        if self._timeout is not None:
            kwargs["timeout"] = self._timeout
        # logpose centralizes retry policy above providers. Leaving the SDK's
        # default retry loop enabled would make attempt counts, backoff, and
        # streaming guarantees differ from the HTTPX-backed providers.
        client = AsyncAnthropic(max_retries=0, **kwargs)
        self._apply_credential(client, credential)
        return client

    async def _current_credential(self) -> Credential:
        """Return a usable credential for this turn.

        The first call builds a
        :class:`~logpose.auth.claude_code.CredentialProvider` (which may read
        the macOS keychain via a subprocess, hence the thread); later calls just
        ask it for a credential, and it refreshes an expiring subscription token
        single-flight.

        The build is double-checked under a lock so concurrent first requests
        share one provider — otherwise each would resolve separately, firing N
        keychain subprocesses and N refresh grants against the same (single-use)
        refresh token.

        Only sources yielding :attr:`REQUIRED_KIND` are considered, so a
        credential of the other kind neither satisfies nor shadows this provider.

        Returns:
            A credential that is not (yet) expired.

        Raises:
            AuthError: If no credential of the required kind could be resolved, or
                a refresh failed. The message names the sibling provider.
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

    async def get_client(self) -> AsyncAnthropic:
        """Return the underlying SDK client, building it on first use.

        When the provider owns the client, the credential is re-checked on every
        call so a refreshed subscription token takes effect without rebuilding the
        connection pool. The credential *kind* is fixed, so unlike an
        all-in-one provider this never has to discard the pool mid-conversation.

        Returns:
            The configured ``AsyncAnthropic`` instance.

        Raises:
            AuthError: If no usable credential could be resolved.
        """
        if not self._owns_client:
            return self._client

        credential = await self._current_credential()
        if self._client is None:
            self._client = self._build_client(credential)
        elif credential.value != self._credential_value:
            self._apply_credential(self._client, credential)
        self._credential_value = credential.value
        return self._client

    async def list_models(self) -> list[str]:
        """Ask Anthropic which models it serves.

        Uses the SDK's own ``models.list``, so it follows whatever the credential is
        entitled to see.

        Returns:
            Model identifiers, newest first as Anthropic orders them.

        Raises:
            AuthError: If no usable credential could be resolved.
            ProviderError: If the request fails.
        """
        client = await self.get_client()
        try:
            page = await client.models.list(limit=_MODEL_PAGE_SIZE)
        except anthropic.AnthropicError as exc:
            raise self._provider_error(exc) from exc
        return [model.id for model in page.data if isinstance(getattr(model, "id", None), str)]

    async def aclose(self) -> None:
        """Close the underlying client if this provider created it."""
        if self._owns_client and self._client is not None:
            await self._client.close()
            self._client = None

    # -- request ------------------------------------------------------------

    def _build_system(self, system: str | None) -> str | list[dict[str, Any]] | None:
        """Build the ``system`` request parameter.

        Args:
            system: The caller's system prompt, if any.

        Returns:
            The value to send, or ``None`` to omit it.
        """
        return system

    def _build_params(self, req: CompletionRequest) -> dict[str, Any]:
        """Translate a :class:`~logpose.providers.base.CompletionRequest` into SDK kwargs.

        Sampling parameters (``temperature``, ``top_p``, ``top_k``) are never
        sent: current models reject them with a 400.

        Args:
            req: The turn to run.

        Returns:
            Keyword arguments for ``client.messages.stream``.
        """
        params: dict[str, Any] = {
            "model": req.model or self.model_default,
            "max_tokens": req.max_tokens or self.max_tokens,
            "messages": [message_to_wire(message) for message in req.messages],
        }
        system = self._build_system(req.system)
        if system:
            params["system"] = system
        if req.tools:
            params["tools"] = [tool_to_wire(spec) for spec in req.tools]
        if self._thinking is not None:
            params["thinking"] = self._thinking
        if req.extra:
            params.update(req.extra)
        return params

    # -- errors -------------------------------------------------------------

    def _secrets(self) -> tuple[str, ...]:
        """Every credential value this provider could have put on the wire.

        The client's own ``api_key`` / ``auth_token`` are included because a
        caller-supplied ``client=`` never goes through credential resolution, so
        the stored values are all ``None`` and would leave an echoed credential
        unredacted.

        Returns:
            The non-empty secrets to scrub, deduplicated.
        """
        candidates = (
            self._credential_value,
            getattr(self._client, "api_key", None),
            getattr(self._client, "auth_token", None),
        )
        seen: dict[str, None] = {}
        for secret in candidates:
            if isinstance(secret, str) and secret:
                seen[secret] = None
        return tuple(seen)

    def _scrub(self, text: str) -> str:
        """Replace any credential value that leaked into ``text`` with a redaction."""
        for secret in self._secrets():
            text = text.replace(secret, redact(secret))
        return text

    def _scrub_in_place(self, exc: BaseException) -> None:
        """Redact credentials inside an SDK exception before it is chained.

        ``raise wrapper from exc`` keeps ``exc`` as ``__cause__``, and every
        traceback renders ``str(exc)`` — so scrubbing only the wrapper still
        writes the raw token to logs, Sentry, and pytest output. The SDK
        exception is about to be discarded as a cause, so rewriting its message
        in place is safe.

        Args:
            exc: The exception that will become ``__cause__``.
        """
        secrets = self._secrets()
        if not secrets:
            return
        exc.args = tuple(
            self._scrub(arg) if isinstance(arg, str) else arg for arg in exc.args
        )
        message = getattr(exc, "message", None)
        if isinstance(message, str):
            with contextlib.suppress(AttributeError):
                exc.message = self._scrub(message)  # type: ignore[attr-defined]

    def _provider_error(self, exc: BaseException) -> ProviderError:
        """Wrap an SDK exception as a :class:`~logpose.errors.ProviderError`.

        Also scrubs ``exc`` itself, so the chained cause cannot leak a
        credential through a traceback.
        """
        self._scrub_in_place(exc)
        if isinstance(exc, anthropic.APIStatusError):
            status: int | None = exc.status_code
            message = self._scrub(str(exc))
            hint = self.SIBLING_HINT if status in (401, 403) else ""
            response = getattr(exc, "response", None)
            raw_headers = getattr(response, "headers", None)
            metadata = _provider_metadata_from_headers(raw_headers)
            request_id = getattr(exc, "request_id", None)
            if isinstance(request_id, str) and request_id:
                metadata["request_id"] = request_id
            return ProviderError(
                f"Anthropic request failed with status {status}: {message}{hint}",
                status_code=status,
                retryable=_status_is_retryable(status),
                **metadata,
            )
        if isinstance(exc, anthropic.APIConnectionError):
            return ProviderError(
                f"Could not reach Anthropic: {self._scrub(str(exc))}",
                status_code=None,
                retryable=True,
            )
        return ProviderError(
            f"Anthropic SDK error: {self._scrub(str(exc))}",
            status_code=None,
            retryable=False,
        )

    # -- streaming ----------------------------------------------------------

    async def stream(self, req: CompletionRequest) -> AsyncIterator[ProviderEvent]:
        """Run one turn and stream its events.

        Emits :class:`~logpose.providers.base.ProviderTextDelta` and
        :class:`~logpose.providers.base.ProviderThinkingDelta` as the model
        generates, then exactly one
        :class:`~logpose.providers.base.CompletionDone` carrying the fully
        assembled assistant message.

        Args:
            req: The turn to run.

        Yields:
            :data:`~logpose.providers.base.ProviderEvent` values.

        Raises:
            AuthError: If no usable credential could be resolved.
            ProviderError: If the upstream request fails.
        """
        client = await self.get_client()
        params = self._build_params(req)

        try:
            async with client.messages.stream(**params) as stream:
                async for event in stream:
                    if getattr(event, "type", None) != "content_block_delta":
                        continue
                    delta: Any = getattr(event, "delta", None)
                    delta_type = getattr(delta, "type", None)
                    if delta_type == "text_delta":
                        yield ProviderTextDelta(text=delta.text)
                    elif delta_type == "thinking_delta":
                        yield ProviderThinkingDelta(text=delta.thinking)
                final = await stream.get_final_message()
        except anthropic.AnthropicError as exc:
            raise self._provider_error(exc) from exc

        content: list[ContentBlock] = []
        for raw_block in final.content:
            block = block_from_wire(raw_block)
            if block is not None:
                content.append(block)

        yield CompletionDone(
            message=Message(role="assistant", content=content),
            stop_reason=map_stop_reason(getattr(final, "stop_reason", None)),
            usage=map_usage(getattr(final, "usage", None)),
        )
