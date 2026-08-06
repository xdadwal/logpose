"""Codex provider — OpenAI's Responses API on a ChatGPT subscription.

Everything about the wire format lives in :mod:`logpose.providers._responses`.
This module is only what makes a request a *Codex CLI* request: the endpoint, the
credential kind, the headers the subscription backend demands, and the identity
line it expects the instructions to open with.

For an OpenAI API key, use :mod:`logpose.providers.openai` instead. The two are
separate providers on purpose — see the ``_responses`` module docstring.
"""

from __future__ import annotations

from typing import Any

from logpose.auth.codex import Credential, require_account_id
from logpose.providers._responses import ResponsesProvider

__all__ = [
    "CHATGPT_BASE_URL",
    "CODEX_CLI_IDENTITY",
    "CODEX_ORIGINATOR",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_MODEL",
    "RESPONSES_BETA_HEADER",
    "CodexProvider",
]

CHATGPT_BASE_URL = "https://chatgpt.com/backend-api/codex"
"""Codex subscription endpoint root (undocumented; may change)."""

RESPONSES_BETA_HEADER = "responses=experimental"
"""``OpenAI-Beta`` value the subscription backend expects."""

CODEX_ORIGINATOR = "codex_cli_rs"
"""``originator`` value identifying a request as the Codex CLI's."""

CODEX_CLI_IDENTITY = "You are Codex, a coding agent based on GPT-5."
"""Instruction line the subscription backend expects requests to open with.

The Codex backend expects requests on a subscription token to look like the CLI's
own: a non-empty ``instructions`` field that opens this way, and an ``originator``
header. Sending them is what makes the subscription path work at all, so
:class:`CodexProvider` sends both by default. See ``compat_codex_cli`` to disable
it, and expect a rejection if you do.

The string is the first sentence of the ``base_instructions`` the CLI caches in
``~/.codex/models_cache.json`` — that is where to re-derive it if it ever drifts.
It is a constant rather than a runtime read of that file on purpose: reading it
would make the request bytes depend on local state, which is both unreproducible
and fatal to prompt caching.
"""

_FALLBACK_INSTRUCTIONS = "You are a helpful assistant."


class CodexProvider(ResponsesProvider):
    """A :class:`~logpose.providers.base.Provider` for Codex on a ChatGPT subscription.

    Accepts **only** a subscription credential — ``codex login``'s
    ``~/.codex/auth.json``, or an explicit ``auth_token=``. An ``$OPENAI_API_KEY``
    neither satisfies nor shadows it; the error says so and names the provider
    that does take one.

    .. code-block:: python

        from logpose import Agent

        agent = Agent("codex")                            # rides `codex login`
        agent = Agent("codex", model="gpt-5.4")
        agent = Agent("codex", reasoning_effort="high")

    Attributes:
        name: Always ``"codex"``.
        compat_codex_cli: Whether requests are dressed as the Codex CLI's.
    """

    name = "codex"
    REQUIRED_KIND = "oauth"
    BASE_URL = CHATGPT_BASE_URL
    MODEL_ENV = "CODEX_MODEL"
    BASE_URL_ENV = "CODEX_BASE_URL"
    SIBLING_HINT = ' Use Agent("openai") for the API-key path.'

    DEFAULT_MODEL = "gpt-5.5"
    """Model used when neither the caller nor ``$CODEX_MODEL`` names one.

    The highest-priority listed model the Codex CLI offers, verified against the
    live subscription backend.
    """

    DEFAULT_MAX_TOKENS = 32000
    """Output ceiling for a turn.

    Twice the Anthropic provider's, because reasoning tokens are billed *inside*
    ``output_tokens`` on this API: a ceiling sized for visible output truncates the
    turn mid-thought. Note that this endpoint rejects the field outright (see
    :attr:`SENDS_MAX_OUTPUT_TOKENS`), so the value only takes effect if a caller
    redirects the provider elsewhere with ``base_url=``.
    """

    SENDS_MAX_OUTPUT_TOKENS = False
    """The subscription backend answers ``400 Unsupported parameter: max_output_tokens``.

    Verified against the live endpoint. An ``Agent``'s ``max_tokens`` is therefore a
    documented no-op here and the model's own budget applies; the API-key provider
    does send and honour it.
    """

    def __init__(
        self,
        *,
        auth_token: str | None = None,
        account_id: str | None = None,
        compat_codex_cli: bool = True,
        **kwargs: Any,
    ) -> None:
        """Configure the Codex backend.

        Args:
            auth_token: Subscription OAuth token, used instead of reading
                ``~/.codex/auth.json``.
            account_id: ChatGPT account to bill requests to, overriding whatever
                the credential store says.
            compat_codex_cli: Dress requests as the Codex CLI's — the identity
                line ahead of ``instructions``, plus ``originator``. On by
                default because the backend expects it; turning it off will
                probably be rejected.
            **kwargs: Forwarded to
                :class:`~logpose.providers._responses.ResponsesProvider`.
        """
        super().__init__(**kwargs)
        self._explicit_auth_token = auth_token
        self._explicit_account_id = account_id
        self.compat_codex_cli = compat_codex_cli

    def __repr__(self) -> str:
        """Return an unambiguous representation. Never includes a credential."""
        return (
            f"CodexProvider(base_url={self.base_url!r}, "
            f"model_default={self.model_default!r}, max_tokens={self.max_tokens!r}, "
            f"compat_codex_cli={self.compat_codex_cli!r})"
        )

    def _resolve_kwargs(self) -> dict[str, Any]:
        """Name the caller's explicit subscription material.

        Returns:
            Keyword arguments for ``CredentialProvider.resolve``.
        """
        return {
            "explicit_auth_token": self._explicit_auth_token,
            "explicit_account_id": self._explicit_account_id,
        }

    def _secrets(self) -> tuple[str, ...]:
        """Credential values that must never appear in an error."""
        secrets = super()._secrets()
        if self._explicit_auth_token:
            secrets += (self._explicit_auth_token,)
        return secrets

    def _headers(self, credential: Credential) -> dict[str, str]:
        """Build request headers for the subscription backend.

        ``chatgpt-account-id`` and ``OpenAI-Beta`` are **not** gated on
        ``compat_codex_cli``: they are part of the subscription wire protocol
        rather than the identity shim, and go out regardless. Only ``originator``
        belongs to the shim.

        Args:
            credential: The subscription credential for this turn.

        Returns:
            The headers to send.

        Raises:
            AuthError: If the credential names no ChatGPT account. Failing here
                beats a 401 with an opaque body.
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {credential.value}",
            "chatgpt-account-id": require_account_id(credential),
            "OpenAI-Beta": RESPONSES_BETA_HEADER,
        }
        if self.compat_codex_cli:
            headers["originator"] = CODEX_ORIGINATOR
        headers.update(self._default_headers)
        return headers

    def _build_instructions(self, system: str | None) -> str:
        """Build the ``instructions`` field, never empty.

        The subscription backend rejects empty ``instructions``, so this always
        returns something. Order mirrors ``ClaudeCodeProvider``: the identity line
        first, the caller's prompt after it.

        Args:
            system: The caller's system prompt, if any.

        Returns:
            The instructions to send.
        """
        identity = CODEX_CLI_IDENTITY if self.compat_codex_cli else None
        parts = [part for part in (identity, self._instructions, system) if part]
        return "\n\n".join(parts) or _FALLBACK_INSTRUCTIONS


DEFAULT_MODEL = CodexProvider.DEFAULT_MODEL
"""Model :class:`CodexProvider` uses when the caller names none."""

DEFAULT_MAX_TOKENS = CodexProvider.DEFAULT_MAX_TOKENS
"""Output ceiling :class:`CodexProvider` defaults to."""
