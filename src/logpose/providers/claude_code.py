"""Claude Code provider — the Anthropic Messages API on a subscription token.

Everything about the wire format lives in
:mod:`logpose.providers._anthropic_base`. This module is only what makes a request
a *Claude Code* request: the credential kind, the OAuth beta header, and the
identity line Anthropic expects the system prompt to open with.

For an Anthropic API key, use :mod:`logpose.providers.anthropic` instead.
"""

from __future__ import annotations

from typing import Any

from logpose.providers._anthropic_base import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    AnthropicBaseProvider,
)

__all__ = [
    "CLAUDE_CODE_IDENTITY",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_MODEL",
    "OAUTH_BETA_HEADER",
    "ClaudeCodeProvider",
]

OAUTH_BETA_HEADER = "oauth-2025-04-20"
"""``anthropic-beta`` value required for subscription OAuth tokens."""

CLAUDE_CODE_IDENTITY = "You are Claude Code, Anthropic's official CLI for Claude."
"""System line prepended to subscription requests.

Anthropic answers a subscription request whose system prompt does not open with
this line with ``HTTP 429 rate_limit_error`` — even when the account has quota to
spare, and with none of the ``anthropic-ratelimit-*`` headers a real limit carries.
Sending it is what makes the subscription path work at all, so
:class:`ClaudeCodeProvider` sends it by default. See ``compat_claude_code`` to
disable it, and expect the 429 if you do.
"""


class ClaudeCodeProvider(AnthropicBaseProvider):
    """A :class:`~logpose.providers.base.Provider` for a Claude Code subscription.

    Accepts **only** a subscription credential — ``$CLAUDE_CODE_OAUTH_TOKEN``, the
    local Claude Code credential store, or an explicit ``auth_token=``. An
    ``$ANTHROPIC_API_KEY`` neither satisfies nor shadows it; the error says so and
    names the provider that does take one.

    .. code-block:: python

        from logpose import Agent

        agent = Agent("claude-code")                       # rides the local login
        agent = Agent("claude-code", model_default="claude-opus-5")

    Attributes:
        name: Always ``"claude-code"``.
        compat_claude_code: Whether the identity line is prepended.
    """

    name = "claude-code"
    REQUIRED_KIND = "oauth"
    AUTH_ATTR = "auth_token"
    UNUSED_AUTH_ATTR = "api_key"
    SDK_HEADERS = {"anthropic-beta": OAUTH_BETA_HEADER}
    SIBLING_HINT = ' Use Agent("anthropic") for the API-key path.'

    def __init__(
        self,
        *,
        auth_token: str | None = None,
        compat_claude_code: bool = True,
        **kwargs: Any,
    ) -> None:
        """Configure the provider.

        Args:
            auth_token: Claude Code subscription OAuth token, sent as
                ``Authorization: Bearer`` with the OAuth beta header. Falls back to
                ``$CLAUDE_CODE_OAUTH_TOKEN`` and then the local credential store.
            compat_claude_code: Prepend :data:`CLAUDE_CODE_IDENTITY` to the system
                prompt. On by default because Anthropic requires it; pass ``False``
                to send a bare subscription request anyway and expect a 429.
            **kwargs: Forwarded to
                :class:`~logpose.providers._anthropic_base.AnthropicBaseProvider`.
        """
        super().__init__(**kwargs)
        self._explicit_auth_token = auth_token
        self.compat_claude_code = compat_claude_code

    def __repr__(self) -> str:
        """Return an unambiguous representation. Never includes a credential."""
        return (
            f"ClaudeCodeProvider(model_default={self.model_default!r}, "
            f"max_tokens={self.max_tokens!r}, "
            f"compat_claude_code={self.compat_claude_code!r})"
        )

    def _resolve_kwargs(self) -> dict[str, Any]:
        """Name the caller's explicit subscription token.

        Returns:
            Keyword arguments for ``CredentialProvider.resolve``.
        """
        return {"explicit_auth_token": self._explicit_auth_token}

    def _secrets(self) -> tuple[str, ...]:
        """Credential values that must never appear in an error."""
        secrets = super()._secrets()
        if self._explicit_auth_token and self._explicit_auth_token not in secrets:
            secrets += (self._explicit_auth_token,)
        return secrets

    def _build_system(self, system: str | None) -> str | list[dict[str, Any]] | None:
        """Build the ``system`` request parameter, opening with the identity line.

        Args:
            system: The caller's system prompt, if any.

        Returns:
            The identity line as the first system block, with the caller's prompt
            following it as a second block. Just the caller's prompt when
            ``compat_claude_code`` is off.
        """
        if not self.compat_claude_code:
            return system
        blocks: list[dict[str, Any]] = [{"type": "text", "text": CLAUDE_CODE_IDENTITY}]
        if system:
            blocks.append({"type": "text", "text": system})
        return blocks
