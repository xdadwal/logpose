"""Anthropic provider — the Messages API with an API key.

The supported, metered path: ``x-api-key`` against ``api.anthropic.com``.
Everything about the wire format lives in
:mod:`logpose.providers._anthropic_base`; this module is only the credential kind
and the auth header.

For a Claude Code subscription, use :mod:`logpose.providers.claude_code` instead.
The two are separate providers on purpose — see the ``_anthropic_base`` module
docstring.
"""

from __future__ import annotations

from typing import Any

from logpose.providers._anthropic_base import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    AnthropicBaseProvider,
)
from logpose.providers._redact import redact

__all__ = [
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_MODEL",
    "AnthropicProvider",
    "redact",
]

# ``redact`` lives in ``logpose.providers._redact`` so other backends can reuse it
# without importing this module (and with it, the anthropic SDK). It stays exported
# here for backwards compatibility.


class AnthropicProvider(AnthropicBaseProvider):
    """A :class:`~logpose.providers.base.Provider` for Anthropic with an API key.

    Accepts **only** an API key — ``api_key=`` or ``$ANTHROPIC_API_KEY``. A Claude
    Code subscription token neither satisfies nor shadows it; the error says so and
    names the provider that does take one.

    .. code-block:: python

        from logpose import Agent

        agent = Agent("anthropic")                          # $ANTHROPIC_API_KEY
        agent = Agent("anthropic", api_key="sk-ant-api03-...")
        agent = Agent("anthropic", model_default="claude-opus-5")

    Attributes:
        name: Always ``"anthropic"``.
    """

    name = "anthropic"
    REQUIRED_KIND = "api_key"
    AUTH_ATTR = "api_key"
    UNUSED_AUTH_ATTR = "auth_token"
    SIBLING_HINT = ' Use Agent("claude-code") for the subscription path.'

    def __init__(self, *, api_key: str | None = None, **kwargs: Any) -> None:
        """Configure the provider.

        ``api_key`` is keyword-only, as everything here is, so no credential can be
        passed positionally into the wrong slot.

        Args:
            api_key: Anthropic API key, sent as ``x-api-key``. Falls back to
                ``$ANTHROPIC_API_KEY``.
            **kwargs: Forwarded to
                :class:`~logpose.providers._anthropic_base.AnthropicBaseProvider`.
        """
        super().__init__(**kwargs)
        self._explicit_api_key = api_key

    def _resolve_kwargs(self) -> dict[str, Any]:
        """Name the caller's explicit API key.

        Returns:
            Keyword arguments for ``CredentialProvider.resolve``.
        """
        return {"explicit_api_key": self._explicit_api_key}

    def _secrets(self) -> tuple[str, ...]:
        """Credential values that must never appear in an error."""
        secrets = super()._secrets()
        if self._explicit_api_key and self._explicit_api_key not in secrets:
            secrets += (self._explicit_api_key,)
        return secrets
