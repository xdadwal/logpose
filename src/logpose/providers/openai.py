"""OpenAI provider — the Responses API with an API key.

The supported, metered path: ``POST https://api.openai.com/v1/responses`` with
``$OPENAI_API_KEY``. Everything about the wire format lives in
:mod:`logpose.providers._responses`; this module is only the endpoint, the
credential kind, and the absence of any Codex CLI dress-up.

Not to be confused with :mod:`logpose.providers.openai_compat`
--------------------------------------------------------------
``openai-compat`` speaks **Chat Completions** (``POST /chat/completions``) — the
API that llama.cpp, vLLM, Ollama, LM Studio and Kimi all clone, and which OpenAI
also still serves. This provider speaks the **Responses API**, which only OpenAI
serves. They are two different APIs, not a modern and a legacy spelling of one:

- Chat Completions has no way to carry a reasoning model's chain of thought across
  a tool call, so multi-step tool use on a reasoning model loses it every turn.
  Responses returns reasoning as an item with an ``encrypted_content`` blob that
  can be resent, which is what keeps it.
- Tool specs, tool results, token-cap fields and usage counters all differ in
  shape.

Pick this one for OpenAI's own models. Pick ``openai-compat`` for anything else
that merely speaks OpenAI's older shape.

For a ChatGPT subscription rather than an API key, use
:mod:`logpose.providers.codex`.
"""

from __future__ import annotations

from logpose.providers._responses import ResponsesProvider

__all__ = [
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_MODEL",
    "OPENAI_BASE_URL",
    "OpenAIProvider",
]

OPENAI_BASE_URL = "https://api.openai.com/v1"
"""Public Responses API root."""


class OpenAIProvider(ResponsesProvider):
    """A :class:`~logpose.providers.base.Provider` for OpenAI's Responses API.

    Accepts **only** an API key — ``api_key=``, ``$OPENAI_API_KEY``, or an
    ``OPENAI_API_KEY`` entry in ``~/.codex/auth.json`` if ``codex login`` wrote
    one there. A ChatGPT subscription token neither satisfies nor shadows it; the
    error says so and names the provider that does take one.

    .. code-block:: python

        from logpose import Agent

        agent = Agent("openai")                              # $OPENAI_API_KEY
        agent = Agent("openai", api_key="sk-proj-...")
        agent = Agent("openai", model="gpt-5.1", reasoning_effort="high")

    Attributes:
        name: Always ``"openai"``.
    """

    name = "openai"
    REQUIRED_KIND = "api_key"
    BASE_URL = OPENAI_BASE_URL
    MODEL_ENV = "OPENAI_RESPONSES_MODEL"
    BASE_URL_ENV = "OPENAI_RESPONSES_BASE_URL"
    SIBLING_HINT = ' Use Agent("codex") for the ChatGPT subscription path.'
    AUTH_FAILURE_HINT = "Check api_key=... or the $OPENAI_API_KEY environment variable."

    DEFAULT_MODEL = "gpt-5.1"
    """Model used when neither the caller nor ``$OPENAI_RESPONSES_MODEL`` names one.

    Deliberately not the Codex provider's default: ChatGPT-backend model slugs and
    public API model ids are separate namespaces, and a provider whose default
    changed with the credential would be a debugging nightmare. Pass ``model=`` for
    anything else.
    """

    DEFAULT_MAX_TOKENS = 32000
    """Output ceiling for a turn.

    Roomy because reasoning tokens are billed *inside* ``output_tokens`` on this
    API, so a ceiling sized for visible output truncates the turn mid-thought and
    comes back ``incomplete`` with nothing usable. Unlike the subscription
    endpoint, this one accepts the field, so the value takes effect.
    """


DEFAULT_MODEL = OpenAIProvider.DEFAULT_MODEL
"""Model :class:`OpenAIProvider` uses when the caller names none."""

DEFAULT_MAX_TOKENS = OpenAIProvider.DEFAULT_MAX_TOKENS
"""Output ceiling :class:`OpenAIProvider` defaults to."""
