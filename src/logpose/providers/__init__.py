"""Provider registry.

Providers are resolved by name so consumers never import a backend directly::

    provider = resolve("anthropic", model="claude-opus-5")

Built-in providers are registered with a factory that imports their module
lazily, so ``import logpose`` never requires a configured backend (or even an
installed optional dependency).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from logpose.errors import LogposeError
from logpose.providers.base import (
    CompletionDone,
    CompletionRequest,
    Provider,
    ProviderEvent,
    ProviderTextDelta,
    ProviderThinkingDelta,
    ToolSpec,
)

__all__ = [
    "CompletionDone",
    "CompletionRequest",
    "Provider",
    "ProviderEvent",
    "ProviderFactory",
    "ProviderTextDelta",
    "ProviderThinkingDelta",
    "ToolSpec",
    "register",
    "resolve",
    "known_providers",
]

ProviderFactory = Callable[..., Provider]
"""Callable that builds a :class:`~logpose.providers.base.Provider` from keyword arguments."""

_REGISTRY: dict[str, ProviderFactory] = {}


def register(name: str, factory: ProviderFactory) -> None:
    """Register a provider factory under ``name``.

    Registering an existing name replaces the previous factory.

    Args:
        name: Identifier callers pass to :func:`resolve`.
        factory: Callable invoked with the keyword arguments given to
            :func:`resolve`; must return a
            :class:`~logpose.providers.base.Provider`.
    """
    _REGISTRY[name] = factory


def known_providers() -> list[str]:
    """List the registered provider names.

    Returns:
        Registered names in sorted order.
    """
    return sorted(_REGISTRY)


def resolve(name: str, **kwargs: Any) -> Provider:
    """Build the provider registered under ``name``.

    Args:
        name: A registered provider name, e.g. ``"anthropic"``.
        **kwargs: Passed straight through to the provider's factory.

    Returns:
        A ready-to-use provider instance.

    Raises:
        LogposeError: If ``name`` is not registered. The message lists the
            names that are.
    """
    try:
        factory = _REGISTRY[name]
    except KeyError:
        known = ", ".join(known_providers()) or "<none>"
        raise LogposeError(f"Unknown provider {name!r}. Known providers: {known}.") from None
    return factory(**kwargs)


def _anthropic_factory(**kwargs: Any) -> Provider:
    """Build the Anthropic provider, importing its module lazily.

    Args:
        **kwargs: Forwarded to ``AnthropicProvider``.

    Returns:
        A configured Anthropic provider.
    """
    from logpose.providers.anthropic import AnthropicProvider

    return AnthropicProvider(**kwargs)


def _docker_factory(**kwargs: Any) -> Provider:
    """Build the Docker Model Runner provider, importing its module lazily.

    Args:
        **kwargs: Forwarded to ``DockerModelsProvider``.

    Returns:
        A configured local-model provider.
    """
    from logpose.providers.openai_compat import DockerModelsProvider

    return DockerModelsProvider(**kwargs)


def _openai_compat_factory(**kwargs: Any) -> Provider:
    """Build a generic OpenAI-compatible provider, importing its module lazily.

    Args:
        **kwargs: Forwarded to ``OpenAICompatProvider``.

    Returns:
        A configured OpenAI-compatible provider.
    """
    from logpose.providers.openai_compat import OpenAICompatProvider

    return OpenAICompatProvider(**kwargs)


register("anthropic", _anthropic_factory)
register("docker", _docker_factory)
register("docker-models", _docker_factory)
register("openai-compat", _openai_compat_factory)
