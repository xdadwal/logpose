"""Provider registry.

Providers are resolved by name so consumers never import a backend directly::

    provider = resolve("openai-compat", base_url="http://localhost:8000/v1")

Built-in providers are registered with a factory that imports their module
lazily, so ``import logpose`` never requires a configured backend.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
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
from logpose.providers.catalog import (
    CredentialKind,
    ProviderInfo,
    ProviderStatus,
    WireApi,
)

__all__ = [
    "CompletionDone",
    "CompletionRequest",
    "CredentialKind",
    "Provider",
    "ProviderEvent",
    "ProviderFactory",
    "ProviderInfo",
    "ProviderStatus",
    "ProviderTextDelta",
    "ProviderThinkingDelta",
    "ToolSpec",
    "WireApi",
    "provider_catalog",
    "provider_info",
    "provider_status",
    "register",
    "resolve",
    "known_providers",
]

ProviderFactory = Callable[..., Provider]
"""Callable that builds a :class:`~logpose.providers.base.Provider` from keyword arguments."""


@dataclass(frozen=True)
class _Registration:
    """A registered factory and the optional metadata describing it."""

    factory: ProviderFactory
    info: ProviderInfo | None = None


_REGISTRY: dict[str, _Registration] = {}


def register(
    name: str,
    factory: ProviderFactory,
    *,
    info: ProviderInfo | None = None,
) -> None:
    """Register a provider factory under ``name``.

    Registering an existing name replaces the previous factory.

    Args:
        name: Identifier callers pass to :func:`resolve`.
        factory: Callable invoked with the keyword arguments given to
            :func:`resolve`; must return a
            :class:`~logpose.providers.base.Provider`.
        info: Optional metadata describing the backend, surfaced by
            :func:`provider_catalog`. Declared here rather than read off the
            provider class so asking what the options are never imports a vendor
            SDK. Omitting it keeps the provider fully usable; it simply will not
            appear in the catalog.
    """
    _REGISTRY[name] = _Registration(factory=factory, info=info)


def known_providers() -> list[str]:
    """List the registered provider names.

    Returns:
        Registered names in sorted order, aliases included. See
        :func:`provider_catalog` for one entry per backend instead.
    """
    return sorted(_REGISTRY)


def provider_catalog() -> tuple[ProviderInfo, ...]:
    """Describe every registered backend that carries metadata.

    Pure data: no credential is read, no module is imported, nothing touches the
    network. Safe to call as often as a UI needs. One entry per *backend* — an
    alias such as ``docker-models`` appears in its owner's
    :attr:`~logpose.ProviderInfo.aliases` rather than as a second entry.

    Providers registered without ``info=`` are omitted, so a third-party backend is
    still resolvable but does not have to describe itself.

    Returns:
        One :class:`~logpose.ProviderInfo` per backend, sorted by name.
    """
    seen: dict[str, ProviderInfo] = {}
    for entry in _REGISTRY.values():
        if entry.info is not None:
            seen[entry.info.name] = entry.info
    return tuple(seen[name] for name in sorted(seen))


def provider_info(name: str) -> ProviderInfo:
    """Describe one backend, by canonical name or alias.

    Args:
        name: A registered provider name.

    Returns:
        The backend's :class:`~logpose.ProviderInfo`.

    Raises:
        LogposeError: If ``name`` is not registered, or is registered without
            metadata. The message distinguishes the two.
    """
    try:
        entry = _REGISTRY[name]
    except KeyError:
        known = ", ".join(known_providers()) or "<none>"
        raise LogposeError(f"Unknown provider {name!r}. Known providers: {known}.") from None
    if entry.info is None:
        raise LogposeError(
            f"Provider {name!r} was registered without metadata, so there is nothing to "
            "describe. Pass info=ProviderInfo(...) to register() to have it appear in the "
            "catalog."
        )
    return entry.info


def _probe(info: ProviderInfo) -> ProviderStatus:
    """Check one backend's endpoint and credential, synchronously.

    Args:
        info: The backend to check.

    Returns:
        Its readiness.
    """
    if info.base_url_env is not None and not os.environ.get(info.base_url_env):
        return ProviderStatus(
            name=info.name,
            ready=False,
            detail=(f"No endpoint configured. Pass base_url=... or set ${info.base_url_env}."),
        )
    if info.credential == "api_key":
        ready = bool(info.env_vars and os.environ.get(info.env_vars[0], "").strip())
        return ProviderStatus(
            name=info.name,
            ready=ready,
            credential="api_key" if ready else "none",
            detail=""
            if ready
            else (
                f"Set ${info.env_vars[0]}."
                if info.env_vars
                else "No credential source configured."
            ),
        )
    return ProviderStatus(name=info.name, ready=True, credential="none")


async def provider_status(names: Iterable[str] | None = None) -> tuple[ProviderStatus, ...]:
    """Report which backends can be used right now.

    Unlike :func:`provider_catalog` this reads the environment to check for each
    backend's endpoint and API key. Nothing is written and no request is sent, so a
    ``ready`` provider can still fail later on a rejected key or a network error.
    Async so a slower readiness check can be added without changing callers.

    Args:
        names: Backends to check, by canonical name or alias. Defaults to every
            backend in the catalog.

    Returns:
        One :class:`~logpose.ProviderStatus` per backend, in catalog order.

    Raises:
        LogposeError: If a requested name is unknown or carries no metadata.
    """
    infos = (
        provider_catalog()
        if names is None
        else tuple(dict.fromkeys(provider_info(name) for name in names))
    )
    return tuple(await asyncio.gather(*(asyncio.to_thread(_probe, info) for info in infos)))


def resolve(name: str, **kwargs: Any) -> Provider:
    """Build the provider registered under ``name``.

    Args:
        name: A registered provider name, e.g. ``"openai-compat"``.
        **kwargs: Passed straight through to the provider's factory.

    Returns:
        A ready-to-use provider instance.

    Raises:
        LogposeError: If ``name`` is not registered. The message lists the
            names that are.
    """
    try:
        entry = _REGISTRY[name]
    except KeyError:
        known = ", ".join(known_providers()) or "<none>"
        raise LogposeError(f"Unknown provider {name!r}. Known providers: {known}.") from None
    return entry.factory(**kwargs)


def _openai_compat_factory(**kwargs: Any) -> Provider:
    """Build a generic OpenAI-compatible provider, importing its module lazily.

    Args:
        **kwargs: Forwarded to ``OpenAICompatProvider``.

    Returns:
        A configured OpenAI-compatible provider.
    """
    from logpose.providers.openai_compat import OpenAICompatProvider

    return OpenAICompatProvider(**kwargs)


# Metadata lives here, not on the provider classes, so provider_catalog() never
# has to import a provider module to answer "what are my options". Each declaration is
# pinned against the class it describes by tests/test_provider_catalog.py.
_OPENAI_COMPAT_INFO = ProviderInfo(
    name="openai-compat",
    summary="Any server speaking the Chat Completions API — llama.cpp, vLLM, Ollama, Kimi.",
    api="chat-completions",
    credential="optional",
    default_model=None,
    env_vars=("OPENAI_BASE_URL", "OPENAI_MODEL", "OPENAI_API_KEY"),
    base_url_env="OPENAI_BASE_URL",
)

register("openai-compat", _openai_compat_factory, info=_OPENAI_COMPAT_INFO)
