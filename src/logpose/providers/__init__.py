"""Provider registry.

Providers are resolved by name so consumers never import a backend directly::

    provider = resolve("anthropic", model="claude-opus-5")

Built-in providers are registered with a factory that imports their module
lazily, so ``import logpose`` never requires a configured backend (or even an
installed optional dependency).
"""

from __future__ import annotations

import asyncio
import importlib
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from logpose.errors import AuthError, LogposeError
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
    """Check one backend's credential, synchronously.

    Run in a worker thread by :func:`provider_status`, because resolving a Claude
    Code credential can shell out to the macOS Keychain.

    Args:
        info: The backend to check.

    Returns:
        Its readiness.
    """
    if info.base_url_env is not None and not os.environ.get(info.base_url_env):
        return ProviderStatus(
            name=info.name,
            ready=False,
            detail=(
                f"No endpoint configured. Pass base_url=... or set ${info.base_url_env}."
            ),
        )
    if info.credential_module is None:
        return ProviderStatus(name=info.name, ready=True, credential="none")

    module = importlib.import_module(info.credential_module)
    try:
        credential = module.resolve_credential(require_kind=info.credential_kind)
        if info.credential_verify is not None:
            getattr(module, info.credential_verify)(credential)
    except AuthError as exc:
        return ProviderStatus(name=info.name, ready=False, detail=str(exc))
    return ProviderStatus(name=info.name, ready=True, credential=credential.kind)


async def provider_status(names: Iterable[str] | None = None) -> tuple[ProviderStatus, ...]:
    """Report which backends can be used right now.

    Unlike :func:`provider_catalog` this **reads credential stores** — environment
    variables, ``~/.codex/auth.json``, and on macOS the Claude Code Keychain entry,
    which costs a subprocess. That is why it is async and separate: a picker wants
    both, but a catalog lookup should never quietly touch the filesystem. Nothing is
    written, and no request is sent, so a ``ready`` provider can still fail later on
    an expired token or a network error.

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
        name: A registered provider name, e.g. ``"anthropic"``.
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


def _anthropic_factory(**kwargs: Any) -> Provider:
    """Build the Anthropic provider, importing its module lazily.

    Args:
        **kwargs: Forwarded to ``AnthropicProvider``.

    Returns:
        A configured Anthropic provider.
    """
    return _optional_provider("logpose.providers.anthropic", "AnthropicProvider", **kwargs)


def _claude_code_factory(**kwargs: Any) -> Provider:
    """Build the Claude Code provider, importing its module lazily.

    Args:
        **kwargs: Forwarded to ``ClaudeCodeProvider``.

    Returns:
        A configured Claude Code provider.
    """
    return _optional_provider("logpose.providers.claude_code", "ClaudeCodeProvider", **kwargs)


def _optional_provider(module_path: str, class_name: str, **kwargs: Any) -> Provider:
    """Construct an Anthropic-backed provider or name the missing extra.

    The package intentionally keeps the vendor SDK optional so applications using
    other providers do not install it. Catch only a missing top-level SDK: an
    unrelated import error inside the provider should remain visible to aid
    diagnosis.
    """
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        if exc.name == "anthropic":
            raise LogposeError(
                "The Anthropic providers require the optional dependency. "
                "Install it with `pip install \"logpose[anthropic]\"`."
            ) from None
        raise
    provider_class = getattr(module, class_name)
    return provider_class(**kwargs)


def _codex_factory(**kwargs: Any) -> Provider:
    """Build the Codex provider, importing its module lazily.

    Args:
        **kwargs: Forwarded to ``CodexProvider``.

    Returns:
        A configured Codex provider.
    """
    from logpose.providers.codex import CodexProvider

    return CodexProvider(**kwargs)


def _openai_factory(**kwargs: Any) -> Provider:
    """Build the OpenAI Responses provider, importing its module lazily.

    Args:
        **kwargs: Forwarded to ``OpenAIProvider``.

    Returns:
        A configured OpenAI provider.
    """
    from logpose.providers.openai import OpenAIProvider

    return OpenAIProvider(**kwargs)


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


# Metadata lives here, not on the provider classes, so provider_catalog() never
# has to import a vendor SDK to answer "what are my options". Each declaration is
# pinned against the class it describes by tests/test_provider_catalog.py.
_ANTHROPIC_INFO = ProviderInfo(
    name="anthropic",
    summary="Anthropic Claude models with an API key.",
    api="messages",
    credential="api_key",
    default_model="claude-opus-5",
    env_vars=("ANTHROPIC_API_KEY",),
    credential_module="logpose.auth.claude_code",
    credential_kind="api_key",
)

_CLAUDE_CODE_INFO = ProviderInfo(
    name="claude-code",
    summary="Anthropic Claude models on a Claude Code subscription.",
    api="messages",
    credential="subscription",
    default_model="claude-opus-5",
    env_vars=("CLAUDE_CODE_OAUTH_TOKEN", "CLAUDE_CONFIG_DIR"),
    officially_supported=False,
    credential_module="logpose.auth.claude_code",
    credential_kind="oauth",
)

_OPENAI_INFO = ProviderInfo(
    name="openai",
    summary="OpenAI models on the Responses API with an API key.",
    api="responses",
    credential="api_key",
    default_model="gpt-5.1",
    env_vars=("OPENAI_API_KEY", "OPENAI_RESPONSES_MODEL", "OPENAI_RESPONSES_BASE_URL"),
    credential_module="logpose.auth.codex",
    credential_kind="api_key",
)

_CODEX_INFO = ProviderInfo(
    name="codex",
    summary="OpenAI models on a ChatGPT subscription, via the Codex backend.",
    api="responses",
    credential="subscription",
    default_model="gpt-5.5",
    env_vars=("CODEX_HOME", "CODEX_MODEL", "CODEX_BASE_URL", "CHATGPT_ACCOUNT_ID"),
    officially_supported=False,
    credential_module="logpose.auth.codex",
    credential_kind="oauth",
    # A token alone is not enough here: the backend rejects a request that does not
    # name a ChatGPT account, so readiness has to include that check.
    credential_verify="require_account_id",
)

_DOCKER_INFO = ProviderInfo(
    name="docker",
    summary="Local models served by Docker Model Runner.",
    api="chat-completions",
    credential="none",
    # Left unset on purpose: with no model= the backend asks the runner for its
    # first model on the first request, so there is no default to report.
    default_model=None,
    aliases=("docker-models",),
    env_vars=("DOCKER_MODEL_RUNNER_URL", "DOCKER_MODEL"),
)

_OPENAI_COMPAT_INFO = ProviderInfo(
    name="openai-compat",
    summary="Any server speaking the Chat Completions API — llama.cpp, vLLM, Ollama, Kimi.",
    api="chat-completions",
    credential="optional",
    default_model=None,
    env_vars=("OPENAI_BASE_URL", "OPENAI_MODEL", "OPENAI_API_KEY"),
    base_url_env="OPENAI_BASE_URL",
)

register("anthropic", _anthropic_factory, info=_ANTHROPIC_INFO)
register("claude-code", _claude_code_factory, info=_CLAUDE_CODE_INFO)
register("codex", _codex_factory, info=_CODEX_INFO)
register("docker", _docker_factory, info=_DOCKER_INFO)
register("openai", _openai_factory, info=_OPENAI_INFO)
register("docker-models", _docker_factory, info=_DOCKER_INFO)
register("openai-compat", _openai_compat_factory, info=_OPENAI_COMPAT_INFO)
