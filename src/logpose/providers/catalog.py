"""Describing the available backends without instantiating any of them.

A consumer building a model picker, a settings screen, or a `--provider` flag needs
to answer three questions, and they have very different costs:

1. *What backends exist, and what does each one need?* — :class:`ProviderInfo`, via
   :func:`logpose.provider_catalog`. Pure data, no I/O, no credentials, safe to call
   in a render loop.
2. *Which ones can I actually use right now?* — :class:`ProviderStatus`, via
   :func:`logpose.provider_status`. Reads credential stores, so it is async and
   opt-in.
3. *What models does this backend serve?* — ``provider.list_models()``, which every
   built-in backend implements. Authoritative, so it needs the network.

Keeping the first one free of I/O is the reason this module holds only dataclasses
and the metadata declarations live in :mod:`logpose.providers` beside
``register()``. Reading them off the provider classes instead would mean importing
every provider module — and with the Anthropic pair, the ``anthropic`` SDK — just to
ask what the options are. ``tests/test_provider_catalog.py`` pins each declaration
against the class it describes so the two cannot drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

__all__ = [
    "CredentialKind",
    "ProviderInfo",
    "ProviderStatus",
    "WireApi",
]

CredentialKind = Literal["api_key", "subscription", "none", "optional"]
"""What a provider needs to authenticate.

``"none"`` is a local backend that needs nothing; ``"optional"`` is a generic
backend whose server may or may not want a key.
"""

WireApi = Literal["messages", "responses", "chat-completions"]
"""Which wire protocol a provider speaks.

Worth surfacing because it is not cosmetic: ``chat-completions`` has no field able
to carry a reasoning model's chain of thought across a tool call, so multi-step tool
use on a reasoning model loses it every turn. ``responses`` and ``messages`` both
preserve it.
"""


@dataclass(frozen=True)
class ProviderInfo:
    """What a provider is, what it needs, and how to reach its model list.

    Attributes:
        name: The canonical name to pass to :func:`logpose.resolve` or ``Agent``.
        aliases: Other registered names resolving to the same backend.
        summary: One line describing the backend, suitable for a UI label.
        api: The wire protocol — see :data:`WireApi`.
        credential: What it needs to authenticate — see :data:`CredentialKind`.
        default_model: The model used when the caller names none, or ``None`` when
            the backend discovers one on its first request (``docker``) or requires
            an explicit choice (``openai-compat``).
        supports_model_discovery: Whether ``provider.list_models()`` asks the
            backend. Every built-in provider does; a third-party one may not.
        env_vars: Environment variables that configure the backend, credential
            first. Presented in the order a user would set them.
        officially_supported: ``False`` for the two subscription backends, which
            ride an undocumented vendor CLI integration and may break without
            notice or conflict with that vendor's terms of service. A UI offering
            these should say so — see the README's disclaimer.
        credential_module: Dotted path of the :mod:`logpose.auth` module that
            resolves this backend's credential, or ``None`` when it needs none.
            Imported lazily, and only by :func:`logpose.provider_status`.
        credential_kind: The ``require_kind`` value passed to that module's
            ``resolve_credential``.
        credential_verify: Name of an extra check in that module to call with the
            resolved credential, if the backend needs more than a token — Codex
            also requires the credential to name a ChatGPT account.
        base_url_env: Environment variable that must be set (or ``base_url=``
            passed) before the backend can be reached at all.
    """

    name: str
    summary: str
    api: WireApi
    credential: CredentialKind
    default_model: str | None = None
    supports_model_discovery: bool = True
    aliases: tuple[str, ...] = ()
    env_vars: tuple[str, ...] = ()
    officially_supported: bool = True
    credential_module: str | None = None
    credential_kind: Literal["api_key", "oauth"] | None = None
    credential_verify: str | None = None
    base_url_env: str | None = None

    @property
    def names(self) -> tuple[str, ...]:
        """Every registered name for this backend, canonical first."""
        return (self.name, *self.aliases)

    @property
    def preserves_reasoning(self) -> bool:
        """Whether a reasoning model's chain of thought survives a tool call.

        Returns:
            ``False`` for Chat Completions, which has no field to carry it.
        """
        return self.api != "chat-completions"


@dataclass(frozen=True)
class ProviderStatus:
    """Whether a provider can be used right now.

    Attributes:
        name: The provider's canonical name.
        ready: Whether a usable credential (and endpoint) was found.
        detail: Why not, phrased as an instruction — the same message the provider
            would have raised. Empty when ready.
        credential: Which kind was found, when one was.
    """

    name: str
    ready: bool
    detail: str = ""
    credential: Literal["api_key", "oauth", "none"] | None = None

    def __bool__(self) -> bool:
        """Report readiness, so a status can be tested directly."""
        return self.ready
