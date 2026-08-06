"""Credential resolution for logpose.

Two backends are supported, each preferring its own vendor's **subscription**
OAuth token and falling back to a plain API key::

    from logpose.auth import CredentialProvider, CodexCredentialProvider

    anthropic = CredentialProvider.resolve()          # Claude Code / Anthropic
    codex = CodexCredentialProvider.resolve()         # Codex / ChatGPT
    credential = await anthropic.get()

The unprefixed names are Anthropic's, for backwards compatibility; Codex's are
prefixed. Both sit on the backend-neutral core in :mod:`logpose.auth._common`.

See :mod:`logpose.auth.claude_code` and :mod:`logpose.auth.codex` for the
precedence rules, the read-only credential-store discovery, and an important note
on why the subscription paths are a terms-of-service gray area. The vendor
``ENV_*`` and ``OAUTH_*`` constants are deliberately not re-exported here — their
unprefixed names collide — so import the submodule when you need them.
"""

from __future__ import annotations

from logpose.auth import codex
from logpose.auth.claude_code import (
    ENV_API_KEY,
    ENV_CONFIG_DIR,
    ENV_OAUTH_TOKEN,
    EXPIRY_SKEW_SECONDS,
    KEYCHAIN_SERVICE,
    OAUTH_CLIENT_ID,
    OAUTH_TOKEN_URL,
    Credential,
    CredentialProvider,
    credentials_file_path,
    load_stored_credential,
    refresh,
    resolve_credential,
)
from logpose.auth.codex import (
    CredentialProvider as CodexCredentialProvider,
)
from logpose.auth.codex import (
    auth_file_path as codex_auth_file_path,
)
from logpose.auth.codex import (
    load_stored_credential as load_codex_credential,
)
from logpose.auth.codex import (
    refresh as refresh_codex,
)
from logpose.auth.codex import (
    require_account_id,
)
from logpose.auth.codex import (
    resolve_credential as resolve_codex_credential,
)

__all__ = [
    "Credential",
    "CredentialProvider",
    "resolve_credential",
    "refresh",
    "load_stored_credential",
    "credentials_file_path",
    "EXPIRY_SKEW_SECONDS",
    "OAUTH_CLIENT_ID",
    "OAUTH_TOKEN_URL",
    "KEYCHAIN_SERVICE",
    "ENV_OAUTH_TOKEN",
    "ENV_API_KEY",
    "ENV_CONFIG_DIR",
    # Codex / ChatGPT
    "codex",
    "CodexCredentialProvider",
    "resolve_codex_credential",
    "load_codex_credential",
    "refresh_codex",
    "codex_auth_file_path",
    "require_account_id",
]
