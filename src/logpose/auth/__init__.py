"""Credential resolution for logpose.

v0.1 resolves Anthropic credentials only, preferring a Claude Code
**subscription** OAuth token and falling back to a plain API key::

    from logpose.auth import CredentialProvider

    provider = CredentialProvider.resolve()
    credential = await provider.get()

See :mod:`logpose.auth.claude_code` for the precedence rules, the read-only
credential-store discovery, and an important note on why the subscription path
is a terms-of-service gray area.
"""

from __future__ import annotations

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
]
