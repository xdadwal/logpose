"""Credential redaction shared by every provider.

Lives in its own module so a provider can redact without importing a sibling
provider's SDK: ``from logpose.providers.anthropic import redact`` would drag the
``anthropic`` package into any backend that borrowed it, breaking the promise
that ``import logpose`` pulls in no vendor SDK.
"""

from __future__ import annotations

__all__ = ["redact", "scrub_exception_in_place"]


def redact(secret: str) -> str:
    """Render a credential as a non-recoverable summary.

    Args:
        secret: The credential value. Never logged verbatim.

    Returns:
        A short prefix (only for values long enough that a prefix is not the
        whole secret) plus the length, e.g. ``"<redacted sk-ant… len=64>"``.
    """
    prefix = f"{secret[:6]}… " if len(secret) > 12 else ""
    return f"<redacted {prefix}len={len(secret)}>"


def scrub_exception_in_place(exc: BaseException, secrets: tuple[str, ...]) -> None:
    """Redact credentials inside an exception before it is chained.

    ``raise wrapper from exc`` keeps ``exc`` as ``__cause__``, and every
    traceback renders ``str(exc)`` — so scrubbing only the wrapper still writes
    the raw token to logs, error trackers, and test output.

    Args:
        exc: The exception that will become ``__cause__``.
        secrets: Credential values to replace with their redactions.
    """
    if not secrets:
        return

    def _scrub(text: str) -> str:
        for secret in secrets:
            text = text.replace(secret, redact(secret))
        return text

    exc.args = tuple(_scrub(arg) if isinstance(arg, str) else arg for arg in exc.args)
    message = getattr(exc, "message", None)
    if isinstance(message, str):
        try:
            exc.message = _scrub(message)  # type: ignore[attr-defined]
        except AttributeError:
            pass
