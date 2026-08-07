"""Tests pinning the public surface of ``logpose``.

Two promises are enforced here.

1. ``logpose.__all__`` is the API. It is checked exactly, so adding or removing a
   name is a deliberate edit to this file rather than an accident, and every
   listed name actually exists and is re-exported from the module that owns it.
2. ``import logpose`` is cheap and side-effect free — in particular it must not
   import the ``anthropic`` SDK, read a credential, or touch the network.
   Backends are resolved lazily by name, so the package imports fine on a
   machine with no configuration at all. That is checked in a *fresh subprocess*
   because the test session itself has already imported everything.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

import logpose

REPO_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_API = {
    "__version__",
    # the loop
    "Agent",
    "Conversation",
    "ToolGate",
    "ToolGateOutcome",
    "ToolGateResult",
    # tools
    "tool",
    "ToolDef",
    "discover_tools",
    # sync facade
    "SyncAgent",
    "run_sync",
    "stream_sync",
    "close_sync",
    # messages
    "Message",
    "Usage",
    "StopReason",
    "ContentBlock",
    "TextBlock",
    "ThinkingBlock",
    "RedactedThinkingBlock",
    "ToolUseBlock",
    "ToolResultBlock",
    "RawBlock",
    # events
    "Event",
    "RunResult",
    "RunEnd",
    "TextDelta",
    "ThinkingDelta",
    "ToolCall",
    "ToolResult",
    "TurnEnd",
    # errors
    "LogposeError",
    "AuthError",
    "ProviderError",
    "MaxIterationsError",
    "ToolSchemaError",
    "ToolExecutionError",
    # provider seam
    "Provider",
    "ProviderEvent",
    "ProviderTextDelta",
    "ProviderThinkingDelta",
    "CompletionDone",
    "CompletionRequest",
    "ToolSpec",
    "register",
    "resolve",
    "known_providers",
    # discovering what is available
    "provider_catalog",
    "provider_info",
    "provider_status",
    "ProviderInfo",
    "ProviderStatus",
    "CredentialKind",
    "WireApi",
    # resilience
    "RetryPolicy",
    "DEFAULT_RETRY_POLICY",
}

# name in logpose -> module that defines it
OWNERS = {
    "Agent": "logpose.agent",
    "Conversation": "logpose.agent",
    "ToolGate": "logpose.agent",
    "ToolGateOutcome": "logpose.agent",
    "ToolGateResult": "logpose.agent",
    "tool": "logpose.tools",
    "ToolDef": "logpose.tools",
    "discover_tools": "logpose.tools",
    "SyncAgent": "logpose.sync",
    "run_sync": "logpose.sync",
    "stream_sync": "logpose.sync",
    "close_sync": "logpose.sync",
    "Message": "logpose.messages",
    "Usage": "logpose.messages",
    "TextBlock": "logpose.messages",
    "ThinkingBlock": "logpose.messages",
    "RedactedThinkingBlock": "logpose.messages",
    "ToolUseBlock": "logpose.messages",
    "ToolResultBlock": "logpose.messages",
    "RawBlock": "logpose.messages",
    "RunResult": "logpose.events",
    "RunEnd": "logpose.events",
    "TextDelta": "logpose.events",
    "ThinkingDelta": "logpose.events",
    "ToolCall": "logpose.events",
    "ToolResult": "logpose.events",
    "TurnEnd": "logpose.events",
    "LogposeError": "logpose.errors",
    "AuthError": "logpose.errors",
    "ProviderError": "logpose.errors",
    "MaxIterationsError": "logpose.errors",
    "ToolSchemaError": "logpose.errors",
    "ToolExecutionError": "logpose.errors",
    "Provider": "logpose.providers.base",
    "CompletionDone": "logpose.providers.base",
    "CompletionRequest": "logpose.providers.base",
    "ToolSpec": "logpose.providers.base",
    "ProviderTextDelta": "logpose.providers.base",
    "ProviderThinkingDelta": "logpose.providers.base",
    "register": "logpose.providers",
    "resolve": "logpose.providers",
    "known_providers": "logpose.providers",
    "provider_catalog": "logpose.providers",
    "provider_info": "logpose.providers",
    "provider_status": "logpose.providers",
    "ProviderInfo": "logpose.providers.catalog",
    "ProviderStatus": "logpose.providers.catalog",
    "CredentialKind": "logpose.providers.catalog",
    "WireApi": "logpose.providers.catalog",
    "RetryPolicy": "logpose.retry",
    "DEFAULT_RETRY_POLICY": "logpose.retry",
}


def _run_python(code: str) -> subprocess.CompletedProcess[str]:
    """Execute ``code`` in a fresh interpreter using this environment."""
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=120,
    )


# ---------------------------------------------------------------------------
# the surface itself
# ---------------------------------------------------------------------------


def test_all_matches_the_documented_surface() -> None:
    assert set(logpose.__all__) == EXPECTED_API


def test_all_has_no_duplicates() -> None:
    assert len(logpose.__all__) == len(set(logpose.__all__))


@pytest.mark.parametrize("name", sorted(EXPECTED_API))
def test_every_exported_name_exists(name: str) -> None:
    assert hasattr(logpose, name), f"logpose.__all__ promises {name!r} but it is missing"


@pytest.mark.parametrize("name", sorted(OWNERS))
def test_exports_are_the_same_objects_as_their_source(name: str) -> None:
    module = __import__(OWNERS[name], fromlist=["_"])
    assert getattr(logpose, name) is getattr(module, name)


def test_star_import_exposes_exactly_the_public_surface() -> None:
    namespace: dict[str, object] = {}
    exec("from logpose import *", namespace)  # noqa: S102 - deliberate API check
    namespace.pop("__builtins__", None)
    assert set(namespace) == EXPECTED_API


def test_version_is_a_string_matching_pyproject() -> None:
    assert isinstance(logpose.__version__, str)
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, flags=re.MULTILINE)
    assert match is not None, "pyproject.toml has no version"
    assert logpose.__version__ == match.group(1)


def test_distribution_metadata_is_publishable() -> None:
    """Regression: the built metadata carried no classifiers and inlined the
    whole MIT licence body into ``License:``.

    On PyPI that renders the licence text in the sidebar, makes the project
    unfilterable by Python version or licence, and reads as "unknown" to SBOM
    and licence scanners that key on ``License-Expression``.
    """
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert re.search(r'^license\s*=\s*"MIT"', text, flags=re.MULTILINE), (
        "license must be a PEP 639 SPDX expression, not a {file = ...} table"
    )
    assert re.search(r"^license-files\s*=", text, flags=re.MULTILINE)
    assert re.search(r"^authors\s*=", text, flags=re.MULTILINE)

    classifiers = re.search(r"^classifiers = \[(.*?)^\]", text, flags=re.MULTILINE | re.DOTALL)
    assert classifiers is not None, "pyproject.toml declares no classifiers"
    body = classifiers.group(1)
    assert "Development Status ::" in body
    assert "Intended Audience ::" in body
    assert "Typing :: Typed" in body
    # Every supported interpreter is advertised, or the package is invisible to
    # anyone filtering PyPI by version.
    requires = re.search(r'^requires-python\s*=\s*">=3\.(\d+)"', text, flags=re.MULTILINE)
    assert requires is not None
    for minor in range(int(requires.group(1)), 14):
        assert f"Programming Language :: Python :: 3.{minor}" in body


def test_the_agentic_entry_points_are_usable_together() -> None:
    """A smoke check that the exported names actually compose."""

    @logpose.tool
    def ping() -> str:
        """Check liveness."""
        return "pong"

    assert isinstance(ping, logpose.ToolDef)
    assert ping.spec().name == "ping"
    assert isinstance(logpose.Conversation([logpose.Message.user("hi")]), logpose.Conversation)
    assert issubclass(logpose.AuthError, logpose.LogposeError)
    assert "anthropic" in logpose.known_providers()


# ---------------------------------------------------------------------------
# import must stay lazy and side-effect free
# ---------------------------------------------------------------------------


def test_importing_logpose_does_not_import_the_anthropic_sdk() -> None:
    code = (
        "import sys\n"
        "import logpose\n"
        "leaked = sorted(\n"
        "    m for m in sys.modules if m == 'anthropic' or m.startswith('anthropic.')\n"
        ")\n"
        "assert not leaked, leaked\n"
        "print('clean')\n"
    )
    result = _run_python(code)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "clean"


def test_importing_the_sync_facade_does_not_import_the_anthropic_sdk() -> None:
    code = (
        "import sys\n"
        "import logpose.sync\n"
        "import logpose.tools\n"
        "import logpose.agent\n"
        "import logpose.auth\n"
        "leaked = sorted(\n"
        "    m for m in sys.modules if m == 'anthropic' or m.startswith('anthropic.')\n"
        ")\n"
        "assert not leaked, leaked\n"
        "print('clean')\n"
    )
    result = _run_python(code)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "clean"


def test_resolving_the_anthropic_provider_is_what_imports_the_sdk() -> None:
    """The laziness is real, not an accident of import ordering."""
    code = (
        "import sys\n"
        "import logpose\n"
        "assert 'anthropic' not in sys.modules\n"
        "logpose.resolve('anthropic', api_key='not-a-real-key')\n"
        "assert 'anthropic' in sys.modules\n"
        "print('lazy')\n"
    )
    result = _run_python(code)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "lazy"


def test_importing_logpose_reads_no_credentials_and_opens_no_loop() -> None:
    code = (
        "import asyncio\n"
        "import os\n"
        "os.environ.pop('ANTHROPIC_API_KEY', None)\n"
        "os.environ.pop('CLAUDE_CODE_OAUTH_TOKEN', None)\n"
        "import logpose\n"
        "try:\n"
        "    asyncio.get_event_loop_policy().get_event_loop().close()\n"
        "except Exception:\n"
        "    pass\n"
        "print(logpose.__version__)\n"
    )
    result = _run_python(code)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == logpose.__version__


def test_public_api_import_does_not_warn() -> None:
    code = (
        "import warnings\n"
        "warnings.simplefilter('error')\n"
        "import logpose\n"
        "from logpose import Agent, SyncAgent, tool, run_sync, stream_sync\n"
        "print('quiet')\n"
    )
    result = _run_python(code)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "quiet"
