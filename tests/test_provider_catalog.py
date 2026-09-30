"""Tests for provider discovery: the catalog, readiness probes, and model listing.

The catalog's whole value is being cheap and truthful, so the two things most
worth pinning are that it needs no I/O and that its declarations match the provider
classes they describe. Nothing here touches a network; credential probes run against
an isolated environment.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

from logpose import (
    ProviderInfo,
    ProviderStatus,
    known_providers,
    provider_catalog,
    provider_info,
    provider_status,
)
from logpose.auth import claude_code as claude_auth
from logpose.auth import codex as codex_auth
from logpose.errors import AuthError, LogposeError
from logpose.providers import register, resolve
from tests.jwt_helpers import make_jwt
from tests.responses_helpers import API_KEY, Recorder, completed, message_item, mock_client, sse

BUILT_INS = {"anthropic", "claude-code", "codex", "docker", "openai", "openai-compat"}


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Hide every ambient credential so probes report what the test set up.

    Returns:
        An empty Codex home, which tests may populate.
    """
    home = tmp_path / "codex-home"
    home.mkdir()
    config = tmp_path / "claude-config"
    config.mkdir()
    for name in (
        "ANTHROPIC_API_KEY",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "CHATGPT_ACCOUNT_ID",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(codex_auth.ENV_CODEX_HOME, str(home))
    monkeypatch.setenv(claude_auth.ENV_CONFIG_DIR, str(config))
    monkeypatch.setattr(claude_auth, "_is_macos", lambda: False)
    return home


def write_codex_auth(home: Path, **tokens: Any) -> None:
    """Write a Codex ``auth.json`` holding a subscription token."""
    payload = {"tokens": {"access_token": make_jwt({"exp": 4_000_000_000}), **tokens}}
    (home / "auth.json").write_text(json.dumps(payload))


# ---------------------------------------------------------------------------
# the catalog is pure
# ---------------------------------------------------------------------------


def test_the_catalog_covers_every_built_in_backend() -> None:
    assert {info.name for info in provider_catalog()} == BUILT_INS


@pytest.mark.parametrize("name", ["anthropic", "claude-code"])
def test_anthropic_providers_name_their_optional_extra_when_sdk_is_missing(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    import logpose.providers as providers

    original = providers.importlib.import_module

    def unavailable(module_path: str, package: str | None = None) -> Any:
        if module_path in {"logpose.providers.anthropic", "logpose.providers.claude_code"}:
            raise ModuleNotFoundError("No module named 'anthropic'", name="anthropic")
        return original(module_path, package)

    monkeypatch.setattr(providers.importlib, "import_module", unavailable)

    with pytest.raises(LogposeError, match=r"logpose\[anthropic\]"):
        resolve(name)


def test_the_catalog_is_sorted_and_has_one_entry_per_backend() -> None:
    names = [info.name for info in provider_catalog()]
    assert names == sorted(names)
    assert len(names) == len(set(names))


def test_an_alias_does_not_get_its_own_entry() -> None:
    """`docker-models` is a spelling of `docker`, not a seventh backend."""
    assert "docker-models" in known_providers()
    assert "docker-models" not in {info.name for info in provider_catalog()}
    docker = provider_info("docker")
    assert docker.aliases == ("docker-models",)
    assert docker.names == ("docker", "docker-models")


def test_provider_info_resolves_an_alias_to_its_owner() -> None:
    assert provider_info("docker-models") is provider_info("docker")


def test_the_catalog_needs_no_credential_and_no_network() -> None:
    """It must be safe to call in a render loop, with nothing configured."""

    def explode(*args: object, **kwargs: object) -> None:
        raise AssertionError("the catalog must not resolve a credential")

    for module in (claude_auth, codex_auth):
        original = module.resolve_credential
        module.resolve_credential = explode  # type: ignore[assignment]
        try:
            assert provider_catalog()
        finally:
            module.resolve_credential = original  # type: ignore[assignment]


def test_reading_the_catalog_imports_no_vendor_sdk() -> None:
    """The reason metadata is declared at register() rather than on the classes."""
    code = (
        "import sys, logpose; "
        "logpose.provider_catalog(); "
        "logpose.provider_info('anthropic'); "
        "assert 'anthropic' not in sys.modules, 'anthropic SDK leaked'; "
        "assert 'httpx' not in sys.modules, 'httpx leaked'; "
        "print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False, timeout=60
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_an_unknown_name_raises_and_lists_the_known_ones() -> None:
    with pytest.raises(LogposeError) as excinfo:
        provider_info("nope")
    assert "anthropic" in str(excinfo.value)


def test_a_provider_registered_without_metadata_is_omitted_but_usable() -> None:
    from logpose.providers import _REGISTRY

    register("bare-test-provider", lambda **kwargs: object())  # type: ignore[arg-type,return-value]
    try:
        assert "bare-test-provider" in known_providers()
        assert "bare-test-provider" not in {i.name for i in provider_catalog()}
        with pytest.raises(LogposeError, match="registered without metadata"):
            provider_info("bare-test-provider")
    finally:
        del _REGISTRY["bare-test-provider"]


# ---------------------------------------------------------------------------
# the declarations match the classes they describe
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "module_path", "class_name"),
    [
        ("anthropic", "logpose.providers.anthropic", "AnthropicProvider"),
        ("claude-code", "logpose.providers.claude_code", "ClaudeCodeProvider"),
        ("codex", "logpose.providers.codex", "CodexProvider"),
        ("openai", "logpose.providers.openai", "OpenAIProvider"),
    ],
)
def test_the_declared_default_model_matches_the_provider_class(
    name: str, module_path: str, class_name: str
) -> None:
    """The one real cost of declaring metadata away from the class: it can drift.

    This is the guard. Importing the provider module here is fine — a test may pull
    in a vendor SDK, the catalog may not.
    """
    import importlib

    provider_class = getattr(importlib.import_module(module_path), class_name)
    declared = provider_info(name).default_model
    if hasattr(provider_class, "DEFAULT_MODEL"):
        assert declared == provider_class.DEFAULT_MODEL
    else:  # the Anthropic pair takes its default from the shared base
        from logpose.providers._anthropic_base import DEFAULT_MODEL

        assert declared == DEFAULT_MODEL


@pytest.mark.parametrize("name", sorted(BUILT_INS))
def test_every_declared_name_is_actually_registered(name: str) -> None:
    info = provider_info(name)
    for registered in info.names:
        assert registered in known_providers()


def test_the_declared_credential_kind_matches_the_provider_class() -> None:
    from logpose.providers.anthropic import AnthropicProvider
    from logpose.providers.claude_code import ClaudeCodeProvider
    from logpose.providers.codex import CodexProvider
    from logpose.providers.openai import OpenAIProvider

    expected = {
        "anthropic": (AnthropicProvider, "api_key"),
        "claude-code": (ClaudeCodeProvider, "oauth"),
        "codex": (CodexProvider, "oauth"),
        "openai": (OpenAIProvider, "api_key"),
    }
    for name, (provider_class, kind) in expected.items():
        assert provider_info(name).credential_kind == kind
        assert provider_class.REQUIRED_KIND == kind


def test_the_subscription_backends_are_flagged_as_unsupported() -> None:
    """A UI offering these should be able to warn without hardcoding names."""
    unsupported = {i.name for i in provider_catalog() if not i.officially_supported}
    assert unsupported == {"claude-code", "codex"}


def test_chat_completions_is_flagged_as_losing_reasoning() -> None:
    lossy = {i.name for i in provider_catalog() if not i.preserves_reasoning}
    assert lossy == {"docker", "openai-compat"}


def test_the_credential_env_var_comes_first() -> None:
    """env_vars is ordered so a UI can show the one that matters most."""
    assert provider_info("anthropic").env_vars[0] == "ANTHROPIC_API_KEY"
    assert provider_info("openai").env_vars[0] == "OPENAI_API_KEY"
    assert provider_info("claude-code").env_vars[0] == "CLAUDE_CODE_OAUTH_TOKEN"


def test_backends_that_discover_a_model_declare_no_default() -> None:
    for name in ("docker", "openai-compat"):
        assert provider_info(name).default_model is None


# ---------------------------------------------------------------------------
# provider_status
# ---------------------------------------------------------------------------


async def test_status_covers_the_catalog_in_order() -> None:
    statuses = await provider_status()
    assert [s.name for s in statuses] == [i.name for i in provider_catalog()]


async def test_a_backend_needing_no_credential_is_always_ready() -> None:
    (docker,) = await provider_status(["docker"])
    assert docker.ready is True
    assert docker.credential == "none"
    assert docker.detail == ""


async def test_a_missing_credential_reports_the_providers_own_message() -> None:
    (anthropic,) = await provider_status(["anthropic"])
    assert anthropic.ready is False
    assert "ANTHROPIC_API_KEY" in anthropic.detail


async def test_a_present_credential_is_ready_and_names_its_kind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-present")
    (anthropic,) = await provider_status(["anthropic"])
    assert anthropic.ready is True
    assert anthropic.credential == "api_key"


async def test_a_wrong_kind_credential_does_not_make_a_backend_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An API key must not make the subscription backend look usable."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-present")
    (claude_code,) = await provider_status(["claude-code"])
    assert claude_code.ready is False
    assert "claude setup-token" in claude_code.detail


async def test_codex_needs_an_account_id_not_just_a_token(isolated_env: Path) -> None:
    """Readiness has to include the extra check, or it would promise too much."""
    write_codex_auth(isolated_env)
    (codex,) = await provider_status(["codex"])
    assert codex.ready is False
    assert "codex login" in codex.detail

    write_codex_auth(isolated_env, account_id="acc_1")
    (codex,) = await provider_status(["codex"])
    assert codex.ready is True
    assert codex.credential == "oauth"


async def test_a_backend_needing_an_endpoint_says_so() -> None:
    (compat,) = await provider_status(["openai-compat"])
    assert compat.ready is False
    assert "OPENAI_BASE_URL" in compat.detail


async def test_a_configured_endpoint_makes_the_generic_backend_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "http://local.test/v1")
    (compat,) = await provider_status(["openai-compat"])
    assert compat.ready is True


async def test_status_accepts_an_alias() -> None:
    (docker,) = await provider_status(["docker-models"])
    assert docker.name == "docker"


async def test_status_deduplicates_a_backend_named_twice() -> None:
    statuses = await provider_status(["docker", "docker-models"])
    assert len(statuses) == 1


async def test_status_raises_on_an_unknown_name() -> None:
    with pytest.raises(LogposeError):
        await provider_status(["nope"])


async def test_a_status_is_truthy_when_ready() -> None:
    assert ProviderStatus(name="x", ready=True)
    assert not ProviderStatus(name="x", ready=False, detail="nope")


async def test_status_never_raises_for_a_missing_credential() -> None:
    """A picker asks about everything at once; one absent credential is not an error."""
    statuses = await provider_status()
    assert any(not s.ready for s in statuses)
    assert all(isinstance(s, ProviderStatus) for s in statuses)


# ---------------------------------------------------------------------------
# list_models
# ---------------------------------------------------------------------------


def models_body(*, data: list[str] | None = None, slugs: list[str] | None = None) -> bytes:
    """Render a GET /models body in either of the two shapes the endpoints use."""
    if data is not None:
        return json.dumps({"object": "list", "data": [{"id": m} for m in data]}).encode()
    return json.dumps({"models": [{"slug": m} for m in (slugs or [])]}).encode()


class ModelsRecorder(Recorder):
    """Recorder that answers /models and streams a turn for anything else."""

    def __init__(self, body: bytes, *, status: int = 200) -> None:
        super().__init__()
        self._models_body = body
        self._models_status = status

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            self.requests.append(request)
            return httpx.Response(self._models_status, content=self._models_body)
        return super().__call__(request)


async def test_the_openai_shape_is_parsed() -> None:
    from logpose.providers.openai import OpenAIProvider

    handler = ModelsRecorder(models_body(data=["gpt-5.1", "gpt-5.1-mini"]))
    provider = OpenAIProvider(
        client=mock_client(handler), base_url="http://x.test/v1", api_key=API_KEY
    )
    assert await provider.list_models() == ["gpt-5.1", "gpt-5.1-mini"]


async def test_the_codex_shape_is_parsed_and_declares_a_client_version() -> None:
    from logpose.providers.codex import CODEX_CLIENT_VERSION, CodexProvider

    handler = ModelsRecorder(models_body(slugs=["gpt-5.5", "gpt-5.4"]))
    provider = CodexProvider(
        client=mock_client(handler),
        base_url="http://x.test/v1",
        auth_token="tok",
        account_id="acc_1",
    )
    assert await provider.list_models() == ["gpt-5.5", "gpt-5.4"]
    assert handler.requests[-1].url.params["client_version"] == CODEX_CLIENT_VERSION


async def test_the_client_version_can_be_overridden() -> None:
    """The backend gates its answer on this, so it is a real knob, not a formality."""
    from logpose.providers.codex import CodexProvider

    handler = ModelsRecorder(models_body(slugs=["gpt-5.5"]))
    provider = CodexProvider(
        client=mock_client(handler),
        base_url="http://x.test/v1",
        auth_token="tok",
        account_id="acc_1",
        client_version="9.9.9",
    )
    await provider.list_models()
    assert handler.requests[-1].url.params["client_version"] == "9.9.9"


@pytest.mark.parametrize("client_version", [None, "0.142.5"])
async def test_codex_discovery_defaults_to_the_current_version_gated_catalog(
    client_version: str | None,
) -> None:
    """An outdated declaration hides current models even with the same credential."""
    from logpose.providers.codex import CodexProvider

    def catalog(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/v1/models"
        models = ["gpt-5.5"]
        if request.url.params.get("client_version") == "0.159.0":
            models = ["gpt-6.1-sol", "gpt-6-astra", "gpt-5.5"]
        return httpx.Response(200, content=models_body(slugs=models))

    kwargs = {} if client_version is None else {"client_version": client_version}
    async with mock_client(catalog) as client:
        provider = CodexProvider(
            client=client,
            base_url="http://x.test/v1",
            auth_token="tok",
            account_id="acc_1",
            **kwargs,
        )
        expected = (
            ["gpt-6.1-sol", "gpt-6-astra", "gpt-5.5"]
            if client_version is None else ["gpt-5.5"]
        )
        assert await provider.list_models() == expected


@pytest.mark.parametrize(
    "payload",
    [b"{}", b"[]", b"null", b'{"data": "nope"}', b'{"models": []}', b'{"data": [{"no_id": 1}]}'],
)
async def test_an_unusable_models_payload_yields_an_empty_list(payload: bytes) -> None:
    """One unfamiliar row must not break discovery for a picker."""
    from logpose.providers.openai import OpenAIProvider

    provider = OpenAIProvider(
        client=mock_client(ModelsRecorder(payload)),
        base_url="http://x.test/v1",
        api_key=API_KEY,
    )
    assert await provider.list_models() == []


async def test_entries_without_an_identifier_are_skipped() -> None:
    from logpose.providers.openai import OpenAIProvider

    body = json.dumps({"data": [{"id": "keep"}, {"no_id": 1}, {"id": 42}]}).encode()
    provider = OpenAIProvider(
        client=mock_client(ModelsRecorder(body)), base_url="http://x.test/v1", api_key=API_KEY
    )
    assert await provider.list_models() == ["keep"]


async def test_a_models_failure_becomes_a_provider_error() -> None:
    from logpose.errors import ProviderError
    from logpose.providers.openai import OpenAIProvider

    provider = OpenAIProvider(
        client=mock_client(ModelsRecorder(b"nope", status=500)),
        base_url="http://x.test/v1",
        api_key=API_KEY,
    )
    with pytest.raises(ProviderError) as excinfo:
        await provider.list_models()
    assert excinfo.value.retryable is True


async def test_list_models_raises_auth_error_with_no_credential() -> None:
    from logpose.providers.openai import OpenAIProvider

    provider = OpenAIProvider(
        client=mock_client(ModelsRecorder(models_body(data=["m"]))), base_url="http://x.test/v1"
    )
    with pytest.raises(AuthError):
        await provider.list_models()


async def test_the_anthropic_pair_lists_models_through_the_sdk() -> None:
    from logpose.providers.claude_code import ClaudeCodeProvider

    class FakeModels:
        def __init__(self) -> None:
            self.limit: int | None = None

        async def list(self, *, limit: int) -> Any:
            self.limit = limit
            entry = type("M", (), {"id": "claude-opus-5"})()
            return type("Page", (), {"data": [entry, type("M", (), {"id": None})()]})()

    class FakeClient:
        def __init__(self) -> None:
            self.models = FakeModels()
            self.api_key = None
            self.auth_token = "tok"

    client = FakeClient()
    provider = ClaudeCodeProvider(client=client)  # type: ignore[arg-type]
    assert await provider.list_models() == ["claude-opus-5"]
    assert client.models.limit == 1000


async def test_every_built_in_provider_exposes_list_models() -> None:
    """The consistency the catalog promises: one call works everywhere."""
    import importlib

    for module_path, class_name in (
        ("logpose.providers.anthropic", "AnthropicProvider"),
        ("logpose.providers.claude_code", "ClaudeCodeProvider"),
        ("logpose.providers.codex", "CodexProvider"),
        ("logpose.providers.openai", "OpenAIProvider"),
        ("logpose.providers.openai_compat", "OpenAICompatProvider"),
        ("logpose.providers.openai_compat", "DockerModelsProvider"),
    ):
        provider_class = getattr(importlib.import_module(module_path), class_name)
        assert callable(getattr(provider_class, "list_models", None)), class_name
        assert provider_info(
            {"AnthropicProvider": "anthropic", "ClaudeCodeProvider": "claude-code",
             "CodexProvider": "codex", "OpenAIProvider": "openai",
             "OpenAICompatProvider": "openai-compat", "DockerModelsProvider": "docker"}[class_name]
        ).supports_model_discovery


def test_provider_info_is_immutable() -> None:
    info = provider_info("codex")
    with pytest.raises(Exception):  # noqa: B017 - dataclasses raise FrozenInstanceError
        info.default_model = "nope"  # type: ignore[misc]


def test_provider_info_is_constructible_by_a_third_party() -> None:
    """Only name, summary, api and credential are required."""
    info = ProviderInfo(
        name="mine", summary="My backend.", api="responses", credential="api_key"
    )
    assert info.default_model is None
    assert info.aliases == ()
    assert info.officially_supported is True


async def test_an_end_to_end_picker_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    """What a consumer actually does: catalog, filter by readiness, list models."""
    monkeypatch.setenv("OPENAI_API_KEY", API_KEY)
    ready = {s.name for s in await provider_status() if s.ready}
    assert "openai" in ready
    assert "anthropic" not in ready

    choices = [i for i in provider_catalog() if i.name in ready and i.officially_supported]
    assert [i.name for i in choices] == ["docker", "openai"]

    from logpose.providers.openai import OpenAIProvider

    handler = ModelsRecorder(models_body(data=["gpt-5.1", "gpt-5.1-mini"]))
    provider = OpenAIProvider(client=mock_client(handler), base_url="http://x.test/v1")
    assert await provider.list_models() == ["gpt-5.1", "gpt-5.1-mini"]
    await drain_noop(provider)


async def drain_noop(provider: Any) -> None:
    """Close a provider built around an injected client."""
    await provider.aclose()


async def test_a_turn_still_works_after_listing_models() -> None:
    """list_models shares the client and credential with stream()."""
    from logpose.providers.openai import OpenAIProvider

    handler = ModelsRecorder(models_body(data=["m"]))
    handler.bodies.append(sse(completed(output=[message_item("hi")])))
    provider = OpenAIProvider(
        client=mock_client(handler), base_url="http://x.test/v1", api_key=API_KEY
    )
    assert await provider.list_models() == ["m"]
    from tests.responses_helpers import drain, final_message

    assert final_message(await drain(provider)).text == "hi"
