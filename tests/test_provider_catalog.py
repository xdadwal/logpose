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
from logpose.errors import LogposeError, ProviderError
from logpose.providers import register
from logpose.providers.openai_compat import DockerModelsProvider, OpenAICompatProvider

BUILT_INS = {"docker", "openai-compat"}


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hide every ambient credential so probes report what the test set up."""
    for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "KEYED_TEST_KEY"):
        monkeypatch.delenv(name, raising=False)


# ---------------------------------------------------------------------------
# the catalog is pure
# ---------------------------------------------------------------------------


def test_the_catalog_covers_every_built_in_backend() -> None:
    assert {info.name for info in provider_catalog()} == BUILT_INS


def test_the_catalog_is_sorted_and_has_one_entry_per_backend() -> None:
    names = [info.name for info in provider_catalog()]
    assert names == sorted(names)
    assert len(names) == len(set(names))


def test_an_alias_does_not_get_its_own_entry() -> None:
    """`docker-models` is a spelling of `docker`, not a separate backend."""
    assert "docker-models" in known_providers()
    assert "docker-models" not in {info.name for info in provider_catalog()}
    docker = provider_info("docker")
    assert docker.aliases == ("docker-models",)
    assert docker.names == ("docker", "docker-models")


def test_provider_info_resolves_an_alias_to_its_owner() -> None:
    assert provider_info("docker-models") is provider_info("docker")


def test_reading_the_catalog_imports_no_provider_module() -> None:
    """The reason metadata is declared at register() rather than on the classes."""
    code = (
        "import sys, logpose; "
        "logpose.provider_catalog(); "
        "logpose.provider_info('openai-compat'); "
        "assert 'logpose.providers.openai_compat' not in sys.modules, 'provider leaked'; "
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
    assert "openai-compat" in str(excinfo.value)


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


@pytest.mark.parametrize("name", sorted(BUILT_INS))
def test_every_declared_name_is_actually_registered(name: str) -> None:
    info = provider_info(name)
    for registered in info.names:
        assert registered in known_providers()


def test_chat_completions_is_flagged_as_losing_reasoning() -> None:
    lossy = {i.name for i in provider_catalog() if not i.preserves_reasoning}
    assert lossy == {"docker", "openai-compat"}


def test_the_credential_env_var_comes_first() -> None:
    """env_vars is ordered so a UI can show the one that matters most."""
    assert provider_info("openai-compat").env_vars[0] == "OPENAI_BASE_URL"


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


@pytest.fixture
def keyed_provider() -> Any:
    """Register a third-party backend that requires an API key."""
    from logpose.providers import _REGISTRY

    register(
        "keyed-test",
        lambda **kwargs: object(),  # type: ignore[arg-type,return-value]
        info=ProviderInfo(
            name="keyed-test",
            summary="Test backend.",
            api="chat-completions",
            credential="api_key",
            env_vars=("KEYED_TEST_KEY",),
        ),
    )
    yield
    del _REGISTRY["keyed-test"]


async def test_a_missing_credential_names_the_variable_to_set(keyed_provider: Any) -> None:
    (keyed,) = await provider_status(["keyed-test"])
    assert keyed.ready is False
    assert "KEYED_TEST_KEY" in keyed.detail


async def test_a_present_credential_is_ready_and_names_its_kind(
    monkeypatch: pytest.MonkeyPatch, keyed_provider: Any
) -> None:
    monkeypatch.setenv("KEYED_TEST_KEY", "present")
    (keyed,) = await provider_status(["keyed-test"])
    assert keyed.ready is True
    assert keyed.credential == "api_key"


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

BASE = "http://x.test/v1"


def models_handler(body: bytes, *, status: int = 200) -> httpx.MockTransport:
    """Answer GET /models with ``body`` and fail anything else."""

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/models"), request.url
        return httpx.Response(status, content=body)

    return httpx.MockTransport(handle)


def compat(body: bytes, *, status: int = 200, **kwargs: Any) -> OpenAICompatProvider:
    client = httpx.AsyncClient(transport=models_handler(body, status=status))
    return OpenAICompatProvider(client=client, base_url=BASE, **kwargs)


def data(*ids: str) -> bytes:
    return json.dumps({"object": "list", "data": [{"id": m} for m in ids]}).encode()


async def test_the_models_list_is_parsed() -> None:
    assert await compat(data("m-1", "m-2")).list_models() == ["m-1", "m-2"]


@pytest.mark.parametrize(
    "payload",
    [b"{}", b"[]", b"null", b'{"data": "nope"}', b'{"data": [{"no_id": 1}]}'],
)
async def test_an_unusable_models_payload_yields_an_empty_list(payload: bytes) -> None:
    """One unfamiliar row must not break discovery for a picker."""
    assert await compat(payload).list_models() == []


async def test_entries_without_an_identifier_are_skipped() -> None:
    body = json.dumps({"data": [{"id": "keep"}, {"no_id": 1}, {"id": 42}]}).encode()
    assert await compat(body).list_models() == ["keep"]


async def test_a_models_failure_becomes_a_provider_error() -> None:
    with pytest.raises(ProviderError) as excinfo:
        await compat(b"nope", status=500).list_models()
    assert excinfo.value.retryable is True


async def test_every_built_in_provider_exposes_list_models() -> None:
    """The consistency the catalog promises: one call works everywhere."""
    for name, provider_class in (
        ("openai-compat", OpenAICompatProvider),
        ("docker", DockerModelsProvider),
    ):
        assert callable(getattr(provider_class, "list_models", None)), name
        assert provider_info(name).supports_model_discovery


def test_provider_info_is_immutable() -> None:
    info = provider_info("openai-compat")
    with pytest.raises(Exception):  # noqa: B017 - dataclasses raise FrozenInstanceError
        info.default_model = "nope"  # type: ignore[misc]


def test_provider_info_is_constructible_by_a_third_party() -> None:
    """Only name, summary, api and credential are required."""
    info = ProviderInfo(
        name="mine", summary="My backend.", api="chat-completions", credential="api_key"
    )
    assert info.default_model is None
    assert info.aliases == ()
    assert info.preserves_reasoning is False


async def test_an_end_to_end_picker_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    """What a consumer actually does: catalog, filter by readiness, list models."""
    monkeypatch.setenv("OPENAI_BASE_URL", BASE)
    ready = {s.name for s in await provider_status() if s.ready}
    assert ready == {"docker", "openai-compat"}

    provider = compat(data("m-1"))
    try:
        assert await provider.list_models() == ["m-1"]
    finally:
        await provider.aclose()
