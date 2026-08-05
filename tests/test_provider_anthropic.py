"""Tests for the Anthropic provider. No network: the SDK client is always faked.

The one exception is the auth tests, which build a *real* ``AsyncAnthropic`` and
inspect the headers it would put on the wire — that behaviour is
environment-dependent and silent, so it has to be asserted against the installed
SDK rather than assumed.
"""

from __future__ import annotations

import asyncio
import time
import traceback
from collections.abc import AsyncIterator, Sequence
from typing import Any

import anthropic
import httpx
import pytest
from anthropic import types as sdk
from anthropic._models import FinalRequestOptions

from logpose.auth import claude_code
from logpose.errors import AuthError, LogposeError, ProviderError
from logpose.messages import (
    Message,
    RawBlock,
    RedactedThinkingBlock,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from logpose.providers import resolve
from logpose.providers.anthropic import (
    CLAUDE_CODE_IDENTITY,
    OAUTH_BETA_HEADER,
    AnthropicProvider,
    redact,
)
from logpose.providers.base import (
    CompletionDone,
    CompletionRequest,
    Provider,
    ProviderTextDelta,
    ProviderThinkingDelta,
    ToolSpec,
)

SIGNATURE = "ErUBCkYIBRgCIkDdQ7/vNdkKgFQ0oX+signature+bytes+must+survive=="
REDACTED_DATA = "EroBCkYIBRgCKkB0aGlzLWlzLWVuY3J5cHRlZA=="

# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------


class FakeStream:
    """Stands in for ``AsyncMessageStreamManager`` / ``AsyncMessageStream``."""

    def __init__(
        self,
        events: Sequence[Any],
        final: sdk.Message | None,
        *,
        error: BaseException | None = None,
    ) -> None:
        self._events = list(events)
        self._final = final
        self._error = error

    async def __aenter__(self) -> FakeStream:
        if self._error is not None:
            raise self._error
        return self

    async def __aexit__(self, *exc_info: object) -> bool:
        return False

    def __aiter__(self) -> AsyncIterator[Any]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[Any]:
        for event in self._events:
            yield event

    async def get_final_message(self) -> sdk.Message:
        assert self._final is not None
        return self._final


class FakeMessages:
    """Records the kwargs the provider builds and replays a scripted stream."""

    def __init__(
        self,
        events: Sequence[Any],
        final: sdk.Message | None,
        *,
        error: BaseException | None = None,
    ) -> None:
        self._events = events
        self._final = final
        self._error = error
        self.calls: list[dict[str, Any]] = []

    def stream(self, **kwargs: Any) -> FakeStream:
        self.calls.append(kwargs)
        return FakeStream(self._events, self._final, error=self._error)


class FakeClient:
    """An ``AsyncAnthropic``-shaped object exposing ``messages.stream`` and auth.

    ``api_key`` / ``auth_token`` mirror the plain attributes the real client
    carries; the scrubbing tests read them the same way the provider does.
    """

    def __init__(
        self,
        events: Sequence[Any] = (),
        final: sdk.Message | None = None,
        *,
        error: BaseException | None = None,
        api_key: str | None = None,
        auth_token: str | None = None,
    ) -> None:
        self.messages = FakeMessages(events, final, error=error)
        self.api_key = api_key
        self.auth_token = auth_token


def sdk_message(
    content: Sequence[Any],
    *,
    stop_reason: str | None = "end_turn",
    usage: sdk.Usage | None = None,
) -> sdk.Message:
    """Build a realistic final ``Message`` as the SDK would hand it back."""
    return sdk.Message.model_construct(
        id="msg_01FAKE",
        content=list(content),
        model="claude-opus-5",
        role="assistant",
        stop_reason=stop_reason,
        stop_sequence=None,
        type="message",
        usage=usage or sdk.Usage(input_tokens=0, output_tokens=0),
    )


def text_delta(text: str) -> sdk.RawContentBlockDeltaEvent:
    return sdk.RawContentBlockDeltaEvent(
        type="content_block_delta", index=0, delta=sdk.TextDelta(type="text_delta", text=text)
    )


def thinking_delta(text: str) -> sdk.RawContentBlockDeltaEvent:
    return sdk.RawContentBlockDeltaEvent(
        type="content_block_delta",
        index=0,
        delta=sdk.ThinkingDelta(type="thinking_delta", thinking=text),
    )


def make_provider(client: FakeClient, **kwargs: Any) -> AnthropicProvider:
    return AnthropicProvider(client=client, **kwargs)  # type: ignore[arg-type]


async def drain(provider: AnthropicProvider, req: CompletionRequest) -> list[Any]:
    return [event async for event in provider.stream(req)]


def simple_request(**kwargs: Any) -> CompletionRequest:
    params: dict[str, Any] = {
        "messages": [Message.user("hi")],
        "model": "claude-opus-5",
        "max_tokens": 1024,
    }
    params.update(kwargs)
    return CompletionRequest(**params)


@pytest.fixture(autouse=True)
def _isolate_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep ambient credentials out of every test by default."""
    for name in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "ANTHROPIC_PROFILE",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture()
def no_stored_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hide the developer's real Claude Code credential store from the resolver."""
    monkeypatch.setattr(claude_code, "load_stored_credential", lambda: None)


# ---------------------------------------------------------------------------
# protocol conformance
# ---------------------------------------------------------------------------


def test_satisfies_provider_protocol() -> None:
    provider = make_provider(FakeClient())
    assert isinstance(provider, Provider)
    assert provider.name == "anthropic"


def test_registry_resolves_to_this_class() -> None:
    provider = resolve("anthropic", api_key="sk-ant-api-test", model_default="claude-opus-5")
    assert isinstance(provider, AnthropicProvider)


def test_repr_never_carries_a_credential() -> None:
    provider = AnthropicProvider(api_key="sk-ant-api-super-secret-value")
    assert "super-secret" not in repr(provider)


# ---------------------------------------------------------------------------
# request translation
# ---------------------------------------------------------------------------


async def test_request_translation_snapshot() -> None:
    client = FakeClient(final=sdk_message([sdk.TextBlock(type="text", text="ok")]))
    provider = make_provider(client)
    tool = ToolSpec(
        name="get_weather",
        description="Get current weather.",
        input_schema={
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
            "additionalProperties": False,
        },
    )
    req = CompletionRequest(
        messages=[
            Message.user("weather in Pune?"),
            Message(
                role="assistant",
                content=[
                    ToolUseBlock(id="toolu_1", name="get_weather", input={"location": "Pune"})
                ],
            ),
            Message(
                role="user",
                content=[ToolResultBlock(tool_use_id="toolu_1", content="31C", is_error=False)],
            ),
        ],
        model="claude-opus-5",
        max_tokens=4096,
        system="You are terse.",
        tools=[tool],
    )

    await drain(provider, req)

    assert client.messages.calls == [
        {
            "model": "claude-opus-5",
            "max_tokens": 4096,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "weather in Pune?"}]},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "get_weather",
                            "input": {"location": "Pune"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": "31C",
                            "is_error": False,
                        }
                    ],
                },
            ],
            "system": "You are terse.",
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get current weather.",
                    "input_schema": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                        "additionalProperties": False,
                    },
                }
            ],
            "thinking": {"type": "adaptive", "display": "summarized"},
        }
    ]


async def test_no_sampling_parameters_are_ever_sent() -> None:
    client = FakeClient(final=sdk_message([sdk.TextBlock(type="text", text="ok")]))
    provider = make_provider(client)

    await drain(provider, simple_request())

    sent = client.messages.calls[0]
    assert "temperature" not in sent
    assert "top_p" not in sent
    assert "top_k" not in sent


async def test_thinking_defaults_to_summarized_adaptive() -> None:
    client = FakeClient(final=sdk_message([]))
    await drain(make_provider(client), simple_request())
    assert client.messages.calls[0]["thinking"] == {"type": "adaptive", "display": "summarized"}


@pytest.mark.parametrize(
    ("thinking", "expected"),
    [
        (None, None),
        ("disabled", {"type": "disabled"}),
        ("adaptive", {"type": "adaptive", "display": "summarized"}),
        ({"type": "adaptive", "display": "omitted"}, {"type": "adaptive", "display": "omitted"}),
    ],
)
async def test_thinking_modes(thinking: Any, expected: dict[str, Any] | None) -> None:
    client = FakeClient(final=sdk_message([]))
    await drain(make_provider(client, thinking=thinking), simple_request())
    assert client.messages.calls[0].get("thinking") == expected


def test_unknown_thinking_mode_is_rejected() -> None:
    with pytest.raises(LogposeError, match="Unknown thinking mode"):
        AnthropicProvider(thinking="enabled", api_key="sk-ant-api-test")


async def test_system_omitted_when_absent() -> None:
    client = FakeClient(final=sdk_message([]))
    await drain(make_provider(client), simple_request())
    assert "system" not in client.messages.calls[0]


async def test_tools_omitted_when_empty() -> None:
    client = FakeClient(final=sdk_message([]))
    await drain(make_provider(client), simple_request())
    assert "tools" not in client.messages.calls[0]


async def test_compat_claude_code_prepends_identity() -> None:
    client = FakeClient(final=sdk_message([]))
    provider = make_provider(client, compat_claude_code=True)

    await drain(provider, simple_request(system="You are terse."))

    assert client.messages.calls[0]["system"] == [
        {"type": "text", "text": CLAUDE_CODE_IDENTITY},
        {"type": "text", "text": "You are terse."},
    ]


async def test_compat_claude_code_off_by_default() -> None:
    client = FakeClient(final=sdk_message([]))
    await drain(make_provider(client), simple_request(system="You are terse."))
    assert client.messages.calls[0]["system"] == "You are terse."


async def test_extra_is_merged_into_the_wire_request() -> None:
    client = FakeClient(final=sdk_message([]))
    provider = make_provider(client)

    await drain(provider, simple_request(extra={"output_config": {"effort": "low"}}))

    assert client.messages.calls[0]["output_config"] == {"effort": "low"}


async def test_defaults_fill_in_for_a_request_without_model_or_max_tokens() -> None:
    client = FakeClient(final=sdk_message([]))
    provider = make_provider(client, model_default="claude-sonnet-5", max_tokens=8000)

    await drain(provider, CompletionRequest(messages=[Message.user("hi")], model="", max_tokens=0))

    assert client.messages.calls[0]["model"] == "claude-sonnet-5"
    assert client.messages.calls[0]["max_tokens"] == 8000


def test_non_positive_max_tokens_is_rejected() -> None:
    with pytest.raises(LogposeError, match="max_tokens"):
        AnthropicProvider(max_tokens=0, api_key="sk-ant-api-test")


# ---------------------------------------------------------------------------
# thinking-block round-trip fidelity
# ---------------------------------------------------------------------------


async def test_thinking_blocks_are_sent_back_verbatim() -> None:
    client = FakeClient(final=sdk_message([]))
    provider = make_provider(client)
    history = Message(
        role="assistant",
        content=[
            ThinkingBlock(thinking="step one", signature=SIGNATURE),
            RedactedThinkingBlock(data=REDACTED_DATA),
            TextBlock(text="done"),
        ],
    )

    await drain(provider, simple_request(messages=[Message.user("hi"), history]))

    assert client.messages.calls[0]["messages"][1]["content"] == [
        {"type": "thinking", "thinking": "step one", "signature": SIGNATURE},
        {"type": "redacted_thinking", "data": REDACTED_DATA},
        {"type": "text", "text": "done"},
    ]


async def test_thinking_signature_survives_the_response_round_trip() -> None:
    final = sdk_message(
        [
            sdk.ThinkingBlock(type="thinking", thinking="reasoned", signature=SIGNATURE),
            sdk.RedactedThinkingBlock(type="redacted_thinking", data=REDACTED_DATA),
            sdk.TextBlock(type="text", text="answer"),
        ]
    )
    provider = make_provider(FakeClient(final=final))

    events = await drain(provider, simple_request())
    done = events[-1]
    assert isinstance(done, CompletionDone)

    thinking = done.message.content[0]
    assert isinstance(thinking, ThinkingBlock)
    assert thinking.signature == SIGNATURE
    redacted = done.message.content[1]
    assert isinstance(redacted, RedactedThinkingBlock)
    assert redacted.data == REDACTED_DATA

    # Losslessly serializable, so the loop can persist and replay the turn.
    assert Message.model_validate(done.message.model_dump()) == done.message


async def test_thinking_blocks_are_not_stripped_from_the_assembled_message() -> None:
    final = sdk_message(
        [
            sdk.ThinkingBlock(type="thinking", thinking="reasoned", signature=SIGNATURE),
            sdk.TextBlock(type="text", text="answer"),
        ]
    )
    provider = make_provider(FakeClient(final=final))

    done = (await drain(provider, simple_request()))[-1]

    assert [block.type for block in done.message.content] == ["thinking", "text"]
    assert done.message.text == "answer"


async def test_unmodelled_blocks_survive_the_round_trip_verbatim() -> None:
    """Regression: unmodelled blocks were dropped, which breaks the one case
    that actually produces ``pause_turn``.

    The documented resume is to re-send the paused assistant turn unchanged; the
    API recognises the trailing ``server_tool_use`` block and continues. Strip
    it and the turn is read as a prefill instead, and rejected.
    """
    payload = {
        "type": "server_tool_use",
        "id": "srvtoolu_1",
        "name": "web_search",
        "input": {"query": "logpose"},
    }
    server_tool_use = sdk.ToolUseBlock.model_construct(**payload)

    final = sdk_message(
        [server_tool_use, sdk.TextBlock(type="text", text="answer")],
        stop_reason="pause_turn",
    )
    provider = make_provider(FakeClient(final=final))

    done = (await drain(provider, simple_request()))[-1]

    assert [block.type for block in done.message.content] == ["raw", "text"]
    raw = done.message.content[0]
    assert isinstance(raw, RawBlock)
    assert raw.block_type == "server_tool_use"
    assert raw.data == payload

    # Losslessly serializable, and re-emitted on the wire byte for byte.
    assert Message.model_validate(done.message.model_dump()) == done.message
    client = FakeClient(final=sdk_message([]))
    await drain(
        make_provider(client),
        simple_request(messages=[Message.user("hi"), done.message]),
    )
    assert client.messages.calls[0]["messages"][1]["content"][0] == payload


async def test_blocks_with_nothing_resendable_are_dropped() -> None:
    class Opaque:
        type = "server_tool_use"  # no model_dump, not a dict: nothing to echo

    final = sdk_message([Opaque(), sdk.TextBlock(type="text", text="answer")])
    provider = make_provider(FakeClient(final=final))

    done = (await drain(provider, simple_request()))[-1]

    assert [block.type for block in done.message.content] == ["text"]


async def test_a_thinking_block_with_no_signature_is_dropped() -> None:
    """Regression: a thinking block truncated by ``max_tokens`` never receives
    its ``signature_delta``, so the SDK leaves ``signature`` at its ``""`` seed.

    Echoing that back presents an unsigned block as a complete one and the API
    rejects the follow-up turn for an invalid signature. The block is
    unusable — a signature cannot be reconstructed — so it must not reach the
    conversation at all.
    """
    final = sdk_message(
        [
            sdk.ThinkingBlock(type="thinking", thinking="Let me consid", signature=""),
            sdk.TextBlock(type="text", text="partial ans"),
        ],
        stop_reason="max_tokens",
    )
    provider = make_provider(FakeClient(final=final))

    done = (await drain(provider, simple_request()))[-1]

    assert [block.type for block in done.message.content] == ["text"]
    assert done.stop_reason == "max_tokens"

    # Nothing carrying an empty signature can therefore reach the wire.
    client = FakeClient(final=sdk_message([]))
    await drain(
        make_provider(client),
        simple_request(messages=[Message.user("hi"), done.message, Message.user("continue")]),
    )
    for message in client.messages.calls[0]["messages"]:
        for block in message["content"]:
            assert block.get("signature") != ""


def test_sending_an_unsupported_block_type_raises() -> None:
    provider = make_provider(FakeClient(final=sdk_message([])))

    class Weird:
        type = "weird"

    with pytest.raises(LogposeError, match="Cannot send content block"):
        provider._build_params(
            simple_request(messages=[Message.model_construct(role="user", content=[Weird()])])
        )


# ---------------------------------------------------------------------------
# tool_use parsing
# ---------------------------------------------------------------------------


async def test_tool_use_input_is_used_as_parsed(no_stored_credential: None) -> None:
    parsed = {"location": "Pune", "unit": "c", "nested": {"n": 1}}
    final = sdk_message(
        [
            sdk.TextBlock(type="text", text="checking"),
            sdk.ToolUseBlock(type="tool_use", id="toolu_9", name="get_weather", input=parsed),
        ],
        stop_reason="tool_use",
    )
    provider = make_provider(FakeClient(final=final))

    done = (await drain(provider, simple_request()))[-1]

    call = done.message.content[1]
    assert isinstance(call, ToolUseBlock)
    assert call.id == "toolu_9"
    assert call.name == "get_weather"
    assert call.input == parsed
    assert done.stop_reason == "tool_use"


async def test_tool_use_with_a_non_dict_input_degrades_to_empty() -> None:
    block = sdk.ToolUseBlock.model_construct(type="tool_use", id="toolu_1", name="t", input="oops")
    provider = make_provider(FakeClient(final=sdk_message([block], stop_reason="tool_use")))

    done = (await drain(provider, simple_request()))[-1]

    assert done.message.content[0].input == {}


# ---------------------------------------------------------------------------
# streaming events
# ---------------------------------------------------------------------------


async def test_delta_events_are_emitted_in_wire_order_then_completion_done() -> None:
    events = [
        thinking_delta("let me "),
        thinking_delta("think"),
        text_delta("Hello"),
        text_delta(", world"),
    ]
    final = sdk_message(
        [
            sdk.ThinkingBlock(type="thinking", thinking="let me think", signature=SIGNATURE),
            sdk.TextBlock(type="text", text="Hello, world"),
        ]
    )
    provider = make_provider(FakeClient(events, final))

    emitted = await drain(provider, simple_request())

    assert emitted[:4] == [
        ProviderThinkingDelta(text="let me "),
        ProviderThinkingDelta(text="think"),
        ProviderTextDelta(text="Hello"),
        ProviderTextDelta(text=", world"),
    ]
    assert isinstance(emitted[4], CompletionDone)
    assert len(emitted) == 5


async def test_non_delta_events_are_ignored() -> None:
    class Noise:
        type = "content_block_start"

    class OtherDelta:
        type = "content_block_delta"

        class delta:  # noqa: N801 - stands in for an SDK model attribute
            type = "input_json_delta"
            partial_json = '{"a":'

    events = [Noise(), OtherDelta(), text_delta("hi")]
    final = sdk_message([sdk.TextBlock(type="text", text="hi")])
    provider = make_provider(FakeClient(events, final))

    emitted = await drain(provider, simple_request())

    assert emitted[0] == ProviderTextDelta(text="hi")
    assert isinstance(emitted[1], CompletionDone)
    assert len(emitted) == 2


async def test_stream_ends_with_exactly_one_completion_done() -> None:
    provider = make_provider(FakeClient([text_delta("a")], sdk_message([])))
    emitted = await drain(provider, simple_request())
    assert sum(isinstance(event, CompletionDone) for event in emitted) == 1


# ---------------------------------------------------------------------------
# usage + stop_reason mapping
# ---------------------------------------------------------------------------


async def test_usage_mapping_includes_cache_fields() -> None:
    usage = sdk.Usage(
        input_tokens=11,
        output_tokens=22,
        cache_read_input_tokens=33,
        cache_creation_input_tokens=44,
    )
    provider = make_provider(FakeClient(final=sdk_message([], usage=usage)))

    done = (await drain(provider, simple_request()))[-1]

    assert done.usage.input_tokens == 11
    assert done.usage.output_tokens == 22
    assert done.usage.cache_read_input_tokens == 33
    assert done.usage.cache_creation_input_tokens == 44


async def test_usage_none_cache_fields_become_zero() -> None:
    usage = sdk.Usage(input_tokens=5, output_tokens=6)
    provider = make_provider(FakeClient(final=sdk_message([], usage=usage)))

    done = (await drain(provider, simple_request()))[-1]

    assert done.usage.cache_read_input_tokens == 0
    assert done.usage.cache_creation_input_tokens == 0


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("end_turn", "end_turn"),
        ("tool_use", "tool_use"),
        ("max_tokens", "max_tokens"),
        ("stop_sequence", "stop_sequence"),
        ("refusal", "refusal"),
        ("pause_turn", "pause_turn"),
        ("model_context_window_exceeded", "max_tokens"),
        (None, "end_turn"),
        ("something_new", "end_turn"),
    ],
)
async def test_stop_reason_mapping(raw: str | None, expected: str) -> None:
    provider = make_provider(FakeClient(final=sdk_message([], stop_reason=raw)))
    done = (await drain(provider, simple_request()))[-1]
    assert done.stop_reason == expected


# ---------------------------------------------------------------------------
# auth wiring — exactly one auth header
# ---------------------------------------------------------------------------


def built_headers(client: anthropic.AsyncAnthropic) -> dict[str, str]:
    """Headers the SDK would actually put on a /v1/messages request."""
    request = client._build_request(
        FinalRequestOptions(method="post", url="/v1/messages", json_data={})
    )
    return {key.lower(): value for key, value in request.headers.items()}


async def test_oauth_mode_sends_only_bearer_even_with_ANTHROPIC_API_KEY_set(
    monkeypatch: pytest.MonkeyPatch,
    no_stored_credential: None,
) -> None:
    # The SDK reads ANTHROPIC_API_KEY when api_key is not supplied; if that
    # leaked through, both auth headers would go out and the API would 401.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api-env-key-must-not-leak")
    provider = AnthropicProvider(auth_token="sk-ant-oat01-subscription-token")

    client = await provider.get_client()
    headers = built_headers(client)

    assert headers["authorization"] == "Bearer sk-ant-oat01-subscription-token"
    assert "x-api-key" not in headers
    assert headers["anthropic-beta"] == OAUTH_BETA_HEADER
    assert client.api_key is None
    assert set(client.auth_headers) == {"Authorization"}
    await provider.aclose()


async def test_api_key_mode_sends_only_x_api_key(
    monkeypatch: pytest.MonkeyPatch,
    no_stored_credential: None,
) -> None:
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "sk-ant-oat01-env-token-must-not-leak")
    provider = AnthropicProvider(api_key="sk-ant-api-byok")

    client = await provider.get_client()
    headers = built_headers(client)

    assert headers["x-api-key"] == "sk-ant-api-byok"
    assert "authorization" not in headers
    assert headers.get("anthropic-beta") != OAUTH_BETA_HEADER
    assert client.auth_token is None
    assert set(client.auth_headers) == {"X-Api-Key"}
    await provider.aclose()


async def test_client_is_cached_across_turns(no_stored_credential: None) -> None:
    provider = AnthropicProvider(api_key="sk-ant-api-byok")
    first = await provider.get_client()
    second = await provider.get_client()
    assert first is second
    await provider.aclose()


async def test_injected_client_is_used_verbatim_and_never_resolves_credentials() -> None:
    fake = FakeClient(final=sdk_message([]))
    provider = make_provider(fake)
    assert await provider.get_client() is fake
    await provider.aclose()  # must not touch a client it does not own
    assert await provider.get_client() is fake


# ---------------------------------------------------------------------------
# credential resolution
# ---------------------------------------------------------------------------


async def test_explicit_auth_token_wins_over_api_key(no_stored_credential: None) -> None:
    provider = AnthropicProvider(api_key="sk-ant-api-key", auth_token="sk-ant-oat01-token")
    client = await provider.get_client()
    assert client.auth_token == "sk-ant-oat01-token"
    assert client.api_key is None
    await provider.aclose()


async def test_env_api_key_is_used_when_nothing_explicit(
    monkeypatch: pytest.MonkeyPatch, no_stored_credential: None
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api-from-env")
    provider = AnthropicProvider()
    client = await provider.get_client()
    assert client.api_key == "sk-ant-api-from-env"
    await provider.aclose()


async def test_subscription_token_outranks_an_ambient_api_key(
    monkeypatch: pytest.MonkeyPatch, no_stored_credential: None
) -> None:
    """Subscription-first: an ANTHROPIC_API_KEY left over from other tooling must
    not silently divert a subscription user onto per-token billing."""
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-from-env")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api-ambient")
    provider = AnthropicProvider()

    client = await provider.get_client()

    assert client.auth_token == "sk-ant-oat01-from-env"
    assert client.api_key is None
    await provider.aclose()


async def test_auth_error_when_no_credential_anywhere(no_stored_credential: None) -> None:
    provider = AnthropicProvider()
    with pytest.raises(AuthError, match="claude setup-token"):
        await provider.get_client()


async def test_constructing_the_provider_never_raises_auth_error(
    no_stored_credential: None,
) -> None:
    AnthropicProvider()  # lazy: resolution is deferred to the first request


async def test_credential_provider_refresh_reaches_the_wire(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An expiring subscription token is refreshed and swapped in, in place."""
    stale = claude_code.Credential(
        kind="oauth",
        value="sk-ant-oat01-stale",
        expires_at=time.time() - 1,
        refresh_token="rt_1",
    )
    fresh = claude_code.Credential(
        kind="oauth",
        value="sk-ant-oat01-refreshed",
        expires_at=time.time() + 3600,
    )
    refreshes = 0

    async def refresher(credential: claude_code.Credential) -> claude_code.Credential:
        nonlocal refreshes
        refreshes += 1
        return fresh

    credentials = claude_code.CredentialProvider(stale, refresher=refresher)
    monkeypatch.setattr(
        claude_code.CredentialProvider,
        "resolve",
        classmethod(lambda cls, *args, **kwargs: credentials),
    )
    provider = AnthropicProvider()

    first = await provider.get_client()
    second = await provider.get_client()

    assert refreshes == 1
    assert first is second, "a refreshed token must not cost a new connection pool"
    assert second.auth_token == "sk-ant-oat01-refreshed"
    assert second.api_key is None
    await provider.aclose()


async def test_credential_resolution_is_deferred_to_the_first_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Constructing the provider must not touch the keychain or the network."""
    calls = 0

    def resolve(cls: object, *args: object, **kwargs: object) -> claude_code.CredentialProvider:
        nonlocal calls
        calls += 1
        return claude_code.CredentialProvider(
            claude_code.Credential(kind="api_key", value="sk-ant-api-lazy")
        )

    monkeypatch.setattr(claude_code.CredentialProvider, "resolve", classmethod(resolve))

    provider = AnthropicProvider()
    assert calls == 0

    await provider.get_client()
    await provider.get_client()
    assert calls == 1, "the CredentialProvider is built once and reused"
    await provider.aclose()


async def test_concurrent_first_requests_resolve_and_refresh_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: the lazy build of the CredentialProvider was unguarded.

    N concurrent first requests each saw ``self._credentials is None``, each
    built its own provider — and therefore its own lock — so the single-flight
    re-check inside ``CredentialProvider.get`` never saw the others. On an
    already-expired subscription token that meant N keychain subprocesses and N
    refresh grants replaying the same single-use refresh token, so all but one
    came back ``invalid_grant``.
    """
    resolves = 0
    refreshes = 0

    async def refresher(credential: claude_code.Credential) -> claude_code.Credential:
        nonlocal refreshes
        refreshes += 1
        await asyncio.sleep(0.01)
        return claude_code.Credential(
            kind="oauth",
            value="sk-ant-oat01-refreshed",
            expires_at=time.time() + 3600,
            refresh_token="rt_2",
        )

    def resolve(cls: object, *args: object, **kwargs: object) -> claude_code.CredentialProvider:
        nonlocal resolves
        resolves += 1
        time.sleep(0.02)  # the keychain subprocess, which is why resolve() is threaded
        return claude_code.CredentialProvider(
            claude_code.Credential(
                kind="oauth",
                value="sk-ant-oat01-expired",
                expires_at=time.time() - 1,
                refresh_token="rt_1",
            ),
            refresher=refresher,
        )

    monkeypatch.setattr(claude_code.CredentialProvider, "resolve", classmethod(resolve))
    provider = AnthropicProvider()

    clients = await asyncio.gather(*(provider.get_client() for _ in range(10)))

    assert resolves == 1
    assert refreshes == 1
    assert len({id(client) for client in clients}) == 1
    assert clients[0].auth_token == "sk-ant-oat01-refreshed"
    await provider.aclose()


# ---------------------------------------------------------------------------
# error mapping
# ---------------------------------------------------------------------------


def status_error(status_code: int, message: str = "boom") -> anthropic.APIStatusError:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(status_code, request=request)
    error_class = anthropic._exceptions.APIStatusError
    for candidate in (
        anthropic.BadRequestError,
        anthropic.AuthenticationError,
        anthropic.PermissionDeniedError,
        anthropic.NotFoundError,
        anthropic.RateLimitError,
        anthropic.InternalServerError,
    ):
        if getattr(candidate, "status_code", None) == status_code:
            error_class = candidate
            break
    return error_class(message, response=response, body=None)


@pytest.mark.parametrize(
    ("status_code", "retryable"),
    [
        (400, False),
        (401, False),
        (403, False),
        (404, False),
        (408, True),
        (409, True),
        (429, True),
        (500, True),
        (529, True),
    ],
)
async def test_status_errors_map_to_provider_error(status_code: int, retryable: bool) -> None:
    provider = make_provider(FakeClient(error=status_error(status_code)))

    with pytest.raises(ProviderError) as excinfo:
        await drain(provider, simple_request())

    assert excinfo.value.status_code == status_code
    assert excinfo.value.retryable is retryable
    assert isinstance(excinfo.value.__cause__, anthropic.APIStatusError)


async def test_connection_error_is_retryable() -> None:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    provider = make_provider(FakeClient(error=anthropic.APIConnectionError(request=request)))

    with pytest.raises(ProviderError) as excinfo:
        await drain(provider, simple_request())

    assert excinfo.value.status_code is None
    assert excinfo.value.retryable is True
    assert isinstance(excinfo.value.__cause__, anthropic.APIConnectionError)


async def test_timeout_is_retryable() -> None:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    provider = make_provider(FakeClient(error=anthropic.APITimeoutError(request=request)))

    with pytest.raises(ProviderError) as excinfo:
        await drain(provider, simple_request())

    assert excinfo.value.retryable is True


async def test_error_raised_mid_stream_is_wrapped_too() -> None:
    class ExplodingMessages(FakeMessages):
        def stream(self, **kwargs: Any) -> FakeStream:
            self.calls.append(kwargs)
            return ExplodingStream()

    class ExplodingStream(FakeStream):
        def __init__(self) -> None:
            super().__init__([], None)

        async def get_final_message(self) -> sdk.Message:
            raise status_error(529, "overloaded")

    client = FakeClient()
    client.messages = ExplodingMessages([], None)
    provider = make_provider(client)

    with pytest.raises(ProviderError) as excinfo:
        await drain(provider, simple_request())

    assert excinfo.value.status_code == 529
    assert excinfo.value.retryable is True


async def test_error_message_never_contains_the_token(no_stored_credential: None) -> None:
    token = "sk-ant-oat01-super-secret-subscription-token"
    provider = AnthropicProvider(auth_token=token)
    await provider.get_client()
    # A server that echoed the credential back in its error body must not be
    # able to smuggle it into our exception message.
    provider._client = FakeClient(error=status_error(401, f"invalid bearer {token}"))

    with pytest.raises(ProviderError) as excinfo:
        await drain(provider, simple_request())

    rendered = f"{excinfo.value.message} {excinfo.value!r} {excinfo.value}"
    assert token not in rendered
    assert "<redacted" in rendered
    # The wrapper is not the only place the token can surface: `raise ... from
    # exc` keeps the SDK exception as __cause__, and every traceback renders
    # str(__cause__). Scrubbing only the wrapper still wrote the raw token to
    # logs, Sentry, and pytest failure output.
    formatted = "".join(traceback.format_exception(excinfo.value))
    assert token not in formatted
    assert "<redacted" in formatted
    cause = excinfo.value.__cause__
    assert cause is not None and token not in str(cause)


async def test_the_real_sdk_client_exposes_the_attributes_scrubbing_reads() -> None:
    """The injected-client scrub reads the live secrets off the client, so the
    attribute names are a contract with the installed SDK, not an assumption."""
    token = "sk-ant-oat01-attribute-probe"
    client = anthropic.AsyncAnthropic(auth_token=token, api_key=None)
    try:
        assert client.auth_token == token
        assert client.api_key is None
    finally:
        await client.close()


@pytest.mark.parametrize(
    ("kwargs", "secret", "echoed"),
    [
        ({"auth_token": "sk-ant-oat01-injected-secret-value"}, "auth_token", "bad bearer token"),
        ({"api_key": "sk-ant-api03-injected-byok-secret"}, "api_key", "invalid x-api-key"),
    ],
)
async def test_an_injected_client_is_scrubbed_too(
    no_stored_credential: None,
    kwargs: dict[str, str],
    secret: str,
    echoed: str,
) -> None:
    """Regression: with ``client=``, ``get_client`` short-circuits before
    credential resolution, so all three stored secrets stayed ``None`` and
    ``_scrub`` was a no-op — a gateway that echoed the presented credential put
    it verbatim into ``ProviderError.message``, which callers log."""
    value = kwargs[secret]
    client = FakeClient(error=status_error(401, f"{echoed} {value}"), **kwargs)
    provider = AnthropicProvider(client=client)  # type: ignore[arg-type]

    with pytest.raises(ProviderError) as excinfo:
        await drain(provider, simple_request())

    formatted = "".join(traceback.format_exception(excinfo.value))
    assert value not in f"{excinfo.value.message} {excinfo.value!r} {formatted}"
    assert "<redacted" in excinfo.value.message


# ---------------------------------------------------------------------------
# end-to-end against a mock transport (real SDK, no network)
#
# The FakeClient tests pin the translation logic; these pin the contract with
# the installed SDK — that it accepts the parameters we build, and that the
# bytes we put on the wire are the ones we think they are.
# ---------------------------------------------------------------------------

SSE_EVENTS: list[tuple[str, dict[str, Any]]] = [
    (
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-5",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 1,
                    "cache_read_input_tokens": 4,
                    "cache_creation_input_tokens": 2,
                },
            },
        },
    ),
    (
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": "", "signature": ""},
        },
    ),
    (
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "hmm "},
        },
    ),
    (
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": SIGNATURE},
        },
    ),
    ("content_block_stop", {"type": "content_block_stop", "index": 0}),
    (
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "text", "text": ""},
        },
    ),
    (
        "content_block_delta",
        {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "Hi"}},
    ),
    ("content_block_stop", {"type": "content_block_stop", "index": 1}),
    (
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 2,
            "content_block": {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "get_weather",
                "input": {},
            },
        },
    ),
    (
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": 2,
            "delta": {"type": "input_json_delta", "partial_json": '{"city": "Pune"}'},
        },
    ),
    ("content_block_stop", {"type": "content_block_stop", "index": 2}),
    (
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use", "stop_sequence": None},
            "usage": {"output_tokens": 7},
        },
    ),
    ("message_stop", {"type": "message_stop"}),
]


def sse_body() -> bytes:
    import json

    return b"".join(
        f"event: {name}\ndata: {json.dumps(payload)}\n\n".encode() for name, payload in SSE_EVENTS
    )


async def test_end_to_end_over_a_mock_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api-env-key-must-not-leak")
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = {key.lower(): value for key, value in request.headers.items()}
        captured["body"] = json.loads(request.read())
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=sse_body()
        )

    http_client = anthropic.DefaultAsyncHttpxClient(transport=httpx.MockTransport(handler))
    client = anthropic.AsyncAnthropic(
        api_key=None,
        auth_token="sk-ant-oat01-subscription-token",
        default_headers={"anthropic-beta": OAUTH_BETA_HEADER},
        http_client=http_client,
    )
    provider = AnthropicProvider(client=client)
    req = CompletionRequest(
        messages=[Message.user("weather?")],
        model="claude-opus-5",
        max_tokens=1024,
        system="be terse",
        tools=[
            ToolSpec(
                name="get_weather",
                description="Get weather.",
                input_schema={"type": "object", "properties": {}},
            )
        ],
    )

    try:
        emitted = await drain(provider, req)
    finally:
        await http_client.aclose()

    assert emitted[0] == ProviderThinkingDelta(text="hmm ")
    assert emitted[1] == ProviderTextDelta(text="Hi")

    done = emitted[2]
    assert isinstance(done, CompletionDone)
    thinking, text, tool_use = done.message.content
    assert isinstance(thinking, ThinkingBlock) and thinking.signature == SIGNATURE
    assert isinstance(text, TextBlock) and text.text == "Hi"
    # The SDK accumulates the input_json_delta stream into a parsed dict.
    assert isinstance(tool_use, ToolUseBlock) and tool_use.input == {"city": "Pune"}
    assert done.stop_reason == "tool_use"
    assert done.usage.input_tokens == 10
    assert done.usage.output_tokens == 7
    assert done.usage.cache_read_input_tokens == 4
    assert done.usage.cache_creation_input_tokens == 2

    body = captured["body"]
    assert body["stream"] is True
    assert body["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert not {"temperature", "top_p", "top_k"} & set(body)

    headers = captured["headers"]
    assert headers["authorization"] == "Bearer sk-ant-oat01-subscription-token"
    assert "x-api-key" not in headers
    assert headers["anthropic-beta"] == OAUTH_BETA_HEADER


async def test_real_sdk_status_error_maps_to_provider_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"type": "error", "error": {"type": "rate_limit_error"}})

    http_client = anthropic.DefaultAsyncHttpxClient(transport=httpx.MockTransport(handler))
    client = anthropic.AsyncAnthropic(
        api_key="sk-ant-api-test", http_client=http_client, max_retries=0
    )
    provider = AnthropicProvider(client=client)

    try:
        with pytest.raises(ProviderError) as excinfo:
            await drain(provider, simple_request())
    finally:
        await http_client.aclose()

    assert excinfo.value.status_code == 429
    assert excinfo.value.retryable is True
    assert isinstance(excinfo.value.__cause__, anthropic.RateLimitError)


def test_redact_never_reveals_the_whole_secret() -> None:
    assert "secret" not in redact("secret")
    assert redact("x" * 40).endswith("len=40>")
    assert "…" in redact("sk-ant-oat01-plenty-of-characters")
