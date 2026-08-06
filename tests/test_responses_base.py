"""Tests for the Responses API machinery shared by the openai and codex providers.

Driven through :class:`~logpose.providers.openai.OpenAIProvider` because it is the
simpler of the two — an API key and nothing else — so what is under test here is
the base class rather than either backend's dress-up. The subscription-specific
and BYOK-specific behaviour lives in ``test_provider_codex.py`` and
``test_provider_openai.py``.

Everything runs against an ``httpx.MockTransport``; nothing here touches a
network or the developer's own credentials.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from logpose import (
    Message,
    RawBlock,
    RedactedThinkingBlock,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from logpose.auth import codex as codex_auth
from logpose.errors import LogposeError, ProviderError
from logpose.providers._responses import messages_to_input
from logpose.providers._toolargs import UNPARSED_ARGUMENTS_KEY
from logpose.providers.base import ProviderTextDelta, ProviderThinkingDelta, ToolSpec
from logpose.providers.openai import OpenAIProvider
from tests.responses_helpers import (
    API_KEY,
    Recorder,
    completed,
    drain,
    final_message,
    function_call_item,
    incomplete,
    item_added,
    item_done,
    message_item,
    mock_client,
    reasoning_delta,
    reasoning_item,
    refusal_item,
    request,
    sse,
    summary_delta,
    text_delta,
)

BASE = "http://responses.test/v1"


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep the developer's credentials and environment out of every test."""
    for name in (
        codex_auth.ENV_API_KEY,
        codex_auth.ENV_ACCOUNT_ID,
        "OPENAI_RESPONSES_MODEL",
        "OPENAI_RESPONSES_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(codex_auth.ENV_CODEX_HOME, str(tmp_path / "empty-codex-home"))


def make_provider(handler: Any, **kwargs: Any) -> OpenAIProvider:
    """Build a provider wired to a MockTransport with an explicit API key."""
    kwargs.setdefault("base_url", BASE)
    kwargs.setdefault("api_key", API_KEY)
    return OpenAIProvider(client=mock_client(handler), **kwargs)


async def run_turn(handler: Any, req: Any = None, **kwargs: Any) -> list[Any]:
    """Build a provider, stream one turn, and return its events."""
    return await drain(make_provider(handler, **kwargs), req)


# ---------------------------------------------------------------------------
# request translation
# ---------------------------------------------------------------------------


def test_user_text_becomes_an_input_text_message_item() -> None:
    assert messages_to_input([Message.user("hello")]) == [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]}
    ]


def test_assistant_text_becomes_an_output_text_message_item() -> None:
    assert messages_to_input([Message.assistant_text("hi there")]) == [
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "hi there"}],
        }
    ]


def test_tool_use_becomes_a_flat_function_call_item_without_the_fc_id() -> None:
    """call_id is the correlation key; the fc_ id is deliberately not resent."""
    message = Message(
        role="assistant",
        content=[ToolUseBlock(id="call_1", name="get_weather", input={"city": "Pune"})],
    )
    items = messages_to_input([message])
    assert items == [
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "get_weather",
            "arguments": '{"city": "Pune"}',
        }
    ]
    assert "id" not in items[0]


def test_tool_results_become_one_function_call_output_each_in_order() -> None:
    message = Message(
        role="user",
        content=[
            ToolResultBlock(tool_use_id="call_1", content="sunny"),
            ToolResultBlock(tool_use_id="call_2", content="rainy"),
        ],
    )
    assert messages_to_input([message]) == [
        {"type": "function_call_output", "call_id": "call_1", "output": "sunny"},
        {"type": "function_call_output", "call_id": "call_2", "output": "rainy"},
    ]


def test_a_failed_tool_result_is_marked_in_the_output_text() -> None:
    """function_call_output has no is_error field, so the signal lives in the text."""
    message = Message(
        role="user",
        content=[ToolResultBlock(tool_use_id="call_1", content="boom", is_error=True)],
    )
    assert messages_to_input([message])[0]["output"] == "ERROR: boom"


def test_a_reasoning_raw_block_is_resent_byte_exact() -> None:
    item = reasoning_item()
    assert messages_to_input([Message(role="assistant", content=[RawBlock(data=item)])]) == [item]


def test_reasoning_stays_immediately_before_the_item_it_reasoned_for() -> None:
    """The API rejects a reasoning item that is not followed by its own item."""
    message = Message(
        role="assistant",
        content=[
            RawBlock(data=reasoning_item()),
            ToolUseBlock(id="call_1", name="t", input={}),
        ],
    )
    assert [item["type"] for item in messages_to_input([message])] == [
        "reasoning",
        "function_call",
    ]


def test_reasoning_ordering_survives_two_full_tool_rounds() -> None:
    history = [
        Message.user("go"),
        Message(
            role="assistant",
            content=[
                RawBlock(data=reasoning_item(item_id="rs_1")),
                ToolUseBlock(id="call_1", name="t", input={}),
            ],
        ),
        Message(role="user", content=[ToolResultBlock(tool_use_id="call_1", content="ok")]),
        Message(
            role="assistant",
            content=[RawBlock(data=reasoning_item(item_id="rs_2")), TextBlock(text="done")],
        ),
    ]
    kinds = [item["type"] for item in messages_to_input(history)]
    assert kinds == [
        "message",
        "reasoning",
        "function_call",
        "function_call_output",
        "reasoning",
        "message",
    ]
    for index, kind in enumerate(kinds):
        if kind == "reasoning":
            assert index + 1 < len(kinds), "a reasoning item must not be last"


@pytest.mark.parametrize(
    "block",
    [
        ThinkingBlock(thinking="anthropic thoughts", signature="sig"),
        RedactedThinkingBlock(data="opaque"),
    ],
)
def test_anthropic_thinking_blocks_are_dropped_not_raised(block: Any) -> None:
    """A Conversation carried over from an Anthropic backend must not crash."""
    message = Message(role="assistant", content=[block, TextBlock(text="hi")])
    assert [item["type"] for item in messages_to_input([message])] == ["message"]


async def test_tools_are_advertised_flat_without_a_nested_function_object() -> None:
    handler = Recorder()
    spec = ToolSpec(name="t", description="d", input_schema={"type": "object"})
    await run_turn(handler, request(tools=[spec]))
    assert handler.last_body["tools"] == [
        {"type": "function", "name": "t", "description": "d", "parameters": {"type": "object"}}
    ]
    assert handler.last_body["tool_choice"] == "auto"
    assert handler.last_body["parallel_tool_calls"] is True


async def test_no_tool_fields_are_sent_when_there_are_no_tools() -> None:
    handler = Recorder()
    await run_turn(handler)
    assert "tools" not in handler.last_body
    assert "tool_choice" not in handler.last_body


async def test_store_is_false_and_encrypted_reasoning_is_included() -> None:
    handler = Recorder()
    await run_turn(handler)
    assert handler.last_body["store"] is False
    assert handler.last_body["include"] == ["reasoning.encrypted_content"]
    assert handler.last_body["stream"] is True


async def test_reasoning_effort_and_summary_are_sent() -> None:
    handler = Recorder()
    await run_turn(handler, reasoning_effort="high", reasoning_summary="detailed")
    assert handler.last_body["reasoning"] == {"effort": "high", "summary": "detailed"}


async def test_no_reasoning_key_is_sent_when_effort_is_none() -> None:
    handler = Recorder()
    await run_turn(handler, reasoning_effort=None)
    assert "reasoning" not in handler.last_body


async def test_the_prompt_cache_key_is_sent_when_given() -> None:
    handler = Recorder()
    await run_turn(handler, prompt_cache_key="conv-7")
    assert handler.last_body["prompt_cache_key"] == "conv-7"


async def test_extra_body_and_request_extra_merge_with_request_extra_winning() -> None:
    handler = Recorder()
    await run_turn(
        handler,
        request(extra={"service_tier": "priority"}),
        extra_body={"service_tier": "default", "top_p": 0.5},
    )
    assert handler.last_body["service_tier"] == "priority"
    assert handler.last_body["top_p"] == 0.5


def test_an_unknown_reasoning_effort_is_rejected_at_construction() -> None:
    with pytest.raises(LogposeError) as excinfo:
        OpenAIProvider(reasoning_effort="ludicrous")
    assert "ludicrous" in str(excinfo.value)


@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
def test_every_documented_reasoning_effort_is_accepted(effort: str) -> None:
    assert OpenAIProvider(reasoning_effort=effort) is not None


def test_a_non_positive_max_tokens_is_rejected_at_construction() -> None:
    with pytest.raises(LogposeError):
        OpenAIProvider(max_tokens=0)


# ---------------------------------------------------------------------------
# streaming assembly
# ---------------------------------------------------------------------------


async def test_text_deltas_stream_and_the_turn_assembles_from_the_output() -> None:
    body = sse(
        text_delta("Hel"),
        text_delta("lo"),
        item_done(message_item("Hello")),
        completed(output=[message_item("Hello")]),
    )
    events = await run_turn(Recorder(body))
    assert [e.text for e in events if isinstance(e, ProviderTextDelta)] == ["Hel", "lo"]
    message = final_message(events)
    assert message.text == "Hello"
    assert message.role == "assistant"


async def test_reasoning_summary_deltas_become_thinking_deltas() -> None:
    body = sse(
        summary_delta("Weighing "),
        summary_delta("options."),
        completed(output=[reasoning_item(), message_item()]),
    )
    events = await run_turn(Recorder(body))
    assert [e.text for e in events if isinstance(e, ProviderThinkingDelta)] == [
        "Weighing ",
        "options.",
    ]


async def test_reasoning_text_deltas_also_become_thinking_deltas() -> None:
    body = sse(reasoning_delta("Visible thought."), completed(output=[message_item()]))
    events = await run_turn(Recorder(body))
    assert [e.text for e in events if isinstance(e, ProviderThinkingDelta)] == [
        "Visible thought."
    ]


async def test_reasoning_is_stored_as_a_raw_block_not_a_thinking_block() -> None:
    item = reasoning_item()
    body = sse(summary_delta("Thinking."), completed(output=[item, message_item()]))
    message = final_message(await run_turn(Recorder(body)))
    assert isinstance(message.content[0], RawBlock)
    assert message.content[0].block_type == "reasoning"
    assert message.content[0].data == item
    assert not any(isinstance(block, ThinkingBlock) for block in message.content)


async def test_a_reasoning_item_with_an_empty_summary_still_round_trips() -> None:
    """At low effort the summary is empty but encrypted_content is still required."""
    item = reasoning_item(summary=None)
    message = final_message(await run_turn(Recorder(sse(completed(output=[item, message_item()])))))
    assert message.content[0] == RawBlock(data=item)


async def test_blocks_are_assembled_in_wire_order() -> None:
    body = sse(completed(output=[reasoning_item(), message_item("hi"), function_call_item()]))
    message = final_message(await run_turn(Recorder(body)))
    assert [block.type for block in message.content] == ["raw", "text", "tool_use"]


async def test_function_call_arguments_are_parsed_from_the_json_string() -> None:
    message = final_message(await run_turn(Recorder(sse(completed(output=[function_call_item()])))))
    call = message.content[0]
    assert isinstance(call, ToolUseBlock)
    assert call.id == "call_1"
    assert call.name == "get_weather"
    assert call.input == {"city": "Pune"}


async def test_malformed_tool_arguments_are_reported_not_raised() -> None:
    body = sse(completed(output=[function_call_item(arguments="{not json")]))
    message = final_message(await run_turn(Recorder(body)))
    call = message.content[0]
    assert isinstance(call, ToolUseBlock)
    assert call.input == {UNPARSED_ARGUMENTS_KEY: "{not json"}


async def test_parallel_function_calls_keep_their_output_order() -> None:
    body = sse(
        completed(
            output=[
                function_call_item(call_id="call_a", name="a"),
                function_call_item(call_id="call_b", name="b"),
            ]
        )
    )
    message = final_message(await run_turn(Recorder(body)))
    assert [block.id for block in message.content] == ["call_a", "call_b"]  # type: ignore[union-attr]


async def test_output_item_added_stubs_never_contribute() -> None:
    """An `added` event carries a hollow item; only `done` may be assembled."""
    hollow = {"type": "message", "id": "msg_1", "role": "assistant", "content": []}
    body = sse(item_added(hollow), text_delta("real"), item_done(message_item("real")))
    message = final_message(await run_turn(Recorder(body)))
    assert message.text == "real"
    assert len(message.content) == 1


async def test_a_populated_terminal_response_overrides_the_accumulated_items() -> None:
    body = sse(
        item_done(message_item("partial")), completed(output=[message_item("authoritative")])
    )
    assert final_message(await run_turn(Recorder(body))).text == "authoritative"


async def test_an_empty_terminal_output_does_not_discard_the_accumulated_items() -> None:
    """Verified live: the subscription backend sends `output: []` on completion.

    Everything real arrives through `response.output_item.done`, so trusting the
    empty array would throw the whole turn away and report end_turn with no text.
    """
    body = sse(
        item_done(reasoning_item(), index=0),
        item_done(message_item("kept"), index=1),
        completed(output=[]),
    )
    message = final_message(await run_turn(Recorder(body)))
    assert message.text == "kept"
    assert [block.type for block in message.content] == ["raw", "text"]


async def test_a_terminal_event_without_output_falls_back_to_the_done_items() -> None:
    body = sse(item_done(message_item("from items")), completed())
    assert final_message(await run_turn(Recorder(body))).text == "from items"


async def test_done_items_are_ordered_by_output_index() -> None:
    body = sse(
        item_done(message_item("second"), index=1),
        item_done(message_item("first"), index=0),
        completed(),
    )
    message = final_message(await run_turn(Recorder(body)))
    assert [block.text for block in message.content] == ["first", "second"]  # type: ignore[union-attr]


async def test_a_body_with_only_data_lines_still_works() -> None:
    """Some proxies strip `event:` lines, so the payload's own type is what counts."""
    payload = json.dumps(completed(output=[message_item("ok")]))
    message = final_message(await run_turn(Recorder(f"data: {payload}\n\n".encode())))
    assert message.text == "ok"


async def test_noise_lines_are_ignored() -> None:
    body = (
        b": keep-alive\n\n"
        b"event: ping\n\n"
        b"data: not json\n\n"
        b"data: \n\n"
        b"data: [DONE]\n\n"
        b'data: {"type": "response.created", "response": {}}\n\n'
        + sse(completed(output=[message_item("ok")]), done=False)
    )
    assert final_message(await run_turn(Recorder(body))).text == "ok"


async def test_a_server_side_tool_item_survives_as_a_raw_block() -> None:
    item = {"type": "web_search_call", "id": "ws_1", "status": "completed"}
    message = final_message(await run_turn(Recorder(sse(completed(output=[item, message_item()])))))
    assert message.content[0] == RawBlock(data=item)


async def test_an_empty_completion_yields_an_empty_message() -> None:
    events = await run_turn(Recorder(sse(completed(output=[]))))
    assert final_message(events).content == []
    assert events[-1].stop_reason == "end_turn"


# ---------------------------------------------------------------------------
# stop reason
# ---------------------------------------------------------------------------


async def test_completed_without_tool_calls_is_end_turn() -> None:
    events = await run_turn(Recorder(sse(completed(output=[message_item()]))))
    assert events[-1].stop_reason == "end_turn"


async def test_completed_with_tool_calls_is_tool_use() -> None:
    """A 'completed' status must not end the run with calls unexecuted."""
    events = await run_turn(Recorder(sse(completed(output=[function_call_item()]))))
    assert events[-1].stop_reason == "tool_use"


async def test_incomplete_max_output_tokens_is_max_tokens() -> None:
    body = sse(incomplete("max_output_tokens", output=[message_item("cut off")]))
    events = await run_turn(Recorder(body))
    assert events[-1].stop_reason == "max_tokens"


async def test_incomplete_content_filter_is_refusal() -> None:
    events = await run_turn(Recorder(sse(incomplete("content_filter", output=[]))))
    assert events[-1].stop_reason == "refusal"


async def test_incomplete_outranks_tool_calls() -> None:
    """A truncated turn's arguments may be cut mid-JSON, so the call must not run."""
    body = sse(incomplete("max_output_tokens", output=[function_call_item(arguments='{"ci')]))
    events = await run_turn(Recorder(body))
    assert events[-1].stop_reason == "max_tokens"


async def test_an_unknown_incomplete_reason_falls_back_to_max_tokens() -> None:
    events = await run_turn(Recorder(sse(incomplete("something_new", output=[]))))
    assert events[-1].stop_reason == "max_tokens"


async def test_a_refusal_content_part_is_a_refusal_and_keeps_its_text() -> None:
    events = await run_turn(Recorder(sse(completed(output=[refusal_item("No.")]))))
    assert events[-1].stop_reason == "refusal"
    assert final_message(events).text == "No."


@pytest.mark.parametrize("status", ["queued", "in_progress", "unheard_of"])
async def test_an_unknown_status_falls_back_to_end_turn(status: str) -> None:
    events = await run_turn(Recorder(sse(completed(output=[message_item()], status=status))))
    assert events[-1].stop_reason == "end_turn"


# ---------------------------------------------------------------------------
# usage
# ---------------------------------------------------------------------------


async def test_usage_subtracts_cached_tokens_from_input() -> None:
    body = sse(
        completed(
            output=[message_item()],
            usage={
                "input_tokens": 100,
                "input_tokens_details": {"cached_tokens": 40},
                "output_tokens": 25,
            },
        )
    )
    usage = (await run_turn(Recorder(body)))[-1].usage
    assert usage.input_tokens == 60
    assert usage.cache_read_input_tokens == 40
    assert usage.output_tokens == 25
    assert usage.cache_creation_input_tokens == 0


async def test_cache_write_tokens_are_reported() -> None:
    """The subscription backend reports these; the public API omits the field."""
    body = sse(
        completed(
            output=[message_item()],
            usage={
                "input_tokens": 30,
                "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 12},
                "output_tokens": 18,
            },
        )
    )
    usage = (await run_turn(Recorder(body)))[-1].usage
    assert usage.cache_creation_input_tokens == 12
    assert usage.input_tokens == 30


async def test_cached_tokens_cannot_exceed_input_tokens() -> None:
    body = sse(
        completed(
            output=[], usage={"input_tokens": 10, "input_tokens_details": {"cached_tokens": 99}}
        )
    )
    usage = (await run_turn(Recorder(body)))[-1].usage
    assert usage.input_tokens == 0
    assert usage.cache_read_input_tokens == 10


async def test_reasoning_tokens_are_not_added_to_anything() -> None:
    """They are already inside output_tokens; adding them would inflate the cost."""
    body = sse(
        completed(
            output=[],
            usage={
                "input_tokens": 5,
                "output_tokens": 30,
                "output_tokens_details": {"reasoning_tokens": 20},
            },
        )
    )
    assert (await run_turn(Recorder(body)))[-1].usage.output_tokens == 30


@pytest.mark.parametrize("usage", [None, {}, "nope"])
async def test_an_unusable_usage_object_reports_zeros(usage: Any) -> None:
    response: dict[str, Any] = {"id": "r", "status": "completed", "output": []}
    if usage is not None:
        response["usage"] = usage
    events = await run_turn(Recorder(sse({"type": "response.completed", "response": response})))
    assert events[-1].usage.input_tokens == 0
    assert events[-1].usage.output_tokens == 0


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------


async def test_a_client_error_is_not_retryable() -> None:
    with pytest.raises(ProviderError) as excinfo:
        await run_turn(Recorder(status=400, text="bad request"))
    assert excinfo.value.status_code == 400
    assert excinfo.value.retryable is False
    assert "bad request" in str(excinfo.value)


@pytest.mark.parametrize("status", [408, 409, 429, 500, 503])
async def test_server_and_rate_limit_errors_are_retryable(status: int) -> None:
    with pytest.raises(ProviderError) as excinfo:
        await run_turn(Recorder(status=status, text="later"))
    assert excinfo.value.retryable is True


async def test_a_connection_failure_names_the_base_url_and_is_retryable() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route")

    with pytest.raises(ProviderError) as excinfo:
        await run_turn(handler)
    assert BASE in str(excinfo.value)
    assert excinfo.value.retryable is True
    assert excinfo.value.status_code is None


async def test_a_response_failed_event_becomes_a_provider_error() -> None:
    """The stream can fail after a 200, so this is not an HTTP status path."""
    body = sse(
        {
            "type": "response.failed",
            "response": {
                "id": "r",
                "status": "failed",
                "error": {"code": "invalid_prompt", "message": "no good"},
            },
        }
    )
    with pytest.raises(ProviderError) as excinfo:
        await run_turn(Recorder(body))
    assert "no good" in str(excinfo.value)
    assert "invalid_prompt" in str(excinfo.value)
    assert excinfo.value.retryable is False


async def test_a_bare_error_event_becomes_a_provider_error() -> None:
    body = sse({"type": "error", "code": "bad_thing", "message": "it broke"})
    with pytest.raises(ProviderError) as excinfo:
        await run_turn(Recorder(body))
    assert "it broke" in str(excinfo.value)


async def test_a_rate_limit_error_event_is_retryable() -> None:
    body = sse({"type": "error", "code": "rate_limit_exceeded", "message": "slow down"})
    with pytest.raises(ProviderError) as excinfo:
        await run_turn(Recorder(body))
    assert excinfo.value.retryable is True


async def test_a_failure_event_with_no_detail_still_raises() -> None:
    with pytest.raises(ProviderError):
        await run_turn(Recorder(sse({"type": "response.failed", "response": {"status": "failed"}})))


async def test_a_truncated_stream_raises_a_retryable_provider_error() -> None:
    with pytest.raises(ProviderError) as excinfo:
        await run_turn(Recorder(sse(text_delta("half a th"), done=False)))
    assert excinfo.value.retryable is True
    assert "terminal event" in str(excinfo.value)


async def test_a_reasoning_ordering_400_surfaces_the_api_message() -> None:
    """The failure mode the call_id decision risks; it must be legible when it hits."""
    message = "Item 'rs_1' of type 'reasoning' was provided without its required following item."
    handler = Recorder(status=400, text=json.dumps({"error": {"message": message}}))
    with pytest.raises(ProviderError) as excinfo:
        await run_turn(handler)
    assert "required following item" in str(excinfo.value)


# ---------------------------------------------------------------------------
# secrets and lifecycle
# ---------------------------------------------------------------------------


async def test_the_api_key_never_appears_in_an_error_body() -> None:
    with pytest.raises(ProviderError) as excinfo:
        await run_turn(Recorder(status=401, text=f"bad key {API_KEY}"))
    assert API_KEY not in str(excinfo.value)
    assert "<redacted" in str(excinfo.value)


async def test_a_credential_never_appears_in_a_stream_failure() -> None:
    body = sse({"type": "error", "code": "x", "message": f"key was {API_KEY}"})
    with pytest.raises(ProviderError) as excinfo:
        await run_turn(Recorder(body))
    assert API_KEY not in str(excinfo.value)


async def test_the_repr_redacts_the_credential() -> None:
    provider = make_provider(Recorder())
    await drain(provider)
    assert API_KEY not in repr(provider)


async def test_aclose_closes_a_client_the_provider_created() -> None:
    provider = OpenAIProvider(api_key=API_KEY)
    client = provider._http()
    await provider.aclose()
    assert client.is_closed


async def test_aclose_leaves_an_injected_client_alone() -> None:
    client = mock_client(Recorder())
    provider = OpenAIProvider(client=client, api_key=API_KEY)
    await provider.aclose()
    assert not client.is_closed
    await client.aclose()


async def test_credential_resolution_is_deferred_to_the_first_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    original = codex_auth.CredentialProvider.resolve

    def counting(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(codex_auth.CredentialProvider, "resolve", counting)
    handler = Recorder(sse(completed(output=[message_item()])), sse(completed(output=[])))
    provider = make_provider(handler)
    assert calls == []
    await drain(provider)
    assert calls == [1]
    await drain(provider)
    assert calls == [1], "the CredentialProvider must be built once, not per request"
