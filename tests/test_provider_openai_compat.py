"""Tests for the OpenAI-compatible provider and the Docker Model Runner backend.

Everything runs against an ``httpx.MockTransport``; nothing here touches a
network or a running Docker Model Runner.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from logpose import (
    Agent,
    Message,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    known_providers,
    resolve,
    tool,
)
from logpose.errors import LogposeError, ProviderError
from logpose.providers.base import CompletionRequest, ToolSpec
from logpose.providers.openai_compat import (
    AUTO_MODEL,
    DEFAULT_DOCKER_BASE_URL,
    UNPARSED_ARGUMENTS_KEY,
    DockerModelsProvider,
    OpenAICompatProvider,
    _messages_to_wire,
    _ThinkTagSplitter,
)

BASE = "http://local.test/v1"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def sse(*chunks: dict[str, Any], done: bool = True) -> bytes:
    """Render chunk dicts as a Chat Completions SSE body."""
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks)
    if done:
        body += "data: [DONE]\n\n"
    return body.encode()


def delta_chunk(**delta: Any) -> dict[str, Any]:
    """One streaming chunk carrying a delta."""
    return {"choices": [{"index": 0, "delta": delta}]}


def finish_chunk(reason: str) -> dict[str, Any]:
    """One streaming chunk carrying a finish_reason."""
    return {"choices": [{"index": 0, "delta": {}, "finish_reason": reason}]}


def usage_chunk(prompt: int, completion: int, cached: int = 0) -> dict[str, Any]:
    """The trailing usage-only chunk emitted with stream_options.include_usage."""
    return {
        "choices": [],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "prompt_tokens_details": {"cached_tokens": cached},
        },
    }


class Recorder:
    """MockTransport handler that records requests and replays scripted bodies."""

    def __init__(self, *bodies: bytes, models: list[str] | None = None) -> None:
        self.bodies = list(bodies)
        self.models = models if models is not None else ["m-first", "m-second"]
        self.requests: list[httpx.Request] = []
        self.model_calls = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            self.model_calls += 1
            return httpx.Response(
                200, json={"object": "list", "data": [{"id": m} for m in self.models]}
            )
        self.requests.append(request)
        body = self.bodies.pop(0) if self.bodies else sse(finish_chunk("stop"))
        return httpx.Response(200, content=body)

    @property
    def last_body(self) -> dict[str, Any]:
        """The most recent chat-completions request body."""
        return json.loads(self.requests[-1].content)

    def body(self, index: int) -> dict[str, Any]:
        """The nth chat-completions request body."""
        return json.loads(self.requests[index].content)


def make_provider(handler: Recorder, **kwargs: Any) -> OpenAICompatProvider:
    """Build a provider wired to a MockTransport."""
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    kwargs.setdefault("base_url", BASE)
    kwargs.setdefault("model", "test-model")
    return OpenAICompatProvider(client=client, **kwargs)


def request(**kwargs: Any) -> CompletionRequest:
    """Build a CompletionRequest with sane defaults."""
    kwargs.setdefault("messages", [Message.user("hi")])
    kwargs.setdefault("model", "test-model")
    kwargs.setdefault("max_tokens", 128)
    return CompletionRequest(**kwargs)


async def drain(provider: OpenAICompatProvider, req: CompletionRequest) -> list[Any]:
    """Collect every event a provider stream yields."""
    return [event async for event in provider.stream(req)]


# ---------------------------------------------------------------------------
# message translation
# ---------------------------------------------------------------------------


def test_system_prompt_leads_the_message_list() -> None:
    wire = _messages_to_wire([Message.user("hello")], "be terse")
    assert wire[0] == {"role": "system", "content": "be terse"}
    assert wire[1] == {"role": "user", "content": "hello"}


def test_tool_results_fan_out_to_one_message_each_in_order() -> None:
    """logpose batches results into one user turn; Chat Completions wants one each."""
    batched = Message(
        role="user",
        content=[
            ToolResultBlock(tool_use_id="a", content="ra"),
            ToolResultBlock(tool_use_id="b", content="rb"),
            ToolResultBlock(tool_use_id="c", content="rc"),
        ],
    )
    wire = _messages_to_wire([batched], None)
    assert [m["role"] for m in wire] == ["tool", "tool", "tool"]
    assert [m["tool_call_id"] for m in wire] == ["a", "b", "c"]
    assert [m["content"] for m in wire] == ["ra", "rb", "rc"]


def test_failed_tool_result_is_marked_in_the_text() -> None:
    """Chat Completions has no is_error field, so the signal must be in the content."""
    msg = Message(
        role="user", content=[ToolResultBlock(tool_use_id="a", content="boom", is_error=True)]
    )
    assert _messages_to_wire([msg], None)[0]["content"] == "ERROR: boom"


def test_assistant_tool_calls_serialize_arguments_as_a_json_string() -> None:
    msg = Message(
        role="assistant",
        content=[
            TextBlock(text="calling"),
            ToolUseBlock(id="tu_1", name="get_weather", input={"city": "Pune"}),
        ],
    )
    entry = _messages_to_wire([msg], None)[0]
    assert entry["role"] == "assistant"
    assert entry["content"] == "calling"
    call = entry["tool_calls"][0]
    assert call == {
        "id": "tu_1",
        "type": "function",
        "function": {"name": "get_weather", "arguments": '{"city": "Pune"}'},
    }


def test_thinking_is_not_sent_back() -> None:
    """No signature to preserve, and several servers reject echoed reasoning."""
    msg = Message(
        role="assistant",
        content=[ThinkingBlock(thinking="secret reasoning"), TextBlock(text="answer")],
    )
    entry = _messages_to_wire([msg], None)[0]
    assert entry["content"] == "answer"
    assert "secret reasoning" not in json.dumps(entry)


def test_empty_assistant_turn_is_dropped() -> None:
    """An assistant turn with only thinking would otherwise become content=''."""
    msg = Message(role="assistant", content=[ThinkingBlock(thinking="only reasoning")])
    assert _messages_to_wire([msg], None) == []


async def test_tools_are_advertised_as_function_tools() -> None:
    handler = Recorder(sse(finish_chunk("stop")))
    provider = make_provider(handler)
    spec = ToolSpec(name="f", description="does f", input_schema={"type": "object"})
    await drain(provider, request(tools=[spec]))
    assert handler.last_body["tools"] == [
        {
            "type": "function",
            "function": {"name": "f", "description": "does f", "parameters": {"type": "object"}},
        }
    ]


async def test_streaming_and_usage_are_always_requested() -> None:
    handler = Recorder(sse(finish_chunk("stop")))
    await drain(make_provider(handler), request())
    body = handler.last_body
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}


async def test_extra_body_and_request_extra_merge_into_the_payload() -> None:
    handler = Recorder(sse(finish_chunk("stop")))
    provider = make_provider(handler, extra_body={"temperature": 0.1, "top_p": 0.5})
    await drain(provider, request(extra={"top_p": 0.9, "seed": 7}))
    body = handler.last_body
    assert body["temperature"] == 0.1
    assert body["seed"] == 7
    assert body["top_p"] == 0.9, "request.extra must win over provider extra_body"


# ---------------------------------------------------------------------------
# streaming
# ---------------------------------------------------------------------------


async def test_text_deltas_stream_and_assemble() -> None:
    handler = Recorder(
        sse(
            delta_chunk(role="assistant"),
            delta_chunk(content="Hel"),
            delta_chunk(content="lo!"),
            finish_chunk("stop"),
            usage_chunk(10, 3),
        )
    )
    events = await drain(make_provider(handler), request())
    assert [e.text for e in events if type(e).__name__ == "ProviderTextDelta"] == ["Hel", "lo!"]
    done = events[-1]
    assert done.message.text == "Hello!"
    assert done.stop_reason == "end_turn"


async def test_reasoning_content_becomes_thinking() -> None:
    handler = Recorder(
        sse(
            delta_chunk(reasoning_content="step 1 "),
            delta_chunk(reasoning_content="step 2"),
            delta_chunk(content="answer"),
            finish_chunk("stop"),
        )
    )
    events = await drain(make_provider(handler), request())
    thinking = [e.text for e in events if type(e).__name__ == "ProviderThinkingDelta"]
    assert thinking == ["step 1 ", "step 2"]
    blocks = events[-1].message.content
    assert isinstance(blocks[0], ThinkingBlock)
    assert blocks[0].thinking == "step 1 step 2"
    assert isinstance(blocks[1], TextBlock)


async def test_tool_call_fragments_assemble_across_chunks() -> None:
    """The real wire shape: id and name arrive once, arguments in pieces."""
    handler = Recorder(
        sse(
            delta_chunk(
                tool_calls=[
                    {
                        "index": 0,
                        "id": "call_abc",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{"},
                    }
                ]
            ),
            delta_chunk(tool_calls=[{"index": 0, "function": {"arguments": '"city"'}}]),
            delta_chunk(tool_calls=[{"index": 0, "function": {"arguments": ':"Pune"}'}}]),
            finish_chunk("tool_calls"),
        )
    )
    events = await drain(make_provider(handler), request())
    done = events[-1]
    assert done.stop_reason == "tool_use"
    block = done.message.content[0]
    assert isinstance(block, ToolUseBlock)
    assert block.id == "call_abc"
    assert block.name == "get_weather"
    assert block.input == {"city": "Pune"}


async def test_parallel_tool_calls_are_keyed_by_index() -> None:
    handler = Recorder(
        sse(
            delta_chunk(
                tool_calls=[
                    {"index": 0, "id": "c0", "function": {"name": "a", "arguments": '{"x":1}'}},
                    {"index": 1, "id": "c1", "function": {"name": "b", "arguments": '{"y":'}},
                ]
            ),
            delta_chunk(tool_calls=[{"index": 1, "function": {"arguments": "2}"}}]),
            finish_chunk("tool_calls"),
        )
    )
    blocks = (await drain(make_provider(handler), request()))[-1].message.content
    assert [(b.id, b.name, b.input) for b in blocks] == [
        ("c0", "a", {"x": 1}),
        ("c1", "b", {"y": 2}),
    ]


async def test_missing_tool_call_id_is_synthesized() -> None:
    """Some servers omit id; the loop still needs a stable handle to match results."""
    handler = Recorder(
        sse(
            delta_chunk(tool_calls=[{"index": 0, "function": {"name": "a", "arguments": "{}"}}]),
            finish_chunk("tool_calls"),
        )
    )
    block = (await drain(make_provider(handler), request()))[-1].message.content[0]
    assert block.id == "call_0"


async def test_unparseable_tool_arguments_are_reported_not_raised() -> None:
    """Small local models emit invalid JSON; the model must get a chance to retry."""
    handler = Recorder(
        sse(
            delta_chunk(
                tool_calls=[
                    {"index": 0, "id": "c", "function": {"name": "a", "arguments": "{city: Pune"}}
                ]
            ),
            finish_chunk("tool_calls"),
        )
    )
    block = (await drain(make_provider(handler), request()))[-1].message.content[0]
    assert block.input == {UNPARSED_ARGUMENTS_KEY: "{city: Pune"}


async def test_non_object_tool_arguments_are_reported() -> None:
    handler = Recorder(
        sse(
            delta_chunk(
                tool_calls=[
                    {"index": 0, "id": "c", "function": {"name": "a", "arguments": "[1,2]"}}
                ]
            ),
            finish_chunk("tool_calls"),
        )
    )
    block = (await drain(make_provider(handler), request()))[-1].message.content[0]
    assert block.input == {UNPARSED_ARGUMENTS_KEY: "[1,2]"}


async def test_usage_subtracts_cached_tokens_from_input() -> None:
    """logpose input_tokens means uncached; prompt_tokens includes the cached part."""
    handler = Recorder(sse(finish_chunk("stop"), usage_chunk(100, 20, cached=60)))
    usage = (await drain(make_provider(handler), request()))[-1].usage
    assert usage.input_tokens == 40
    assert usage.cache_read_input_tokens == 60
    assert usage.output_tokens == 20


async def test_cached_tokens_cannot_exceed_prompt_tokens() -> None:
    handler = Recorder(sse(finish_chunk("stop"), usage_chunk(10, 1, cached=999)))
    usage = (await drain(make_provider(handler), request()))[-1].usage
    assert usage.input_tokens == 0
    assert usage.cache_read_input_tokens == 10


@pytest.mark.parametrize(
    ("reason", "expected"),
    [("stop", "end_turn"), ("length", "max_tokens"), ("content_filter", "refusal")],
)
async def test_finish_reason_mapping(reason: str, expected: str) -> None:
    handler = Recorder(sse(finish_chunk(reason)))
    assert (await drain(make_provider(handler), request()))[-1].stop_reason == expected


async def test_tool_calls_win_over_a_missing_finish_reason() -> None:
    """Dropping to end_turn here would silently discard the calls."""
    handler = Recorder(
        sse(
            delta_chunk(tool_calls=[{"index": 0, "id": "c", "function": {"name": "a"}}]),
            done=True,
        )
    )
    assert (await drain(make_provider(handler), request()))[-1].stop_reason == "tool_use"


async def test_tool_calls_win_over_a_stop_finish_reason() -> None:
    handler = Recorder(
        sse(
            delta_chunk(tool_calls=[{"index": 0, "id": "c", "function": {"name": "a"}}]),
            finish_chunk("stop"),
        )
    )
    assert (await drain(make_provider(handler), request()))[-1].stop_reason == "tool_use"


async def test_malformed_and_noise_lines_are_ignored() -> None:
    body = (
        b": keep-alive comment\n\n"
        b"event: ping\n\n"
        b"data: not json at all\n\n"
        b"data: []\n\n"
        + sse(delta_chunk(content="ok"), finish_chunk("stop"))
    )
    events = await drain(make_provider(Recorder(body)), request())
    assert events[-1].message.text == "ok"


# ---------------------------------------------------------------------------
# <think> tags
# ---------------------------------------------------------------------------


def test_think_splitter_handles_tags_split_across_chunks() -> None:
    splitter = _ThinkTagSplitter()
    assert splitter.feed("<thi") == ("", "")
    assert splitter.feed("nk>reason") == ("", "reason")
    assert splitter.feed("ing</thi") == ("", "ing")
    assert splitter.feed("nk>visible") == ("visible", "")
    assert splitter.flush() == ("", "")


def test_think_splitter_treats_unterminated_block_as_thinking() -> None:
    """Text inside an unclosed <think> is emitted as reasoning, not withheld."""
    splitter = _ThinkTagSplitter()
    assert splitter.feed("<think>never closed") == ("", "never closed")
    assert splitter.flush() == ("", "")


def test_think_splitter_flushes_a_trailing_partial_tag() -> None:
    """A dangling '<thi' is held back mid-stream, then released as plain text."""
    splitter = _ThinkTagSplitter()
    assert splitter.feed("abc<thi") == ("abc", ""), "the partial tag must be withheld"
    assert splitter.flush() == ("<thi", "")


def test_think_splitter_passes_plain_text_through() -> None:
    splitter = _ThinkTagSplitter()
    assert splitter.feed("just text") == ("just text", "")


async def test_think_tags_in_content_become_thinking_blocks() -> None:
    handler = Recorder(
        sse(
            delta_chunk(content="<think>weighing options</think>"),
            delta_chunk(content="Final answer."),
            finish_chunk("stop"),
        )
    )
    done = (await drain(make_provider(handler), request()))[-1]
    blocks = done.message.content
    assert isinstance(blocks[0], ThinkingBlock)
    assert blocks[0].thinking == "weighing options"
    assert done.message.text == "Final answer."


async def test_think_tag_parsing_can_be_disabled() -> None:
    handler = Recorder(sse(delta_chunk(content="<think>x</think>y"), finish_chunk("stop")))
    done = (await drain(make_provider(handler, parse_think_tags=False), request()))[-1]
    assert done.message.text == "<think>x</think>y"


# ---------------------------------------------------------------------------
# model discovery
# ---------------------------------------------------------------------------


async def test_auto_model_resolves_to_the_first_served_model() -> None:
    handler = Recorder(sse(finish_chunk("stop")), models=["docker.io/ai/gemma4:latest", "other"])
    provider = make_provider(handler, model=AUTO_MODEL)
    await drain(provider, request(model=AUTO_MODEL))
    assert handler.last_body["model"] == "docker.io/ai/gemma4:latest"


async def test_auto_model_is_discovered_only_once() -> None:
    handler = Recorder(sse(finish_chunk("stop")), sse(finish_chunk("stop")))
    provider = make_provider(handler, model=AUTO_MODEL)
    await drain(provider, request(model=AUTO_MODEL))
    await drain(provider, request(model=AUTO_MODEL))
    assert handler.model_calls == 1


async def test_explicit_model_skips_discovery_entirely() -> None:
    handler = Recorder(sse(finish_chunk("stop")))
    await drain(make_provider(handler, model="pinned"), request(model="pinned"))
    assert handler.model_calls == 0
    assert handler.last_body["model"] == "pinned"


async def test_no_served_models_raises_an_actionable_error() -> None:
    handler = Recorder(sse(finish_chunk("stop")), models=[])
    provider = make_provider(handler, model=AUTO_MODEL)
    with pytest.raises(ProviderError, match="no available models"):
        await drain(provider, request(model=AUTO_MODEL))


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------


async def test_client_error_is_not_retryable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "bad model"}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAICompatProvider(base_url=BASE, model="m", client=client)
    with pytest.raises(ProviderError) as excinfo:
        await drain(provider, request())
    assert excinfo.value.status_code == 400
    assert excinfo.value.retryable is False


async def test_http_failure_carries_normalized_retry_metadata() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            json={"error": {"code": "rate_limit_exceeded", "message": "slow down"}},
            headers={"x-request-id": "req_compat", "retry-after": "2.5"},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAICompatProvider(base_url=BASE, model="m", client=client)
    with pytest.raises(ProviderError) as excinfo:
        await drain(provider, request())

    error = excinfo.value
    assert error.error_code == "rate_limit_exceeded"
    assert error.request_id == "req_compat"
    assert error.retry_after == 2.5


@pytest.mark.parametrize("status", [429, 500, 503])
async def test_server_and_rate_limit_errors_are_retryable(status: int) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text="nope")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAICompatProvider(base_url=BASE, model="m", client=client)
    with pytest.raises(ProviderError) as excinfo:
        await drain(provider, request())
    assert excinfo.value.retryable is True


async def test_connection_failure_explains_how_to_start_docker() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = DockerModelsProvider(model="m", client=client)
    with pytest.raises(ProviderError) as excinfo:
        await drain(provider, request())
    message = str(excinfo.value)
    assert "docker desktop enable model-runner" in message
    assert excinfo.value.retryable is True


async def test_api_key_never_appears_in_an_error_body() -> None:
    secret = "sk-super-secret-key-value-123456"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text=f"bad key {secret}")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAICompatProvider(base_url=BASE, model="m", api_key=secret, client=client)
    with pytest.raises(ProviderError) as excinfo:
        await drain(provider, request())
    assert secret not in str(excinfo.value)
    assert "<redacted" in str(excinfo.value)


def test_repr_redacts_the_api_key() -> None:
    secret = "sk-super-secret-key-value-123456"
    provider = OpenAICompatProvider(base_url=BASE, model="m", api_key=secret)
    assert secret not in repr(provider)


def test_missing_base_url_fails_at_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    with pytest.raises(LogposeError, match="OPENAI_BASE_URL"):
        OpenAICompatProvider(model="m")


# ---------------------------------------------------------------------------
# auth wiring
# ---------------------------------------------------------------------------


async def test_bearer_header_is_sent_when_a_key_is_configured() -> None:
    handler = Recorder(sse(finish_chunk("stop")))
    provider = make_provider(handler, api_key="sk-abc")
    await drain(provider, request())
    assert handler.requests[-1].headers["authorization"] == "Bearer sk-abc"


async def test_docker_ignores_an_ambient_openai_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A developer's OpenAI key must never be shipped to a server on localhost."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-leak")
    handler = Recorder(sse(finish_chunk("stop")))
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = DockerModelsProvider(model="m", client=client)
    await drain(provider, request())
    assert "authorization" not in handler.requests[-1].headers


async def test_generic_provider_does_read_openai_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-key")
    handler = Recorder(sse(finish_chunk("stop")))
    provider = make_provider(handler)
    await drain(provider, request())
    assert handler.requests[-1].headers["authorization"] == "Bearer sk-env-key"


# ---------------------------------------------------------------------------
# docker defaults and registration
# ---------------------------------------------------------------------------


def test_docker_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DOCKER_MODEL_RUNNER_URL", raising=False)
    monkeypatch.delenv("DOCKER_MODEL", raising=False)
    provider = DockerModelsProvider()
    assert provider.base_url == DEFAULT_DOCKER_BASE_URL
    assert provider.model_default == AUTO_MODEL
    assert provider.name == "docker"


def test_docker_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOCKER_MODEL_RUNNER_URL", "http://elsewhere:9000/engines/v1")
    monkeypatch.setenv("DOCKER_MODEL", "qwen3")
    provider = DockerModelsProvider()
    assert provider.base_url == "http://elsewhere:9000/engines/v1"
    assert provider.model_default == "qwen3"


def test_providers_are_registered_under_their_names() -> None:
    for name in ("docker", "docker-models", "openai-compat"):
        assert name in known_providers()
    assert isinstance(resolve("docker", model="m"), DockerModelsProvider)


def test_importing_logpose_does_not_import_httpx() -> None:
    """The provider module is lazy; httpx must not ride along with `import logpose`."""
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-c", "import sys, logpose; print('httpx' in sys.modules)"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert out.stdout.strip() == "False"


# ---------------------------------------------------------------------------
# end-to-end through the Agent
# ---------------------------------------------------------------------------


@tool
def add(a: int, b: int) -> str:
    """Add two numbers.

    Args:
        a: First number.
        b: Second number.
    """
    return str(a + b)


async def test_agent_drives_a_full_tool_round_trip() -> None:
    """The whole stack: loop -> provider -> wire -> tool -> wire -> answer."""
    turn1 = sse(
        delta_chunk(reasoning_content="need to add"),
        delta_chunk(
            tool_calls=[
                {"index": 0, "id": "c1", "function": {"name": "add", "arguments": '{"a":2,"b":3}'}}
            ]
        ),
        finish_chunk("tool_calls"),
        usage_chunk(50, 10),
    )
    turn2 = sse(delta_chunk(content="The answer is 5."), finish_chunk("stop"), usage_chunk(70, 6))
    handler = Recorder(turn1, turn2)
    provider = make_provider(handler)

    agent = Agent(provider, tools=[add], system="be terse")
    result = await agent.run("what is 2+3?")

    assert result.text == "The answer is 5."
    assert result.stop_reason == "end_turn"
    assert result.iterations == 2
    assert result.usage.input_tokens == 120
    assert result.usage.output_tokens == 16

    # The second request must carry the assistant tool_call and its tool result.
    second = handler.body(1)
    roles = [m["role"] for m in second["messages"]]
    assert roles == ["system", "user", "assistant", "tool"]
    assert second["messages"][2]["tool_calls"][0]["function"]["name"] == "add"
    assert second["messages"][3] == {"role": "tool", "tool_call_id": "c1", "content": "5"}
    # Reasoning stays in logpose history but is never re-sent.
    assert "need to add" not in json.dumps(second)


async def test_agent_reports_a_failed_tool_back_to_the_model() -> None:
    @tool
    def explode(x: int) -> str:
        """Always fails.

        Args:
            x: ignored.
        """
        raise RuntimeError("tool blew up")

    turn1 = sse(
        delta_chunk(
            tool_calls=[
                {"index": 0, "id": "c1", "function": {"name": "explode", "arguments": '{"x":1}'}}
            ]
        ),
        finish_chunk("tool_calls"),
    )
    turn2 = sse(delta_chunk(content="Sorry, that failed."), finish_chunk("stop"))
    handler = Recorder(turn1, turn2)
    agent = Agent(make_provider(handler), tools=[explode])

    result = await agent.run("go")
    assert result.stop_reason == "end_turn"
    tool_message = handler.body(1)["messages"][-1]
    assert tool_message["role"] == "tool"
    assert tool_message["content"].startswith("ERROR: ")
