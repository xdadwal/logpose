"""SSE and item builders for the Responses API suites.

Not named ``test_*`` so pytest does not collect it, matching
:mod:`tests.fake_provider`. Shared by :mod:`tests.test_responses_base`,
:mod:`tests.test_provider_codex`, and :mod:`tests.test_provider_openai`.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from logpose import Message
from logpose.providers.base import CompletionDone, CompletionRequest

OAUTH_TOKEN = "oauth-" + "S3CR3T" * 20
API_KEY = "sk-proj-" + "K3YV4L" * 20
ACCOUNT_ID = "acc_test"


def sse(*events: dict[str, Any], done: bool = True) -> bytes:
    """Render events as a Responses SSE body, ``event:`` lines included."""
    body = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)
    if done:
        body += "data: [DONE]\n\n"
    return body.encode()


def text_delta(text: str) -> dict[str, Any]:
    """A visible-output delta."""
    return {"type": "response.output_text.delta", "delta": text}


def summary_delta(text: str) -> dict[str, Any]:
    """A reasoning-summary delta."""
    return {"type": "response.reasoning_summary_text.delta", "delta": text}


def reasoning_delta(text: str) -> dict[str, Any]:
    """A visible-reasoning delta, emitted by some models instead of a summary."""
    return {"type": "response.reasoning_text.delta", "delta": text}


def item_done(item: dict[str, Any], index: int = 0) -> dict[str, Any]:
    """A completed output item."""
    return {"type": "response.output_item.done", "output_index": index, "item": item}


def item_added(item: dict[str, Any], index: int = 0) -> dict[str, Any]:
    """A freshly-opened output item, typically only partly populated."""
    return {"type": "response.output_item.added", "output_index": index, "item": item}


def completed(
    *,
    output: list[dict[str, Any]] | None = None,
    usage: dict[str, Any] | None = None,
    status: str = "completed",
) -> dict[str, Any]:
    """The terminal response event."""
    response: dict[str, Any] = {"id": "resp_1", "status": status}
    if output is not None:
        response["output"] = output
    if usage is not None:
        response["usage"] = usage
    return {"type": "response.completed", "response": response}


def incomplete(reason: str, **response_fields: Any) -> dict[str, Any]:
    """A terminal response that ran out of room."""
    response: dict[str, Any] = {
        "id": "resp_1",
        "status": "incomplete",
        "incomplete_details": {"reason": reason},
    }
    response.update(response_fields)
    return {"type": "response.incomplete", "response": response}


def message_item(text: str = "hello") -> dict[str, Any]:
    """An assistant message output item."""
    return {
        "type": "message",
        "id": "msg_1",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text}],
    }


def refusal_item(text: str = "I can't help with that.") -> dict[str, Any]:
    """An assistant message whose content part is a refusal."""
    return {
        "type": "message",
        "id": "msg_1",
        "role": "assistant",
        "content": [{"type": "refusal", "refusal": text}],
    }


def function_call_item(
    *,
    name: str = "get_weather",
    arguments: str = '{"city": "Pune"}',
    call_id: str = "call_1",
) -> dict[str, Any]:
    """A function-call output item, complete with both ids the API returns."""
    return {
        "type": "function_call",
        "id": "fc_1",
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
    }


def reasoning_item(
    *, item_id: str = "rs_1", summary: str | None = "Thinking."
) -> dict[str, Any]:
    """A reasoning output item, carrying the blob that must be resent verbatim."""
    parts = [{"type": "summary_text", "text": summary}] if summary is not None else []
    return {
        "type": "reasoning",
        "id": item_id,
        "summary": parts,
        "encrypted_content": "gAAAAABencrypted",
    }


class Recorder:
    """MockTransport handler that records requests and replays scripted bodies."""

    def __init__(self, *bodies: bytes, status: int = 200, text: str | None = None) -> None:
        self.bodies = list(bodies)
        self.status = status
        self.text = text
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.text is not None:
            return httpx.Response(self.status, text=self.text)
        body = self.bodies.pop(0) if self.bodies else sse(completed(output=[message_item()]))
        return httpx.Response(self.status, content=body)

    @property
    def last_body(self) -> dict[str, Any]:
        """The most recent request body."""
        return json.loads(self.requests[-1].content)

    @property
    def last_headers(self) -> httpx.Headers:
        """The most recent request's headers."""
        return self.requests[-1].headers

    @property
    def last_url(self) -> str:
        """The most recent request's URL."""
        return str(self.requests[-1].url)

    def body(self, index: int) -> dict[str, Any]:
        """The nth request body."""
        return json.loads(self.requests[index].content)


def mock_client(handler: Any) -> httpx.AsyncClient:
    """Build an AsyncClient whose transport never touches the network."""
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def request(**kwargs: Any) -> CompletionRequest:
    """Build a CompletionRequest with sane defaults."""
    kwargs.setdefault("messages", [Message.user("hi")])
    kwargs.setdefault("model", "gpt-test")
    kwargs.setdefault("max_tokens", 128)
    return CompletionRequest(**kwargs)


async def drain(provider: Any, req: CompletionRequest | None = None) -> list[Any]:
    """Collect every event a provider stream yields."""
    target = req if req is not None else request()
    return [event async for event in provider.stream(target)]


def final_message(events: list[Any]) -> Message:
    """The assembled assistant turn from a drained stream."""
    done = events[-1]
    assert isinstance(done, CompletionDone)
    return done.message
