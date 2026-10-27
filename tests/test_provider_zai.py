"""Coding Plan requests preserve reasoning across the actual agent/tool loop."""

import json

import httpx
import pytest

from logpose import Agent, provider_status, resolve, tool
from logpose.errors import AuthError, LogposeError
from tests.test_provider_openai_compat import delta_chunk, finish_chunk, sse


@pytest.mark.asyncio
async def test_coding_tool_loop_preserves_reasoning_and_uses_own_credentials(monkeypatch):
    monkeypatch.setenv("ZAI_KEY", "zai-test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "wrong-key")
    bodies = []

    def handler(request):
        assert str(request.url) == "https://api.z.ai/api/coding/paas/v4/chat/completions"
        assert request.headers["authorization"] == "Bearer zai-test-key"
        bodies.append(json.loads(request.content))
        if len(bodies) == 1:
            return httpx.Response(
                200,
                content=sse(
                    delta_chunk(reasoning_content="First\n"),
                    delta_chunk(reasoning_content="then verify."),
                    delta_chunk(
                        tool_calls=[
                            {
                                "index": 0,
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "get_fact", "arguments": "{}"},
                            }
                        ]
                    ),
                    finish_chunk("tool_calls"),
                ),
            )
        return httpx.Response(
            200, content=sse(delta_chunk(content="Verified."), finish_chunk("stop"))
        )

    @tool
    def get_fact() -> str:
        """Return a deterministic fact."""
        return "42"

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = resolve("zai", client=client)
        result = await Agent(provider, tools=[get_fact]).run("Verify the fact")
    assert result.text == "Verified."
    assert bodies[0]["model"] == "glm-5.3-flash"
    assert bodies[0]["reasoning_effort"] == "low"
    assert bodies[0]["thinking"] == {"type": "enabled", "clear_thinking": False}
    assert bodies[1]["messages"][1]["reasoning_content"] == "First\nthen verify."
    assert bodies[1]["messages"][2] == {"role": "tool", "tool_call_id": "call-1", "content": "42"}
    assert "zai-test-key" not in repr(provider)


@pytest.mark.parametrize("effort", ["medium", "xhigh", "", "invalid"])
def test_unsupported_effort_is_not_silently_mapped(effort):
    with pytest.raises(LogposeError, match="low.*high.*max"):
        resolve("zai", reasoning_effort=effort)


@pytest.mark.asyncio
async def test_readiness_and_auth_never_fall_back_to_openai_key(monkeypatch):
    monkeypatch.delenv("ZAI_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-key")
    (status,) = await provider_status(["zai"])
    assert not status.ready and "ZAI_KEY" in status.detail
    provider = resolve("zai")
    with pytest.raises(AuthError, match="ZAI_KEY"):
        await provider.list_models()
    monkeypatch.setenv("ZAI_KEY", "zai-test-key")
    (status,) = await provider_status(["zai"])
    assert status.ready and status.credential == "api_key"


@pytest.mark.asyncio
async def test_request_effort_override_is_validated_before_dispatch():
    from logpose.providers.base import CompletionRequest

    provider = resolve("zai", api_key="test-key")
    request = CompletionRequest(
        messages=[], model="glm-5.3-flash", max_tokens=100, extra={"reasoning_effort": "medium"}
    )
    with pytest.raises(LogposeError, match="low.*high.*max"):
        provider._build_body(request, request.model)
