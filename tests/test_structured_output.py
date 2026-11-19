"""End-to-end tests for structured output through the agent loop."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from logpose.agent import Agent, Conversation
from logpose.errors import (
    LogposeError,
    MaxIterationsError,
    OutputRefusalError,
    OutputValidationError,
)
from logpose.events import RunEnd
from logpose.messages import TextBlock
from logpose.output import response_format_for
from logpose.tools import tool
from tests.fake_provider import FakeProvider, ScriptedTurn, tool_call

VALID = '{"city":"Pune","temp_c":31}'
FENCED = '```json\n{"city":"Pune","temp_c":31}\n```'


class Answer(BaseModel):
    city: str
    temp_c: float


@tool
def lookup(city: str) -> str:
    """Look up the weather for a city."""
    return "31C"


def _agent(provider: FakeProvider, **kwargs: Any) -> Agent:
    return Agent(provider, model="test-model", **kwargs)


async def test_valid_answer_returns_typed_output() -> None:
    provider = FakeProvider([ScriptedTurn.text(VALID)])
    result = await _agent(provider).run("x", output_type=Answer)
    assert isinstance(result.output, Answer)
    assert result.output.city == "Pune"
    assert result.text == VALID
    assert provider.last_request.extra["response_format"] == response_format_for(Answer)


async def test_no_output_type_is_unchanged() -> None:
    provider = FakeProvider([ScriptedTurn.text(VALID)])
    result = await _agent(provider).run("x")
    assert "response_format" not in provider.last_request.extra
    assert result.output is None


async def test_agent_default_output_type_applies() -> None:
    provider = FakeProvider([ScriptedTurn.text(VALID)])
    result = await _agent(provider, output_type=Answer).run("x")
    assert isinstance(result.output, Answer)


async def test_per_call_none_disables_agent_default() -> None:
    provider = FakeProvider([ScriptedTurn.text(VALID)])
    result = await _agent(provider, output_type=Answer).run("x", output_type=None)
    assert "response_format" not in provider.last_request.extra
    assert result.output is None


async def test_correction_turn_fixes_invalid_answer() -> None:
    provider = FakeProvider([ScriptedTurn.text('{"city":"Pune"}'), ScriptedTurn.text(VALID)])
    result = await _agent(provider).run("x", output_type=Answer)
    assert result.output is not None
    assert result.output.temp_c == 31
    assert provider.call_count == 2
    last = provider.requests[1].messages[-1]
    assert last.role == "user"
    assert len(last.content) == 1 and isinstance(last.content[0], TextBlock)
    assert "temp_c" in last.text
    assert "did not match the required JSON schema" in last.text
    assert provider.requests[1].extra["response_format"] == response_format_for(Answer)


async def test_correction_message_does_not_echo_values() -> None:
    provider = FakeProvider(
        [ScriptedTurn.text('{"city":"SECRET-VALUE","temp_c":"hot"}'), ScriptedTurn.text(VALID)]
    )
    await _agent(provider).run("x", output_type=Answer)
    assert "SECRET-VALUE" not in provider.requests[1].messages[-1].text
    assert "hot" not in provider.requests[1].messages[-1].text


async def test_successful_correction_is_removed_from_conversation() -> None:
    conv = Conversation()
    provider = FakeProvider([ScriptedTurn.text('{"city":"Pune"}'), ScriptedTurn.text(VALID)])
    result = await _agent(provider).run("x", conversation=conv, output_type=Answer)
    assert [m.role for m in conv.messages] == ["user", "assistant"]
    assert conv.messages[-1].text == VALID
    assert [m.role for m in result.messages] == ["user", "assistant"]


async def test_failed_correction_keeps_full_history() -> None:
    conv = Conversation()
    provider = FakeProvider([ScriptedTurn.text('{"city":"Pune"}')], repeat_last=True)
    with pytest.raises(OutputValidationError) as info:
        await _agent(provider).run("x", conversation=conv, output_type=Answer)
    roles = ["user", "assistant", "user", "assistant"]
    assert [m.role for m in conv.messages] == roles
    assert [m.role for m in info.value.messages] == roles


async def test_output_retries_zero_fails_immediately() -> None:
    provider = FakeProvider([ScriptedTurn.text('{"city":"Pune"}')])
    with pytest.raises(OutputValidationError) as info:
        await _agent(provider).run("x", output_type=Answer, output_retries=0)
    exc = info.value
    assert provider.call_count == 1
    assert "temp_c" in str(exc.errors)
    assert "the server may not enforce strict json_schema" in str(exc)
    assert exc.reason == "invalid"
    assert exc.raw_text == '{"city":"Pune"}'
    assert exc.output_type is Answer


async def test_agent_level_output_retries_zero() -> None:
    provider = FakeProvider([ScriptedTurn.text('{"city":"Pune"}')])
    with pytest.raises(OutputValidationError):
        await _agent(provider, output_type=Answer, output_retries=0).run("x")
    assert provider.call_count == 1


async def test_fenced_json_gets_a_correction_then_fails() -> None:
    provider = FakeProvider([ScriptedTurn.text(FENCED), ScriptedTurn.text(FENCED)])
    with pytest.raises(OutputValidationError):
        await _agent(provider).run("x", output_type=Answer)
    assert provider.call_count == 2


async def test_empty_final_text_is_corrected() -> None:
    provider = FakeProvider([ScriptedTurn.text(""), ScriptedTurn.text(VALID)])
    result = await _agent(provider).run("x", output_type=Answer)
    assert isinstance(result.output, Answer)
    assert provider.call_count == 2


async def test_refusal_raises_without_correction() -> None:
    provider = FakeProvider([ScriptedTurn.text("", refusal="no")])
    with pytest.raises(OutputRefusalError) as info:
        await _agent(provider).run("x", output_type=Answer)
    assert info.value.refusal == "no"
    assert provider.call_count == 1


async def test_refusal_stop_reason_raises() -> None:
    provider = FakeProvider([ScriptedTurn.text("", stop_reason="refusal")])
    with pytest.raises(OutputRefusalError) as info:
        await _agent(provider).run("x", output_type=Answer)
    assert info.value.refusal == ""
    assert provider.call_count == 1


async def test_refusal_without_output_type_is_ignored() -> None:
    provider = FakeProvider([ScriptedTurn.text("sorry", refusal="no")])
    result = await _agent(provider).run("x")
    assert result.text == "sorry"
    assert result.output is None


async def test_cut_off_answer_raises_max_tokens() -> None:
    provider = FakeProvider([ScriptedTurn.text('{"city":"Pu', stop_reason="max_tokens")])
    with pytest.raises(OutputValidationError) as info:
        await _agent(provider).run("x", output_type=Answer)
    assert info.value.reason == "max_tokens"
    assert "raise max_tokens" in str(info.value)
    assert provider.call_count == 1


async def test_tools_then_structured_answer() -> None:
    provider = FakeProvider(
        [
            ScriptedTurn.tool_use(tool_call("lookup", {"city": "Pune"}, id="t1")),
            ScriptedTurn.text(VALID),
        ]
    )
    result = await _agent(provider, tools=[lookup]).run("x", output_type=Answer)
    assert isinstance(result.output, Answer)
    assert all(r.extra["response_format"] == response_format_for(Answer) for r in provider.requests)


async def test_correction_hitting_max_iterations_raises_max_iterations() -> None:
    provider = FakeProvider([ScriptedTurn.text('{"city":"Pune"}')], repeat_last=True)
    with pytest.raises(MaxIterationsError):
        await _agent(provider, max_iterations=1).run("x", output_type=Answer)


async def test_extra_response_format_conflict() -> None:
    provider = FakeProvider([ScriptedTurn.text(VALID)])
    agent = _agent(provider, extra={"response_format": {}}, output_type=Answer)
    with pytest.raises(LogposeError, match="response_format"):
        await agent.run("x")
    assert provider.call_count == 0


async def test_extra_response_format_without_output_type_is_passed_through() -> None:
    provider = FakeProvider([ScriptedTurn.text(VALID)])
    await _agent(provider, extra={"response_format": {"type": "text"}}).run("x")
    assert provider.last_request.extra["response_format"] == {"type": "text"}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"output_type": dict},
        {"output_type": Answer(city="a", temp_c=1)},
        {"output_retries": -1},
        {"output_retries": 1.5},
        {"output_retries": True},
    ],
)
async def test_invalid_output_type_and_retries_rejected(kwargs: dict[str, Any]) -> None:
    provider = FakeProvider([ScriptedTurn.text(VALID)])
    with pytest.raises(LogposeError):
        _agent(provider, **kwargs)
    agent = _agent(provider)
    with pytest.raises(LogposeError):
        await agent.run("x", **kwargs)
    with pytest.raises(LogposeError):
        agent.stream("x", **kwargs)
    assert provider.call_count == 0


async def test_stream_run_end_carries_output() -> None:
    provider = FakeProvider([ScriptedTurn.text(VALID)])
    events = [e async for e in _agent(provider).stream("x", output_type=Answer)]
    ends = [e for e in events if isinstance(e, RunEnd)]
    assert len(ends) == 1
    assert isinstance(ends[0].result.output, Answer)
