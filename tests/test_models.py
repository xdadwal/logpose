"""Tests for the provider-neutral message model."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from logpose.messages import (
    CONTENT_BLOCK_ADAPTER,
    Message,
    RedactedThinkingBlock,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
)


def _every_block_message() -> Message:
    """Build an assistant message containing one of every block type."""
    return Message(
        role="assistant",
        content=[
            ThinkingBlock(thinking="let me check the weather", signature="sig-abc123"),
            RedactedThinkingBlock(data="ZW5jcnlwdGVk"),
            TextBlock(text="Looking that up."),
            ToolUseBlock(id="toolu_1", name="get_weather", input={"location": "Pune", "unit": "c"}),
            ToolResultBlock(tool_use_id="toolu_1", content="31C and clear", is_error=False),
        ],
    )


class TestContentBlockRoundTrip:
    """dump -> load must be lossless for every block type."""

    def test_round_trip_preserves_every_block(self) -> None:
        original = _every_block_message()

        restored = Message.model_validate(original.model_dump())

        assert restored == original
        assert [type(b) for b in restored.content] == [
            ThinkingBlock,
            RedactedThinkingBlock,
            TextBlock,
            ToolUseBlock,
            ToolResultBlock,
        ]

    def test_round_trip_preserves_thinking_signature(self) -> None:
        # The Anthropic API rejects turns whose thinking blocks were altered,
        # so the signature must survive a dump/load cycle byte for byte.
        original = _every_block_message()

        restored = Message.model_validate(original.model_dump())

        thinking = restored.content[0]
        assert isinstance(thinking, ThinkingBlock)
        assert thinking.signature == "sig-abc123"
        assert thinking.thinking == "let me check the weather"

    def test_round_trip_preserves_redacted_thinking_data(self) -> None:
        original = _every_block_message()

        restored = Message.model_validate(original.model_dump())

        redacted = restored.content[1]
        assert isinstance(redacted, RedactedThinkingBlock)
        assert redacted.data == "ZW5jcnlwdGVk"

    def test_round_trip_through_json(self) -> None:
        original = _every_block_message()

        restored = Message.model_validate_json(original.model_dump_json())

        assert restored == original

    def test_thinking_signature_defaults_to_none(self) -> None:
        block = ThinkingBlock(thinking="hmm")

        assert block.signature is None
        assert ThinkingBlock.model_validate(block.model_dump()) == block

    def test_tool_use_input_survives_nested_structures(self) -> None:
        block = ToolUseBlock(
            id="toolu_2",
            name="query",
            input={"filters": [{"field": "city", "in": ["Pune", "Goa"]}], "limit": 10},
        )

        restored = ToolUseBlock.model_validate(block.model_dump())

        assert restored.input == block.input

    @pytest.mark.parametrize(
        "block",
        [
            TextBlock(text="hi"),
            ThinkingBlock(thinking="reasoning", signature="sig"),
            RedactedThinkingBlock(data="cipher"),
            ToolUseBlock(id="t1", name="tool", input={}),
            ToolResultBlock(tool_use_id="t1", content="ok"),
        ],
    )
    def test_discriminated_union_selects_the_right_class(self, block: BaseModel) -> None:
        dumped = block.model_dump()

        restored = CONTENT_BLOCK_ADAPTER.validate_python(dumped)

        assert type(restored) is type(block)
        assert restored == block

    def test_unknown_block_type_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CONTENT_BLOCK_ADAPTER.validate_python({"type": "image", "data": "..."})

    def test_tool_result_is_error_defaults_to_false(self) -> None:
        block = ToolResultBlock(tool_use_id="t1", content="ok")

        assert block.is_error is False
        assert ToolResultBlock.model_validate(block.model_dump()).is_error is False


class TestMessageConstructors:
    """Convenience constructors and the text property."""

    def test_user_builds_a_single_text_block(self) -> None:
        msg = Message.user("What's the weather in Pune?")

        assert msg.role == "user"
        assert msg.content == [TextBlock(text="What's the weather in Pune?")]

    def test_assistant_text_builds_a_single_text_block(self) -> None:
        msg = Message.assistant_text("It's 31C.")

        assert msg.role == "assistant"
        assert msg.content == [TextBlock(text="It's 31C.")]

    def test_text_concatenates_only_text_blocks(self) -> None:
        msg = Message(
            role="assistant",
            content=[
                ThinkingBlock(thinking="ignored", signature="sig"),
                TextBlock(text="Hello, "),
                ToolUseBlock(id="t1", name="noop", input={}),
                TextBlock(text="world."),
                RedactedThinkingBlock(data="ignored"),
            ],
        )

        assert msg.text == "Hello, world."

    def test_text_is_empty_without_text_blocks(self) -> None:
        msg = Message(role="assistant", content=[ToolUseBlock(id="t1", name="noop", input={})])

        assert msg.text == ""

    def test_text_of_convenience_constructors(self) -> None:
        assert Message.user("hi").text == "hi"
        assert Message.assistant_text("yo").text == "yo"

    def test_invalid_role_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Message(role="system", content=[])  # type: ignore[arg-type]


class TestUsage:
    """Token accounting arithmetic."""

    def test_defaults_are_zero(self) -> None:
        usage = Usage()

        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.cache_read_input_tokens == 0
        assert usage.cache_creation_input_tokens == 0

    def test_add_sums_every_field(self) -> None:
        first = Usage(
            input_tokens=10,
            output_tokens=3,
            cache_read_input_tokens=100,
            cache_creation_input_tokens=7,
        )
        second = Usage(
            input_tokens=5,
            output_tokens=11,
            cache_read_input_tokens=1,
            cache_creation_input_tokens=2,
        )

        total = first + second

        assert total == Usage(
            input_tokens=15,
            output_tokens=14,
            cache_read_input_tokens=101,
            cache_creation_input_tokens=9,
        )

    def test_add_does_not_mutate_operands(self) -> None:
        first = Usage(input_tokens=1)
        second = Usage(input_tokens=2)

        first + second

        assert first.input_tokens == 1
        assert second.input_tokens == 2

    def test_add_is_usable_with_sum_across_turns(self) -> None:
        turns = [Usage(input_tokens=i, output_tokens=1) for i in range(4)]

        total = sum(turns, Usage())

        assert total == Usage(input_tokens=6, output_tokens=4)

    def test_builtin_sum_without_explicit_start(self) -> None:
        turns = [Usage(output_tokens=2), Usage(output_tokens=3)]

        assert sum(turns) == Usage(output_tokens=5)  # type: ignore[call-overload]

    def test_add_with_non_usage_is_not_implemented(self) -> None:
        with pytest.raises(TypeError):
            Usage() + 1  # type: ignore[operator]

    def test_round_trip(self) -> None:
        usage = Usage(input_tokens=1, output_tokens=2, cache_read_input_tokens=3)

        assert Usage.model_validate(usage.model_dump()) == usage
