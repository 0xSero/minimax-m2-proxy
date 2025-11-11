"""Tests for Anthropic↔OpenAI message conversion helpers."""

import json

from proxy.models import (
    AnthropicContentBlock,
    AnthropicMessage,
    anthropic_messages_to_openai,
)


def test_anthropic_messages_to_openai_serializes_tool_arguments() -> None:
    message = AnthropicMessage(
        role="assistant",
        content=[
            AnthropicContentBlock(type="text", text="Calling a tool"),
            AnthropicContentBlock(
                type="tool_use",
                id="toolu_123",
                name="get_weather",
                input={"location": "San Francisco", "unit": "celsius"},
            ),
        ],
    )

    converted = anthropic_messages_to_openai([message])
    tool_call = converted[0]["tool_calls"][0]["function"]

    assert isinstance(tool_call["arguments"], str)
    assert json.loads(tool_call["arguments"]) == {
        "location": "San Francisco",
        "unit": "celsius",
    }


def test_anthropic_messages_to_openai_preserves_json_string_arguments() -> None:
    # Note: Anthropic API spec requires input to be a dict, not a JSON string
    # This test verifies that dict input is properly serialized to JSON string
    input_dict = {"query": "pizza"}
    message = AnthropicMessage(
        role="assistant",
        content=[
            AnthropicContentBlock(
                type="tool_use",
                id="toolu_456",
                name="search",
                input=input_dict,
            )
        ],
    )

    converted = anthropic_messages_to_openai([message])
    tool_call = converted[0]["tool_calls"][0]["function"]

    # Arguments should be a JSON string in OpenAI format
    assert isinstance(tool_call["arguments"], str)
    assert json.loads(tool_call["arguments"]) == input_dict


def test_anthropic_messages_to_openai_includes_thinking_blocks() -> None:
    message = AnthropicMessage(
        role="assistant",
        content=[
            AnthropicContentBlock(type="thinking", thinking="Consider options"),
            AnthropicContentBlock(type="text", text="Final answer."),
        ],
    )

    converted = anthropic_messages_to_openai([message])
    assert converted[0]["content"] == "<think>Consider options</think>\nFinal answer."


def test_anthropic_messages_to_openai_handles_tool_only_messages() -> None:
    message = AnthropicMessage(
        role="assistant",
        content=[
            AnthropicContentBlock(
                type="tool_use",
                id="call_123",
                name="lookup",
                input={"query": "hi"},
            )
        ],
    )

    converted = anthropic_messages_to_openai([message])
    assert converted[0]["content"] == ""
    assert converted[0]["tool_calls"][0]["function"]["name"] == "lookup"


def test_anthropic_messages_to_openai_flattens_tool_result_blocks() -> None:
    message = AnthropicMessage(
        role="user",
        content=[
            AnthropicContentBlock(
                type="tool_result",
                tool_use_id="call_123",
                content=[{"type": "text", "text": '{"temp": "24C"}'}],
            )
        ],
    )

    converted = anthropic_messages_to_openai([message])
    assert converted[0]["content"] == 'Tool result for call_123: {"temp": "24C"}'
