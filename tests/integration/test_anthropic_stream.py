"""Integration-style tests for Anthropic streaming."""

import json

import pytest

from parsers.tools import tool_calls_to_minimax_xml
from proxy.main import AnthropicChatRequest, AnthropicMessage, stream_anthropic_response


def decode_event(chunk: str) -> dict:
    lines = [line for line in chunk.splitlines() if line]
    data_line = next(line for line in lines if line.startswith("data: "))
    return json.loads(data_line[len("data: ") :])


@pytest.mark.asyncio
async def test_stream_anthropic_response_emits_tool_use_from_xml(stub_tabby):
    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "bash",
                "arguments": json.dumps({"command": "ls"}, ensure_ascii=False),
            },
        }
    ]
    xml_block = tool_calls_to_minimax_xml(tool_calls)
    content = f"Planning</think>\n{xml_block}"

    stub_tabby.queue_stream(
        [
            {
                "choices": [
                    {
                        "delta": {"content": content},
                        "finish_reason": None,
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ]
            },
        ]
    )

    request = AnthropicChatRequest(
        model="minimax-m2",
        messages=[AnthropicMessage(role="user", content="List files")],
        stream=True,
        tools=[
            {
                "name": "bash",
                "description": "Run a shell command.",
                "input_schema": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            }
        ],
    )

    events = []
    async for chunk in stream_anthropic_response(request, session_id=None):
        if chunk.startswith("event: "):
            events.append(decode_event(chunk))

    tool_use_starts = [
        event
        for event in events
        if event["type"] == "content_block_start"
        and event["content_block"]["type"] == "tool_use"
    ]
    assert tool_use_starts
    assert tool_use_starts[0]["content_block"]["name"] == "bash"

    input_deltas = [
        event
        for event in events
        if event["type"] == "content_block_delta"
        and event["delta"]["type"] == "input_json_delta"
    ]
    assert any("\"command\"" in event["delta"]["partial_json"] for event in input_deltas)

    message_deltas = [event for event in events if event["type"] == "message_delta"]
    assert message_deltas[-1]["delta"]["stop_reason"] == "tool_use"
