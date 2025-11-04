"""Integration-style tests for Anthropic formatting helpers."""

import pytest

from proxy.main import AnthropicChatRequest, AnthropicMessage, complete_anthropic_response


@pytest.mark.asyncio
async def test_complete_anthropic_response_emits_thinking_block(stub_tabby):
    stub_tabby.queue_chat_response(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Tracing options</think>\nHere is the plan.",
                    }
                }
            ]
        }
    )

    request = AnthropicChatRequest(
        model="minimax-m2",
        messages=[AnthropicMessage(role="user", content="Draft a plan.")],
    )

    response = await complete_anthropic_response(request, session_id=None)

    blocks = response["content"]
    assert blocks[0]["type"] == "thinking"
    assert blocks[0]["thinking"].strip() == "Tracing options"
    assert blocks[1]["type"] == "text"
    assert blocks[1]["text"].strip() == "Here is the plan."
