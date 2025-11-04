"""Integration-style tests for OpenAI completion helpers."""

import json

import pytest

from proxy.main import OpenAIChatRequest, OpenAIMessage, complete_openai_response


@pytest.mark.asyncio
async def test_complete_openai_response_with_reasoning_split(stub_tabby):
    stub_tabby.queue_chat_response(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Planning next steps</think>\nProvide summary.",
                    }
                }
            ]
        }
    )

    request = OpenAIChatRequest(
        model="minimax-m2",
        messages=[OpenAIMessage(role="user", content="Summarize this.")],
        extra_body={"reasoning_split": True},
    )

    result = await complete_openai_response(request, session_id=None)

    message = result["choices"][0]["message"]
    assert message["content"].strip() == "Provide summary."
    assert message["reasoning_details"][0]["text"].strip() == "Planning next steps"


@pytest.mark.asyncio
async def test_complete_openai_response_with_tool_call(stub_tabby):
    stub_tabby.queue_chat_response(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": (
                            "Checking data</think>\n"
                            "<minimax:tool_call>"
                            "<invoke name=\"lookup\"><parameter name=\"id\">7</parameter></invoke>"
                            "</minimax:tool_call>"
                        ),
                    }
                }
            ]
        }
    )

    request = OpenAIChatRequest(
        model="minimax-m2",
        messages=[OpenAIMessage(role="user", content="Fetch id=7.")],
    )

    result = await complete_openai_response(request, session_id=None)
    message = result["choices"][0]["message"]
    tool_call = message["tool_calls"][0]
    assert tool_call["function"]["name"] == "lookup"
    assert json.loads(tool_call["function"]["arguments"])["id"] == "7"
