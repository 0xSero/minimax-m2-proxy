"""Integration-style tests for OpenAI streaming."""

import json

import pytest

from proxy.main import OpenAIChatRequest, OpenAIMessage, stream_openai_response


def decode_sse_chunk(chunk: str) -> dict:
    assert chunk.startswith("data: ")
    payload = chunk[len("data: ") :].strip()
    if payload == "[DONE]":
        return {"done": True}
    return json.loads(payload)


@pytest.mark.asyncio
async def test_stream_openai_response_emits_reasoning_details(stub_tabby):
    stub_tabby.queue_stream(
        [
            {
                "choices": [
                    {
                        "delta": {
                            "content": "Analyzing options</think>\nSuggestion: use tool.",
                        },
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

    request = OpenAIChatRequest(
        model="minimax-m2",
        messages=[OpenAIMessage(role="user", content="How should I proceed?")],
        extra_body={"reasoning_split": True},
    )

    stream = stream_openai_response(request, session_id=None)
    chunks = []
    async for chunk in stream:
        chunks.append(chunk)

    assert chunks[-1].strip() == "data: [DONE]"

    decoded = [decode_sse_chunk(chunk) for chunk in chunks[:-1]]
    first_delta = decoded[0]["choices"][0]["delta"]

    assert first_delta["reasoning_details"][0]["text"].strip() == "Analyzing options"
    assert first_delta["content"].strip() == "Suggestion: use tool."
