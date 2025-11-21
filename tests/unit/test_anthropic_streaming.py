import json
import pytest

from proxy.main import stream_anthropic_response
from proxy.models import AnthropicChatRequest, AnthropicMessage


class DummyTabbyClient:
    async def extract_streaming_content(self, *args, **kwargs):
        yield {
            "choices": [
                {
                    "delta": {
                        "content": "<think>Reason through</think>\nVisible reply",
                        "reasoning_content": "",
                        "tool_calls": None,
                    },
                    "finish_reason": None,
                }
            ]
        }
        yield {
            "choices": [
                {
                    "delta": {
                        "content": (
                            "<minimax:tool_call><invoke name=\"test_tool\">"
                            "<parameter name=\"value\">42</parameter></invoke>"
                            "</minimax:tool_call>"
                        ),
                        "reasoning_content": "",
                        "tool_calls": None,
                    },
                    "finish_reason": "stop",
                }
            ]
        }


@pytest.mark.asyncio
async def test_stream_anthropic_response_uses_parser(monkeypatch):
    from proxy import main as proxy_main

    monkeypatch.setattr(proxy_main, "tabby_client", DummyTabbyClient())

    request = AnthropicChatRequest(
        model="minimax-m2",
        max_tokens=256,
        stream=True,
        messages=[AnthropicMessage(role="user", content="Hi")],
    )

    events = []
    async for event in stream_anthropic_response(request, session_id=None):
        events.append(event)

    thinking_events = [evt for evt in events if '"type": "thinking"' in evt]
    tool_events = [evt for evt in events if '"type": "tool_use"' in evt]

    assert thinking_events, "Expected thinking block events in Anthropic stream"
    assert tool_events, "Expected tool_use events parsed from MiniMax XML"

    for evt in tool_events:
        data_line = next((line for line in evt.splitlines() if line.startswith("data: ")), None)
        assert data_line, "Tool_use event missing data payload"
        payload = json.loads(data_line[len("data: ") :])
        content_block = payload.get("content_block", {})
        assert content_block.get("id") is not None, "tool_use id must not be null"
        assert content_block.get("name") is not None, "tool_use name must not be null"
