"""Unit tests for history normalization."""

from proxy.main import normalize_openai_history
from parsers.tools import tool_calls_to_minimax_xml


def test_normalize_openai_history_keeps_tool_calls_and_strips_xml() -> None:
    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "lookup", "arguments": "{\"id\": 7}"},
        }
    ]
    xml_block = tool_calls_to_minimax_xml(tool_calls)
    messages = [
        {
            "role": "assistant",
            "content": f"Planning</think>\n{xml_block}",
            "tool_calls": tool_calls,
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
    ]

    normalized = normalize_openai_history(messages)
    assistant = normalized[0]

    assert assistant["tool_calls"][0]["function"]["arguments"] == {"id": 7}
    assert "<minimax:tool_call>" not in (assistant.get("content") or "")
