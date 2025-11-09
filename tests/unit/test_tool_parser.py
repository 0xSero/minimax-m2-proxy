"""Tests for the MiniMax tool call parser."""

import json

from parsers.tools import parse_tool_calls


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }
]


def test_parse_tool_call_handles_spaced_parameter_attributes() -> None:
    xml = """
    <minimax:tool_call>
      <invoke name="web_search">
        <parameter   name =  "query"  >
          ramen near Shibuya
        </parameter>
      </invoke>
    </minimax:tool_call>
    """

    result = parse_tool_calls(xml, TOOLS)
    assert result["tools_called"] is True
    args = json.loads(result["tool_calls"][0]["function"]["arguments"])
    assert args["query"].strip() == "ramen near Shibuya"


def test_parse_tool_call_handles_spaced_invoke_attributes() -> None:
    xml = """
    <minimax:tool_call>
      <invoke    name =  "web_search"   >
        <parameter name="query">tokyo weather</parameter>
      </invoke>
    </minimax:tool_call>
    """

    result = parse_tool_calls(xml, TOOLS)
    assert result["tools_called"] is True
    args = json.loads(result["tool_calls"][0]["function"]["arguments"])
    assert args["query"] == "tokyo weather"
