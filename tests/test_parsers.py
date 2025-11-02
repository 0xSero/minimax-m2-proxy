"""Unit tests for MiniMax-M2 parsers"""

import pytest
import json
from parsers.tools import ToolCallParser
from parsers.streaming import SimpleStreamingParser
from parsers.reasoning import ensure_think_wrapped
from formatters.openai import OpenAIFormatter


class TestToolCallParser:
    """Test suite for ToolCallParser"""

    def setup_method(self):
        """Setup test fixtures"""
        self.parser = ToolCallParser()

    def test_parse_single_tool_call(self):
        """Test parsing a single tool call"""
        text = """<minimax:tool_call>
<invoke name="get_weather">
<parameter name="location">San Francisco</parameter>
<parameter name="unit">celsius</parameter>
</invoke>
</minimax:tool_call>"""

        result = self.parser.parse_tool_calls(text)

        assert result["tools_called"] is True
        assert len(result["tool_calls"]) == 1

        tool_call = result["tool_calls"][0]
        assert tool_call["type"] == "function"
        assert tool_call["function"]["name"] == "get_weather"

        args = json.loads(tool_call["function"]["arguments"])
        assert args["location"] == "San Francisco"
        assert args["unit"] == "celsius"

    def test_parse_multiple_tool_calls(self):
        """Test parsing multiple tool calls"""
        text = """<minimax:tool_call>
<invoke name="get_weather">
<parameter name="location">Tokyo</parameter>
</invoke>
<invoke name="get_time">
<parameter name="timezone">Asia/Tokyo</parameter>
</invoke>
</minimax:tool_call>"""

        result = self.parser.parse_tool_calls(text)

        assert result["tools_called"] is True
        assert len(result["tool_calls"]) == 2

        assert result["tool_calls"][0]["function"]["name"] == "get_weather"
        assert result["tool_calls"][1]["function"]["name"] == "get_time"

    def test_parse_with_content_before(self):
        """Test parsing with text content before tool call"""
        text = """Let me check the weather for you.

<minimax:tool_call>
<invoke name="get_weather">
<parameter name="location">London</parameter>
</invoke>
</minimax:tool_call>"""

        result = self.parser.parse_tool_calls(text)

        assert result["tools_called"] is True
        assert result["content"] == "Let me check the weather for you."
        assert len(result["tool_calls"]) == 1

    def test_parse_no_tool_calls(self):
        """Test parsing text without tool calls"""
        text = "This is just regular text with no tool calls."

        result = self.parser.parse_tool_calls(text)

        assert result["tools_called"] is False
        assert result["tool_calls"] == []
        assert result["content"] == text

    def test_think_preservation_when_opening_missing(self):
        """Ensure parser restores <think> tag when only </think> is present."""
        text = "Reasoning in progress</think>\n<minimax:tool_call><invoke name=\"foo\"></invoke></minimax:tool_call>"

        result = self.parser.parse_tool_calls(text)

        assert result["tools_called"] is True
        assert result["content"].startswith("<think>")
        assert result["content"].rstrip().endswith("</think>")

    def test_type_inference_integer(self):
        """Test integer type inference"""
        text = """<minimax:tool_call>
<invoke name="test_func">
<parameter name="count">42</parameter>
</invoke>
</minimax:tool_call>"""

        tools = [{
            "type": "function",
            "function": {
                "name": "test_func",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "count": {"type": "integer"}
                    }
                }
            }
        }]

        result = self.parser.parse_tool_calls(text, tools)
        args = json.loads(result["tool_calls"][0]["function"]["arguments"])

        assert args["count"] == 42
        assert isinstance(args["count"], int)

    def test_type_inference_float(self):
        """Test float type inference"""
        text = """<minimax:tool_call>
<invoke name="test_func">
<parameter name="value">3.14</parameter>
</invoke>
</minimax:tool_call>"""

        tools = [{
            "type": "function",
            "function": {
                "name": "test_func",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "number"}
                    }
                }
            }
        }]

        result = self.parser.parse_tool_calls(text, tools)
        args = json.loads(result["tool_calls"][0]["function"]["arguments"])

        assert args["value"] == 3.14
        assert isinstance(args["value"], float)

    def test_type_inference_boolean(self):
        """Test boolean type inference"""
        text = """<minimax:tool_call>
<invoke name="test_func">
<parameter name="enabled">true</parameter>
<parameter name="disabled">false</parameter>
</invoke>
</minimax:tool_call>"""

        tools = [{
            "type": "function",
            "function": {
                "name": "test_func",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "enabled": {"type": "boolean"},
                        "disabled": {"type": "boolean"}
                    }
                }
            }
        }]

        result = self.parser.parse_tool_calls(text, tools)
        args = json.loads(result["tool_calls"][0]["function"]["arguments"])

        assert args["enabled"] is True
        assert args["disabled"] is False

    def test_type_inference_null(self):
        """Test null type inference"""
        text = """<minimax:tool_call>
<invoke name="test_func">
<parameter name="optional">null</parameter>
</invoke>
</minimax:tool_call>"""

        result = self.parser.parse_tool_calls(text)
        args = json.loads(result["tool_calls"][0]["function"]["arguments"])

        assert args["optional"] is None

    def test_type_inference_json_object(self):
        """Test JSON object parsing"""
        text = """<minimax:tool_call>
<invoke name="test_func">
<parameter name="config">{"key": "value", "count": 10}</parameter>
</invoke>
</minimax:tool_call>"""

        tools = [{
            "type": "function",
            "function": {
                "name": "test_func",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "config": {"type": "object"}
                    }
                }
            }
        }]

        result = self.parser.parse_tool_calls(text, tools)
        args = json.loads(result["tool_calls"][0]["function"]["arguments"])

        assert isinstance(args["config"], dict)
        assert args["config"]["key"] == "value"
        assert args["config"]["count"] == 10

    def test_type_inference_json_array(self):
        """Test JSON array parsing"""
        text = """<minimax:tool_call>
<invoke name="test_func">
<parameter name="items">["apple", "banana", "orange"]</parameter>
</invoke>
</minimax:tool_call>"""

        tools = [{
            "type": "function",
            "function": {
                "name": "test_func",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "items": {"type": "array"}
                    }
                }
            }
        }]

        result = self.parser.parse_tool_calls(text, tools)
        args = json.loads(result["tool_calls"][0]["function"]["arguments"])

        assert isinstance(args["items"], list)
        assert len(args["items"]) == 3
        assert "apple" in args["items"]

    def test_quoted_function_name(self):
        """Test function name with quotes"""
        text = """<minimax:tool_call>
<invoke name="get_weather">
<parameter name="loc">NYC</parameter>
</invoke>
</minimax:tool_call>"""

        result = self.parser.parse_tool_calls(text)

        assert result["tool_calls"][0]["function"]["name"] == "get_weather"

    def test_has_tool_calls(self):
        """Test has_tool_calls detection"""
        assert self.parser.has_tool_calls("<minimax:tool_call>test</minimax:tool_call>")
        assert not self.parser.has_tool_calls("No tool calls here")

    def test_extract_content_without_tools(self):
        """Test content extraction without tool XML"""
        text = """Here is some text.

<minimax:tool_call>
<invoke name="test">
<parameter name="x">1</parameter>
</invoke>
</minimax:tool_call>

And more text after."""

        content = self.parser.extract_content_without_tools(text)

        assert "<minimax:tool_call>" not in content
        assert "Here is some text." in content
        assert "And more text after." in content

    def test_multiline_parameter_value(self):
        """Test parameter with multiline value"""
        text = """<minimax:tool_call>
<invoke name="write_code">
<parameter name="code">
def hello():
    print("world")
</parameter>
</invoke>
</minimax:tool_call>"""

        result = self.parser.parse_tool_calls(text)
        args = json.loads(result["tool_calls"][0]["function"]["arguments"])

        assert "def hello():" in args["code"]
        assert 'print("world")' in args["code"]

    def test_empty_parameter(self):
        """Test empty parameter value"""
        text = """<minimax:tool_call>
<invoke name="test">
<parameter name="empty"></parameter>
</invoke>
</minimax:tool_call>"""

        result = self.parser.parse_tool_calls(text)
        args = json.loads(result["tool_calls"][0]["function"]["arguments"])

        assert args["empty"] == ""

    def test_tool_call_with_think_blocks(self):
        """Test tool call mixed with think blocks (think blocks should be preserved)"""
        text = """<think>I need to get the weather</think>

<minimax:tool_call>
<invoke name="get_weather">
<parameter name="location">Paris</parameter>
</invoke>
</minimax:tool_call>"""

        result = self.parser.parse_tool_calls(text)

        # Think blocks should be in content
        assert result["tools_called"] is True
        assert "<think>" in result["content"]
        assert len(result["tool_calls"]) == 1

    def test_tool_call_preserves_think_and_followup_text(self):
        """Ensure reasoning and final message survive parsing"""
        text = """<think>I should look this up</think>

<minimax:tool_call>
<invoke name="web_search">
<parameter name="query">interleaved thinking</parameter>
</invoke>
</minimax:tool_call>

Here is what I found.
"""

        result = self.parser.parse_tool_calls(text)

        assert result["tools_called"] is True
        assert result["content"].startswith("<think>")
        assert "Here is what I found." in result["content"]
        assert result["content"].rstrip().endswith("Here is what I found.")


class TestStreamingAndFormatting:
    """Tests for streaming parser and OpenAI formatter compliance"""

    def test_simple_streaming_parser_preserves_think(self):
        parser = SimpleStreamingParser()
        sample = """<think>Let me reason</think>

<minimax:tool_call>
<invoke name="web_search">
<parameter name="query">interleaved thinking</parameter>
</invoke>
</minimax:tool_call>"""

        result = parser.process_chunk(sample)

        assert result is not None
        assert result["type"] == "tool_calls"
        assert result["content"].startswith("<think>")
        assert result["content"].rstrip().endswith("</think>")

    def test_openai_streaming_chunk_includes_index(self):
        chunk = OpenAIFormatter.format_streaming_chunk(
            tool_calls=[{"function": {"name": "web_search"}}]
        )

        payload = json.loads(chunk.split("data: ", 1)[1].strip())
        call = payload["choices"][0]["delta"]["tool_calls"][0]

        assert call["index"] == 0
        assert call["function"]["name"] == "web_search"

    def test_openai_tool_call_stream_structure(self):
        tool_call = {
            "id": "call_example",
            "type": "function",
            "function": {
                "name": "web_search",
                "arguments": "{\"query\": \"interleaved thinking\"}"
            }
        }

        chunks = OpenAIFormatter.format_tool_call_stream(tool_call, index=0)

        assert len(chunks) == 2

        first = json.loads(chunks[0].split("data: ", 1)[1].strip())
        first_call = first["choices"][0]["delta"]["tool_calls"][0]
        assert first_call["id"] == "call_example"
        assert first_call["type"] == "function"
        assert first_call["index"] == 0
        assert first_call["function"]["name"] == "web_search"

        second = json.loads(chunks[1].split("data: ", 1)[1].strip())
        second_call = second["choices"][0]["delta"]["tool_calls"][0]
        assert "arguments" in second_call["function"]
        assert second_call["index"] == 0


def test_ensure_think_wrapped_adds_missing_opening():
    """Helper adds opening <think> when only </think> is present."""
    text = "   internal chain of thought</think>"
    normalized, inserted = ensure_think_wrapped(text)

    assert inserted is True
    assert normalized.strip().startswith("<think>")
    assert normalized.strip().endswith("</think>")


def test_ensure_think_wrapped_idempotent():
    """Applying helper twice never duplicates tags."""
    text = "<think>\nreasoning</think>"
    first, inserted_first = ensure_think_wrapped(text)
    second, inserted_second = ensure_think_wrapped(first)

    assert inserted_first is False
    assert inserted_second is False
    assert first == second == text
