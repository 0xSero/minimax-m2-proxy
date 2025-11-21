#!/usr/bin/env python3
"""Tests for MiniMax tool parsing utilities."""

import json

from parsers.tools import extract_json_tool_calls


def test_extract_json_tool_calls_strips_trailing_payload() -> None:
    message = "<think>analysis</think>\n[{\"name\": \"web_search\", \"parameters\": {\"query\": \"something\"}}]"
    content, tool_calls = extract_json_tool_calls(message)

    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "web_search"
    assert json.loads(tool_calls[0]["function"]["arguments"])["query"] == "something"
    assert content.endswith("</think>")


def test_extract_json_tool_calls_returns_none_when_invalid() -> None:
    message = "<think>analysis</think>\n[{\"name\": 123}]"
    content, tool_calls = extract_json_tool_calls(message)
    assert tool_calls is None
    assert content == message
