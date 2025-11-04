"""Unit tests for the streaming parser."""

import pytest

from parsers.streaming import StreamingParser


def test_streaming_parser_emits_reasoning_delta_once_closed() -> None:
    parser = StreamingParser()

    assert parser.process_chunk("Working through options ") is None

    result = parser.process_chunk("and deciding</think>\nFinal answer.")
    assert result is not None
    assert result["type"] == "content"
    assert result["reasoning_delta"].strip() == "Working through options and deciding"
    assert result["content_delta"].strip() == "Final answer."


def test_streaming_parser_handles_tool_calls() -> None:
    parser = StreamingParser()

    first = parser.process_chunk("Consider response</think>\n")
    assert first is not None
    assert first["type"] == "content"

    second = parser.process_chunk(
        "<minimax:tool_call><invoke name=\"foo\">"
        "<parameter name=\"value\">42</parameter></invoke></minimax:tool_call>"
    )
    assert second is not None
    assert second["type"] == "tool_calls"
    assert second["tool_calls"][0]["function"]["name"] == "foo"


def test_streaming_parser_flushes_remaining_content() -> None:
    parser = StreamingParser()

    parser.process_chunk("Direct reply without tools.")
    final = parser.flush_pending()
    assert final is not None
    assert final["content_delta"].strip() == "Direct reply without tools."

