"""Focused tests covering SimpleStreamingParser reasoning behaviour."""

from parsers.streaming import SimpleStreamingParser


def test_streaming_parser_emits_think_block_when_closing_detected():
    parser = SimpleStreamingParser()

    chunks = [
        "Working through the options... ",
        "deciding next steps</think>\n",
        "<minimax:tool_call><invoke name=\"foo\"></invoke></minimax:tool_call>"
    ]

    outputs = []
    for chunk in chunks:
        result = parser.process_chunk(chunk)
        if result:
            outputs.append(result)

    content_event = outputs[0]
    assert content_event["type"] == "content"
    assert content_event["delta"].startswith("<think>")
    assert "</think>" in content_event["delta"]

    tool_event = outputs[1]
    assert tool_event["type"] == "tool_calls"
    assert len(tool_event["tool_calls"]) == 1
    assert tool_event.get("content") is None


def test_streaming_parser_flushes_plain_content_on_finalize():
    parser = SimpleStreamingParser()

    assert parser.process_chunk("Direct answer without thinking.") is None

    tail = parser.flush_pending()
    assert tail == "Direct answer without thinking."


def test_streaming_parser_think_without_tools():
    parser = SimpleStreamingParser()

    result = parser.process_chunk("Reasoning path</think>\nFinal reply.")

    assert result is not None
    assert result["type"] == "content"
    assert result["delta"].startswith("<think>")
    assert result["delta"].rstrip().endswith("Final reply.")


def test_streaming_parser_does_not_duplicate_think_block():
    parser = SimpleStreamingParser()

    first = parser.process_chunk("Interpreting request</think>\n")
    assert first is not None
    assert first["type"] == "content"
    assert first["delta"].startswith("<think>")

    second = parser.process_chunk("<minimax:tool_call><invoke name=\"foo\"></invoke></minimax:tool_call>")
    # Tool event should not resend reasoning already emitted
    assert second is not None
    assert second["type"] == "tool_calls"
    assert second.get("content") is None


def test_streaming_parser_wraps_reasoning_without_closing():
    parser = SimpleStreamingParser()

    # Chunk lacks </think>; the wrap should still add a well-formed block when forced
    result = parser.process_chunk("Analyzing request for more context.")
    assert result is None

    # Force tool call to trigger flush without actual closing tag
    result = parser.process_chunk("<minimax:tool_call><invoke name=\"foo\"></invoke></minimax:tool_call>")
    assert result is not None
    assert result["type"] == "tool_calls"
    assert result.get("content") is not None
    assert result["content"].startswith("<think>")
    assert result["content"].rstrip().endswith("</think>")
