"""Tests for tool parsing helpers."""

import json

from parsers.tools import parse_tool_calls


def _extract_single_argument(xml: str) -> str:
    """Helper to pull the single tool call argument dict as JSON string."""
    result = parse_tool_calls(xml)
    assert result["tools_called"], "Expected tool call to be detected"
    tool_call = result["tool_calls"][0]
    return tool_call["function"]["arguments"]


def test_parse_tool_calls_unescapes_newlines() -> None:
    xml = (
        "<minimax:tool_call>\n"
        '<invoke name="db_custom_query">\n'
        '<parameter name="sql">SELECT \\n  id as application_id</parameter>\n'
        "</invoke>\n"
        "</minimax:tool_call>"
    )

    args_json = _extract_single_argument(xml)
    sql = json.loads(args_json)["sql"]
    assert sql == "SELECT \n  id as application_id"


def test_parse_tool_calls_handles_multiple_backslashes() -> None:
    xml = (
        "<minimax:tool_call>\n"
        '<invoke name="db_custom_query">\n'
        '<parameter name="sql">SELECT \\\\nFROM dual</parameter>\n'
        "</invoke>\n"
        "</minimax:tool_call>"
    )

    args_json = _extract_single_argument(xml)
    sql = json.loads(args_json)["sql"]
    assert sql == "SELECT \nFROM dual"


def test_parse_tool_calls_preserves_unknown_escape_sequences() -> None:
    xml = (
        "<minimax:tool_call>\n"
        '<invoke name="db_custom_query">\n'
        '<parameter name="sql">C\\\\logs\\\\app</parameter>\n'
        "</invoke>\n"
        "</minimax:tool_call>"
    )

    args_json = _extract_single_argument(xml)
    sql = json.loads(args_json)["sql"]
    assert sql == "C\\logs\\app"
