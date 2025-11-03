"""Message transformation utilities for MiniMax compatibility

MiniMax doesn't natively support role: "tool" messages in the OpenAI format.
This module transforms tool result messages into a format MiniMax can understand.
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


def transform_messages_for_minimax(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Transform OpenAI-style messages for MiniMax compatibility.

    Key transformations:
    1. Convert role: "tool" messages to role: "user" messages with formatted content
    2. Track tool_call_id to function_name mappings from assistant messages

    Args:
        messages: List of message dictionaries in OpenAI format

    Returns:
        Transformed message list compatible with MiniMax

    Example:
        Input:
        [
            {"role": "assistant", "tool_calls": [{"id": "call_123", "function": {"name": "get_weather", ...}}]},
            {"role": "tool", "tool_call_id": "call_123", "content": "Temperature: 25C"},
            {"role": "user", "content": "thanks!"}
        ]

        Output:
        [
            {"role": "assistant", "tool_calls": [{"id": "call_123", "function": {"name": "get_weather", ...}}]},
            {"role": "user", "content": "Tool Result (get_weather):\\nTemperature: 25C"},
            {"role": "user", "content": "thanks!"}
        ]
    """
    if not messages:
        return messages

    # Build mapping of tool_call_id -> function_name
    call_id_to_function = {}
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tool_call in msg["tool_calls"]:
                call_id = tool_call.get("id")
                function_name = tool_call.get("function", {}).get("name")
                if call_id and function_name:
                    call_id_to_function[call_id] = function_name

    # Transform messages
    transformed = []
    tool_messages_found = 0

    for msg in messages:
        if msg.get("role") == "tool":
            tool_messages_found += 1

            # Get tool call details
            tool_call_id = msg.get("tool_call_id", "")
            content = msg.get("content", "")
            function_name = call_id_to_function.get(tool_call_id, "unknown_function")

            # Transform to user message with clear formatting
            transformed.append({
                "role": "user",
                "content": f"Tool Result ({function_name}):\n{content}"
            })

            logger.debug(
                f"Transformed tool message: call_id={tool_call_id}, "
                f"function={function_name}, content_len={len(content)}"
            )
        else:
            # Pass through other messages unchanged
            transformed.append(msg)

    if tool_messages_found > 0:
        logger.info(f"Transformed {tool_messages_found} tool result message(s) to user format")

    return transformed


def find_function_name_for_call_id(
    call_id: str,
    messages: List[Dict[str, Any]]
) -> Optional[str]:
    """
    Find the function name corresponding to a tool_call_id.

    Args:
        call_id: The tool call ID to look up
        messages: List of messages to search

    Returns:
        Function name if found, None otherwise
    """
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tool_call in msg["tool_calls"]:
                if tool_call.get("id") == call_id:
                    return tool_call.get("function", {}).get("name")
    return None
