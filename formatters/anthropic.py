"""Anthropic format converter for MiniMax-M2 responses"""

import json
import time
import uuid
from typing import Dict, Any, List, Optional


class AnthropicFormatter:
    """Convert MiniMax-M2 responses to Anthropic Messages format"""

    @staticmethod
    def format_complete_response(
        content: Optional[str],
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        model: str = "minimax-m2",
        stop_reason: str = "end_turn"
    ) -> Dict[str, Any]:
        """
        Format a complete (non-streaming) response.

        Args:
            content: Response content (with <think> blocks preserved)
            tool_calls: List of tool calls in OpenAI format (will be converted)
            model: Model name
            stop_reason: One of: end_turn, max_tokens, stop_sequence, tool_use

        Returns:
            Anthropic Messages response
        """
        content_blocks = []

        # Add text content (keeping <think> blocks as-is)
        if content:
            content_blocks.append({
                "type": "text",
                "text": content
            })

        # Convert tool calls to Anthropic format
        if tool_calls:
            stop_reason = "tool_use"
            for tool_call in tool_calls:
                # OpenAI format: {id, type: "function", function: {name, arguments}}
                # Anthropic format: {type: "tool_use", id, name, input}
                content_blocks.append({
                    "type": "tool_use",
                    "id": tool_call["id"],
                    "name": tool_call["function"]["name"],
                    "input": json.loads(tool_call["function"]["arguments"])
                })

        return {
            "id": f"msg_{uuid.uuid4().hex}",
            "type": "message",
            "role": "assistant",
            "content": content_blocks,
            "model": model,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0
            }
        }

    @staticmethod
    def format_streaming_event(
        event_type: str,
        data: Dict[str, Any]
    ) -> str:
        """
        Format Anthropic SSE event.

        Event types:
        - message_start
        - content_block_start
        - content_block_delta
        - content_block_stop
        - message_delta
        - message_stop
        """
        event_data = {
            "type": event_type,
            **data
        }

        return f"event: {event_type}\ndata: {json.dumps(event_data, ensure_ascii=False)}\n\n"

    @staticmethod
    def format_message_start(model: str = "minimax-m2") -> str:
        """Format message_start event"""
        return AnthropicFormatter.format_streaming_event(
            "message_start",
            {
                "message": {
                    "id": f"msg_{uuid.uuid4().hex}",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": model,
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0}
                }
            }
        )

    @staticmethod
    def format_content_block_start(index: int, block_type: str = "text") -> str:
        """Format content_block_start event"""
        block = {"type": block_type}
        if block_type == "text":
            block["text"] = ""
        return AnthropicFormatter.format_streaming_event(
            "content_block_start",
            {"index": index, "content_block": block}
        )

    @staticmethod
    def format_content_block_delta(index: int, delta: str) -> str:
        """Format content_block_delta event for text"""
        return AnthropicFormatter.format_streaming_event(
            "content_block_delta",
            {
                "index": index,
                "delta": {"type": "text_delta", "text": delta}
            }
        )

    @staticmethod
    def format_tool_use_delta(index: int, tool_call: Dict[str, Any]) -> str:
        """Format tool_use block"""
        return AnthropicFormatter.format_streaming_event(
            "content_block_start",
            {
                "index": index,
                "content_block": {
                    "type": "tool_use",
                    "id": tool_call["id"],
                    "name": tool_call["function"]["name"],
                    "input": json.loads(tool_call["function"]["arguments"])
                }
            }
        )

    @staticmethod
    def format_content_block_stop(index: int) -> str:
        """Format content_block_stop event"""
        return AnthropicFormatter.format_streaming_event(
            "content_block_stop",
            {"index": index}
        )

    @staticmethod
    def format_message_delta(stop_reason: str = "end_turn") -> str:
        """Format message_delta event"""
        return AnthropicFormatter.format_streaming_event(
            "message_delta",
            {
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": 0}
            }
        )

    @staticmethod
    def format_message_stop() -> str:
        """Format message_stop event"""
        return AnthropicFormatter.format_streaming_event("message_stop", {})

    @staticmethod
    def format_error(error_message: str, error_type: str = "api_error") -> Dict[str, Any]:
        """Format an error response"""
        return {
            "type": "error",
            "error": {
                "type": error_type,
                "message": error_message
            }
        }
