"""OpenAI format converter for MiniMax-M2 responses"""

import json
import time
import uuid
from typing import Dict, Any, List, Optional


class OpenAIFormatter:
    """Convert MiniMax-M2 responses to OpenAI Chat Completions format"""

    @staticmethod
    def format_complete_response(
        content: Optional[str],
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        model: str = "minimax-m2",
        finish_reason: str = "stop"
    ) -> Dict[str, Any]:
        """
        Format a complete (non-streaming) response.

        Args:
            content: Response content (with <think> blocks preserved)
            tool_calls: List of tool calls in OpenAI format
            model: Model name
            finish_reason: One of: stop, length, tool_calls, content_filter

        Returns:
            OpenAI Chat Completion response
        """
        message: Dict[str, Any] = {"role": "assistant"}

        if tool_calls:
            # OpenAI spec: content can be None or string when tool_calls present
            message["content"] = content
            message["tool_calls"] = tool_calls
            finish_reason = "tool_calls"
        else:
            message["content"] = content or ""

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:29]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": message,
                "logprobs": None,
                "finish_reason": finish_reason
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }

    @staticmethod
    def format_streaming_chunk(
        delta: Optional[str] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        finish_reason: Optional[str] = None,
        model: str = "minimax-m2"
    ) -> str:
        """
        Format a streaming chunk in Server-Sent Events format.

        Args:
            delta: Text delta
            tool_calls: Tool call deltas
            finish_reason: Set on final chunk
            model: Model name

        Returns:
            SSE formatted string: "data: {json}\n\n"
        """
        delta_content: Dict[str, Any] = {}

        if delta is not None:
            delta_content["content"] = delta

        if tool_calls is not None:
            delta_content["tool_calls"] = tool_calls

        chunk = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:29]}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": delta_content,
                "logprobs": None,
                "finish_reason": finish_reason
            }]
        }

        return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

    @staticmethod
    def format_streaming_done() -> str:
        """Format the final [DONE] message for streaming"""
        return "data: [DONE]\n\n"

    @staticmethod
    def format_error(error_message: str, error_type: str = "api_error") -> Dict[str, Any]:
        """Format an error response"""
        return {
            "error": {
                "message": error_message,
                "type": error_type,
                "param": None,
                "code": None
            }
        }
