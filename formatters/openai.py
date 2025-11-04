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
        finish_reason: str = "stop",
        reasoning_text: Optional[str] = None,
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
            message["content"] = content
            message["tool_calls"] = tool_calls
            finish_reason = "tool_calls"
        else:
            message["content"] = content or ""

        if reasoning_text:
            message["reasoning_details"] = [
                {"type": "chain_of_thought", "text": reasoning_text}
            ]

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
        model: str = "minimax-m2",
        reasoning_delta: Optional[str] = None,
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

        if reasoning_delta is not None:
            delta_content["reasoning_details"] = [
                {"type": "chain_of_thought", "text": reasoning_delta}
            ]

        if tool_calls is not None:
            normalized_calls = []
            for idx, call in enumerate(tool_calls):
                call_entry = {}
                for key, value in call.items():
                    if key == "function" and isinstance(value, dict):
                        call_entry[key] = dict(value)
                    else:
                        call_entry[key] = value
                if "index" not in call_entry:
                    call_entry["index"] = idx
                normalized_calls.append(call_entry)
            delta_content["tool_calls"] = normalized_calls

        chunk: Dict[str, Any] = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:29]}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": delta_content or {},
                "logprobs": None,
                "finish_reason": finish_reason
            }]
        }

        return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

    @staticmethod
    def format_tool_call_stream(tool_call: Dict[str, Any], index: int, model: str = "minimax-m2") -> List[str]:
        """Create streaming chunks for a single tool call following OpenAI spec."""
        chunks: List[str] = []

        call_id = tool_call.get("id")
        call_type = tool_call.get("type")
        function = tool_call.get("function", {})

        # First chunk announces the tool call id/type. Include the function name
        # immediately so downstream clients that validate the field as a string
        # don't fail on an empty placeholder object.
        id_delta: Dict[str, Any] = {"index": index, "function": {}}
        if call_id is not None:
            id_delta["id"] = call_id
        if call_type is not None:
            id_delta["type"] = call_type
        if name := function.get("name"):
            id_delta["function"]["name"] = name
        else:
            id_delta["function"]["name"] = ""
        chunks.append(OpenAIFormatter.format_streaming_chunk(tool_calls=[id_delta], model=model))

        # Second chunk carries the arguments payload
        arguments = function.get("arguments")
        if arguments:  # Only send if arguments is truthy
            chunks.append(OpenAIFormatter.format_streaming_chunk(
                tool_calls=[{"index": index, "function": {"arguments": arguments}}],
                model=model
            ))

        return chunks

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
