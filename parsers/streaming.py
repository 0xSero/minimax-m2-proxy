"""Streaming state machine for MiniMax-M2 responses

Handles incremental parsing of tool calls and thinking blocks
"""

from enum import Enum
from typing import Dict, Any, Optional, List
from .tools import ToolCallParser


class StreamingState(Enum):
    """Streaming parser states"""
    CONTENT = "content"
    TOOL_CALL = "tool_call"


class StreamingParser:
    """Stateful parser for streaming MiniMax-M2 responses"""

    def __init__(self):
        self.tool_parser = ToolCallParser()
        self.reset()

    def reset(self):
        """Reset streaming state for a new message"""
        self.state = StreamingState.CONTENT
        self.accumulated_text = ""
        self.previous_text = ""
        self.in_tool_call = False
        self.tool_call_started = False
        self.sent_tool_calls = []
        self.tools = None

    def set_tools(self, tools: Optional[List[Dict]]):
        """Set tools configuration for type conversion"""
        self.tools = tools

    def process_delta(
        self,
        delta_text: str
    ) -> Dict[str, Any]:
        """
        Process a streaming delta and return formatted chunk.

        For simplicity in Phase 1, we'll use a simple approach:
        - Stream regular content as-is (including <think> blocks)
        - When we detect tool calls, buffer and send complete tool info

        Returns:
            {
                "type": "content" | "tool_calls",
                "delta": str,  # for content
                "tool_calls": [...]  # for tool calls
            }
        """
        # Update accumulated text
        self.accumulated_text += delta_text

        # Check if we're entering a tool call
        if not self.tool_call_started and self.tool_parser.tool_call_start_token in self.accumulated_text:
            self.tool_call_started = True
            self.in_tool_call = True

            # Extract content before tool call
            tool_start_idx = self.accumulated_text.find(self.tool_parser.tool_call_start_token)
            content_before = self.accumulated_text[:tool_start_idx]

            if content_before and content_before != self.previous_text:
                self.previous_text = content_before
                return {
                    "type": "content",
                    "delta": content_before[len(self.previous_text) - len(content_before):]
                }

        # Check if we're exiting a tool call
        if self.in_tool_call and self.tool_parser.tool_call_end_token in self.accumulated_text:
            # Parse complete tool calls
            result = self.tool_parser.parse_tool_calls(self.accumulated_text, self.tools)

            if result["tools_called"]:
                # Send tool calls that haven't been sent yet
                new_tool_calls = result["tool_calls"][len(self.sent_tool_calls):]
                if new_tool_calls:
                    self.sent_tool_calls.extend(new_tool_calls)
                    return {
                        "type": "tool_calls",
                        "tool_calls": new_tool_calls
                    }

            self.in_tool_call = False

        # Regular content streaming
        if not self.in_tool_call:
            # Only send new delta
            if len(self.accumulated_text) > len(self.previous_text):
                new_content = self.accumulated_text[len(self.previous_text):]
                self.previous_text = self.accumulated_text

                # Don't send if it's just the tool call marker
                if self.tool_parser.tool_call_start_token not in new_content:
                    return {
                        "type": "content",
                        "delta": new_content
                    }

        # No change to send
        return {"type": "none"}


class SimpleStreamingParser:
    """
    Simplified streaming parser for Phase 1.

    Strategy: Buffer text until we have complete tool calls,
    then send everything at once. This is simpler than
    incremental tool call streaming.
    """

    def __init__(self):
        self.tool_parser = ToolCallParser()
        self.accumulated = ""
        self.content_sent = 0
        self.tools_sent = False
        self.tools = None

    def set_tools(self, tools: Optional[List[Dict]]):
        self.tools = tools

    def process_chunk(self, chunk_text: str) -> Optional[Dict[str, Any]]:
        """
        Process a chunk of text.

        Returns dict with:
        - type: "content" | "tool_calls"
        - delta: str (for content)
        - tool_calls: list (for tool calls)
        """
        self.accumulated += chunk_text

        # If we have complete tool calls, send them once
        if not self.tools_sent and self.tool_parser.has_tool_calls(self.accumulated):
            if self.tool_parser.tool_call_end_token in self.accumulated:
                result = self.tool_parser.parse_tool_calls(self.accumulated, self.tools)
                if result["tools_called"]:
                    self.tools_sent = True
                    # Send content before tools (if any)
                    if result["content"]:
                        self.content_sent = len(result["content"])
                    return {
                        "type": "tool_calls",
                        "tool_calls": result["tool_calls"],
                        "content": result["content"]
                    }
            # Still building tool call, don't send anything yet
            return None

        # Send regular content deltas
        if not self.tools_sent or self.tool_parser.tool_call_end_token in self.accumulated:
            # Send new content that hasn't been sent
            new_content = self.accumulated[self.content_sent:]

            # Don't send tool call XML
            if self.tool_parser.tool_call_start_token in new_content:
                # Trim to before tool call
                tool_idx = new_content.find(self.tool_parser.tool_call_start_token)
                new_content = new_content[:tool_idx]

            if new_content:
                self.content_sent += len(new_content)
                return {
                    "type": "content",
                    "delta": new_content
                }

        return None

    def get_final_content(self) -> str:
        """Get final content with tool calls removed"""
        if self.tool_parser.has_tool_calls(self.accumulated):
            result = self.tool_parser.parse_tool_calls(self.accumulated, self.tools)
            return result.get("content", "")
        return self.accumulated
