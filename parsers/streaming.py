"""Simple streaming parser for MiniMax-M2 responses

Strategy: Stream content immediately until tool calls are detected,
then buffer until complete, parse, and continue.

Critical: TabbyAPI's chat template puts <think> in generation prompt,
so model outputs "</think>" without opening tag. We add it at stream start
when we detect </think> exists.
"""

from typing import Dict, Any, Optional, List
from .tools import parse_tool_calls


class StreamingParser:
    """Simple buffering parser for streaming responses"""

    def __init__(self):
        self.buffer = ""
        self.sent_position = 0  # How much we've already sent
        self.tools_sent = False
        self.tools = None
        self.think_status_determined = False  # Have we determined if </think> will appear?
        self.has_think_closing = False  # Will </think> appear in response?
        self.think_opening_sent = False  # Have we sent <think> opening?

    def set_tools(self, tools: Optional[List[Dict]]):
        """Set tools for type conversion during parsing"""
        self.tools = tools

    def has_tool_calls(self) -> bool:
        """Check if tool calls were detected"""
        return self.tools_sent

    def process_chunk(self, chunk: str) -> Optional[Dict[str, Any]]:
        """
        Process incoming chunk.

        Returns:
            - {"type": "content", "delta": str} for content
            - {"type": "tool_calls", "tool_calls": [...]} for tool calls
            - None if no output to send yet
        """
        self.buffer += chunk

        # Check if think status is determined
        # We MUST wait until we see </think> OR a tool call starts to know if we need <think> opening
        if not self.think_status_determined:
            if "</think>" in self.buffer:
                self.think_status_determined = True
                self.has_think_closing = True
            elif "<minimax:tool_call>" in self.buffer:
                # Tool call started without seeing </think> - no think block in this response
                self.think_status_determined = True
                self.has_think_closing = False
            else:
                # Haven't determined yet - buffer more
                return None

        # Check if we're inside a tool call block
        has_start = "<minimax:tool_call>" in self.buffer
        has_end = "</minimax:tool_call>" in self.buffer

        if has_start and not has_end:
            # Tool call started but not finished - send content before tool call, then pause
            if not self.tools_sent:
                tool_start_idx = self.buffer.find("<minimax:tool_call>")
                content_before = self.buffer[:tool_start_idx]

                if len(content_before) > self.sent_position:
                    delta = content_before[self.sent_position:]
                    # Prepend <think> tag if this is first send and </think> exists
                    if self.sent_position == 0 and not self.think_opening_sent and self.has_think_closing:
                        self.think_opening_sent = True
                        delta = "<think>\n" + delta
                    self.sent_position = len(content_before)
                    return {"type": "content", "delta": delta}

            # Pause - waiting for tool call to complete
            return None

        if has_start and has_end and not self.tools_sent:
            # Complete tool call - parse and send
            result = parse_tool_calls(self.buffer, self.tools)

            if result["tools_called"]:
                self.tools_sent = True

                # Send any remaining content before tool call (if we haven't already)
                tool_start_idx = self.buffer.find("<minimax:tool_call>")
                content_before = self.buffer[:tool_start_idx]

                if len(content_before) > self.sent_position:
                    # Still have content to send before tool call
                    delta = content_before[self.sent_position:]
                    # Prepend <think> tag if this is first send and </think> exists
                    if self.sent_position == 0 and not self.think_opening_sent and self.has_think_closing:
                        self.think_opening_sent = True
                        delta = "<think>\n" + delta
                    self.sent_position = tool_start_idx
                    return {"type": "content", "delta": delta}

                # All content before tool call was already sent, now send tool calls
                self.sent_position = len(self.buffer)
                return {
                    "type": "tool_calls",
                    "tool_calls": result["tool_calls"]
                }

        # No tool call, or tool call already sent - stream content normally
        if len(self.buffer) > self.sent_position:
            delta = self.buffer[self.sent_position:]
            # Prepend <think> tag if this is first send and </think> exists
            if self.sent_position == 0 and not self.think_opening_sent and self.has_think_closing:
                self.think_opening_sent = True
                delta = "<think>\n" + delta
            self.sent_position = len(self.buffer)
            return {"type": "content", "delta": delta}

        return None

    def flush_pending(self) -> Optional[str]:
        """Flush any remaining buffered content at stream end"""
        if len(self.buffer) > self.sent_position:
            remaining = self.buffer[self.sent_position:]
            self.sent_position = len(self.buffer)
            return remaining
        return None

    def get_final_content(self) -> str:
        """Get complete final content with tool calls removed"""
        if "<minimax:tool_call>" in self.buffer:
            result = parse_tool_calls(self.buffer, self.tools)
            return result.get("content") or ""
        return self.buffer
