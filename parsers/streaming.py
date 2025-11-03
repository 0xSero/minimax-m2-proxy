"""Streaming state machine for MiniMax-M2 responses

Handles incremental parsing of tool calls and thinking blocks
"""

import logging
from enum import Enum
from typing import Dict, Any, Optional, List
from .tools import ToolCallParser
from .reasoning import ensure_think_wrapped


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

    def has_tool_calls(self) -> bool:
        """Check if any tool calls were detected and sent during streaming"""
        return len(self.sent_tool_calls) > 0

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
        self.cursor = 0
        self.tools_sent = False
        self.tools = None
        self.pending_buffer = ""
        self.ready_to_emit = False
        self.emitted_content = False
        self.synthetic_think_close_sent = False
        self.logger: Optional[logging.Logger] = None

        # Default threshold before assuming the model won't emit </think>.
        self.REASONING_BUFFER_LIMIT = 1024

    def set_tools(self, tools: Optional[List[Dict]]):
        self.tools = tools

    def set_logger(self, logger: logging.Logger):
        """Attach a debug logger for streaming diagnostics."""
        self.logger = logger

    def has_tool_calls(self) -> bool:
        """Check if any tool calls were detected and sent during streaming"""
        return self.tools_sent

    def _log(self, message: str, **context):
        if self.logger:
            if context:
                self.logger.debug("%s %s", message, context)
            else:
                self.logger.debug(message)

    def _maybe_flush_buffer(self, force_plain: bool = False) -> Optional[str]:
        """Drain pending buffered content when safe to emit."""
        if not self.pending_buffer:
            return None

        if not self.ready_to_emit:
            if "</think>" in self.pending_buffer:
                content, _ = ensure_think_wrapped(self.pending_buffer)
                self.pending_buffer = ""
                self.ready_to_emit = True
                self._log("emit_think_on_close", content_preview=content[:120])
                return content

            if force_plain or len(self.pending_buffer) >= self.REASONING_BUFFER_LIMIT:
                out = self.pending_buffer
                if "</think>" not in out and self.tool_parser.tool_call_start_token in self.accumulated:
                    wrapped, _ = ensure_think_wrapped(out + "</think>")
                    out = wrapped
                    self.synthetic_think_close_sent = True
                    self._log("emit_synthetic_close", content_preview=out[:120])
                self.pending_buffer = ""
                self.ready_to_emit = True
                self._log("emit_plain_buffer", reason="force" if force_plain else "limit", size=len(out))
                return out

            return None

        out = self.pending_buffer
        self.pending_buffer = ""
        self._log("emit_ready_buffer", size=len(out))
        return out

    def flush_pending(self) -> Optional[str]:
        """Force flush any buffered text (used at stream end)."""
        return self._maybe_flush_buffer(force_plain=True)

    def process_chunk(self, chunk_text: str) -> Optional[Dict[str, Any]]:
        """
        Process a chunk of text.

        Returns dict with:
        - type: "content" | "tool_calls"
        - delta: str (for content)
        - tool_calls: list (for tool calls)

        IMPORTANT: Pauses streaming when inside a tool call block until closing tag is received.
        """
        self.accumulated += chunk_text

        # Track new text for possible content flushing while avoiding duplicate
        # processing across chunks.
        new_segment = self.accumulated[self.cursor:]
        self.cursor = len(self.accumulated)

        if new_segment:
            if self.synthetic_think_close_sent and "</think>" in new_segment:
                new_segment = new_segment.replace("</think>", "", 1)
                self.synthetic_think_close_sent = False
                self._log("drop_real_close_after_synthetic")
            self.pending_buffer += new_segment
            self._log("append_segment", size=len(new_segment))

        # Check if we have complete tool calls
        if not self.tools_sent and self.tool_parser.has_tool_calls(self.accumulated):
            if self.tool_parser.tool_call_end_token in self.accumulated:
                # Complete tool call block received - parse and send
                result = self.tool_parser.parse_tool_calls(self.accumulated, self.tools)
                if result["tools_called"]:
                    self.tools_sent = True
                    content_delta = self._maybe_flush_buffer(force_plain=True)
                    if content_delta and self.tool_parser.tool_call_start_token in content_delta:
                        content_delta, _ = content_delta.split(self.tool_parser.tool_call_start_token, 1)
                        content_delta = content_delta.rstrip()
                        if not self.emitted_content:
                            if "</think>" not in content_delta:
                                content_delta = content_delta.rstrip("\n") + "\n</think>"
                        else:
                            content_delta = None
                    if content_delta == "":
                        content_delta = None
                    if content_delta:
                        self.emitted_content = True
                        self._log("tool_event_with_content", content_preview=content_delta[:120])
                    elif not self.emitted_content and result.get("content"):
                        content_delta = result["content"]
                        self.emitted_content = True
                        self._log("tool_event_with_parser_content", content_preview=content_delta[:120])
                    else:
                        self._log("tool_event_no_content")
                    return {
                        "type": "tool_calls",
                        "tool_calls": result["tool_calls"],
                        "content": content_delta
                    }
            else:
                # Still building tool call - PAUSE streaming, don't send anything
                return None

        # Check if we're currently inside a tool call block (started but not finished)
        tool_call_started = self.tool_parser.tool_call_start_token in self.accumulated
        tool_call_finished = self.tool_parser.tool_call_end_token in self.accumulated
        in_tool_call_block = tool_call_started and not tool_call_finished

        # Only send content if we're NOT inside a tool call block
        if not in_tool_call_block:
            content_delta = self._maybe_flush_buffer()
            if content_delta:
                # Remove any pending tool XML fragments from the chunk
                if self.tool_parser.tool_call_start_token in content_delta:
                    idx = content_delta.find(self.tool_parser.tool_call_start_token)
                    content_delta = content_delta[:idx]

                if content_delta:
                    self.emitted_content = True
                    self._log("content_delta_emitted", delta_preview=content_delta[:120])
                    return {
                        "type": "content",
                        "delta": content_delta
                    }

        return None

    def get_final_content(self) -> str:
        """Get final content with tool calls removed"""
        if self.tool_parser.has_tool_calls(self.accumulated):
            result = self.tool_parser.parse_tool_calls(self.accumulated, self.tools)
            content = result.get("content")
            return content if content is not None else ""
        return self.accumulated
