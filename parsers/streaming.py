"""Simple streaming parser for MiniMax-M2 responses."""

from typing import Any, Dict, List, Optional

from .reasoning import split_think
from .tools import parse_tool_calls


class StreamingParser:
    """Simple buffering parser for streaming responses."""

    def __init__(self) -> None:
        self.buffer = ""
        self.sent_position = 0  # How much raw content we've already emitted
        self.tools_sent = False
        self.tools: Optional[List[Dict[str, Any]]] = None
        self.think_status_determined = False  # Have we determined if </think> will appear?
        self.has_think_closing = False  # Will </think> appear in response?
        self.think_opening_sent = False  # Have we sent <think> opening?
        self.reasoning_len_sent = 0  # Number of reasoning characters already emitted
        self.last_tool_calls: Optional[List[Dict[str, Any]]] = None

    def set_tools(self, tools: Optional[List[Dict[str, Any]]]) -> None:
        """Set tools for type conversion during parsing."""
        self.tools = tools

    def has_tool_calls(self) -> bool:
        """Check if tool calls were detected."""
        return self.tools_sent

    def _compute_reasoning_delta(self) -> str:
        """Return incremental reasoning text (inside <think>...</think>)."""
        if "</think>" not in self.buffer and "<think>" not in self.buffer:
            return ""

        closing_idx = self.buffer.find("</think>")
        if closing_idx == -1:
            # Think block still streaming; avoid emitting until we know it exists
            return ""

        reasoning_segment = self.buffer[:closing_idx]
        if reasoning_segment.startswith("<think>"):
            reasoning_segment = reasoning_segment[len("<think>") :]

        if len(reasoning_segment) <= self.reasoning_len_sent:
            return ""

        delta = reasoning_segment[self.reasoning_len_sent :]
        self.reasoning_len_sent = len(reasoning_segment)
        return delta

    def process_chunk(self, chunk: str) -> Optional[Dict[str, Any]]:
        """
        Process incoming chunk and return delta information.

        Returns dict with keys:
            - type: "content" or "tool_calls" or "reasoning"
            - raw_delta: Raw text delta (with <think> tags) when applicable
            - content_delta: Visible text delta (outside <think>)
            - reasoning_delta: Reasoning text delta
            - tool_calls: Parsed tool call payload when available
        """
        self.buffer += chunk
        reasoning_delta = self._compute_reasoning_delta()

        # Determine think status before emitting content
        if not self.think_status_determined:
            if "</think>" in self.buffer:
                self.think_status_determined = True
                self.has_think_closing = True
            elif "<minimax:tool_call>" in self.buffer:
                self.think_status_determined = True
                self.has_think_closing = False
            else:
                # Still waiting for confirmation; only emit reasoning if available
                if reasoning_delta:
                    return {"type": "reasoning", "reasoning_delta": reasoning_delta}
                return None

        has_start = "<minimax:tool_call>" in self.buffer
        has_end = "</minimax:tool_call>" in self.buffer

        if has_start and not has_end:
            if not self.tools_sent:
                tool_start_idx = self.buffer.find("<minimax:tool_call>")
                content_before = self.buffer[:tool_start_idx]

                if len(content_before) > self.sent_position:
                    delta = content_before[self.sent_position :]
                    if self.sent_position == 0 and not self.think_opening_sent and self.has_think_closing:
                        self.think_opening_sent = True
                        delta = "<think>\n" + delta
                    self.sent_position = len(content_before)
                    think_text, visible_text = split_think(delta)
                    result: Dict[str, Any] = {
                        "type": "content",
                        "raw_delta": delta,
                        "content_delta": visible_text or None,
                    }
                    aggregate_reasoning = reasoning_delta or think_text
                    if aggregate_reasoning:
                        result["reasoning_delta"] = aggregate_reasoning
                    return result

            return {"type": "reasoning", "reasoning_delta": reasoning_delta} if reasoning_delta else None

        if has_start and has_end and not self.tools_sent:
            result = parse_tool_calls(self.buffer, self.tools)

            if result["tools_called"]:
                self.tools_sent = True
                self.last_tool_calls = result["tool_calls"]

                tool_start_idx = self.buffer.find("<minimax:tool_call>")
                content_before = self.buffer[:tool_start_idx]

                if len(content_before) > self.sent_position:
                    delta = content_before[self.sent_position :]
                    if self.sent_position == 0 and not self.think_opening_sent and self.has_think_closing:
                        self.think_opening_sent = True
                        delta = "<think>\n" + delta
                    self.sent_position = tool_start_idx
                    think_text, visible_text = split_think(delta)
                    response: Dict[str, Any] = {
                        "type": "content",
                        "raw_delta": delta,
                        "content_delta": visible_text or None,
                    }
                    aggregate_reasoning = reasoning_delta or think_text
                    if aggregate_reasoning:
                        response["reasoning_delta"] = aggregate_reasoning
                    return response

                tool_end_idx = self.buffer.find("</minimax:tool_call>")
                if tool_end_idx == -1:
                    tool_end_idx = len(self.buffer)
                else:
                    tool_end_idx += len("</minimax:tool_call>")

                tool_delta = self.buffer[self.sent_position:tool_end_idx]
                self.sent_position = tool_end_idx
                payload: Dict[str, Any] = {
                    "type": "tool_calls",
                    "tool_calls": result["tool_calls"],
                }
                if tool_delta:
                    payload["raw_delta"] = tool_delta
                if reasoning_delta:
                    payload["reasoning_delta"] = reasoning_delta
                return payload

        if len(self.buffer) > self.sent_position:
            delta = self.buffer[self.sent_position :]
            if self.sent_position == 0 and not self.think_opening_sent and self.has_think_closing:
                self.think_opening_sent = True
                delta = "<think>\n" + delta
            self.sent_position = len(self.buffer)
            think_text, visible_text = split_think(delta)

            result: Dict[str, Any] = {
                "type": "content",
                "raw_delta": delta,
                "content_delta": visible_text or None,
            }

            aggregate_reasoning = reasoning_delta or think_text
            if aggregate_reasoning:
                result["reasoning_delta"] = aggregate_reasoning

            return result

        return {"type": "reasoning", "reasoning_delta": reasoning_delta} if reasoning_delta else None

    def flush_pending(self) -> Optional[Dict[str, Any]]:
        """Flush any remaining buffered content at stream end."""
        if len(self.buffer) > self.sent_position:
            remaining = self.buffer[self.sent_position :]
            self.sent_position = len(self.buffer)
            think_text, visible_text = split_think(remaining)
            result: Dict[str, Any] = {
                "raw_delta": remaining,
                "content_delta": visible_text or None,
            }
            if think_text:
                result["reasoning_delta"] = think_text
            return result
        return None

    def get_final_content(self) -> str:
        """Get complete final content as streamed (including tool blocks)."""
        return self.buffer

    def get_last_tool_calls(self) -> Optional[List[Dict[str, Any]]]:
        """Return the most recently parsed tool calls, if any."""
        return self.last_tool_calls
