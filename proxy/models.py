"""Pydantic models for API requests and responses"""

import json
from typing import List, Optional, Dict, Any, Literal, Union

from pydantic import BaseModel, Field, field_validator


# OpenAI Models
class OpenAIMessage(BaseModel):
    """OpenAI message format - accepts both string and Anthropic-style array content"""
    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    reasoning_details: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

    @field_validator('content', mode='before')
    @classmethod
    def flatten_content(cls, v):
        """Flatten Anthropic-style content arrays to strings for OpenAI compatibility"""
        if isinstance(v, list):
            # Extract text from Anthropic content blocks
            text_parts = []
            for block in v:
                if isinstance(block, dict):
                    # Handle text blocks
                    if block.get('type') == 'text' and 'text' in block:
                        text_parts.append(block['text'])
                    # Handle simple text dict
                    elif 'text' in block:
                        text_parts.append(block['text'])
            return '\n\n'.join(text_parts) if text_parts else ''
        return v


class OpenAITool(BaseModel):
    """OpenAI tool/function definition"""
    type: Literal["function"] = "function"
    function: Dict[str, Any]


class OpenAIChatRequest(BaseModel):
    """OpenAI Chat Completions request"""
    model: str = "minimax-m2"
    messages: List[OpenAIMessage]
    max_tokens: Optional[int] = Field(None, ge=1)
    temperature: Optional[float] = Field(1.0, gt=0.0, le=1.0)
    top_p: Optional[float] = Field(1.0, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(None, ge=0)
    n: Optional[int] = Field(1, ge=1)
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    tools: Optional[List[OpenAITool]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    extra_body: Optional[Dict[str, Any]] = None


# Anthropic Models
class AnthropicContentBlock(BaseModel):
    """Anthropic content block"""
    type: Literal["text", "image", "tool_use", "tool_result", "thinking"]
    text: Optional[str] = None
    thinking: Optional[str] = None
    source: Optional[Dict[str, Any]] = None  # For image
    id: Optional[str] = None  # For tool_use
    name: Optional[str] = None  # For tool_use
    input: Optional[Dict[str, Any]] = None  # For tool_use
    tool_use_id: Optional[str] = None  # For tool_result
    content: Optional[Union[str, List[Dict[str, Any]]]] = None  # For tool_result

    @field_validator('input', mode='before')
    @classmethod
    def normalize_input(cls, v):
        """Normalize tool_use input field - convert empty strings to empty dict"""
        if v == "":
            return {}
        return v


class AnthropicMessage(BaseModel):
    """Anthropic message format"""
    role: Literal["user", "assistant"]
    content: Union[str, List[AnthropicContentBlock]]


class AnthropicTool(BaseModel):
    """Anthropic tool definition"""
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any]


class AnthropicChatRequest(BaseModel):
    """Anthropic Messages request"""
    model: str = "minimax-m2"
    messages: List[AnthropicMessage]
    max_tokens: int = Field(4096, ge=1)
    system: Optional[Union[str, List[Dict[str, Any]]]] = None
    temperature: Optional[float] = Field(1.0, gt=0.0, le=1.0)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(None, ge=0)
    stream: bool = False
    stop_sequences: Optional[List[str]] = None
    tools: Optional[List[AnthropicTool]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    thinking: Optional[Dict[str, Any]] = None  # For extended thinking


# Conversion helpers
def anthropic_tools_to_openai(tools: Optional[List[AnthropicTool]]) -> Optional[List[Dict[str, Any]]]:
    """Convert Anthropic tools to OpenAI format"""
    if not tools:
        return None

    openai_tools = []
    for tool in tools:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.input_schema
            }
        })
    return openai_tools


def _serialize_tool_arguments(input_payload: Optional[Union[str, Dict[str, Any], List[Any]]]) -> str:
    """Convert Anthropic tool_use input payloads to OpenAI-style JSON strings."""
    if input_payload is None:
        return "{}"

    if isinstance(input_payload, str):
        stripped = input_payload.strip()
        if not stripped:
            return "{}"
        try:
            json.loads(input_payload)
            return input_payload
        except json.JSONDecodeError:
            return json.dumps(stripped, ensure_ascii=False)

    try:
        return json.dumps(input_payload, ensure_ascii=False)
    except (TypeError, ValueError):
        return json.dumps(str(input_payload), ensure_ascii=False)


def _stringify_tool_result_content(content: Optional[Union[str, List[Dict[str, Any]]]]) -> str:
    """Flatten tool_result content payloads to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for entry in content:
            if isinstance(entry, dict):
                text_value = entry.get("text")
                if text_value:
                    parts.append(text_value)
        if parts:
            return "\n".join(parts)
        try:
            return json.dumps(content, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(content)
    return str(content)


def anthropic_messages_to_openai(messages: List[AnthropicMessage]) -> List[Dict[str, Any]]:
    """Convert Anthropic messages to OpenAI format

    CRITICAL: MiniMax-M2 chat template does NOT support role="tool".
    Tool results must be sent as role="user" messages as per MiniMax docs.
    """
    openai_messages = []

    for msg in messages:
        if isinstance(msg.content, str):
            # Simple text message
            openai_messages.append({
                "role": msg.role,
                "content": msg.content
            })
        else:
            # Content blocks - merge text and thinking blocks in order
            text_segments: List[str] = []
            tool_calls = []
            tool_results = []

            for block in msg.content:
                if block.type == "thinking" and block.thinking:
                    text_segments.append(f"<think>{block.thinking}</think>")
                elif block.type == "text" and block.text:
                    text_segments.append(block.text)
                elif block.type == "image":
                    text_segments.append("[image]")
                elif block.type == "tool_use":
                    tool_calls.append({
                        "id": block.id,
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": _serialize_tool_arguments(block.input)
                        }
                    })
                elif block.type == "tool_result":
                    # Collect tool results - will be formatted as user message content
                    tool_results.append({
                        "tool_use_id": block.tool_use_id,
                        "content": _stringify_tool_result_content(block.content)
                    })

            # Build message
            has_text = any(segment for segment in text_segments)
            if has_text or tool_calls:
                message = {
                    "role": msg.role,
                    "content": "\n".join(segment for segment in text_segments if segment) if has_text else ""
                }
                if tool_calls:
                    message["tool_calls"] = tool_calls
                openai_messages.append(message)

            # Tool results become user messages (NOT role="tool")
            # This matches MiniMax's expected format
            if tool_results:
                for result in tool_results:
                    openai_messages.append({
                        "role": "user",
                        "content": f"Tool result for {result['tool_use_id']}: {result['content']}"
                    })

    return openai_messages
