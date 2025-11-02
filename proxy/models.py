"""Pydantic models for API requests and responses"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal, Union


# OpenAI Models
class OpenAIMessage(BaseModel):
    """OpenAI message format"""
    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class OpenAITool(BaseModel):
    """OpenAI tool/function definition"""
    type: Literal["function"] = "function"
    function: Dict[str, Any]


class OpenAIChatRequest(BaseModel):
    """OpenAI Chat Completions request"""
    model: str = "minimax-m2"
    messages: List[OpenAIMessage]
    max_tokens: Optional[int] = Field(None, ge=1)
    temperature: Optional[float] = Field(1.0, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(1.0, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(None, ge=0)
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    tools: Optional[List[OpenAITool]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None


# Anthropic Models
class AnthropicContentBlock(BaseModel):
    """Anthropic content block"""
    type: Literal["text", "image", "tool_use", "tool_result"]
    text: Optional[str] = None
    source: Optional[Dict[str, Any]] = None  # For image
    id: Optional[str] = None  # For tool_use
    name: Optional[str] = None  # For tool_use
    input: Optional[Dict[str, Any]] = None  # For tool_use
    tool_use_id: Optional[str] = None  # For tool_result
    content: Optional[Union[str, List[Dict[str, Any]]]] = None  # For tool_result


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
    temperature: Optional[float] = Field(1.0, ge=0.0, le=1.0)
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


def anthropic_messages_to_openai(messages: List[AnthropicMessage]) -> List[Dict[str, Any]]:
    """Convert Anthropic messages to OpenAI format"""
    openai_messages = []

    for msg in messages:
        if isinstance(msg.content, str):
            # Simple text message
            openai_messages.append({
                "role": msg.role,
                "content": msg.content
            })
        else:
            # Content blocks - merge text blocks
            text_parts = []
            tool_calls = []

            for block in msg.content:
                if block.type == "text" and block.text:
                    text_parts.append(block.text)
                elif block.type == "tool_use":
                    tool_calls.append({
                        "id": block.id,
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": block.input
                        }
                    })
                elif block.type == "tool_result":
                    # Tool result becomes a tool role message
                    openai_messages.append({
                        "role": "tool",
                        "tool_call_id": block.tool_use_id,
                        "content": block.content if isinstance(block.content, str) else str(block.content)
                    })

            if text_parts or tool_calls:
                message = {
                    "role": msg.role,
                    "content": "\n".join(text_parts) if text_parts else None
                }
                if tool_calls:
                    message["tool_calls"] = tool_calls
                openai_messages.append(message)

    return openai_messages
