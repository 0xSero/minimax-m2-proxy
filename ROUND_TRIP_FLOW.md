# Anthropic ↔ MiniMax Round-Trip Flow

This document traces the complete data flow from Anthropic format through the proxy to MiniMax and back, showing what's preserved and what's lost at each step.

## Architecture Overview

```
Anthropic Client
    ↓ (Anthropic Messages API format)
/v1/messages endpoint
    ↓ (anthropic_messages_to_openai)
OpenAI format
    ↓ (transform_messages_for_minimax)
OpenAI format (tool → user)
    ↓ (HTTP POST to TabbyAPI)
TabbyAPI /v1/chat/completions
    ↓ (apply_chat_template)
MiniMax prompt with message markers
    ↓ (model inference)
MiniMax XML output with <think> blocks
    ↓ (TabbyAPI response)
OpenAI format (raw XML in content)
    ↓ (ensure_think_wrapped + parse_tool_calls)
Parsed content + tool calls
    ↓ (anthropic_formatter)
Anthropic Messages API format
    ↓ (HTTP response)
Anthropic Client
```

## Detailed Flow: Tool Calling Example

### Step 1: Client Request → Proxy

**Input (Anthropic format):**
```json
POST /v1/messages
{
  "model": "minimax-m2",
  "messages": [{"role": "user", "content": "What's the weather in SF?"}],
  "tools": [{
    "name": "get_weather",
    "description": "Get weather",
    "input_schema": {
      "type": "object",
      "properties": {
        "location": {"type": "string"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
      },
      "required": ["location", "unit"]
    }
  }]
}
```

### Step 2: Anthropic → OpenAI Conversion

**Code:** `proxy/models.py:98-143` - `anthropic_messages_to_openai()` and `anthropic_tools_to_openai()`

**Output (OpenAI format):**
```json
{
  "messages": [{"role": "user", "content": "What's the weather in SF?"}],
  "tools": [{
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "Get weather",
      "parameters": {
        "type": "object",
        "properties": {
          "location": {"type": "string"},
          "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
        },
        "required": ["location", "unit"]
      }
    }
  }]
}
```

**Changes:**
- ✅ Preserved: Message content, tool definitions, parameter schemas
- 🔄 Converted: `input_schema` → `parameters`
- 🔄 Wrapped: Tool in `{type: "function", function: {...}}`

### Step 3: Message Transformation

**Code:** `proxy/message_transformer.py:13-85` - `transform_messages_for_minimax()`

**Effect:** No change yet (no tool results in this request)

### Step 4: Proxy → TabbyAPI

**HTTP Request:**
```http
POST http://localhost:8000/v1/chat/completions
Content-Type: application/json

{
  "model": "minimax-m2",
  "messages": [{"role": "user", "content": "What's the weather in SF?"}],
  "tools": [...],
  "tool_choice": "auto",
  "temperature": 1.0,
  "max_tokens": 4096
}
```

### Step 5: TabbyAPI Applies Chat Template

**Code:** `chat_template.jinja` in model directory

**Rendered Prompt:**
```
]~!b[]~b]system
You are a helpful assistant.

# Tools
You may call one or more tools to assist with the user query.
Here are the tools available in JSONSchema format:

<tools>
<tool>{"name": "get_weather", "description": "Get weather", "parameters": {...}}</tool>
</tools>

When making tool calls, use XML format to invoke tools and pass parameters:

<minimax:tool_call>
<invoke name="tool-name-1">
<parameter name="param-key-1">param-value-1</parameter>
<parameter name="param-key-2">param-value-2</parameter>
...
</invoke>
[e~[
]~b]user
What's the weather in SF?[e~[
]~b]ai
<think>
```

**Changes:**
- ➕ Added: Message markers (`]~b]system`, `]~b]user`, `]~b]ai`, `[e~[`)
- ➕ Added: Tool instructions and XML format example
- ➕ Added: Opening `<think>` tag for generation

### Step 6: MiniMax Model Generates

**Model Output:**
```xml
The user wants weather for San Francisco. I'll use the get_weather tool with celsius units.
</think>

<minimax:tool_call>
<invoke name="get_weather">
<parameter name="location">San Francisco, CA</parameter>
<parameter name="unit">celsius</parameter>
</invoke>
</minimax:tool_call>
```

**Note:** Opening `<think>` tag is NOT in the completion (it's part of the prompt)

### Step 7: TabbyAPI → Proxy Response

**HTTP Response:**
```json
{
  "id": "cmpl-xxx",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "minimax-m2",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "The user wants weather for San Francisco. I'll use the get_weather tool with celsius units.\n</think>\n\n<minimax:tool_call>\n<invoke name=\"get_weather\">\n<parameter name=\"location\">San Francisco, CA</parameter>\n<parameter name=\"unit\">celsius</parameter>\n</invoke>\n</minimax:tool_call>"
    },
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
}
```

**Issues:**
- ⚠️ Missing opening `<think>` tag (model didn't generate it, was in prompt)
- ⚠️ `finish_reason` is "stop" instead of "tool_calls"
- ⚠️ Token counts are 0 (TabbyAPI doesn't track them)

### Step 8: Proxy Parses Response

**Code:** `parsers/reasoning.py:1-34` - `ensure_think_wrapped()`

```python
# Detects </think> without opening tag, adds it:
raw_content = "<think>\nThe user wants weather for San Francisco...\n</think>\n\n<minimax:tool_call>..."
```

**Code:** `parsers/tools.py:161-227` - `parse_tool_calls()`

```python
result = {
  "tools_called": True,
  "tool_calls": [{
    "id": "call_abc123",  # Generated UUID
    "type": "function",
    "function": {
      "name": "get_weather",
      "arguments": '{"location": "San Francisco, CA", "unit": "celsius"}'
    }
  }],
  "content": "<think>\nThe user wants weather for San Francisco. I'll use the get_weather tool with celsius units.\n</think>"
}
```

**Changes:**
- ✅ Preserved: Think block content, tool parameters
- ➕ Added: Missing opening `<think>` tag
- ➕ Added: Tool call ID (UUID)
- 🔄 Converted: XML parameters → JSON arguments
- ❌ Removed: XML tags from content

### Step 9: OpenAI → Anthropic Conversion

**Code:** `formatters/anthropic.py:13-65` - `format_complete_response()`

```python
response = {
  "id": "msg_xyz789",
  "type": "message",
  "role": "assistant",
  "content": [
    {
      "type": "text",
      "text": "<think>\nThe user wants weather for San Francisco. I'll use the get_weather tool with celsius units.\n</think>"
    },
    {
      "type": "tool_use",
      "id": "call_abc123",
      "name": "get_weather",
      "input": {
        "location": "San Francisco, CA",
        "unit": "celsius"
      }
    }
  ],
  "model": "minimax-m2",
  "stop_reason": "tool_use",
  "stop_sequence": null,
  "usage": {"input_tokens": 0, "output_tokens": 0}
}
```

**Changes:**
- ✅ Preserved: Think blocks (in text content), tool calls
- 🔄 Converted: Tool calls to `tool_use` blocks
- 🔄 Converted: JSON arguments → `input` object
- ✅ Fixed: `stop_reason` is now "tool_use"

### Step 10: Proxy → Client Response

Client receives proper Anthropic Messages format with think blocks and tool calls!

---

## Round-Trip 2: Tool Result → Final Answer

### Client Sends Tool Result

**Request:**
```json
POST /v1/messages
{
  "messages": [
    {"role": "user", "content": "What's the weather in SF?"},
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": "<think>...</think>"},
        {"type": "tool_use", "id": "call_abc123", "name": "get_weather", "input": {...}}
      ]
    },
    {
      "role": "user",
      "content": [
        {
          "type": "tool_result",
          "tool_use_id": "call_abc123",
          "content": "Temperature: 18°C, Sunny"
        }
      ]
    }
  ]
}
```

### Anthropic → OpenAI Conversion

**Code:** `proxy/models.py:126-132`

```python
# tool_result block becomes role: "tool" message
openai_messages.append({
  "role": "tool",
  "tool_call_id": "call_abc123",
  "content": "Temperature: 18°C, Sunny"
})
```

### Message Transformation for MiniMax

**Code:** `proxy/message_transformer.py:69-72`

```python
# MiniMax doesn't understand role: "tool", convert to role: "user"
transformed_messages.append({
  "role": "user",
  "content": "Tool Result (get_weather):\nTemperature: 18°C, Sunny"
})
```

**This is critical!** MiniMax official docs say it doesn't support `role: "tool"`, must use `role: "user"`.

### MiniMax Generates Final Response

**Model Output:**
```
Great! I have the weather data. Let me respond naturally.
</think>

The current weather in San Francisco is 18°C and sunny!
```

### Client Receives Final Answer

```json
{
  "content": [
    {
      "type": "text",
      "text": "<think>\nGreat! I have the weather data. Let me respond naturally.\n</think>\n\nThe current weather in San Francisco is 18°C and sunny!"
    }
  ],
  "stop_reason": "end_turn"
}
```

---

## What's Preserved ✅

1. **Message content** - All text preserved exactly
2. **Think blocks** - Kept verbatim in text content
3. **Tool calls** - Correctly converted between formats
4. **Tool parameters** - Type conversion works (string, int, float, bool, arrays, objects)
5. **Tool results** - Transformed but semantically identical
6. **Multi-turn conversations** - Full history maintained
7. **Function names** - Preserved across all conversions
8. **Stop reasons** - Correctly mapped (end_turn, tool_use, max_tokens)

## What's Lost ❌

1. **Token counts** - Set to 0 (TabbyAPI doesn't provide accurate counts)
2. **Original tool call IDs** - We generate new UUIDs when parsing XML
3. **Anthropic thinking config** - `thinking` parameter in request is ignored
4. **Anthropic system blocks** - System messages are flattened to strings
5. **Image content** - Not supported (MiniMax is text-only)
6. **Anthropic content block IDs** - Not preserved across round-trips

## What's Transformed 🔄

1. **Message markers** - Added by TabbyAPI template, removed by proxy
2. **Tool call format** - XML ↔ JSON conversion
3. **Tool result role** - `role: "tool"` → `role: "user"` (required by MiniMax)
4. **Opening think tags** - Added by proxy when missing
5. **Tool definitions** - `input_schema` ↔ `parameters`

## Known Issues ⚠️

1. **Token counts are always 0** - TabbyAPI doesn't expose token counts
2. **Think tags must be balanced** - Proxy adds missing opening tags
3. **Tool IDs change** - UUIDs generated by proxy, not preserved from model
4. **System message format** - Anthropic system blocks flattened to plain text

## Recommendations

1. ✅ **Current architecture is correct** - Using OpenAI as intermediate format leverages TabbyAPI infrastructure
2. ✅ **Tool result transformation is necessary** - MiniMax doesn't support `role: "tool"`
3. ✅ **Think block preservation works** - Content is passed through unchanged
4. 🔧 **Could improve:** Add token counting (estimate from tokenizer)
5. 🔧 **Could improve:** Preserve tool call IDs if TabbyAPI provides them
