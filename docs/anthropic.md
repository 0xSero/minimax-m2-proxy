# Anthropic-Compatible Guide

This guide explains how to use the MiniMax-M2 proxy through the Anthropic Messages API. The proxy exposes `POST /v1/messages` and emits Anthropic-style content blocks so existing SDK clients work out of the box.

## Endpoint and URL

- Base URL: `http://<proxy-host>:8001`
- Endpoint: `POST /v1/messages`
- Model name: `MiniMax-M2`

The proxy also accepts requests on `/anthropic/v1/messages` if you prefer Anthropic’s official base path (configure a reverse proxy accordingly).

## Minimal request

```python
import anthropic

client = anthropic.Anthropic(base_url="http://localhost:8001", api_key="dummy")

message = client.messages.create(
    model="MiniMax-M2",
    max_tokens=512,
    messages=[
        {"role": "user", "content": [{"type": "text", "text": "Outline a launch plan."}]}
    ],
)

for block in message.content:
    if block.type == "thinking":
        print("Thinking:", block.thinking)
    elif block.type == "text":
        print("Assistant:", block.text)
```

## Response structure

The proxy mirrors Anthropic’s block types:

- `thinking` blocks contain the contents of `<think>…</think>`.
- `text` blocks contain visible content.
- `tool_use` blocks map to each function call.
- Streaming events follow the `message_start` → `content_block_start`/`delta` → `content_block_stop` → `message_stop` pattern.

Example reply:

```json
{
  "content": [
    {
      "type": "thinking",
      "thinking": "Research competitors before drafting."
    },
    {
      "type": "text",
      "text": "Here is a launch outline..."
    }
  ],
  "stop_reason": "end_turn"
}
```

## Tool calls

### Declaring tools

Anthropic tools use `name`, `description`, and `input_schema`:

```python
tools = [
    {
        "name": "get_weather",
        "description": "Fetch weather for a city.",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location", "unit"]
        }
    }
]

message = client.messages.create(
    model="MiniMax-M2",
    max_tokens=512,
    messages=[...],
    tools=tools,
)
```

### Handling `tool_use`

When MiniMax decides to call a tool, you will receive a block similar to:

```json
{
  "type": "tool_use",
  "id": "toolu_01",
  "name": "get_weather",
  "input": {
    "location": "Tokyo",
    "unit": "celsius"
  }
}
```

Execute your function, then append a new user message with a `tool_result` block referencing the `tool_use_id`:

```python
messages.append({
    "role": "assistant",
    "content": response.content,  # the original thinking/text/tool_use blocks
})
messages.append({
    "role": "user",
    "content": [{
        "type": "tool_result",
        "tool_use_id": tool_block.id,
        "content": [{"type": "text", "text": json.dumps(result)}],
    }],
})
```

Resubmit the updated history to `/v1/messages` for the follow-up turn. Do not omit the assistant turn—the proxy can optionally repair missing history (see [sessions](./sessions.md)), but complete transcripts yield better behaviour.

## Streaming

Streaming is fully supported through SSE. The Anthropic SDK handles this automatically:

```python
stream = client.messages.create(
    model="MiniMax-M2",
    messages=[...],
    stream=True,
)

for event in stream:
    if event.type == "content_block_delta":
        if event.delta.type == "thinking_delta":
            print("Thinking:", event.delta.thinking, end="", flush=True)
        elif event.delta.type == "text_delta":
            print("Text:", event.delta.text, end="", flush=True)
```

The proxy emits a separate thinking block before visible text, ensuring your UI can highlight reasoning while it arrives.

## Parameter handling

- `temperature` must be in `(0.0, 1.0]`.
- The proxy ignores unsupported fields (`container`, `mcp_servers`, etc.).
- `max_tokens`, `top_p`, `metadata`, and `tool_choice` behave as they do in Anthropic’s API.

## History best practices

MiniMax expects every assistant response to be appended back into the conversation, including thinking blocks and all tool metadata. Follow this pattern:

1. After each assistant reply, append the entire `message.content` list to history.
2. After executing a tool, send a `tool_result` block referencing the `tool_use_id`.
3. If your client cannot maintain perfect history, enable [session repair](./sessions.md) and set a `conversation_id` header so the proxy can fill gaps.

With these steps you can reuse your existing Anthropic-compatible client to talk to MiniMax-M2, retaining its rich reasoning and tool-calling abilities. Explore the [tool calling deep dive](./tool-calling.md) for more details on argument parsing and result formats.
