# OpenAI-Compatible Guide

This guide covers everything you need to integrate the MiniMax-M2 proxy via the OpenAI Chat Completions API (`POST /v1/chat/completions`). The proxy mirrors OpenAI’s schema so your existing client should work after pointing it at the proxy URL.

## Endpoint and URL

- Base URL: `http://<proxy-host>:8001/v1`
- Endpoint: `POST /chat/completions`
- Models: use `"minimax-m2"` (or any alias you configured in TabbyAPI)

## Minimal request

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8001/v1", api_key="dummy")

response = client.chat.completions.create(
    model="minimax-m2",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Summarise MiniMax-M2 in one paragraph."},
    ],
)

print(response.choices[0].message.content)
```

> **Authentication**  
> If you enable `AUTH_API_KEY` (see [configuration](./configuration.md)), set `api_key` accordingly. Otherwise the proxy ignores the token.

## Response shape

- `message.content` contains the full assistant message, including `<think>` blocks unless you enable reasoning split.
- When tool calls are present, `message.tool_calls` mirrors the standard OpenAI structure.
- `finish_reason` becomes `"tool_calls"` whenever the model requests a tool.

Example:

```json
{
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "<think>Comparing weather APIs...</think>\nLet's call get_weather.",
        "tool_calls": [
          {
            "id": "call_123",
            "type": "function",
            "function": {
              "name": "get_weather",
              "arguments": "{\"location\": \"Tokyo\", \"unit\": \"celsius\"}"
            }
          }
        ]
      },
      "finish_reason": "tool_calls"
    }
  ]
}
```

## Reasoning Split (`reasoning_split=True`)

MiniMax requires clients to keep the full assistant turn (with `<think>`) in history. Some OpenAI SDK clients prefer `reasoning_details`, so the proxy supports that flag.

```python
response = client.chat.completions.create(
    model="minimax-m2",
    messages=[...],
    extra_body={"reasoning_split": True},
)

print(response.choices[0].message.reasoning_details[0]["text"])
print(response.choices[0].message.content)
```

With the flag:

- `message.reasoning_details` contains the stripped thinking text.
- `message.content` omits `<think>` so you can display visible content directly.
- Internally, the proxy still stores the merged message (including thinking) when [session repair](./sessions.md) is enabled, ensuring consistency.

Streaming responses propagate reasoning deltas via `delta.reasoning_details`.

## Tool calls

### Declaring tools

Provide tools using the standard OpenAI schema:

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather by city",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location", "unit"],
            },
        },
    }
]

response = client.chat.completions.create(
    model="minimax-m2",
    messages=[...],
    tools=tools,
)
```

The proxy forwards the definitions to MiniMax-M2 and converts the XML response into the `tool_calls` array shown earlier.

### Executing tools

When a tool call arrives:

1. Parse the arguments with `json.loads(tool_call.function.arguments)`.
2. Execute your function.
3. Append a `{"role": "tool", "tool_call_id": ..., "content": ...}` message to history.
4. Resend the conversation to `/chat/completions`.

**Do not drop `<think>`**. Always append the full assistant message (including reasoning) before the tool result. If your UI hides it, still keep the raw data in memory.

See [Tool Calling Deep Dive](./tool-calling.md) for detailed examples.

## Streaming

The proxy supports SSE streaming identical to OpenAI:

```python
stream = client.chat.completions.create(..., stream=True)
for chunk in stream:
    delta = chunk.choices[0].delta
    if delta.get("reasoning_details"):
        print("Thinking:", delta.reasoning_details[0]["text"], end="")
    if delta.get("content"):
        print("Content:", delta.content, end="")
```

During tool calls, the proxy yields `tool_calls` deltas containing IDs, names, and argument fragments. Use the same assembly logic you would for GPT models.

## Multi-turn conversations

MiniMax requires strict history handling:

- Append every assistant message exactly as returned (with `tool_calls` and thinking).
- Append each tool result as a `role: tool` message referencing `tool_call_id`.
- Resend the full conversation on subsequent turns.

If your client occasionally drops assistant messages or reasoning, consider enabling [session repair](./sessions.md) so the proxy can patch the history automatically.

## Additional validation

- `temperature` must be in `(0.0, 1.0]`.
- `n` is fixed to `1`.
- Unsupported OpenAI parameters (e.g. `presence_penalty`) are safely ignored.

With these guidelines your OpenAI-compatible client should interoperate seamlessly with MiniMax-M2 through the proxy. Next, explore the [tool calling deep dive](./tool-calling.md) for more complex workflows or jump to [Anthropic integration](./anthropic.md) if you also support that API.
