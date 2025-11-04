# Tool Calling Deep Dive

MiniMax-M2 emits XML tool calls of the form:

```xml
<minimax:tool_call>
  <invoke name="search_web">
    <parameter name="query">best restaurants in Tokyo</parameter>
  </invoke>
</minimax:tool_call>
```

The proxy parses these blocks and translates them into OpenAI `tool_calls` or Anthropic `tool_use` entries. This page explains how the translation works and how to wire your tool execution logic.

## Declaring tools

Define tools using the native schema of the API you consume:

- **OpenAI** — `{"type": "function", "function": {...}}`
- **Anthropic** — `{"name": "...", "input_schema": {...}}`

The proxy forwards the JSON schema to MiniMax so it can format arguments correctly (strings, numbers, booleans, nested JSON, etc.).

## Parsing results

### OpenAI response

```json
"tool_calls": [
  {
    "id": "call_f3a...",
    "type": "function",
    "function": {
      "name": "search_web",
      "arguments": "{\"query\": \"best restaurants in Tokyo\"}"
    }
  }
]
```

### Anthropic response

```json
{
  "type": "tool_use",
  "id": "toolu_01",
  "name": "search_web",
  "input": {
    "query": "best restaurants in Tokyo"
  }
}
```

The proxy ensures argument types match your schema:

- Integers and floats remain numeric.
- Booleans become `true`/`false`.
- Objects and arrays are parsed from JSON strings in the XML when present.

## Executing tools

OpenAI example:

```python
tool_call = response.choices[0].message.tool_calls[0]
args = json.loads(tool_call.function.arguments)
result = search_web(**args)

messages.append(response.choices[0].message.model_dump())
messages.append({
    "role": "tool",
    "tool_call_id": tool_call.id,
    "content": json.dumps(result),
})

client.chat.completions.create(model="minimax-m2", messages=messages)
```

Anthropic example:

```python
assistant_turn = message.content
tool_block = next(block for block in assistant_turn if block.type == "tool_use")

result = search_web(**tool_block.input)

messages.append({"role": "assistant", "content": assistant_turn})
messages.append({
    "role": "user",
    "content": [{
        "type": "tool_result",
        "tool_use_id": tool_block.id,
        "content": [{"type": "text", "text": json.dumps(result)}],
    }],
})

client.messages.create(model="MiniMax-M2", messages=messages)
```

## Returning structured results

Tool results are plain text by default. If you want richer output:

- OpenAI: include structured JSON in the `content` string or adopt a custom schema your client understands.
- Anthropic: `tool_result.content` can be a list of blocks (`text`, `json`, etc.) so you can return multiple representations.

MiniMax expects the tool result to appear immediately before the next assistant call. Sending multiple tool results per assistant is fine—just ensure each `tool_call_id`/`tool_use_id` matches the original.

## Multiple tool calls in one message

MiniMax can produce multiple `<invoke>` elements inside a single `<minimax:tool_call>`. The proxy emits a `tool_calls` array with one entry per invocation. Execute them sequentially; after each result, append the corresponding tool message. Only when all tool calls are satisfied should you request the next assistant turn.

## Error handling

If a tool fails:

1. Return a descriptive error string in the tool result.
2. Let the model decide how to proceed—it may call another tool or answer with a fallback.
3. Optionally record the error in your telemetry.

The proxy does not attempt to recover from tool-side errors; they are treated as normal tool results.

## Debugging

Enable streaming debug logs to inspect raw XML and parser decisions:

```bash
ENABLE_STREAMING_DEBUG=true STREAMING_DEBUG_PATH=stream.log scripts/proxy.sh dev
```

Each streaming chunk and parsed tool event is recorded, making it easy to trace argument conversion issues or unexpected XML.

With these practices, you can safely expose MiniMax-M2 tool calling through your existing OpenAI or Anthropic integration while retaining type fidelity and reasoning integrity.
