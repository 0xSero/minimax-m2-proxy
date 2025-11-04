# Introduction

MiniMax-M2 is a 456B MoE model that speaks a bespoke protocol: its reasoning is wrapped in `<think>` tags and tool calls are returned as XML `<minimax:tool_call>` blocks. Most clients, SDKs, and agent frameworks cannot parse those formats directly.

The MiniMax-M2 proxy sits between your client (OpenAI/Anthropic compatible) and a backend running MiniMax-M2 (e.g. TabbyAPI or vLLM). It:

- Exposes the familiar OpenAI `POST /v1/chat/completions` and Anthropic `POST /v1/messages` endpoints.
- Forwards your original messages to MiniMax-M2.
- Parses `<think>` blocks and `<minimax:tool_call>` XML, converting them into native JSON (`reasoning_details`, `tool_calls`, `tool_use`, etc.).
- Streams deltas so your UI can display ongoing thinking or tool-call arguments in real time.

## When to use it

Use the proxy if you already integrate with OpenAI or Anthropic APIs and want to try MiniMax-M2 without refactoring your client. The proxy is particularly helpful for:

- IDE co-pilots needing reliable tool-calling.
- Workflow automation tools that must maintain reasoning chains across turns.
- Custom chat applications that prefer standard API contracts.

## Requirements

- Python 3.11+
- A MiniMax-M2 backend (TabbyAPI, vLLM, or another host running the official MiniMax chat template).
- Your client speaks either the OpenAI or Anthropic messaging format.

## Architecture overview

```
Your Client (OpenAI/Anthropic SDK)
        │
        ▼
MiniMax-M2 Proxy  ──► TabbyAPI/vLLM ──► MiniMax-M2 Model
        │
        ▼
Standard JSON responses with thinking + tool calls
```

The proxy is stateless by default. Optional session features allow it to repair missing assistant turns to prevent broken reasoning chains when clients forget to append the last assistant message.
