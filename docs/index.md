# MiniMax-M2 Proxy Documentation

Welcome! This documentation explains how to integrate MiniMax-M2 into your own client applications using the MiniMax-M2 proxy. If you are building a custom developer tool, an editor extension, or any UX that already speaks either the OpenAI or Anthropic API surface, this proxy gives you compatible endpoints and the glue to translate MiniMax’s XML tool-calling format into standard JSON.

To get the best experience, work through the sections in order:

1. [Introduction](./introduction.md) — what the proxy does and the pieces involved.
2. [Quick Start](./quickstart.md) — run the proxy locally and send your first request.
3. [OpenAI-Compatible Guide](./openai.md) — how to use the `/v1/chat/completions` endpoint, support `reasoning_split`, and handle tool calls.
4. [Anthropic-Compatible Guide](./anthropic.md) — how to use the `/v1/messages` endpoint with thinking/tool blocks.
5. [Tool Calling Deep Dive](./tool-calling.md) — translating XML `<minimax:tool_call>` output into JSON tool invocations and returning results to the model.
6. [Sessions & History Repair](./sessions.md) — optional stateful features that help keep reasoning intact when clients drop assistant turns.
7. [Configuration Reference](./configuration.md) — environment variables, feature flags, and security knobs.
8. [Troubleshooting](./troubleshooting.md) — common issues and how to diagnose them.

If you are in a hurry, jump to the quick start and then into whichever API you plan to consume. Each guide contains a succinct checklist of requirements from MiniMax’s docs so you never lose crucial reasoning or tool metadata between turns.
