# Troubleshooting

This guide lists common integration issues and how to resolve them. If the symptom you see isn’t covered, enable debug logging (see [configuration](./configuration.md)) and share the relevant snippets.

## Proxy starts but returns 500 errors

- Check backend connectivity: `curl http://localhost:8000/health`.
- Ensure the MiniMax chat template is active. The proxy expects to see `<think>` and `<minimax:tool_call>` in responses.
- Set `LOG_RAW_RESPONSES=true` and inspect the logs for backend error messages.

## Tool calls missing or arguments malformed

- Confirm your tool schema includes correct types (`integer`, `number`, `boolean`, etc.). Without type hints MiniMax emits strings.
- Inspect the raw XML by enabling `ENABLE_STREAMING_DEBUG=true` and view `STREAMING_DEBUG_PATH`.
- Verify you append the full assistant message (including tool metadata) to the next request.

## Thinking content disappears between turns

- Make sure your client (or the proxy sessions feature) preserves `<think>` content.
- If you rely on `reasoning_split=True`, ensure you store both `message.content` and `message.reasoning_details` in your conversation history.
- Enable [session repair](./sessions.md) to patch missing assistant turns automatically.

## Anthropic client sees no `thinking` block

- Check `ENABLE_ANTHROPIC_THINKING_BLOCKS` is `true`.
- Confirm your streaming loop handles `content_block_start` events for the `thinking` type before text.

## Streaming stops prematurely

- Look for `finish_reason` in emitted chunks—`tool_calls` indicates MiniMax wants you to execute tools.
- Make sure your streaming consumer drains the final `[DONE]` message; otherwise, the HTTP connection may linger.
- If the backend disconnects unexpectedly, increase `TABBY_TIMEOUT`.

## Tool results rejected

- When session repair and validation are enabled, the proxy ensures every tool result references a previous tool call. Double-check `tool_call_id` (OpenAI) or `tool_use_id` (Anthropic).
- If you intentionally send a tool result without a prior assistant message, disable `REQUIRE_TOOL_LINK_VALIDATION` or supply a session ID so the proxy can recover.

## Authentication failures

- If `AUTH_API_KEY` is set, you must send `Authorization: Bearer <key>` with each request. SDKs often expose this as `api_key`.
- Verify no extra whitespace or quotes appear in the environment variable.

## Still stuck?

- Re-run with `LOG_LEVEL=DEBUG`.
- Capture the request/response pair (scrub secrets) and open an issue in the repository.
- Check [Quick Start](./quickstart.md) to ensure your environment matches the expected setup.

Remember: most issues trace back to missing assistant history or incorrect tool schemas. Validating those two behaviors resolves the majority of integration problems.
