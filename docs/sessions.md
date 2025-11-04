# Sessions & History Repair

The proxy is stateless by default—it forwards the `messages` array exactly as you send it. Some clients, however, drop the last assistant message or strip `<think>` blocks when persisting conversation history. MiniMax-M2 relies on those details to continue reasoning, so missing context can derail tool use.

To mitigate this, the proxy offers an optional session store that can detect and repair gaps in the history.

## When to enable

Enable sessions if:

- Your client intermittently forgets to append the assistant turn or reasoning.
- You are integrating with third-party tools (e.g. editors) that you cannot readily modify.
- You want server-side validation that tool results match the previous tool call.

## Configuration

Set the following environment variables (see [configuration](./configuration.md) for the full list):

```
SESSION_STORE_ENABLED=true
SESSION_STORE_BACKEND=sqlite        # or "memory"
SESSION_STORE_PATH=conversations.db # file or :memory:
SESSION_TTL_SECONDS=3600
MAX_MESSAGES_PER_SESSION=8
REQUIRE_SESSION_FOR_REPAIR=true
```

- `memory` backend keeps data in-process (reset on restart).
- `sqlite` persists messages across restarts and is suitable for single-node deployments.

## Providing a session ID

Session repair only activates when the proxy can identify the conversation. Supply one of:

- HTTP header `X-Session-Id: <id>`
- Query parameter `?conversation_id=<id>`
- For OpenAI requests, `extra_body.conversation_id`

If `REQUIRE_SESSION_FOR_REPAIR` is `true` and no ID is present, the proxy logs a warning and skips repair.

## How repair works

1. On each request, the proxy loads the stored assistant message for the session (if any).
2. It compares the incoming `messages` array to ensure the last assistant turn is present.
3. If missing, the proxy injects the stored assistant message before any `role: tool`/`tool_result` entries and logs the action.
4. After completing the turn, it stores the new assistant message (including `<think>`/`tool_calls` or `reasoning_details`) for future repairs.

This lightweight approach preserves MiniMax’s reasoning chain without fully owning your conversation state.

## Validation helpers

With sessions enabled the proxy can also:

- Reject tool results that do not reference a known `tool_call_id`.
- Ensure reasoning content stays intact even when clients request `reasoning_split`.
- Provide metrics on how often repairs occur.

## Best practices

- Prefer to fix the client whenever possible—session repair is a safety net, not a substitute for correct behaviour.
- Use short TTLs (`SESSION_TTL_SECONDS`) to avoid stale history.
- In clustered deployments, point multiple proxy instances at the same SQLite database or implement a custom session backend (e.g., Redis) using the same interface.

By combining session IDs with the proxy’s repair logic you can maintain MiniMax’s interleaved reasoning guarantees even when upstream clients are imperfect.
