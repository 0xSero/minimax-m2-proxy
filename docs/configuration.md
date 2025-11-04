# Configuration Reference

The proxy is configured via environment variables (or `.env` file when using `pydantic-settings`). This page documents the key options.

## Server settings

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Interface uvicorn binds to. |
| `PORT` | `8001` | Port for the FastAPI app. |
| `TABBY_URL` | `http://localhost:8000` | Base URL of your MiniMax backend (TabbyAPI/vLLM). |
| `TABBY_TIMEOUT` | `300` | Timeout in seconds for backend requests. |

The helper script `scripts/proxy.sh` reads `HOST`, `PORT`, and `PYTHON_BIN`.

## Feature flags

| Variable | Default | Purpose |
|----------|---------|---------|
| `ENABLE_THINKING_PASSTHROUGH` | `true` | Preserve `<think>` tags in responses when `reasoning_split` is not requested. |
| `ENABLE_TOOL_TRANSLATION` | `true` | Convert `<minimax:tool_call>` XML into JSON tool calls. |
| `ENABLE_CHINESE_CHAR_BLOCKING` | `true` | Filter a list of banned Chinese characters (workaround for tokenizer bleed). |

Disable these only if you know your backend or client handles raw XML/Unicode safely.

## Session store (optional)

See [Sessions & History Repair](./sessions.md) for flow details.

| Variable | Default | Description |
|----------|---------|-------------|
| `SESSION_STORE_ENABLED` | `false` | Enable repairing missing assistant turns. |
| `SESSION_STORE_BACKEND` | `sqlite` | `memory` or `sqlite`. |
| `SESSION_STORE_PATH` | `conversations.db` | SQLite file path (ignored for memory backend). |
| `SESSION_TTL_SECONDS` | `3600` | Retention window for session history. |
| `MAX_MESSAGES_PER_SESSION` | `8` | Max assistant messages stored per session. |
| `REQUIRE_SESSION_FOR_REPAIR` | `true` | If `true`, repair occurs only when a session ID is provided. |

## Reasoning & thinking presentation

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_REASONING_SPLIT` | `true` | Allow OpenAI clients to request `extra_body.reasoning_split`. |
| `ENABLE_ANTHROPIC_THINKING_BLOCKS` | `true` | Emit separate `thinking` blocks in Anthropic responses. |

## Validation

| Variable | Default | Description |
|----------|---------|-------------|
| `REQUIRE_TOOL_LINK_VALIDATION` | `true` | Verify that tool results correspond to known tool calls when sessions are enabled. |

## Logging & debugging

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Logging verbosity (`DEBUG`, `INFO`, etc.). |
| `LOG_RAW_RESPONSES` | `false` | Dump raw backend responses to logs (useful for troubleshooting). |
| `ENABLE_STREAMING_DEBUG` | `false` | Emit detailed logs for streaming parser decisions. |
| `STREAMING_DEBUG_PATH` | *unset* | Optional file path where streaming trace logs are written. |

## Security

| Variable | Default | Description |
|----------|---------|-------------|
| `AUTH_API_KEY` | *unset* | If set, the proxy requires `Authorization: Bearer <key>` on requests. |

Use this for simple shared-secret protection when exposing the proxy externally.

## Recommended `.env` skeleton

```
TABBY_URL=http://localhost:8000
HOST=0.0.0.0
PORT=8001
LOG_LEVEL=INFO

# Session repair (optional)
SESSION_STORE_ENABLED=true
SESSION_STORE_BACKEND=sqlite
SESSION_STORE_PATH=conversations.db

# Debugging
ENABLE_STREAMING_DEBUG=false
STREAMING_DEBUG_PATH=
```

Restart the proxy after changing environment variables. For production deployments, integrate these values via systemd unit files, Docker environment variables, or your orchestrator of choice.
