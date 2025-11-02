# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

MiniMax-M2 Proxy is a translation proxy that converts MiniMax-M2's custom XML-based tool calling format to standard OpenAI and Anthropic API formats. The proxy enables MiniMax-M2 (456B MoE model) to work seamlessly with any OpenAI/Anthropic-compatible tool or framework without modifications.

## Common Commands

### Development

```bash
# Setup environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run proxy (development mode with auto-reload)
uvicorn proxy.main:app --reload --port 8001

# Run proxy (simple script)
./start_proxy.sh

# Run proxy in background with logging
nohup uvicorn proxy.main:app --host 0.0.0.0 --port 8001 > proxy.log 2>&1 &
```

### Testing

```bash
# Run all unit tests
pytest tests/test_parsers.py -v

# Run specific test
pytest tests/test_parsers.py::TestToolCallParser::test_parse_single_tool_call -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# End-to-end tests (requires TabbyAPI running on port 8000)
python tests/test_tool_calling.py
python tests/e2e_test.py
```

### Code Quality

```bash
# Format code
black . --line-length 100

# Lint code
ruff check .

# Auto-fix linting issues
ruff check . --fix
```

### Production Deployment

```bash
# Install systemd service
sudo cp minimax-m2-proxy.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl start minimax-m2-proxy
sudo systemctl enable minimax-m2-proxy

# Check service status
sudo systemctl status minimax-m2-proxy
sudo journalctl -u minimax-m2-proxy -f
```

## Architecture

### Request Flow

```
Client (OpenAI/Anthropic SDK)
    ↓
FastAPI endpoints (/v1/chat/completions or /v1/messages)
    ↓
Format conversion (Anthropic → OpenAI if needed)
    ↓
TabbyAPI/vLLM client (HTTP request to backend)
    ↓
MiniMax-M2 model (generates XML tool calls)
    ↓
ToolCallParser (XML → JSON conversion)
    ↓
Formatter (OpenAI or Anthropic response format)
    ↓
Client receives standard API response
```

### Core Components

**proxy/main.py** - FastAPI application with dual API endpoints
- `/v1/chat/completions` - OpenAI-compatible endpoint (streaming & non-streaming)
- `/v1/messages` - Anthropic-compatible endpoint (streaming & non-streaming)
- `/health` - Health check for proxy and backend
- `/v1/models`, `/v1/model` - Pass-through endpoints to backend
- Global instances: `tabby_client`, `tool_parser`, `openai_formatter`, `anthropic_formatter`
- Lifespan manager handles client initialization and cleanup

**proxy/client.py** - TabbyClient HTTP client wrapper
- Manages async HTTP connections to TabbyAPI/vLLM backend
- Provides `chat_completion()` for non-streaming requests
- Provides `extract_streaming_content()` for SSE streaming
- Handles health checks and connection management

**proxy/models.py** - Pydantic request/response models
- `OpenAIChatRequest` - OpenAI chat completion request schema
- `AnthropicChatRequest` - Anthropic messages request schema
- Conversion utilities: `anthropic_tools_to_openai()`, `anthropic_messages_to_openai()`
- Type validation for all API payloads

**proxy/config.py** - Environment configuration using pydantic-settings
- Loads settings from `.env` file
- Key settings: `TABBY_URL`, `TABBY_TIMEOUT`, `HOST`, `PORT`, `LOG_LEVEL`, `LOG_RAW_RESPONSES`

**parsers/tools.py** - XML tool call parser (adapted from vLLM)
- `ToolCallParser.parse_tool_calls()` - Extract tool calls from complete text
- Parses `<minimax:tool_call>` with `<invoke>` and `<parameter>` tags
- Type conversion based on tool schema (int, float, bool, JSON objects/arrays)
- Generates unique tool call IDs (`call_<uuid>`)
- Returns dict with `tools_called`, `tool_calls`, and `content` (without XML blocks)

**parsers/streaming.py** - Streaming state machine
- `SimpleStreamingParser.process_chunk()` - Process incremental text chunks
- Buffers text until complete tool calls detected
- Sends content deltas in real-time, tool calls when complete
- Separates content from `<minimax:tool_call>` XML blocks

**formatters/openai.py** - OpenAI format generator
- `format_complete_response()` - Non-streaming Chat Completion response
- `format_streaming_chunk()` - SSE chunks with proper delta format
- `format_tool_call_stream()` - Multi-chunk tool call streaming (id → name → arguments)
- Generates response IDs, timestamps, and usage statistics

**formatters/anthropic.py** - Anthropic format generator
- `format_complete_response()` - Messages API response with content blocks
- Event-based streaming: `message_start`, `content_block_start`, `content_block_delta`, `content_block_stop`, `message_delta`, `message_stop`
- Converts tool calls to `tool_use` blocks with proper structure
- Maps finish reasons: `stop` → `end_turn`, `tool_calls` → `tool_use`

### Key Design Decisions

**Stateless Proxy**: Each request is independent with no conversation history management. The proxy only translates formats in real-time.

**Think Block Preservation**: MiniMax-M2's `<think>` blocks are kept verbatim in responses to maintain transparency into the model's reasoning process.

**Type Inference**: Parameter types are converted based on tool schemas. Without schema info, defaults to string. Supports: string, integer, number (float), boolean, null, object, array.

**Streaming Strategy**: Simple buffering approach - stream content immediately, buffer tool calls until complete. More robust than incremental tool call streaming for Phase 1.

**Dual API Support**: Same backend, two frontend formats. Anthropic requests are converted to OpenAI format internally, then response is converted back to Anthropic format.

## Testing Architecture

**tests/test_parsers.py** - 16+ unit tests covering:
- Single and multiple tool call parsing
- Type conversion (int, float, bool, null, JSON)
- Edge cases (empty parameters, special characters, nested objects)
- Content extraction without tool blocks
- Streaming parser state machine

**tests/test_tool_calling.py** - E2E integration tests (requires live backend):
- Think block preservation
- Single tool call round-trip (OpenAI format)
- Multi-step tool calling (tool → result → final answer)
- Anthropic format compatibility
- Streaming with tool calls

Test conventions:
- Use `pytest` with async support (`pytest-asyncio`)
- Fixtures in `setup_method()`
- Descriptive test names: `test_<action>_<scenario>`
- Assert on structure first, then values

## Configuration

Environment variables (`.env`):
- `TABBY_URL` - Backend endpoint (default: http://localhost:8000)
- `TABBY_TIMEOUT` - Request timeout in seconds (default: 300)
- `HOST` - Proxy bind address (default: 0.0.0.0)
- `PORT` - Proxy port (default: 8001)
- `LOG_LEVEL` - Logging level (default: INFO)
- `LOG_RAW_RESPONSES` - Debug flag to log raw TabbyAPI responses (default: false)

## MiniMax-M2 XML Format

MiniMax-M2 outputs tool calls in this XML structure:

```xml
<think>
Internal reasoning goes here...
</think>

<minimax:tool_call>
<invoke name="function_name">
<parameter name="param1">value1</parameter>
<parameter name="param2">value2</parameter>
</invoke>
<invoke name="another_function">
<parameter name="param">value</parameter>
</invoke>
</minimax:tool_call>

Optional text response after tool calls.
```

Key parsing rules:
- Multiple `<invoke>` blocks within one `<minimax:tool_call>` = multiple tool calls
- Parameter values are strings by default, converted using tool schema
- `<think>` blocks are preserved in final response content
- Tool call XML is removed from content, only reasoning/text remains

## Dependencies

Core:
- **FastAPI** - ASGI web framework with async support
- **uvicorn** - ASGI server for production
- **httpx** - Async HTTP client for backend communication
- **pydantic** - Data validation and settings management

Dev/Testing:
- **pytest** - Test framework with async support
- **pytest-asyncio** - Async test support
- **pytest-cov** - Coverage reporting
- **black** - Code formatter (100 char line length)
- **ruff** - Fast Python linter

## Common Development Tasks

### Adding a new tool parsing feature

1. Update `parsers/tools.py` - Modify `ToolCallParser._parse_single_invoke()` or regex patterns
2. Add unit tests in `tests/test_parsers.py`
3. Test with E2E tests in `tests/test_tool_calling.py`

### Supporting a new API format

1. Create new formatter in `formatters/<format>.py`
2. Add Pydantic models in `proxy/models.py`
3. Add endpoint in `proxy/main.py` with streaming/non-streaming handlers
4. Add conversion utilities if needed

### Debugging tool call parsing

1. Set `LOG_RAW_RESPONSES=true` in `.env`
2. Check logs for raw TabbyAPI output
3. Verify chat template is loaded in TabbyAPI (check TabbyAPI logs)
4. Test regex patterns in Python REPL with sample XML

### Performance optimization

Current bottleneck is model inference, not proxy. Proxy adds ~50ms overhead for non-streaming, ~10ms for streaming first token.

To optimize:
- Increase TabbyAPI `max_batch_size` in config
- Monitor GPU utilization with `nvidia-smi`
- Use streaming for better perceived latency
- Check network latency between proxy and backend
