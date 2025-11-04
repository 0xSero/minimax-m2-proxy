# MiniMax-M2 Proxy

A translation proxy that makes [MiniMax-M2](https://huggingface.co/MiniMaxAI/MiniMax-Text-01) compatible with standard OpenAI and Anthropic APIs.

## Why This Exists

MiniMax-M2 is a powerful 229B MoE model with exceptional reasoning capabilities, but it uses a **custom XML-based format** for tool calling that most UIs and frameworks don't understand:

```xml
<think>Let me analyze this request...</think>

<minimax:tool_call>
<invoke name="get_weather">
<parameter name="location">Tokyo</parameter>
<parameter name="unit">celsius</parameter>
</invoke>
</minimax:tool_call>
```

Standard tools like:
- OpenAI Python SDK
- Anthropic Python SDK
- LangChain / LlamaIndex
- OpenWebUI / LibreChat
- Continue.dev / Aider

**Cannot parse this format.** They expect OpenAI's `tool_calls` array or Anthropic's `tool_use` blocks.

### The Problem

Without this proxy, you have two bad options:

1. **Use a generic chat template** - MiniMax-M2 loses its tool-calling abilities entirely
2. **Write custom parsers** - Every UI/framework needs to implement MiniMax-M2-specific XML parsing

### The Solution

This proxy sits between your application and MiniMax-M2, automatically translating:

- **Tool Calls**: `<minimax:tool_call>` XML → OpenAI/Anthropic JSON format
- **Thinking Blocks**: Preserves `<think>` blocks for transparency
- **Streaming**: Full SSE support for both OpenAI and Anthropic formats
- **Type Safety**: Converts parameter types (int, float, bool, JSON) based on tool schemas

Now you can use MiniMax-M2 with **any standard tool** without modifications.

## Architecture

```
┌─────────────────────────┐
│  Your App / UI          │
│  (OpenAI/Anthropic SDK) │
└──────────┬──────────────┘
           │ Standard API Format
           ▼
┌─────────────────────────┐
│  MiniMax-M2 Proxy       │  ← This project
│  (Port 8001)            │
│                         │
│  • Parse XML tool calls │
│  • Preserve <think>     │
│  • Format translation   │
│  • Type conversion      │
└──────────┬──────────────┘
           │ MiniMax format
           ▼
┌─────────────────────────┐
│  TabbyAPI / vLLM        │
│  (Port 8000)            │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  MiniMax-M2 Model       │
│  (456B MoE)             │
└─────────────────────────┘
```

## Quick Start

### 1. Prerequisites

- Python 3.11+
- [TabbyAPI](https://github.com/theroyallab/tabbyAPI) or vLLM running MiniMax-M2 on port 8000
- MiniMax-M2 chat template configured (see [templates](#chat-templates))

### 2. Install

```bash
git clone https://github.com/0xSero/minimax-m2-proxy.git
cd minimax-m2-proxy

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure

```bash
cp .env.example .env
# Edit .env if needed (defaults to localhost:8000)
```

### 4. Run

```bash
# Development
uvicorn proxy.main:app --reload --port 8001

# Production (systemd)
sudo cp minimax-m2-proxy.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl start minimax-m2-proxy
sudo systemctl enable minimax-m2-proxy
```

### 5. Use with OpenAI SDK

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8001/v1",
    api_key="not-needed"
)

# Tool calling works automatically!
response = client.chat.completions.create(
    model="minimax-m2",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location"]
            }
        }
    }]
)

# Model returns tool calls in standard OpenAI format
if response.choices[0].message.tool_calls:
    print(response.choices[0].message.tool_calls[0].function.name)
    # Output: get_weather
```

### 6. Use with Anthropic SDK

```python
import anthropic

client = anthropic.Anthropic(
    base_url="http://localhost:8001",
    api_key="not-needed"
)

message = client.messages.create(
    model="minimax-m2",
    max_tokens=500,
    messages=[{"role": "user", "content": "Search for MiniMax AI"}],
    tools=[{
        "name": "web_search",
        "description": "Search the web",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"]
        }
    }]
)

# Model returns tool_use blocks in Anthropic format
for block in message.content:
    if block.type == "tool_use":
        print(f"{block.name}: {block.input}")
```

## Features

### ✅ Dual API Support

- **OpenAI**: `/v1/chat/completions` (streaming & non-streaming)
- **Anthropic**: `/v1/messages` (streaming & non-streaming)
- **Pass-through**: `/v1/models`, `/v1/model` endpoints

### ✅ Smart XML Parsing

Based on [vLLM's minimax_m2_tool_parser](https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/tool_parsers/minimax_m2_tool_parser.py):

- Parses `<minimax:tool_call>` with multiple `<invoke>` blocks
- Extracts `<parameter>` tags with type inference
- Handles nested JSON objects and arrays
- Generates unique tool call IDs

### ✅ Type Conversion

Automatically converts parameter types based on tool schemas:

```python
# XML: <parameter name="count">42</parameter>
# Schema: {"count": {"type": "integer"}}
# Result: {"count": 42}  # ← int, not string

# Supports: int, float, bool, null, JSON objects, arrays
```

### ✅ Thinking Preservation

MiniMax-M2's `<think>` blocks are kept verbatim in responses:

```
<think>
The user wants weather data. I should call the get_weather tool
with location="Paris" and use celsius since that's standard in France.
</think>

Sure! Let me check the weather in Paris for you.
```

This transparency lets you see the model's reasoning. Optionally hide them in your UI layer.

### ✅ Production Ready

- FastAPI async framework
- Pydantic type validation
- systemd service file included
- Comprehensive test suite (16+ unit tests, E2E tests)
- Health check endpoints

## Translation Examples

### Single Tool Call

**MiniMax-M2 Native Output:**
```xml
<minimax:tool_call>
<invoke name="get_weather">
<parameter name="location">Tokyo</parameter>
<parameter name="unit">celsius</parameter>
</invoke>
</minimax:tool_call>
```

**OpenAI Format:**
```json
{
  "choices": [{
    "message": {
      "tool_calls": [{
        "id": "call_abc123",
        "type": "function",
        "function": {
          "name": "get_weather",
          "arguments": "{\"location\": \"Tokyo\", \"unit\": \"celsius\"}"
        }
      }]
    },
    "finish_reason": "tool_calls"
  }]
}
```

**Anthropic Format:**
```json
{
  "content": [{
    "type": "tool_use",
    "id": "call_abc123",
    "name": "get_weather",
    "input": {
      "location": "Tokyo",
      "unit": "celsius"
    }
  }],
  "stop_reason": "tool_use"
}
```

### Multiple Tool Calls

**MiniMax-M2:**
```xml
<minimax:tool_call>
<invoke name="get_weather">
<parameter name="location">Paris</parameter>
</invoke>
<invoke name="get_weather">
<parameter name="location">New York</parameter>
</invoke>
</minimax:tool_call>
```

**Proxy Output:**
- OpenAI: `tool_calls` array with 2 elements
- Anthropic: `content` array with 2 `tool_use` blocks

Both correctly preserve multiple tool calls in a single turn.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | OpenAI chat completions (with tool parsing) |
| `/v1/messages` | POST | Anthropic messages (with tool parsing) |
| `/v1/models` | GET | List available models (pass-through) |
| `/v1/model` | GET | Get current model info (pass-through) |
| `/health` | GET | Proxy + backend health check |
| `/` | GET | Service info |

## Configuration

Environment variables (`.env`):

```bash
# Backend settings
TABBY_URL=http://localhost:8000       # TabbyAPI/vLLM endpoint
TABBY_TIMEOUT=300                     # Request timeout in seconds

# Server settings
HOST=0.0.0.0
PORT=8001

# Feature flags
ENABLE_THINKING_PASSTHROUGH=true      # Keep <think> blocks
ENABLE_TOOL_TRANSLATION=true          # Parse tool calls

# Logging
LOG_LEVEL=INFO
LOG_RAW_RESPONSES=false               # Debug: log raw TabbyAPI responses
ENABLE_STREAMING_DEBUG=false          # Debug: trace streaming deltas and parser decisions
STREAMING_DEBUG_PATH=proxy_stream.log # Optional file to persist streaming trace
```

## Chat Templates

MiniMax-M2 requires a specific chat template for tool calling. If you're using TabbyAPI:

1. Download the [chat_template.jinja](https://huggingface.co/MiniMaxAI/MiniMax-Text-01/blob/main/chat_template.jinja) from Hugging Face
2. Place it in your model directory
3. TabbyAPI will auto-load it on model initialization

The template teaches the model to output:
- `<think>...</think>` for reasoning
- `<minimax:tool_call>` with `<invoke>` and `<parameter>` tags

## Development Workflow

### Running the Proxy Locally

We ship a small helper script for common tasks:

```bash
# Auto-reload development server (default host/port: 0.0.0.0:8001)
scripts/proxy.sh dev

# Production-style foreground server
scripts/proxy.sh serve
```

You can override the Python binary or bind address via environment variables:

```bash
PYTHON_BIN=python3.11 HOST=127.0.0.1 PORT=9000 scripts/proxy.sh dev
```

### Tests

The test harness now uses `pytest` with lightweight stubs for TabbyAPI, so we no longer
depend on a running backend when exercising unit or integration behaviour.

```bash
# Run the full suite
scripts/proxy.sh test

# Pass extra arguments through to pytest
scripts/proxy.sh test -k openai --maxfail=1
```

Tests are organised under `tests/`:

- `tests/unit/` covers pure helpers (reasoning split, streaming parser, session store)
- `tests/integration/` hits the FastAPI endpoint helpers with stubbed Tabby responses

## Documentation

Detailed integration guides live under [`docs/`](./docs/index.md). Start with the
[introduction](./docs/introduction.md) and [quick start](./docs/quickstart.md), then jump to the
[OpenAI](./docs/openai.md) or [Anthropic](./docs/anthropic.md) walkthrough depending on your client.

## Project Structure

```
minimax-m2-proxy/
├── proxy/
│   ├── main.py              # FastAPI app, endpoints
│   ├── config.py            # Environment configuration
│   ├── models.py            # Pydantic request/response models
│   └── client.py            # TabbyAPI HTTP client
├── parsers/
│   ├── tools.py             # XML → JSON tool call parser
│   └── streaming.py         # Streaming state machine
├── formatters/
│   ├── openai.py            # OpenAI format generator
│   └── anthropic.py         # Anthropic format generator
├── tests/
│   ├── conftest.py                  # Shared pytest fixtures (stubbed Tabby client)
│   ├── unit/
│   │   ├── test_reasoning.py        # Reasoning helpers and think splitting
│   │   ├── test_session_store.py    # History repair logic
│   │   └── test_streaming_parser.py # Streaming parser behaviour
│   └── integration/
│       ├── test_anthropic_completion.py
│       ├── test_openai_completion.py
│       └── test_openai_stream.py
├── requirements.txt         # Python dependencies
├── .env.example             # Configuration template
├── minimax-m2-proxy.service # systemd service file
└── README.md                # This file
```

## Troubleshooting

### Proxy won't start

```bash
# Check if port is in use
lsof -i :8001

# Verify TabbyAPI is running
curl http://localhost:8000/health

# Check logs
tail -f proxy.log  # if running with nohup
sudo journalctl -u minimax-m2-proxy -f  # if using systemd
```

### Tool calls not parsing

1. **Check chat template**: Ensure TabbyAPI has `chat_template.jinja` configured
2. **Verify raw output**: Set `LOG_RAW_RESPONSES=true` in `.env` and check logs
3. **Test XML format**: Model should output `<minimax:tool_call>` with proper structure
4. **Validate tools**: Ensure tool schemas are valid JSONSchema

### Streaming think blocks look wrong

1. Enable `ENABLE_STREAMING_DEBUG=true` and optionally set `STREAMING_DEBUG_PATH`.
2. Reproduce the request – the proxy writes every Tabby SSE line (`tabby_sse`), parsed chunk (`tabby_chunk`), and parser decision (`emit_*`, `tool_event_*`).
3. Inspect the trace to see whether Tabby ever sent `<think>`, whether the proxy synthesised a closing tag, and whether a tool delta dropped the function name.
4. If Tabby never emits `<think>`, verify its chat template matches the [MiniMax template](https://huggingface.co/MiniMaxAI/MiniMax-M2/blob/main/chat_template.jinja) and that `--chat-template` is pointed at it.
5. Share the trace when filing issues; it includes enough context to detect regressions the unit tests miss.

### Streaming is slow

- Increase TabbyAPI's `max_batch_size` in config
- Check GPU utilization (`nvidia-smi`)
- Reduce `max_tokens` in requests
- Monitor network latency between proxy and TabbyAPI

### Type conversion errors

The parser uses tool schemas to convert types. If you see strings instead of numbers:

```python
# ❌ Missing schema - parser defaults to string
tools = [{"type": "function", "function": {"name": "calc"}}]

# ✅ Include schema - parser converts to int
tools = [{
    "type": "function",
    "function": {
        "name": "calc",
        "parameters": {
            "type": "object",
            "properties": {
                "num": {"type": "integer"}  # ← parser sees this
            }
        }
    }
}]
```

## Roadmap

**Phase 1 (Current)**: Stateless proxy
- ✅ XML tool call parsing
- ✅ OpenAI & Anthropic format translation
- ✅ Streaming support
- ✅ Think block preservation
- ✅ Type inference

**Phase 2 (Future)**:
- Conversation history management
- Context window optimization
- Token usage tracking
- Request caching
- Load balancing across multiple backends
- Authentication & rate limiting

## Performance

Tested with MiniMax-M2-EXL3 (456B MoE @ 4.0 BPW) on RTX 4090:

- **Non-streaming**: ~50ms proxy overhead
- **Streaming**: ~10ms first token delay
- **Throughput**: Bottleneck is model inference, not proxy
- **Memory**: ~50MB proxy memory footprint

## Alternatives Considered

### Why not LiteLLM?

[LiteLLM](https://github.com/BerriAI/litellm) is excellent for standard models but:
- Doesn't support custom XML tool formats
- Would require forking and maintaining model-specific logic
- This lightweight proxy is simpler and MiniMax-M2-specific

### Why not modify every UI?

You could patch each tool (OpenWebUI, Continue, etc.) to parse `<minimax:tool_call>`, but:
- Duplicate work across projects
- Hard to maintain across updates
- This proxy provides one canonical implementation

### Why not change MiniMax-M2's template?

Using a generic template loses tool-calling capabilities entirely. MiniMax-M2 was trained with XML format - changing it degrades performance.

## Contributing

This is a focused tool for a specific use case. Feel free to:
- Open issues for bugs
- Submit PRs for bugfixes
- Fork for your own modifications

For major features, consider discussing in an issue first.

## License

MIT - See LICENSE file

## Acknowledgments

- **Tool parsing logic**: Adapted from [vLLM's minimax_m2_tool_parser.py](https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/tool_parsers/minimax_m2_tool_parser.py)
- **Backend**: Designed for [TabbyAPI](https://github.com/theroyallab/tabbyAPI) with [ExLlama v3](https://github.com/turboderp/exllamav2)
- **Model**: [MiniMax-Text-01](https://huggingface.co/MiniMaxAI/MiniMax-Text-01) by MiniMax AI

## Support

- **Issues**: [GitHub Issues](https://github.com/0xSero/minimax-m2-proxy/issues)
- **Model Info**: [MiniMax-Text-01 on Hugging Face](https://huggingface.co/MiniMaxAI/MiniMax-Text-01)
- **TabbyAPI**: [TabbyAPI Docs](https://github.com/theroyallab/tabbyAPI)

---

**TL;DR**: MiniMax-M2 uses XML for tool calls. Most tools expect JSON. This proxy translates automatically so you can use MiniMax-M2 with any standard OpenAI/Anthropic-compatible tool.
