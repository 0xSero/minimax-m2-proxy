# Quick Start

Follow this checklist to get the proxy running locally and verify everything end-to-end.

## 1. Clone and install

```bash
git clone https://github.com/0xSero/minimax-m2-proxy.git
cd minimax-m2-proxy

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you use Poetry:

```bash
poetry install
poetry shell
```

## 2. Configure the backend

Run TabbyAPI or another MiniMax host that exposes `POST /v1/chat/completions`. Ensure the official MiniMax chat template is active. The proxy assumes the backend emits `<think>` and `<minimax:tool_call>` blocks.

Set `TABBY_URL` if the backend is not at `http://localhost:8000`.

## 3. Start the proxy

```bash
scripts/proxy.sh dev
```

This runs the FastAPI app on `http://0.0.0.0:8001` with auto-reload. For production-style serving use `scripts/proxy.sh serve`.

## 4. Send a test request

Using the OpenAI SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8001/v1", api_key="dummy")

response = client.chat.completions.create(
    model="minimax-m2",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Summarise the MiniMax proxy."},
    ],
)

print(response.choices[0].message.content)
```

You should see the assistant’s answer, with `<think>` preserved if you inspect the raw JSON.

## 5. Enable reasoning split (optional)

Some OpenAI SDK clients request `reasoning_split=True` so thinking appears in `reasoning_details`. The proxy supports that flag:

```python
response = client.chat.completions.create(
    model="minimax-m2",
    messages=[ ... ],
    extra_body={"reasoning_split": True},
)

print(response.choices[0].message.reasoning_details[0]["text"])
```

## 6. Anthropic client check

```python
import anthropic

client = anthropic.Anthropic(base_url="http://localhost:8001", api_key="dummy")

msg = client.messages.create(
    model="minimax-m2",
    messages=[{"role": "user", "content": [{"type": "text", "text": "Hi!"}]}],
)

for block in msg.content:
    print(block.type, getattr(block, block.type, None))
```

You will receive separate `thinking`, `text`, and optional `tool_use` blocks, matching Anthropic’s schema.

Next steps: choose the guide that matches your client type—[OpenAI](./openai.md) or [Anthropic](./anthropic.md)—and learn how to handle multi-turn tool flows.
