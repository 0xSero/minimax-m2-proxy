#!/usr/bin/env python3
"""Test raw non-streaming response from TabbyAPI"""

import httpx
import json

# Test TabbyAPI directly
url = "http://localhost:8000/v1/chat/completions"

data = {
    "model": "minimax-m2",
    "messages": [
        {"role": "user", "content": "Think about the number 5 briefly"}
    ],
    "stream": False,
    "max_tokens": 200,
    "temperature": 1.0
}

print("=" * 80)
print("RAW NON-STREAMING RESPONSE FROM TABBYAPI")
print("=" * 80)

response = httpx.post(url, json=data, timeout=30)
result = response.json()

content = result["choices"][0]["message"]["content"]

print(f"\nContent length: {len(content)}")
print(f"Starts with <think>: {content.strip().startswith('<think>')}")
print(f"Contains </think>: {'</think>' in content}")
print(f"\nFirst 100 chars:")
print(repr(content[:100]))

# Now test through the proxy
print("\n" + "=" * 80)
print("RESPONSE FROM PROXY")
print("=" * 80)

proxy_url = "http://localhost:8001/v1/chat/completions"
proxy_response = httpx.post(proxy_url, json=data, timeout=30)
proxy_result = proxy_response.json()

proxy_content = proxy_result["choices"][0]["message"]["content"]

print(f"\nContent length: {len(proxy_content)}")
print(f"Starts with <think>: {proxy_content.strip().startswith('<think>')}")
print(f"Contains </think>: {'</think>' in proxy_content}")
print(f"\nFirst 100 chars:")
print(repr(proxy_content[:100]))

# Check for doubles
think_open_count = proxy_content.count("<think>")
think_close_count = proxy_content.count("</think>")

print(f"\n<think> count: {think_open_count}")
print(f"</think> count: {think_close_count}")

if think_open_count > 1:
    print("\n❌ DOUBLE THINK BLOCKS DETECTED!")
elif think_open_count == 1:
    print("\n✅ Single think block (correct)")
