#!/usr/bin/env python3
"""Test to see if TabbyAPI is returning </think> tags"""

import httpx
import json

# Direct TabbyAPI test
url = "http://localhost:8000/v1/chat/completions"

print("=" * 80)
print("RAW TABBYAPI OUTPUT - Looking for </think> tags")
print("=" * 80)

data = {
    "model": "minimax-m2",
    "messages": [{"role": "user", "content": "Think about Python briefly"}],
    "stream": False,
    "max_tokens": 300
}

response = httpx.post(url, json=data, timeout=30)
result = response.json()
content = result["choices"][0]["message"]["content"]

print(f"\nContent length: {len(content)}")
print(f"Contains </think>: {'</think>' in content}")
print(f"Contains <think>: {'<think>' in content}")
print(f"\nFirst 200 chars:")
print(content[:200])
print(f"\nLast 200 chars:")
print(content[-200:])

# Count tags
closing_count = content.count("</think>")
opening_count = content.count("<think>")

print(f"\n</think> tags: {closing_count}")
print(f"<think> tags: {opening_count}")

# Now test through proxy
print("\n" + "=" * 80)
print("PROXY OUTPUT")
print("=" * 80)

proxy_url = "http://localhost:8001/v1/chat/completions"
proxy_response = httpx.post(proxy_url, json=data, timeout=30)
proxy_result = proxy_response.json()
proxy_content = proxy_result["choices"][0]["message"]["content"]

print(f"\nContent length: {len(proxy_content)}")
print(f"Contains </think>: {'</think>' in proxy_content}")
print(f"Contains <think>: {'<think>' in proxy_content}")
print(f"\nFirst 200 chars:")
print(proxy_content[:200])

proxy_closing = proxy_content.count("</think>")
proxy_opening = proxy_content.count("<think>")

print(f"\n</think> tags: {proxy_closing}")
print(f"<think> tags: {proxy_opening}")

if proxy_closing > 0 and proxy_opening == 0:
    print("\n❌ BUG: Closing tag without opening tag!")
elif proxy_opening > 1:
    print(f"\n❌ BUG: {proxy_opening} opening tags (double think blocks)")
else:
    print("\n✅ Think blocks OK")
