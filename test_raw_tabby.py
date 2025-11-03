#!/usr/bin/env python3
"""Direct test of TabbyAPI to see raw XML output"""

import requests
import json

url = "http://localhost:8000/v1/chat/completions"

tools = [{
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"]
        }
    }
}]

payload = {
    "model": "minimax-m2",
    "messages": [
        {"role": "user", "content": "Search for 'interleaved thinking' and explain it"}
    ],
    "tools": tools,
    "max_tokens": 2000
}

print("Sending request directly to TabbyAPI...")
print("="*80)

response = requests.post(url, json=payload)
data = response.json()

print("\nFULL RAW RESPONSE:")
print(json.dumps(data, indent=2))

print("\n" + "="*80)
print("\nCONTENT ONLY:")
content = data["choices"][0]["message"].get("content", "")
print(repr(content))

print("\n" + "="*80)
if "<think>" in content:
    print("✅ <think> opening tag FOUND in raw content")
else:
    print("❌ <think> opening tag NOT FOUND in raw content")

if "</think>" in content:
    print("✅ </think> closing tag FOUND in raw content")
else:
    print("❌ </think> closing tag NOT FOUND in raw content")

if "<minimax:tool_call>" in content:
    print("✅ <minimax:tool_call> opening tag FOUND in raw content")
else:
    print("❌ <minimax:tool_call> opening tag NOT FOUND in raw content")

print("="*80)
