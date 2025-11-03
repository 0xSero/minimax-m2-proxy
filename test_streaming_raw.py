#!/usr/bin/env python3
"""Test streaming from TabbyAPI to see when <think> appears/disappears"""

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
        {"role": "user", "content": "Search for 'test' using the tool"}
    ],
    "tools": tools,
    "max_tokens": 500,
    "stream": True
}

print("=" * 80)
print("Streaming request to TabbyAPI...")
print("=" * 80)

response = requests.post(url, json=payload, stream=True)

chunk_num = 0
accumulated = ""

for line in response.iter_lines():
    if line:
        line_str = line.decode('utf-8')
        if line_str.startswith('data: '):
            data_str = line_str[6:]
            if data_str == '[DONE]':
                print("\n[DONE]")
                break

            try:
                chunk = json.loads(data_str)
                chunk_num += 1

                if 'choices' in chunk and len(chunk['choices']) > 0:
                    delta = chunk['choices'][0].get('delta', {})
                    content = delta.get('content', '')

                    if content:
                        accumulated += content
                        print(f"\nChunk {chunk_num}:")
                        print(f"  Delta: {repr(content)}")
                        print(f"  Accumulated so far: {repr(accumulated[:100])}...")

                        # Check for think tags in this chunk
                        if '<think>' in content:
                            print(f"  ✅ FOUND <think> opening tag in this chunk!")
                        if '</think>' in content:
                            print(f"  ✅ FOUND </think> closing tag in this chunk!")

                        # Check what came right before accumulated
                        if len(accumulated) > 10:
                            last_chars = accumulated[-20:]
                            print(f"  Last 20 chars: {repr(last_chars)}")

            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                print(f"Line: {line_str}")

print("\n" + "=" * 80)
print("\nFINAL ACCUMULATED TEXT:")
print(repr(accumulated))
print("\n" + "=" * 80)

if '<think>' in accumulated:
    print("✅ <think> opening tag IS in accumulated text")
else:
    print("❌ <think> opening tag NOT in accumulated text")

if '</think>' in accumulated:
    print("✅ </think> closing tag IS in accumulated text")
else:
    print("❌ </think> closing tag NOT in accumulated text")

print("=" * 80)
