#!/usr/bin/env python3
"""Test to see raw chunk data from TabbyAPI"""

import httpx
import json

url = "http://localhost:8000/v1/chat/completions"

data = {
    "model": "minimax-m2",
    "messages": [
        {"role": "user", "content": "Think about the number 5 briefly"}
    ],
    "stream": True,
    "max_tokens": 200,
    "temperature": 1.0
}

print("=" * 80)
print("RAW CHUNKS FROM TABBYAPI (first 10 content chunks)")
print("=" * 80)

chunk_count = 0
content_count = 0

with httpx.stream("POST", url, json=data, timeout=30) as response:
    for line in response.iter_lines():
        if line.startswith("data: "):
            chunk_count += 1
            data_str = line[6:]  # Remove "data: " prefix
            
            if data_str == "[DONE]":
                break
                
            try:
                chunk_data = json.loads(data_str)
                
                if "choices" in chunk_data and chunk_data["choices"]:
                    delta = chunk_data["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    
                    if content:
                        content_count += 1
                        if content_count <= 10:
                            print(f"\nChunk {content_count}:")
                            print(f"  Raw content: {repr(content)}")
                            print(f"  Starts with <think>: {content.startswith('<think>')}")
                            
            except json.JSONDecodeError:
                pass

print(f"\n\nTotal chunks: {chunk_count}, Content chunks: {content_count}")
