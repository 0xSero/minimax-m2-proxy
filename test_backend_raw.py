#!/usr/bin/env python3
"""Test backend directly to see what it sends"""

import json
from openai import OpenAI

# Point directly to backend
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

print("Testing backend directly...\n")

stream = client.chat.completions.create(
    model="minimax-m2",
    messages=[
        {"role": "user", "content": "What is 2+2?"}
    ],
    stream=True
)

print("=== Raw Backend Chunks ===")
for i, chunk in enumerate(stream):
    if i < 10:  # Only show first 10 chunks
        print(f"\nChunk {i}:")
        print(json.dumps(chunk.model_dump(), indent=2))
    if i == 10:
        print("\n... (truncated)")
        break
