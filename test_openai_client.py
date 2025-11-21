#!/usr/bin/env python3
"""Test what the OpenAI client actually parses"""

from openai import OpenAI
import json

client = OpenAI(
    base_url="http://localhost:8001/v1",
    api_key="dummy"
)

print("Testing OpenAI client parsing...\n")

stream = client.chat.completions.create(
    model="minimax-m2",
    messages=[{"role": "user", "content": "What is 2+2?"}],
    stream=True,
    extra_body={"reasoning_split": True}
)

for i, chunk in enumerate(stream):
    if i < 5:
        print(f"\nChunk {i}:")
        print(f"  Type: {type(chunk)}")
        if chunk.choices:
            delta = chunk.choices[0].delta
            print(f"  Delta type: {type(delta)}")
            print(f"  Delta dict: {delta.model_dump()}")
            print(f"  Has reasoning_content attr: {hasattr(delta, 'reasoning_content')}")
            if hasattr(delta, 'reasoning_content'):
                print(f"  reasoning_content value: {delta.reasoning_content}")
    if i == 5:
        print("\n... (truncated)")
        break
