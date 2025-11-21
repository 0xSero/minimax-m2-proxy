#!/usr/bin/env python3
"""Test streaming with <think> blocks for vllm/sglang"""

import json
from openai import OpenAI

# Point to proxy
client = OpenAI(
    base_url="http://localhost:8001/v1",
    api_key="dummy"
)

print("Testing streaming with <think> blocks...\n")

# Test simple question that should trigger thinking
stream = client.chat.completions.create(
    model="minimax-m2",
    messages=[
        {"role": "user", "content": "What is 2+2? Think through this step by step."}
    ],
    stream=True,
    extra_body={"reasoning_split": True}
)

print("=== Visible Content ===")
visible_parts = []
reasoning_parts = []

for chunk in stream:
    if chunk.choices:
        delta = chunk.choices[0].delta

        # Check for reasoning content
        if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
            reasoning_parts.append(delta.reasoning_content)
            print(f"[REASONING] {delta.reasoning_content}", end='', flush=True)

        # Check for visible content
        if delta.content:
            visible_parts.append(delta.content)
            print(delta.content, end='', flush=True)

print("\n\n=== Summary ===")
print(f"Reasoning length: {len(''.join(reasoning_parts))} chars")
print(f"Visible content length: {len(''.join(visible_parts))} chars")

if reasoning_parts:
    print("\n✅ Reasoning detected correctly!")
else:
    print("\n⚠️  No reasoning detected - may be included in visible content")

print("\nFull reasoning:")
print(''.join(reasoning_parts))
