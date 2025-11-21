#!/usr/bin/env python3
"""Test Anthropic API endpoint"""

import anthropic

client = anthropic.Anthropic(
    base_url="http://localhost:8001",
    api_key="dummy"
)

print("Testing Anthropic API with streaming...\n")

with client.messages.stream(
    model="minimax-m2",
    max_tokens=1024,
    messages=[{"role": "user", "content": "What is 2+2? Think step by step."}]
) as stream:
    print("=== Thinking ===")
    for text in stream.text_stream:
        print(text, end="", flush=True)

print("\n\n=== Final Message ===")
message = stream.get_final_message()
print(f"Role: {message.role}")
print(f"Content blocks: {len(message.content)}")
for i, block in enumerate(message.content):
    print(f"\nBlock {i}: {block.type}")
    if block.type == "thinking":
        print(f"  Thinking: {block.thinking[:100]}...")
    elif block.type == "text":
        print(f"  Text: {block.text[:100]}...")
    elif block.type == "tool_use":
        print(f"  Tool: {block.name}")
