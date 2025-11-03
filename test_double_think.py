#!/usr/bin/env python3
"""Test to detect double think blocks"""

from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8001/v1",
    api_key="dummy"
)

print("Testing for double think blocks...\n")

stream = client.chat.completions.create(
    model="minimax-m2",
    messages=[
        {"role": "user", "content": "Think about the number 42 and explain why it's interesting"}
    ],
    stream=True,
    temperature=1.0,
    max_tokens=500
)

content_parts = []
for chunk in stream:
    if chunk.choices and len(chunk.choices) > 0:
        choice = chunk.choices[0]
        delta = choice.delta
        if delta.content:
            content_parts.append(delta.content)
            print(delta.content, end='', flush=True)

full_content = "".join(content_parts)

print("\n\n" + "=" * 80)
print("ANALYSIS:")
print("=" * 80)

# Count <think> tags
think_open_count = full_content.count("<think>")
think_close_count = full_content.count("</think>")

print(f"Opening <think> tags: {think_open_count}")
print(f"Closing </think> tags: {think_close_count}")

if think_open_count > 1:
    print(f"\n❌ DOUBLE THINK BLOCKS DETECTED! Found {think_open_count} opening tags")
    # Show the first 200 chars
    print(f"\nFirst 200 chars:\n{full_content[:200]}")
elif think_open_count == 1:
    print("\n✅ Single think block found (correct)")
else:
    print("\n⚠️  No think blocks found")
