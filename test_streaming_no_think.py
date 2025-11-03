#!/usr/bin/env python3
"""Test streaming when model doesn't use think blocks"""

from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8001/v1",
    api_key="dummy"
)

print("=" * 80)
print("Testing streaming WITHOUT think blocks")
print("=" * 80)

stream = client.chat.completions.create(
    model="minimax-m2",
    messages=[
        {"role": "user", "content": "Say hello in 5 words exactly"}
    ],
    stream=True,
    max_tokens=50
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

think_open_count = full_content.count("<think>")
think_close_count = full_content.count("</think>")

print(f"Opening <think> tags: {think_open_count}")
print(f"Closing </think> tags: {think_close_count}")
print(f"\nFull content:\n{full_content}")

if think_open_count > 0 and think_close_count == 0:
    print("\n❌ BUG: Spurious <think> tag added to non-thinking response!")
elif think_open_count > 1:
    print(f"\n❌ DOUBLE THINK: {think_open_count} opening tags")
else:
    print("\n✅ No think block issues")
