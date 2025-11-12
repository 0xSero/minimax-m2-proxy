#!/usr/bin/env python3
"""Test to reproduce the length issue"""

import anthropic

# Point to the proxy
client = anthropic.Anthropic(
    api_key="test-key",
    base_url="http://localhost:8001"
)

print("=" * 80)
print("Testing with very low max_tokens to trigger length issue")
print("=" * 80)

try:
    # Test 1: Very low max_tokens that should trigger length issue
    response = client.messages.create(
        model="minimax-m2",
        max_tokens=10,  # Very low to force truncation
        messages=[
            {"role": "user", "content": "Write a long essay about the history of computers"}
        ]
    )

    print(f"\nTest 1 - Low max_tokens (10):")
    print(f"Response stop_reason: {response.stop_reason}")
    print(f"Content blocks: {len(response.content)}")

    for i, block in enumerate(response.content):
        print(f"\nBlock {i}: {block.type}")
        if block.type == "text":
            print(f"  Text: '{block.text}'")
        elif block.type == "thinking":
            print(f"  Thinking length: {len(block.thinking)} chars")
            print(f"  Thinking preview: {block.thinking[:100]}...")

    # Test 2: Reasonable max_tokens
    print("\n" + "=" * 80)
    print("Test 2 - Normal max_tokens (1024)")
    print("=" * 80)

    response2 = client.messages.create(
        model="minimax-m2",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": "What's 2+2?"}
        ]
    )

    print(f"\nResponse stop_reason: {response2.stop_reason}")
    print(f"Content blocks: {len(response2.content)}")

    for i, block in enumerate(response2.content):
        print(f"\nBlock {i}: {block.type}")
        if block.type == "text":
            print(f"  Text: '{block.text}'")
        elif block.type == "thinking":
            print(f"  Thinking preview: {block.thinking[:100]}...")

except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
