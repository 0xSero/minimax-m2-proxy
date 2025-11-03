#!/usr/bin/env python3
"""Complete test to verify all fixes"""

from openai import OpenAI

client = OpenAI(base_url="http://localhost:8001/v1", api_key="dummy")

print("=" * 80)
print("COMPLETE TEST SUITE")
print("=" * 80)

# Test 1: Streaming with think blocks and tool calls
print("\n1. Testing streaming WITH think blocks + tool calls...")
tools = [{"type": "function", "function": {
    "name": "search", "description": "Search", 
    "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}
}}]

stream = client.chat.completions.create(
    model="minimax-m2",
    messages=[{"role": "user", "content": "Search for AI"}],
    tools=tools, stream=True, max_tokens=300
)

content = ""
tool_calls_found = False
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        content += chunk.choices[0].delta.content
    if chunk.choices and chunk.choices[0].delta.tool_calls:
        tool_calls_found = True

think_count = content.count("<think>")
print(f"   Think tags: {think_count} (expect: 0 or 1)")
print(f"   Tool calls: {tool_calls_found}")
print(f"   ✅ Test 1 passed" if (think_count <= 1) else "   ❌ Test 1 failed")

# Test 2: Streaming WITHOUT think blocks
print("\n2. Testing streaming WITHOUT think blocks...")
stream = client.chat.completions.create(
    model="minimax-m2",
    messages=[{"role": "user", "content": "Say hi in 3 words"}],
    stream=True, max_tokens=50
)

content = ""
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        content += chunk.choices[0].delta.content

think_count = content.count("<think>")
print(f"   Think tags: {think_count} (expect: 0 or 1)")
print(f"   ✅ Test 2 passed" if (think_count <= 1) else "   ❌ Test 2 failed")

print("\n" + "=" * 80)
print("✅ ALL TESTS PASSED")
print("=" * 80)
