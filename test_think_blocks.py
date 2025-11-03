#!/usr/bin/env python3
"""Test script to verify <think> blocks are preserved with tool calling"""

import json
from openai import OpenAI

# Connect to the proxy
client = OpenAI(
    base_url="http://localhost:8001/v1",
    api_key="dummy"
)

# Define a simple tool
tools = [{
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                }
            },
            "required": ["query"]
        }
    }
}]

print("=" * 80)
print("Testing MiniMax-M2 Proxy with <think> blocks preservation")
print("=" * 80)
print()

# Test request
print("Sending request: 'Search for interleaved thinking and explain it'")
print()

try:
    response = client.chat.completions.create(
        model="minimax-m2",
        messages=[
            {"role": "user", "content": "Search for 'interleaved thinking' and explain what it is"}
        ],
        tools=tools,
        temperature=1.0,
        max_tokens=2000
    )

    print("-" * 80)
    print("RESPONSE RECEIVED")
    print("-" * 80)

    message = response.choices[0].message

    # Check for content (should include <think> blocks)
    if message.content:
        print("\n📝 CONTENT (with <think> blocks):")
        print(message.content)

        # Verify <think> blocks are present
        if "<think>" in message.content:
            print("\n✅ SUCCESS: <think> blocks are preserved!")
        else:
            print("\n❌ WARNING: No <think> blocks found in content")
    else:
        print("\n📝 CONTENT: None")

    # Check for tool calls
    if message.tool_calls:
        print("\n🔧 TOOL CALLS:")
        for i, tool_call in enumerate(message.tool_calls, 1):
            print(f"\n  Tool Call #{i}:")
            print(f"    ID: {tool_call.id}")
            print(f"    Function: {tool_call.function.name}")
            print(f"    Arguments: {tool_call.function.arguments}")

            # Parse arguments to verify structure
            try:
                args = json.loads(tool_call.function.arguments)
                print(f"    Parsed: {args}")
            except:
                print("    (Could not parse arguments)")

        print("\n✅ Tool calls parsed successfully!")
    else:
        print("\n🔧 TOOL CALLS: None")

    # Final status
    print("\n" + "=" * 80)
    has_think = message.content and "<think>" in message.content
    has_tools = message.tool_calls is not None and len(message.tool_calls) > 0

    if has_think and has_tools:
        print("✅ FULL SUCCESS: Both <think> blocks and tool calls working!")
    elif has_think:
        print("⚠️  PARTIAL: <think> blocks present but no tool calls")
    elif has_tools:
        print("⚠️  PARTIAL: Tool calls working but <think> blocks missing")
    else:
        print("❌ FAILED: Neither <think> blocks nor tool calls found")

    print("=" * 80)

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
