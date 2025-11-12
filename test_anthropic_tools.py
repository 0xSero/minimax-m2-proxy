#!/usr/bin/env python3
"""Test Anthropic tool calling through the proxy"""

import anthropic

# Point to the proxy
client = anthropic.Anthropic(
    api_key="test-key",
    base_url="http://localhost:8001"
)

# Define a simple tool
tools = [
    {
        "name": "get_weather",
        "description": "Get weather information for a location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city name"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit"
                }
            },
            "required": ["location"]
        }
    }
]

print("=" * 80)
print("Testing Anthropic tool calling through proxy")
print("=" * 80)

try:
    response = client.messages.create(
        model="minimax-m2",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": "What's the weather in Tokyo?"}
        ],
        tools=tools
    )

    print(f"\nResponse stop_reason: {response.stop_reason}")
    print(f"Content blocks: {len(response.content)}")

    for i, block in enumerate(response.content):
        print(f"\nBlock {i}: {block.type}")
        if block.type == "text":
            print(f"  Text: {block.text[:100]}...")
        elif block.type == "tool_use":
            print(f"  Tool: {block.name}")
            print(f"  ID: {block.id}")
            print(f"  Input: {block.input}")
            print(f"  Input type: {type(block.input)}")

except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("Check proxy.log for detailed logging")
print("=" * 80)
