#!/usr/bin/env python3
"""Test Anthropic tool calling with explicit tool_choice"""

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
print("Testing Anthropic tool calling with tool_choice='any'")
print("=" * 80)

try:
    response = client.messages.create(
        model="minimax-m2",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": "What's the weather in Tokyo in celsius?"}
        ],
        tools=tools,
        tool_choice={"type": "any"}  # Force tool usage
    )

    print(f"\nResponse stop_reason: {response.stop_reason}")
    print(f"Content blocks: {len(response.content)}")

    for i, block in enumerate(response.content):
        print(f"\nBlock {i}: {block.type}")
        if block.type == "text":
            print(f"  Text: {block.text}")
        elif block.type == "thinking":
            print(f"  Thinking: {block.thinking[:200]}...")
        elif block.type == "tool_use":
            print(f"  Tool: {block.name}")
            print(f"  ID: {block.id}")
            print(f"  Input: {block.input}")

    # Now send tool result and get final response
    if response.stop_reason == "tool_use":
        print("\n" + "=" * 80)
        print("Sending tool result...")
        print("=" * 80)

        # Build messages with tool result
        messages = [
            {"role": "user", "content": "What's the weather in Tokyo in celsius?"}
        ]

        # Add assistant response with tool use
        messages.append({
            "role": "assistant",
            "content": response.content
        })

        # Add tool result
        tool_use_block = [b for b in response.content if b.type == "tool_use"][0]
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_block.id,
                    "content": "The weather in Tokyo is 22°C and sunny."
                }
            ]
        })

        final_response = client.messages.create(
            model="minimax-m2",
            max_tokens=1024,
            messages=messages,
            tools=tools
        )

        print(f"\nFinal response stop_reason: {final_response.stop_reason}")
        print(f"Final content blocks: {len(final_response.content)}")

        for i, block in enumerate(final_response.content):
            print(f"\nBlock {i}: {block.type}")
            if block.type == "text":
                print(f"  Text: {block.text}")
            elif block.type == "thinking":
                print(f"  Thinking: {block.thinking[:200]}...")

except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("Check proxy.log for detailed logging")
print("=" * 80)
