#!/usr/bin/env python3
"""Test the exact flow that Chatbox AI uses"""

import anthropic

client = anthropic.Anthropic(
    api_key="test-key",
    base_url="http://localhost:8001"
)

tools = [
    {
        "name": "web_search",
        "description": "Search the web",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"]
        }
    }
]

print("Step 1: User asks a question that needs web search")
response1 = client.messages.create(
    model="minimax-m2",
    max_tokens=4096,
    tools=tools,
    tool_choice={"type": "any"},
    messages=[
        {"role": "user", "content": "What is the weather in Paris?"}
    ]
)

print(f"Response 1 stop_reason: {response1.stop_reason}")
print(f"Content blocks: {len(response1.content)}")

tool_use_block = None
for block in response1.content:
    print(f"  Block type: {block.type}")
    if block.type == "tool_use":
        print(f"    Tool: {block.name}")
        print(f"    Input: {block.input}")
        tool_use_block = block

if tool_use_block:
    print("\nStep 2: Send tool result back (simulating Chatbox AI)")

    # Build the messages exactly like Chatbox AI does
    messages = [
        {"role": "user", "content": "What is the weather in Paris?"},
        {
            "role": "assistant",
            "content": response1.content  # This includes text + tool_use blocks
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_block.id,
                    "content": "The weather in Paris is 15°C and cloudy."
                }
            ]
        }
    ]

    print("Sending messages with tool result...")
    try:
        response2 = client.messages.create(
            model="minimax-m2",
            max_tokens=4096,
            tools=tools,
            messages=messages
        )

        print(f"\n✅ SUCCESS!")
        print(f"Response 2 stop_reason: {response2.stop_reason}")
        for block in response2.content:
            if block.type == "text":
                print(f"Final text: {block.text[:100]}")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
