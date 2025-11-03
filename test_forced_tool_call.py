#!/usr/bin/env python3
"""Test that forces a tool call to verify format"""

import asyncio
import httpx
import json


async def test_forced_tool_call():
    """Test with a request that should definitely trigger a tool call"""
    print("=== Testing Forced Tool Call ===\n")

    async with httpx.AsyncClient(timeout=60.0) as client:
        # Use tool_choice to force the tool call
        response = await client.post(
            "http://localhost:8001/v1/messages",
            json={
                "model": "minimax-m2",
                "max_tokens": 500,
                "messages": [
                    {"role": "user", "content": "Use the get_weather tool to check Tokyo's weather in celsius."}
                ],
                "tools": [
                    {
                        "name": "get_weather",
                        "description": "Get weather for a location",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string", "description": "City name"},
                                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                            },
                            "required": ["location"]
                        }
                    }
                ],
                # Try to force tool use
                "tool_choice": {"type": "tool", "name": "get_weather"}
            }
        )

        print(f"Status Code: {response.status_code}\n")
        result = response.json()
        print("Full Response:")
        print(json.dumps(result, indent=2, ensure_ascii=False))

        print("\n" + "="*60)
        print("Content Blocks:")
        for i, block in enumerate(result.get("content", [])):
            print(f"\nBlock {i}:")
            print(f"  Type: {block.get('type')}")
            if block.get("type") == "text":
                print(f"  Text: {block.get('text')[:500]}")
            elif block.get("type") == "tool_use":
                print(f"  Tool: {block.get('name')}")
                print(f"  ID: {block.get('id')}")
                print(f"  Input: {json.dumps(block.get('input'), indent=4)}")


async def test_simple_tool_call():
    """Test with clear instructions"""
    print("\n\n=== Testing Simple Tool Call Request ===\n")

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "http://localhost:8001/v1/messages",
            json={
                "model": "minimax-m2",
                "max_tokens": 500,
                "messages": [
                    {
                        "role": "user",
                        "content": "Call get_weather with location='Tokyo' and unit='celsius'. Do not ask for clarification, just call the function."
                    }
                ],
                "tools": [
                    {
                        "name": "get_weather",
                        "description": "Get weather for a location",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string"},
                                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                            },
                            "required": ["location"]
                        }
                    }
                ]
            }
        )

        print(f"Status Code: {response.status_code}\n")
        result = response.json()
        print("Full Response:")
        print(json.dumps(result, indent=2, ensure_ascii=False))

        # Check stop_reason
        print(f"\nStop Reason: {result.get('stop_reason')}")
        print(f"Expected: 'tool_use' if tool was called, 'end_turn' if not")


async def main():
    await test_forced_tool_call()
    await test_simple_tool_call()


if __name__ == "__main__":
    asyncio.run(main())
