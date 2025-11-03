#!/usr/bin/env python3
"""Quick test of Anthropic endpoint to see actual response format"""

import asyncio
import httpx
import json


async def test_anthropic_nonstreaming():
    """Test non-streaming Anthropic endpoint"""
    print("=== Testing Anthropic Non-Streaming ===\n")

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "http://localhost:8001/v1/messages",
            json={
                "model": "minimax-m2",
                "max_tokens": 500,
                "messages": [
                    {"role": "user", "content": "Please describe what the minimax-m2-proxy project does."}
                ]
            }
        )

        print(f"Status Code: {response.status_code}\n")
        print(f"Headers: {dict(response.headers)}\n")

        result = response.json()
        print("Full Response:")
        print(json.dumps(result, indent=2, ensure_ascii=False))

        print("\n" + "="*60)
        print("Content Blocks:")
        for i, block in enumerate(result.get("content", [])):
            print(f"\nBlock {i}:")
            print(f"  Type: {block.get('type')}")
            if block.get("type") == "text":
                print(f"  Text: {block.get('text')[:200]}...")


async def test_anthropic_streaming():
    """Test streaming Anthropic endpoint"""
    print("\n\n=== Testing Anthropic Streaming ===\n")

    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream(
            "POST",
            "http://localhost:8001/v1/messages",
            json={
                "model": "minimax-m2",
                "max_tokens": 500,
                "messages": [
                    {"role": "user", "content": "Say hello and describe yourself briefly."}
                ],
                "stream": True
            }
        ) as response:
            print(f"Status Code: {response.status_code}\n")
            print("Streaming Events:\n")

            event_count = 0
            async for line in response.aiter_lines():
                if line.strip():
                    print(line)
                    event_count += 1
                    if event_count > 50:  # Limit output
                        print("\n... (truncated)")
                        break


async def test_anthropic_with_tools():
    """Test Anthropic with tool calling"""
    print("\n\n=== Testing Anthropic with Tools ===\n")

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "http://localhost:8001/v1/messages",
            json={
                "model": "minimax-m2",
                "max_tokens": 500,
                "messages": [
                    {"role": "user", "content": "What's the weather in Tokyo?"}
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
                ]
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
                print(f"  Text: {block.get('text')}")
            elif block.get("type") == "tool_use":
                print(f"  Tool: {block.get('name')}")
                print(f"  ID: {block.get('id')}")
                print(f"  Input: {json.dumps(block.get('input'), indent=4)}")


async def main():
    """Run all tests"""
    await test_anthropic_nonstreaming()
    await test_anthropic_streaming()
    await test_anthropic_with_tools()


if __name__ == "__main__":
    asyncio.run(main())
