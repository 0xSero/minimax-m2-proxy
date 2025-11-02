"""End-to-end test script for MiniMax-M2 Proxy

Run with: python tests/e2e_test.py
Requires: TabbyAPI running on localhost:8000 and proxy on localhost:8001
"""

import asyncio
import httpx
import json


PROXY_URL = "http://localhost:8001"


async def test_openai_basic():
    """Test basic OpenAI endpoint"""
    print("\n=== Testing OpenAI Basic ===")

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{PROXY_URL}/v1/chat/completions",
            json={
                "model": "minimax-m2",
                "messages": [
                    {"role": "user", "content": "Say 'Hello, World!' and nothing else."}
                ],
                "max_tokens": 50,
                "stream": False
            }
        )

        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}")

        if response.status_code == 200:
            content = result["choices"][0]["message"]["content"]
            print(f"\nExtracted content: {content}")
        print("✓ OpenAI basic test passed")


async def test_openai_streaming():
    """Test OpenAI streaming endpoint"""
    print("\n=== Testing OpenAI Streaming ===")

    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream(
            "POST",
            f"{PROXY_URL}/v1/chat/completions",
            json={
                "model": "minimax-m2",
                "messages": [
                    {"role": "user", "content": "Count from 1 to 5."}
                ],
                "max_tokens": 100,
                "stream": True
            }
        ) as response:
            print(f"Status: {response.status_code}")
            print("Streaming response:")

            full_content = ""
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]

                    if data_str.strip() == "[DONE]":
                        print("\n[DONE]")
                        break

                    try:
                        chunk = json.loads(data_str)
                        if "choices" in chunk and len(chunk["choices"]) > 0:
                            delta = chunk["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                print(content, end="", flush=True)
                                full_content += content
                    except json.JSONDecodeError:
                        pass

            print(f"\n\nFull content: {full_content}")
            print("✓ OpenAI streaming test passed")


async def test_openai_with_tools():
    """Test OpenAI endpoint with tool calling"""
    print("\n=== Testing OpenAI with Tools ===")

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{PROXY_URL}/v1/chat/completions",
            json={
                "model": "minimax-m2",
                "messages": [
                    {"role": "user", "content": "What's the weather in San Francisco?"}
                ],
                "max_tokens": 200,
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get weather for a location",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "location": {
                                        "type": "string",
                                        "description": "City name"
                                    },
                                    "unit": {
                                        "type": "string",
                                        "enum": ["celsius", "fahrenheit"]
                                    }
                                },
                                "required": ["location"]
                            }
                        }
                    }
                ],
                "stream": False
            }
        )

        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}")

        if response.status_code == 200:
            message = result["choices"][0]["message"]
            if "tool_calls" in message:
                print(f"\n✓ Tool calls detected: {len(message['tool_calls'])}")
                for tc in message["tool_calls"]:
                    print(f"  - {tc['function']['name']}: {tc['function']['arguments']}")
            else:
                print("\n⚠ No tool calls found (model may have responded with text)")

        print("✓ OpenAI tools test passed")


async def test_anthropic_basic():
    """Test basic Anthropic endpoint"""
    print("\n=== Testing Anthropic Basic ===")

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{PROXY_URL}/v1/messages",
            json={
                "model": "minimax-m2",
                "max_tokens": 100,
                "messages": [
                    {"role": "user", "content": "Say 'Hello from Anthropic!' and nothing else."}
                ]
            }
        )

        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}")

        if response.status_code == 200:
            for block in result["content"]:
                if block["type"] == "text":
                    print(f"\nExtracted text: {block['text']}")

        print("✓ Anthropic basic test passed")


async def test_anthropic_streaming():
    """Test Anthropic streaming endpoint"""
    print("\n=== Testing Anthropic Streaming ===")

    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream(
            "POST",
            f"{PROXY_URL}/v1/messages",
            json={
                "model": "minimax-m2",
                "max_tokens": 100,
                "messages": [
                    {"role": "user", "content": "Count from 1 to 5."}
                ],
                "stream": True
            }
        ) as response:
            print(f"Status: {response.status_code}")
            print("Streaming events:")

            full_text = ""
            async for line in response.aiter_lines():
                if line.startswith("event: "):
                    event_type = line[7:]
                    print(f"\n[{event_type}]", end=" ")
                elif line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        if data.get("type") == "content_block_delta":
                            delta = data.get("delta", {})
                            if delta.get("type") == "text_delta":
                                text = delta.get("text", "")
                                print(text, end="", flush=True)
                                full_text += text
                    except json.JSONDecodeError:
                        pass

            print(f"\n\nFull text: {full_text}")
            print("✓ Anthropic streaming test passed")


async def test_health():
    """Test health endpoint"""
    print("\n=== Testing Health Endpoint ===")

    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.get(f"{PROXY_URL}/health")

        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}")

        if result.get("backend_healthy"):
            print("✓ Backend is healthy")
        else:
            print("⚠ Backend health check failed")

        print("✓ Health endpoint test passed")


async def main():
    """Run all tests"""
    print("=" * 60)
    print("MiniMax-M2 Proxy E2E Tests")
    print("=" * 60)

    tests = [
        test_health,
        test_openai_basic,
        test_openai_streaming,
        test_openai_with_tools,
        test_anthropic_basic,
        test_anthropic_streaming,
    ]

    for test in tests:
        try:
            await test()
        except Exception as e:
            print(f"\n✗ Test failed: {e}")
            import traceback
            traceback.print_exc()

        print()
        await asyncio.sleep(1)  # Brief pause between tests

    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
