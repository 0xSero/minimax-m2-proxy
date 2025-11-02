"""
Comprehensive tool calling tests for MiniMax-M2 Proxy

Tests:
1. Think blocks preservation
2. Single tool call
3. Multi-step tool calling (tool -> result -> final answer)
4. Multiple tools in one turn
"""

import asyncio
import httpx
import json
from typing import Dict, Any


PROXY_URL = "http://localhost:8001"
TIMEOUT = 120.0


# Define realistic tools
WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name or location"
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
}

CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Perform mathematical calculations",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate"
                }
            },
            "required": ["expression"]
        }
    }
}

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return"
                }
            },
            "required": ["query"]
        }
    }
}


def simulate_tool_execution(tool_name: str, arguments: Dict[str, Any]) -> str:
    """Simulate tool execution and return realistic results"""
    if tool_name == "get_weather":
        location = arguments.get("location", "Unknown")
        unit = arguments.get("unit", "celsius")
        symbol = "°C" if unit == "celsius" else "°F"
        temp = 22 if unit == "celsius" else 72
        return json.dumps({
            "location": location,
            "temperature": temp,
            "unit": symbol,
            "condition": "Partly cloudy",
            "humidity": 65,
            "wind_speed": 15
        })

    elif tool_name == "calculate":
        expression = arguments.get("expression", "")
        try:
            # Simple evaluation (in real scenario, use safer parser)
            result = eval(expression.replace("^", "**"))
            return json.dumps({"expression": expression, "result": result})
        except:
            return json.dumps({"error": "Invalid expression"})

    elif tool_name == "web_search":
        query = arguments.get("query", "")
        num_results = arguments.get("num_results", 3)
        return json.dumps({
            "query": query,
            "results": [
                {"title": f"Result {i+1} for '{query}'", "snippet": f"Information about {query}..."}
                for i in range(num_results)
            ]
        })

    return json.dumps({"error": "Unknown tool"})


async def test_think_blocks():
    """Test that think blocks are preserved in responses"""
    print("\n" + "="*60)
    print("TEST 1: Think Blocks Preservation")
    print("="*60)

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        response = await client.post(
            f"{PROXY_URL}/v1/chat/completions",
            json={
                "model": "minimax-m2",
                "messages": [
                    {"role": "user", "content": "Explain why the sky is blue in simple terms"}
                ],
                "max_tokens": 200
            }
        )

        result = response.json()
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            content = result["choices"][0]["message"]["content"]
            print(f"\nResponse content:\n{content}\n")

            # Check for think blocks
            if "<think>" in content:
                print("✅ Think blocks PRESERVED in response")
                # Extract thinking
                think_start = content.find("<think>")
                think_end = content.find("</think>")
                if think_start != -1 and think_end != -1:
                    thinking = content[think_start+7:think_end]
                    print(f"\nModel's thinking:\n{thinking}\n")
            else:
                print("⚠️  No think blocks found (model may not have generated them)")
        else:
            print(f"❌ Error: {result}")


async def test_single_tool_call():
    """Test single tool call with XML parsing"""
    print("\n" + "="*60)
    print("TEST 2: Single Tool Call")
    print("="*60)

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        response = await client.post(
            f"{PROXY_URL}/v1/chat/completions",
            json={
                "model": "minimax-m2",
                "messages": [
                    {"role": "user", "content": "What's the weather in Tokyo?"}
                ],
                "tools": [WEATHER_TOOL],
                "max_tokens": 150
            }
        )

        result = response.json()
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            message = result["choices"][0]["message"]
            print(f"\nMessage: {json.dumps(message, indent=2)}\n")

            if "tool_calls" in message and message["tool_calls"]:
                print(f"✅ Tool calls detected: {len(message['tool_calls'])}")
                for tc in message["tool_calls"]:
                    print(f"\nTool: {tc['function']['name']}")
                    print(f"Arguments: {tc['function']['arguments']}")

                    # Verify it's proper JSON
                    try:
                        args = json.loads(tc['function']['arguments'])
                        print(f"✅ Arguments parsed successfully: {args}")
                    except json.JSONDecodeError as e:
                        print(f"❌ Invalid JSON in arguments: {e}")
            else:
                print("⚠️  No tool calls found")
                if message.get("content"):
                    print(f"Content: {message['content']}")
        else:
            print(f"❌ Error: {result}")


async def test_multistep_tool_calling():
    """Test full multi-step tool calling flow"""
    print("\n" + "="*60)
    print("TEST 3: Multi-Step Tool Calling Flow")
    print("="*60)

    conversation = [
        {"role": "user", "content": "What's the weather in San Francisco and calculate 25 * 4?"}
    ]

    tools = [WEATHER_TOOL, CALCULATOR_TOOL]

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        # Step 1: Initial request
        print("\n--- Step 1: User Request ---")
        response = await client.post(
            f"{PROXY_URL}/v1/chat/completions",
            json={
                "model": "minimax-m2",
                "messages": conversation,
                "tools": tools,
                "max_tokens": 300
            }
        )

        result = response.json()
        print(f"Status: {response.status_code}")

        if response.status_code != 200:
            print(f"❌ Error: {result}")
            return

        message = result["choices"][0]["message"]
        conversation.append({
            "role": "assistant",
            "content": message.get("content"),
            "tool_calls": message.get("tool_calls")
        })

        print(f"Assistant response: {message.get('content', '(no content)')}")

        if "tool_calls" in message and message["tool_calls"]:
            print(f"\n✅ Assistant made {len(message['tool_calls'])} tool call(s):")

            # Step 2: Execute tools and add results
            for tc in message["tool_calls"]:
                tool_name = tc["function"]["name"]
                arguments = json.loads(tc["function"]["arguments"])

                print(f"\n  Tool: {tool_name}")
                print(f"  Args: {arguments}")

                # Simulate tool execution
                tool_result = simulate_tool_execution(tool_name, arguments)
                print(f"  Result: {tool_result}")

                # Add tool result to conversation
                conversation.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": tool_result
                })

            # Step 3: Get final response
            print("\n--- Step 2: Final Response with Tool Results ---")
            response = await client.post(
                f"{PROXY_URL}/v1/chat/completions",
                json={
                    "model": "minimax-m2",
                    "messages": conversation,
                    "tools": tools,
                    "max_tokens": 300
                }
            )

            result = response.json()
            print(f"Status: {response.status_code}")

            if response.status_code == 200:
                final_message = result["choices"][0]["message"]
                print(f"\n✅ Final Answer:\n{final_message.get('content', '(no content)')}\n")

                # Check for think blocks in final answer
                if "<think>" in final_message.get('content', ''):
                    print("✅ Think blocks preserved in final answer")
            else:
                print(f"❌ Error in final response: {result}")
        else:
            print("\n⚠️  No tool calls made, direct answer:")
            print(message.get("content", "(no content)"))


async def test_anthropic_tool_calling():
    """Test tool calling with Anthropic format"""
    print("\n" + "="*60)
    print("TEST 4: Anthropic Format Tool Calling")
    print("="*60)

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        response = await client.post(
            f"{PROXY_URL}/v1/messages",
            json={
                "model": "minimax-m2",
                "max_tokens": 200,
                "messages": [
                    {"role": "user", "content": "Search for 'MiniMax AI model' and give me 2 results"}
                ],
                "tools": [
                    {
                        "name": "web_search",
                        "description": "Search the web",
                        "input_schema": SEARCH_TOOL["function"]["parameters"]
                    }
                ]
            }
        )

        result = response.json()
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            print(f"\nResponse: {json.dumps(result, indent=2)}\n")

            # Check for tool_use blocks
            tool_uses = [block for block in result.get("content", []) if block.get("type") == "tool_use"]

            if tool_uses:
                print(f"✅ Anthropic tool_use blocks found: {len(tool_uses)}")
                for tool_use in tool_uses:
                    print(f"\n  Tool: {tool_use.get('name')}")
                    print(f"  Input: {json.dumps(tool_use.get('input'), indent=2)}")
            else:
                print("⚠️  No tool_use blocks found")
        else:
            print(f"❌ Error: {result}")


async def test_streaming_with_tools():
    """Test streaming responses with tool calls"""
    print("\n" + "="*60)
    print("TEST 5: Streaming with Tool Calls")
    print("="*60)

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        print("\nStreaming request with tools...")

        async with client.stream(
            "POST",
            f"{PROXY_URL}/v1/chat/completions",
            json={
                "model": "minimax-m2",
                "messages": [
                    {"role": "user", "content": "Calculate 15 + 27"}
                ],
                "tools": [CALCULATOR_TOOL],
                "max_tokens": 150,
                "stream": True
            }
        ) as response:
            print(f"Status: {response.status_code}\n")

            full_content = ""
            tool_calls_buffer = {}

            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]

                    if data_str.strip() == "[DONE]":
                        print("\n\n[DONE]")
                        break

                    try:
                        chunk = json.loads(data_str)
                        if "choices" in chunk and len(chunk["choices"]) > 0:
                            delta = chunk["choices"][0].get("delta", {})

                            # Content streaming
                            if "content" in delta and delta["content"]:
                                print(delta["content"], end="", flush=True)
                                full_content += delta["content"]

                            # Tool call streaming
                            if "tool_calls" in delta:
                                for tc_delta in delta["tool_calls"]:
                                    idx = tc_delta.get("index", 0)
                                    if idx not in tool_calls_buffer:
                                        tool_calls_buffer[idx] = {
                                            "id": tc_delta.get("id", ""),
                                            "name": "",
                                            "arguments": ""
                                        }

                                    if "function" in tc_delta:
                                        if "name" in tc_delta["function"]:
                                            tool_calls_buffer[idx]["name"] = tc_delta["function"]["name"]
                                        if "arguments" in tc_delta["function"]:
                                            tool_calls_buffer[idx]["arguments"] += tc_delta["function"]["arguments"]

                    except json.JSONDecodeError:
                        pass

            if tool_calls_buffer:
                print(f"\n\n✅ Tool calls received during streaming:")
                for idx, tc in tool_calls_buffer.items():
                    print(f"\n  [{idx}] {tc['name']}")
                    print(f"      Args: {tc['arguments']}")

            if full_content:
                print(f"\n\n✅ Content received: {len(full_content)} chars")


async def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("MiniMax-M2 Proxy - Comprehensive Tool Calling Tests")
    print("="*60)

    tests = [
        ("Think Blocks", test_think_blocks),
        ("Single Tool Call", test_single_tool_call),
        ("Multi-Step Tool Calling", test_multistep_tool_calling),
        ("Anthropic Format", test_anthropic_tool_calling),
        ("Streaming with Tools", test_streaming_with_tools),
    ]

    results = []

    for name, test_func in tests:
        try:
            await test_func()
            results.append((name, "✅ PASSED"))
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, f"❌ FAILED: {str(e)}"))

        await asyncio.sleep(2)  # Brief pause between tests

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for name, status in results:
        print(f"{status} - {name}")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
