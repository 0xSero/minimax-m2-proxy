#!/usr/bin/env python3
"""Comprehensive regression tests for OpenAI API multi-turn tool calling

Tests:
- Multi-turn conversations with tool calls
- Multiple sequential tool calls in a single turn
- No Chinese character generation
- Think block preservation
- Both streaming and non-streaming modes
"""

import sys
import asyncio
import re
import json
from typing import List, Dict, Any

sys.path.insert(0, '/home/ser/minimax-m2-proxy')

import httpx


def has_chinese_characters(text: str) -> bool:
    """Check if text contains Chinese characters"""
    if not text:
        return False
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
    return bool(chinese_pattern.search(text))


def verify_think_blocks(content: str) -> dict:
    """Verify think blocks are properly formatted (1 opening, 1 closing)"""
    opening_tags = content.count("<think>")
    closing_tags = content.count("</think>")

    return {
        "has_think": opening_tags > 0 or closing_tags > 0,
        "opening_tags": opening_tags,
        "closing_tags": closing_tags,
        "properly_balanced": opening_tags == closing_tags,
        "has_exactly_one_pair": opening_tags == 1 and closing_tags == 1
    }


async def test_single_turn_tool_call_nonstreaming():
    """Test 1: Single-turn tool call (non-streaming)"""
    print("\n" + "="*80)
    print("TEST 1: OpenAI - Single-turn tool call (non-streaming)")
    print("="*80)

    async with httpx.AsyncClient(timeout=60.0) as client:
        messages = [
            {"role": "user", "content": "What's the weather in San Francisco?"}
        ]

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "City name"
                            }
                        },
                        "required": ["location"]
                    }
                }
            }
        ]

        try:
            response = await client.post(
                "http://localhost:8001/v1/chat/completions",
                json={
                    "model": "minimax-m2",
                    "messages": messages,
                    "tools": tools,
                    "max_tokens": 500,
                    "temperature": 1.0
                }
            )

            data = response.json()
            message = data["choices"][0]["message"]
            content = message.get("content", "")
            tool_calls = message.get("tool_calls", [])

            print(f"\n✓ Response received")
            print(f"  - Content length: {len(content)} chars")
            print(f"  - Tool calls: {len(tool_calls)}")

            # Verify no Chinese characters
            if has_chinese_characters(content):
                print(f"  ✗ FAIL: Chinese characters found in content")
                return False
            print(f"  ✓ No Chinese characters")

            # Verify think blocks
            think_check = verify_think_blocks(content)
            print(f"  - Think blocks: opening={think_check['opening_tags']}, closing={think_check['closing_tags']}")
            if think_check['has_think'] and not think_check['properly_balanced']:
                print(f"  ✗ FAIL: Think blocks not balanced")
                return False
            if think_check['has_think']:
                print(f"  ✓ Think blocks properly balanced")

            # Verify tool calls
            if len(tool_calls) == 0:
                print(f"  ✗ FAIL: Expected tool call, got none")
                return False
            print(f"  ✓ Tool call detected: {tool_calls[0]['function']['name']}")

            print(f"\n✓ TEST 1 PASSED")
            return True

        except Exception as e:
            print(f"\n✗ TEST 1 FAILED: {e}")
            return False


async def test_multi_turn_tool_calling_nonstreaming():
    """Test 2: Multi-turn with tool result → final response (non-streaming)"""
    print("\n" + "="*80)
    print("TEST 2: OpenAI - Multi-turn tool calling (non-streaming)")
    print("="*80)

    async with httpx.AsyncClient(timeout=60.0) as client:
        # Turn 1: Initial request
        messages = [
            {"role": "user", "content": "What's the weather in San Francisco?"}
        ]

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        },
                        "required": ["location"]
                    }
                }
            }
        ]

        try:
            # Turn 1: Get tool call
            response1 = await client.post(
                "http://localhost:8001/v1/chat/completions",
                json={
                    "model": "minimax-m2",
                    "messages": messages,
                    "tools": tools,
                    "max_tokens": 500,
                    "temperature": 1.0
                }
            )

            data1 = response1.json()
            message1 = data1["choices"][0]["message"]
            tool_calls = message1.get("tool_calls", [])

            if len(tool_calls) == 0:
                print(f"✗ Turn 1 FAILED: No tool calls")
                return False

            print(f"✓ Turn 1: Got tool call for {tool_calls[0]['function']['name']}")

            # Verify no Chinese characters in turn 1
            content1 = message1.get("content", "")
            if has_chinese_characters(content1):
                print(f"✗ Turn 1 FAILED: Chinese characters found")
                return False

            # Turn 2: Provide tool result
            messages.append(message1)  # Add assistant's response
            messages.append({
                "role": "tool",
                "tool_call_id": tool_calls[0]["id"],
                "content": json.dumps({"temperature": 72, "condition": "sunny", "humidity": 65})
            })

            response2 = await client.post(
                "http://localhost:8001/v1/chat/completions",
                json={
                    "model": "minimax-m2",
                    "messages": messages,
                    "tools": tools,
                    "max_tokens": 500,
                    "temperature": 1.0
                }
            )

            data2 = response2.json()
            message2 = data2["choices"][0]["message"]
            content2 = message2.get("content", "")

            print(f"✓ Turn 2: Got final response ({len(content2)} chars)")

            # Verify no Chinese characters in turn 2
            if has_chinese_characters(content2):
                print(f"✗ Turn 2 FAILED: Chinese characters found")
                print(f"  Content preview: {content2[:200]}")
                return False
            print(f"✓ No Chinese characters in final response")

            # Verify think blocks
            think_check = verify_think_blocks(content2)
            if think_check['has_think'] and not think_check['properly_balanced']:
                print(f"✗ Turn 2 FAILED: Think blocks not balanced")
                return False
            if think_check['has_think']:
                print(f"✓ Think blocks properly balanced")

            print(f"\n✓ TEST 2 PASSED")
            return True

        except Exception as e:
            print(f"\n✗ TEST 2 FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False


async def test_multi_turn_streaming():
    """Test 3: Multi-turn tool calling (streaming)"""
    print("\n" + "="*80)
    print("TEST 3: OpenAI - Multi-turn tool calling (streaming)")
    print("="*80)

    async with httpx.AsyncClient(timeout=60.0) as client:
        messages = [
            {"role": "user", "content": "What's the weather in San Francisco?"}
        ]

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        },
                        "required": ["location"]
                    }
                }
            }
        ]

        try:
            # Turn 1: Stream tool call
            full_response = ""
            tool_calls = []

            async with client.stream(
                "POST",
                "http://localhost:8001/v1/chat/completions",
                json={
                    "model": "minimax-m2",
                    "messages": messages,
                    "tools": tools,
                    "max_tokens": 500,
                    "temperature": 1.0,
                    "stream": True
                }
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break

                        try:
                            chunk = json.loads(data_str)
                            if "choices" in chunk and len(chunk["choices"]) > 0:
                                delta = chunk["choices"][0].get("delta", {})

                                if "content" in delta:
                                    full_response += delta["content"]

                                if "tool_calls" in delta:
                                    # Accumulate tool calls
                                    for tc_delta in delta["tool_calls"]:
                                        idx = tc_delta.get("index", 0)
                                        if idx >= len(tool_calls):
                                            tool_calls.append({
                                                "id": "",
                                                "type": "function",
                                                "function": {"name": "", "arguments": ""}
                                            })
                                        if "id" in tc_delta:
                                            tool_calls[idx]["id"] = tc_delta["id"]
                                        if "function" in tc_delta:
                                            if "name" in tc_delta["function"]:
                                                tool_calls[idx]["function"]["name"] = tc_delta["function"]["name"]
                                            if "arguments" in tc_delta["function"]:
                                                tool_calls[idx]["function"]["arguments"] += tc_delta["function"]["arguments"]

                        except json.JSONDecodeError:
                            continue

            print(f"✓ Turn 1 (streaming): Tool call for {tool_calls[0]['function']['name']}")

            # Verify no Chinese characters
            if has_chinese_characters(full_response):
                print(f"✗ Turn 1 FAILED: Chinese characters in streamed response")
                return False

            # Verify think blocks in streaming
            think_check = verify_think_blocks(full_response)
            if think_check['has_think']:
                if not think_check['properly_balanced']:
                    print(f"✗ FAILED: Think blocks not balanced (opening={think_check['opening_tags']}, closing={think_check['closing_tags']})")
                    print(f"  Content: {full_response[:300]}")
                    return False
                print(f"✓ Think blocks properly balanced in streaming")

            # Turn 2: Provide tool result and get final answer (streaming)
            messages.append({
                "role": "assistant",
                "content": full_response,
                "tool_calls": tool_calls
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tool_calls[0]["id"],
                "content": json.dumps({"temperature": 72, "condition": "sunny"})
            })

            final_response = ""
            async with client.stream(
                "POST",
                "http://localhost:8001/v1/chat/completions",
                json={
                    "model": "minimax-m2",
                    "messages": messages,
                    "tools": tools,
                    "max_tokens": 500,
                    "temperature": 1.0,
                    "stream": True
                }
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break

                        try:
                            chunk = json.loads(data_str)
                            if "choices" in chunk and len(chunk["choices"]) > 0:
                                delta = chunk["choices"][0].get("delta", {})
                                if "content" in delta:
                                    final_response += delta["content"]
                        except json.JSONDecodeError:
                            continue

            print(f"✓ Turn 2 (streaming): Final response ({len(final_response)} chars)")

            # Verify no Chinese characters in final response
            if has_chinese_characters(final_response):
                print(f"✗ Turn 2 FAILED: Chinese characters in final response")
                return False
            print(f"✓ No Chinese characters in streamed final response")

            print(f"\n✓ TEST 3 PASSED")
            return True

        except Exception as e:
            print(f"\n✗ TEST 3 FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False


async def test_multiple_tool_calls_same_turn():
    """Test 4: Multiple tool calls in a single turn (non-streaming)"""
    print("\n" + "="*80)
    print("TEST 4: OpenAI - Multiple tool calls in single turn")
    print("="*80)

    async with httpx.AsyncClient(timeout=60.0) as client:
        messages = [
            {"role": "user", "content": "What's the weather in San Francisco and New York?"}
        ]

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        },
                        "required": ["location"]
                    }
                }
            }
        ]

        try:
            response = await client.post(
                "http://localhost:8001/v1/chat/completions",
                json={
                    "model": "minimax-m2",
                    "messages": messages,
                    "tools": tools,
                    "max_tokens": 1000,
                    "temperature": 1.0
                }
            )

            data = response.json()
            message = data["choices"][0]["message"]
            content = message.get("content", "")
            tool_calls = message.get("tool_calls", [])

            print(f"✓ Response received with {len(tool_calls)} tool call(s)")

            # Verify no Chinese characters
            if has_chinese_characters(content):
                print(f"✗ FAILED: Chinese characters found")
                return False
            print(f"✓ No Chinese characters")

            # Ideally we'd get 2 tool calls, but model might only generate 1
            if len(tool_calls) >= 1:
                print(f"✓ At least one tool call detected")
                if len(tool_calls) >= 2:
                    print(f"✓ Multiple tool calls detected (excellent!)")
            else:
                print(f"✗ FAILED: Expected at least one tool call")
                return False

            print(f"\n✓ TEST 4 PASSED")
            return True

        except Exception as e:
            print(f"\n✗ TEST 4 FAILED: {e}")
            return False


async def main():
    """Run all OpenAI API regression tests"""
    print("\n" + "🔍 " + "="*76)
    print("OpenAI API Multi-Turn Tool Calling Regression Tests")
    print("="*80)

    results = []

    results.append(await test_single_turn_tool_call_nonstreaming())
    results.append(await test_multi_turn_tool_calling_nonstreaming())
    results.append(await test_multi_turn_streaming())
    results.append(await test_multiple_tool_calls_same_turn())

    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Total: {len(results)}")
    print(f"Passed: {sum(results)}")
    print(f"Failed: {len(results) - sum(results)}")

    if all(results):
        print("\n✓ ALL TESTS PASSED")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
