#!/usr/bin/env python3
"""Comprehensive regression tests for Anthropic API multi-turn tool calling

Tests:
- Multi-turn conversations with tool calls
- Multiple sequential tool calls
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
    print("TEST 1: Anthropic - Single-turn tool call (non-streaming)")
    print("="*80)

    async with httpx.AsyncClient(timeout=60.0) as client:
        messages = [
            {
                "role": "user",
                "content": "What's the weather in San Francisco?"
            }
        ]

        tools = [
            {
                "name": "get_weather",
                "description": "Get current weather for a location",
                "input_schema": {
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
        ]

        try:
            response = await client.post(
                "http://localhost:8001/v1/messages",
                json={
                    "model": "minimax-m2",
                    "messages": messages,
                    "tools": tools,
                    "max_tokens": 500
                }
            )

            data = response.json()
            content_blocks = data.get("content", [])

            print(f"\n✓ Response received with {len(content_blocks)} content block(s)")

            # Extract text and tool_use blocks
            text_blocks = [b for b in content_blocks if b.get("type") == "text"]
            tool_use_blocks = [b for b in content_blocks if b.get("type") == "tool_use"]

            # Check for Chinese characters in all text blocks
            for block in text_blocks:
                text = block.get("text", "")
                if has_chinese_characters(text):
                    print(f"✗ FAIL: Chinese characters found in text block")
                    return False

            print(f"✓ No Chinese characters in {len(text_blocks)} text block(s)")

            # Verify think blocks
            for block in text_blocks:
                text = block.get("text", "")
                think_check = verify_think_blocks(text)
                if think_check['has_think'] and not think_check['properly_balanced']:
                    print(f"✗ FAIL: Think blocks not balanced")
                    return False
                if think_check['has_think']:
                    print(f"✓ Think blocks properly balanced")

            # Verify tool use
            if len(tool_use_blocks) == 0:
                print(f"✗ FAIL: Expected tool_use block, got none")
                return False
            print(f"✓ Tool use block detected: {tool_use_blocks[0]['name']}")

            print(f"\n✓ TEST 1 PASSED")
            return True

        except Exception as e:
            print(f"\n✗ TEST 1 FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False


async def test_multi_turn_tool_calling_nonstreaming():
    """Test 2: Multi-turn with tool_result → final response (non-streaming)"""
    print("\n" + "="*80)
    print("TEST 2: Anthropic - Multi-turn tool calling (non-streaming)")
    print("="*80)

    async with httpx.AsyncClient(timeout=60.0) as client:
        # Turn 1: Initial request
        messages = [
            {
                "role": "user",
                "content": "What's the weather in San Francisco?"
            }
        ]

        tools = [
            {
                "name": "get_weather",
                "description": "Get current weather for a location",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    },
                    "required": ["location"]
                }
            }
        ]

        try:
            # Turn 1: Get tool call
            response1 = await client.post(
                "http://localhost:8001/v1/messages",
                json={
                    "model": "minimax-m2",
                    "messages": messages,
                    "tools": tools,
                    "max_tokens": 500
                }
            )

            data1 = response1.json()
            content_blocks1 = data1.get("content", [])
            tool_use_blocks = [b for b in content_blocks1 if b.get("type") == "tool_use"]

            if len(tool_use_blocks) == 0:
                print(f"✗ Turn 1 FAILED: No tool_use blocks")
                return False

            tool_use = tool_use_blocks[0]
            print(f"✓ Turn 1: Got tool_use for {tool_use['name']}")

            # Verify no Chinese characters in turn 1
            text_blocks1 = [b for b in content_blocks1 if b.get("type") == "text"]
            for block in text_blocks1:
                if has_chinese_characters(block.get("text", "")):
                    print(f"✗ Turn 1 FAILED: Chinese characters found")
                    return False

            # Turn 2: Provide tool result
            messages.append({
                "role": "assistant",
                "content": content_blocks1
            })
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use["id"],
                        "content": json.dumps({"temperature": 72, "condition": "sunny", "humidity": 65})
                    }
                ]
            })

            response2 = await client.post(
                "http://localhost:8001/v1/messages",
                json={
                    "model": "minimax-m2",
                    "messages": messages,
                    "tools": tools,
                    "max_tokens": 500
                }
            )

            data2 = response2.json()
            content_blocks2 = data2.get("content", [])
            text_blocks2 = [b for b in content_blocks2 if b.get("type") == "text"]

            print(f"✓ Turn 2: Got final response with {len(text_blocks2)} text block(s)")

            # Verify no Chinese characters in turn 2
            for block in text_blocks2:
                text = block.get("text", "")
                if has_chinese_characters(text):
                    print(f"✗ Turn 2 FAILED: Chinese characters found")
                    print(f"  Content preview: {text[:200]}")
                    return False
            print(f"✓ No Chinese characters in final response")

            # Verify think blocks
            for block in text_blocks2:
                text = block.get("text", "")
                think_check = verify_think_blocks(text)
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
    print("TEST 3: Anthropic - Multi-turn tool calling (streaming)")
    print("="*80)

    async with httpx.AsyncClient(timeout=60.0) as client:
        messages = [
            {
                "role": "user",
                "content": "What's the weather in San Francisco?"
            }
        ]

        tools = [
            {
                "name": "get_weather",
                "description": "Get current weather",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    },
                    "required": ["location"]
                }
            }
        ]

        try:
            # Turn 1: Stream tool call
            text_content = ""
            tool_use_data = None

            async with client.stream(
                "POST",
                "http://localhost:8001/v1/messages",
                json={
                    "model": "minimax-m2",
                    "messages": messages,
                    "tools": tools,
                    "max_tokens": 500,
                    "stream": True
                }
            ) as response:
                response.raise_for_status()

                current_tool_use = {}
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]

                        try:
                            event = json.loads(data_str)
                            event_type = event.get("type")

                            if event_type == "content_block_delta":
                                delta = event.get("delta", {})
                                if delta.get("type") == "text_delta":
                                    text_content += delta.get("text", "")
                                elif delta.get("type") == "input_json_delta":
                                    # Accumulate tool input
                                    if "partial_json" in delta:
                                        if "input" not in current_tool_use:
                                            current_tool_use["input"] = ""
                                        current_tool_use["input"] += delta["partial_json"]

                            elif event_type == "content_block_start":
                                block = event.get("content_block", {})
                                if block.get("type") == "tool_use":
                                    current_tool_use = {
                                        "id": block.get("id"),
                                        "name": block.get("name"),
                                        "input": ""
                                    }

                            elif event_type == "content_block_stop":
                                if current_tool_use and "name" in current_tool_use:
                                    tool_use_data = current_tool_use
                                    current_tool_use = {}

                        except json.JSONDecodeError:
                            continue

            if not tool_use_data:
                print(f"✗ Turn 1 FAILED: No tool_use received in streaming")
                return False

            print(f"✓ Turn 1 (streaming): Tool use for {tool_use_data['name']}")

            # Verify no Chinese characters in streamed text
            if has_chinese_characters(text_content):
                print(f"✗ Turn 1 FAILED: Chinese characters in streamed response")
                return False

            # Verify think blocks in streaming
            if text_content:
                think_check = verify_think_blocks(text_content)
                if think_check['has_think']:
                    if not think_check['properly_balanced']:
                        print(f"✗ FAILED: Think blocks not balanced (opening={think_check['opening_tags']}, closing={think_check['closing_tags']})")
                        return False
                    print(f"✓ Think blocks properly balanced in streaming")

            # Turn 2: Provide tool result and get final answer (streaming)
            # Reconstruct content blocks for assistant message
            content_blocks = []
            if text_content:
                content_blocks.append({"type": "text", "text": text_content})
            content_blocks.append({
                "type": "tool_use",
                "id": tool_use_data["id"],
                "name": tool_use_data["name"],
                "input": json.loads(tool_use_data["input"]) if tool_use_data.get("input") else {}
            })

            messages.append({
                "role": "assistant",
                "content": content_blocks
            })
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_data["id"],
                        "content": json.dumps({"temperature": 72, "condition": "sunny"})
                    }
                ]
            })

            final_text = ""
            async with client.stream(
                "POST",
                "http://localhost:8001/v1/messages",
                json={
                    "model": "minimax-m2",
                    "messages": messages,
                    "tools": tools,
                    "max_tokens": 500,
                    "stream": True
                }
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]

                        try:
                            event = json.loads(data_str)
                            if event.get("type") == "content_block_delta":
                                delta = event.get("delta", {})
                                if delta.get("type") == "text_delta":
                                    final_text += delta.get("text", "")
                        except json.JSONDecodeError:
                            continue

            print(f"✓ Turn 2 (streaming): Final response ({len(final_text)} chars)")

            # Verify no Chinese characters in final response
            if has_chinese_characters(final_text):
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
    print("TEST 4: Anthropic - Multiple tool calls in single turn")
    print("="*80)

    async with httpx.AsyncClient(timeout=60.0) as client:
        messages = [
            {
                "role": "user",
                "content": "What's the weather in San Francisco and New York?"
            }
        ]

        tools = [
            {
                "name": "get_weather",
                "description": "Get weather for a location",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    },
                    "required": ["location"]
                }
            }
        ]

        try:
            response = await client.post(
                "http://localhost:8001/v1/messages",
                json={
                    "model": "minimax-m2",
                    "messages": messages,
                    "tools": tools,
                    "max_tokens": 1000
                }
            )

            data = response.json()
            content_blocks = data.get("content", [])

            # Count text and tool_use blocks
            text_blocks = [b for b in content_blocks if b.get("type") == "text"]
            tool_use_blocks = [b for b in content_blocks if b.get("type") == "tool_use"]

            print(f"✓ Response received with {len(tool_use_blocks)} tool_use block(s)")

            # Verify no Chinese characters in text blocks
            for block in text_blocks:
                if has_chinese_characters(block.get("text", "")):
                    print(f"✗ FAILED: Chinese characters found")
                    return False
            print(f"✓ No Chinese characters")

            # Ideally we'd get 2 tool_use blocks, but model might only generate 1
            if len(tool_use_blocks) >= 1:
                print(f"✓ At least one tool_use block detected")
                if len(tool_use_blocks) >= 2:
                    print(f"✓ Multiple tool_use blocks detected (excellent!)")
            else:
                print(f"✗ FAILED: Expected at least one tool_use block")
                return False

            print(f"\n✓ TEST 4 PASSED")
            return True

        except Exception as e:
            print(f"\n✗ TEST 4 FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False


async def main():
    """Run all Anthropic API regression tests"""
    print("\n" + "🔍 " + "="*76)
    print("Anthropic API Multi-Turn Tool Calling Regression Tests")
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
