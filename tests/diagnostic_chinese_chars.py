#!/usr/bin/env python3
"""Diagnostic test to identify where Chinese characters originate

Tests multiple scenarios to isolate the Chinese character issue:
1. Direct TabbyAPI call (no proxy)
2. Proxy OpenAI endpoint
3. Proxy Anthropic endpoint
4. With and without tools
5. Streaming vs non-streaming
"""

import sys
import asyncio
import json
import re
sys.path.insert(0, '/home/ser/minimax-m2-proxy')

from proxy.client import TabbyClient
import httpx


def has_chinese_characters(text: str) -> bool:
    """Check if text contains Chinese characters"""
    if not text:
        return False
    # Unicode range for CJK (Chinese, Japanese, Korean) characters
    chinese_pattern = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf\u20000-\u2a6df\u2a700-\u2b73f\u2b740-\u2b81f\u2b820-\u2ceaf\uf900-\ufaff\u3300-\u33ff\ufe30-\ufe4f\uf900-\ufaff\u2f800-\u2fa1f]')
    return bool(chinese_pattern.search(text))


def extract_chinese_chars(text: str) -> list:
    """Extract all Chinese characters from text"""
    chinese_pattern = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf\u20000-\u2a6df\u2a700-\u2b73f\u2b740-\u2b81f\u2b820-\u2ceaf\uf900-\ufaff\u3300-\u33ff\ufe30-\ufe4f\uf900-\ufaff\u2f800-\u2fa1f]')
    return chinese_pattern.findall(text)


async def test_direct_tabbyapi_no_tools():
    """Test 1: Direct TabbyAPI without proxy, no tools"""
    print("\n" + "="*80)
    print("TEST 1: Direct TabbyAPI - No Tools")
    print("="*80)

    client = TabbyClient("http://localhost:8000")

    messages = [
        {"role": "user", "content": "Explain what a REST API is in 3 sentences."}
    ]

    try:
        response = await client.chat_completion(
            messages=messages,
            model="minimax-m2",
            max_tokens=200,
            temperature=1.0,
            top_p=0.95,
            top_k=40,
            add_generation_prompt=True
        )

        content = response["choices"][0]["message"].get("content", "")

        print(f"\nResponse length: {len(content)} chars")
        print(f"Has Chinese chars: {has_chinese_characters(content)}")

        if has_chinese_characters(content):
            chinese_chars = extract_chinese_chars(content)
            print(f"Chinese chars found: {chinese_chars}")
            print(f"Count: {len(chinese_chars)}")

        print(f"\nFull content:\n{content}")

    except Exception as e:
        print(f"ERROR: {e}")
    finally:
        await client.close()


async def test_direct_tabbyapi_with_tools():
    """Test 2: Direct TabbyAPI without proxy, with tools"""
    print("\n" + "="*80)
    print("TEST 2: Direct TabbyAPI - With Tools")
    print("="*80)

    client = TabbyClient("http://localhost:8000")

    messages = [
        {"role": "user", "content": "What's the weather in San Francisco?"}
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
        response = await client.chat_completion(
            messages=messages,
            model="minimax-m2",
            max_tokens=500,
            temperature=1.0,
            top_p=0.95,
            top_k=40,
            tools=tools,
            add_generation_prompt=True
        )

        content = response["choices"][0]["message"].get("content", "")

        print(f"\nResponse length: {len(content)} chars")
        print(f"Has Chinese chars: {has_chinese_characters(content)}")

        if has_chinese_characters(content):
            chinese_chars = extract_chinese_chars(content)
            print(f"Chinese chars found: {chinese_chars}")
            print(f"Count: {len(chinese_chars)}")

        print(f"\nFull content (first 500 chars):\n{content[:500]}")

    except Exception as e:
        print(f"ERROR: {e}")
    finally:
        await client.close()


async def test_proxy_openai_with_tools():
    """Test 3: Through proxy OpenAI endpoint"""
    print("\n" + "="*80)
    print("TEST 3: Proxy OpenAI Endpoint - With Tools")
    print("="*80)

    async with httpx.AsyncClient() as client:
        messages = [
            {"role": "user", "content": "What's the weather in San Francisco?"}
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
                    "max_tokens": 500,
                    "temperature": 1.0
                },
                timeout=60.0
            )

            data = response.json()
            content = data["choices"][0]["message"].get("content", "")

            print(f"\nResponse length: {len(content)} chars")
            print(f"Has Chinese chars: {has_chinese_characters(content)}")

            if has_chinese_characters(content):
                chinese_chars = extract_chinese_chars(content)
                print(f"Chinese chars found: {chinese_chars}")
                print(f"Count: {len(chinese_chars)}")

            print(f"\nFull content (first 500 chars):\n{content[:500]}")

        except Exception as e:
            print(f"ERROR: {e}")


async def test_streaming_for_chinese_chars():
    """Test 4: Streaming to see when Chinese chars appear"""
    print("\n" + "="*80)
    print("TEST 4: Streaming Analysis")
    print("="*80)

    client = TabbyClient("http://localhost:8000")

    messages = [
        {"role": "user", "content": "Describe the main features of Python programming language."}
    ]

    try:
        full_response = ""
        chunk_count = 0

        async for chunk in client.extract_streaming_content(
            messages=messages,
            model="minimax-m2",
            max_tokens=300,
            temperature=1.0,
            top_p=0.95,
            top_k=40,
            add_generation_prompt=True
        ):
            if "choices" in chunk and len(chunk["choices"]) > 0:
                delta = chunk["choices"][0].get("delta", {})
                content_delta = delta.get("content", "")

                if content_delta:
                    chunk_count += 1
                    full_response += content_delta

                    # Check each chunk for Chinese chars
                    if has_chinese_characters(content_delta):
                        chinese_chars = extract_chinese_chars(content_delta)
                        print(f"\n⚠️  CHINESE CHARS IN CHUNK {chunk_count}!")
                        print(f"Chunk: {repr(content_delta)}")
                        print(f"Chinese chars: {chinese_chars}")

        print(f"\nTotal chunks: {chunk_count}")
        print(f"Total response length: {len(full_response)} chars")
        print(f"Has Chinese chars overall: {has_chinese_characters(full_response)}")

        if has_chinese_characters(full_response):
            chinese_chars = extract_chinese_chars(full_response)
            print(f"All Chinese chars: {chinese_chars}")
            print(f"Total count: {len(chinese_chars)}")

    except Exception as e:
        print(f"ERROR: {e}")
    finally:
        await client.close()


async def test_without_add_generation_prompt():
    """Test 5: Without add_generation_prompt to see if that's the cause"""
    print("\n" + "="*80)
    print("TEST 5: Without add_generation_prompt")
    print("="*80)

    client = TabbyClient("http://localhost:8000")

    messages = [
        {"role": "user", "content": "What are the benefits of using Docker?"}
    ]

    try:
        response = await client.chat_completion(
            messages=messages,
            model="minimax-m2",
            max_tokens=200,
            temperature=1.0,
            top_p=0.95,
            top_k=40,
            add_generation_prompt=False  # Key difference
        )

        content = response["choices"][0]["message"].get("content", "")

        print(f"\nResponse length: {len(content)} chars")
        print(f"Has Chinese chars: {has_chinese_characters(content)}")

        if has_chinese_characters(content):
            chinese_chars = extract_chinese_chars(content)
            print(f"Chinese chars found: {chinese_chars}")
            print(f"Count: {len(chinese_chars)}")

        print(f"\nFull content:\n{content}")

    except Exception as e:
        print(f"ERROR: {e}")
    finally:
        await client.close()


async def main():
    print("\n" + "🔍 DIAGNOSTIC TEST: Chinese Character Generation")
    print("Testing multiple scenarios to isolate the issue...")

    await test_direct_tabbyapi_no_tools()
    await test_direct_tabbyapi_with_tools()
    await test_proxy_openai_with_tools()
    await test_streaming_for_chinese_chars()
    await test_without_add_generation_prompt()

    print("\n" + "="*80)
    print("DIAGNOSTIC COMPLETE")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())
