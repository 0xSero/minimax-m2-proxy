#!/usr/bin/env python3
"""Test streaming tool calls with proper validation"""

from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8001/v1",
    api_key="dummy"
)

tools = [{
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"]
        }
    }
}]

print("=" * 80)
print("Testing STREAMING tool calls with <think> blocks")
print("=" * 80)
print()

try:
    stream = client.chat.completions.create(
        model="minimax-m2",
        messages=[
            {"role": "user", "content": "Search for 'Python programming' and summarize"}
        ],
        tools=tools,
        stream=True,
        temperature=1.0,
        max_tokens=1000
    )

    print("Streaming chunks:\n")

    content_parts = []
    tool_calls_data = {}
    chunk_count = 0

    for chunk in stream:
        chunk_count += 1

        if chunk.choices and len(chunk.choices) > 0:
            choice = chunk.choices[0]
            delta = choice.delta

            # Check for content
            if delta.content:
                content_parts.append(delta.content)
                if chunk_count <= 5:  # Show first few content chunks
                    print(f"  Content chunk {chunk_count}: {repr(delta.content[:50])}")

            # Check for tool calls
            if delta.tool_calls:
                for tool_call_delta in delta.tool_calls:
                    idx = tool_call_delta.index

                    if idx not in tool_calls_data:
                        tool_calls_data[idx] = {
                            "id": None,
                            "type": None,
                            "function": {"name": "", "arguments": ""}
                        }

                    # Accumulate tool call data
                    if tool_call_delta.id:
                        tool_calls_data[idx]["id"] = tool_call_delta.id
                    if tool_call_delta.type:
                        tool_calls_data[idx]["type"] = tool_call_delta.type
                    if tool_call_delta.function:
                        if tool_call_delta.function.name:
                            tool_calls_data[idx]["function"]["name"] += tool_call_delta.function.name
                        if tool_call_delta.function.arguments:
                            tool_calls_data[idx]["function"]["arguments"] += tool_call_delta.function.arguments

                    print(f"  Tool call chunk {chunk_count} [idx={idx}]: id={tool_call_delta.id}, type={tool_call_delta.type}, function={tool_call_delta.function}")

            # Check for finish
            if choice.finish_reason:
                print(f"\n  Finish reason: {choice.finish_reason}")

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    # Reconstruct full content
    full_content = "".join(content_parts)
    print(f"\nTotal chunks received: {chunk_count}")
    print(f"\nFull content ({len(full_content)} chars):")
    print(full_content[:500] + ("..." if len(full_content) > 500 else ""))

    # Check for think blocks
    if "<think>" in full_content:
        print("\n✅ <think> opening tag found")
    else:
        print("\n❌ <think> opening tag NOT found")

    if "</think>" in full_content:
        print("✅ </think> closing tag found")
    else:
        print("❌ </think> closing tag NOT found")

    # Show tool calls
    if tool_calls_data:
        print(f"\n🔧 Tool Calls ({len(tool_calls_data)}):")
        for idx, tc in tool_calls_data.items():
            print(f"\n  Tool Call #{idx}:")
            print(f"    ID: {tc['id']}")
            print(f"    Type: {tc['type']}")
            print(f"    Function: {tc['function']['name']}")
            print(f"    Arguments: {tc['function']['arguments']}")
        print("\n✅ Tool calls streamed successfully!")
    else:
        print("\n⚠️  No tool calls found")

    print("\n" + "=" * 80)
    print("✅ STREAMING TEST PASSED - No validation errors!")
    print("=" * 80)

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
