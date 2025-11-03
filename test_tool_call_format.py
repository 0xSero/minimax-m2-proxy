#!/usr/bin/env python3
"""Test tool call format to find the string issue"""

import httpx
import json

url = "http://localhost:8001/v1/chat/completions"

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

data = {
    "model": "minimax-m2",
    "messages": [
        {"role": "user", "content": "Search for Python"}
    ],
    "tools": tools,
    "stream": True,
    "max_tokens": 500
}

print("=" * 80)
print("Testing tool call format in streaming")
print("=" * 80)

with httpx.stream("POST", url, json=data, timeout=60) as response:
    for line in response.iter_lines():
        if line.startswith("data: "):
            data_str = line[6:]
            
            if data_str == "[DONE]":
                break
                
            try:
                chunk = json.loads(data_str)
                
                if "choices" in chunk and chunk["choices"]:
                    delta = chunk["choices"][0].get("delta", {})
                    
                    # Check for tool calls
                    if "tool_calls" in delta:
                        for tc in delta["tool_calls"]:
                            print(f"\nTool call chunk:")
                            print(f"  Index: {tc.get('index')}")
                            print(f"  ID: {tc.get('id')}")
                            print(f"  Type: {tc.get('type')}")
                            
                            if "function" in tc:
                                func = tc["function"]
                                print(f"  Function: {func}")
                                print(f"    Name type: {type(func.get('name'))}")
                                print(f"    Name value: {repr(func.get('name'))}")
                                print(f"    Arguments type: {type(func.get('arguments'))}")
                                print(f"    Arguments value: {repr(func.get('arguments'))}")
                                
            except json.JSONDecodeError as e:
                print(f"JSON error: {e}")
