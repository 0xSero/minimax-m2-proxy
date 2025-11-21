#!/usr/bin/env python3
"""Test what the proxy actually sends in streaming mode"""

import httpx

url = "http://localhost:8001/v1/chat/completions"
payload = {
    "model": "minimax-m2",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "stream": True,
    "extra_body": {"reasoning_split": True}
}

print("Testing raw proxy output...\n")

with httpx.stream("POST", url, json=payload, timeout=60) as response:
    for i, line in enumerate(response.iter_lines()):
        if i < 15:  # Only show first 15 lines
            print(f"Line {i}: {line}")
        if i == 15:
            print("... (truncated)")
            break
