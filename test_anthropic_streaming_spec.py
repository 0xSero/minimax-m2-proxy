#!/usr/bin/env python3
"""Test to understand Anthropic's tool use streaming format"""

import anthropic

# Use real Anthropic API to see correct format
client = anthropic.Anthropic(api_key="sk-ant-api03-test")  # Will fail but we can see docs

# According to Anthropic docs, tool_use blocks are sent like this:
# 1. content_block_start with type="tool_use", id, name, input={}
# 2. content_block_delta with type="input_json_delta", partial_json
# 3. content_block_stop

# Our current implementation sends:
# 1. content_block_start with type="tool_use", id, name, input={complete}
# 2. content_block_stop

# This means Chatbox AI might be expecting incremental input_json_delta events!

print("""
FOUND THE BUG!

According to Anthropic's streaming spec, tool_use blocks should be sent as:

1. content_block_start:
   {
     "type": "content_block_start",
     "index": 2,
     "content_block": {
       "type": "tool_use",
       "id": "toolu_123",
       "name": "get_weather",
       "input": {}  # <-- EMPTY initially!
     }
   }

2. content_block_delta (one or more):
   {
     "type": "content_block_delta",
     "index": 2,
     "delta": {
       "type": "input_json_delta",
       "partial_json": '{"location"'  # <-- Incremental JSON
     }
   }

3. content_block_delta (continued):
   {
     "type": "content_block_delta",
     "index": 2,
     "delta": {
       "type": "input_json_delta",
       "partial_json": ': "Paris"}'  # <-- Complete the JSON
     }
   }

4. content_block_stop:
   {
     "type": "content_block_stop",
     "index": 2
   }

But we're currently sending:
- content_block_start with input={"query": "0xSero"}  # <-- Complete input!
- content_block_stop

This means Chatbox AI sees:
1. Tool use started with input={"query": "0xSero"}
2. Tool use immediately stopped

BUT if Chatbox is expecting input_json_delta events and we never send them,
it might default to using the empty {} from content_block_start!
""")
