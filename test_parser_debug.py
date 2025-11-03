#!/usr/bin/env python3
"""Test the tool call parser with array parameters"""

import json
import sys
sys.path.insert(0, '/home/ser/minimax-m2-proxy')

from parsers.tools import ToolCallParser

# Test XML that mimics what the model generates
test_xml = """<think>
I need to run ls command
</think>

<minimax:tool_call>
<invoke name="shell">
<parameter name="command">["ls", "-la"]</parameter>
</invoke>
</minimax:tool_call>"""

# Tool schema like Codex would send
tools = [{
    "type": "function",
    "function": {
        "name": "shell",
        "description": "Execute shell command",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Command and arguments as array"
                }
            },
            "required": ["command"]
        }
    }
}]

parser = ToolCallParser()
result = parser.parse_tool_calls(test_xml, tools)

print("=" * 80)
print("INPUT XML:")
print(test_xml)
print("\n" + "=" * 80)
print("PARSED RESULT:")
print(json.dumps(result, indent=2))

if result["tool_calls"]:
    tool_call = result["tool_calls"][0]
    print("\n" + "=" * 80)
    print("TOOL CALL DETAILS:")
    print(f"Function name: {tool_call['function']['name']}")
    print(f"Arguments (raw string): {tool_call['function']['arguments']}")
    print(f"Arguments length: {len(tool_call['function']['arguments'])}")

    # Try to parse it like Codex would
    try:
        args = json.loads(tool_call['function']['arguments'])
        print(f"\nParsed arguments: {args}")
        print(f"Command type: {type(args['command'])}")
        print(f"Command value: {args['command']}")
    except Exception as e:
        print(f"\n❌ ERROR parsing arguments: {e}")
