#!/usr/bin/env python3
"""Quick test to see what the parser outputs for the apply_patch XML"""

from parsers.tools import ToolCallParser

# Simulate the XML that the model is generating based on the logs
xml_content = '''<minimax:tool_call>
<invoke name="apply_patch">
<parameter name="command">["apply_patch", "patch content here"]</parameter>
</invoke>
</minimax:tool_call>'''

parser = ToolCallParser()
result = parser.parse_tool_calls(xml_content)

print("=== Parser Output ===")
print(f"tools_called: {result['tools_called']}")
print(f"\nNumber of tool_calls: {len(result['tool_calls'])}")
print(f"\nTool calls:")
for i, tool_call in enumerate(result['tool_calls']):
    print(f"\n  Tool Call #{i+1}:")
    print(f"    ID: {tool_call['id']}")
    print(f"    Type: {tool_call['type']}")
    print(f"    Function name: {tool_call['function']['name']}")
    print(f"    Arguments (raw): {tool_call['function']['arguments']}")
    print(f"    Arguments (type): {type(tool_call['function']['arguments'])}")
