# MiniMax-M2 Chat Template Compliance Verification

**Date**: 2025-11-03
**Template Location**: `/mnt/llm_models/MiniMaxAI_MiniMax-M2-EXL3/chat_template.jinja`
**Status**: ✅ Fully Compliant

## Message Markers (All Present)

The chat template includes all required MiniMax-specific message markers:

- **`]~!b[]~b]system`** (Line 61) - System message marker
- **`]~b]ai`** (Line 85) - Assistant message marker
- **`]~b]tool`** (Line 153) - Tool result marker
- **`]~b]user`** (Line 171) - User message marker
- **`[e~[`** (Lines 78, 146, 167, 173) - End of block marker

These markers are automatically added by TabbyAPI when applying the chat template via `tokenizer.apply_chat_template(messages, tools=tools)`.

## Tool Call Format (XML-Based)

The template correctly implements MiniMax's XML tool calling format:

```xml
<minimax:tool_call>
<invoke name="function_name">
<parameter name="param1">value1</parameter>
<parameter name="param2">value2</parameter>
</invoke>
</minimax:tool_call>
```

**Template implementation**:
- Lines 2-3: Defines `<minimax:tool_call>` wrapper tokens
- Lines 104-141: Renders assistant tool calls in proper XML format
- Lines 69-76: Shows tool call format example in system prompt

## Think Block Handling

The template preserves and automatically adds `<think>` blocks for reasoning:

- **Line 179**: Automatically adds opening `<think>` tag to generation prompt
- **Lines 87-99**: Preserves `<think>` blocks from assistant messages
- **Lines 92-95**: Extracts reasoning content from existing messages

This ensures the model always starts reasoning in `<think>` mode even though TabbyAPI doesn't include the opening tag in the completion (it's part of the prompt).

## Tool Result Format

The template has **native support** for `role: "tool"` messages (Lines 148-168):

```
]~b]tool
<response>tool result content</response>
[e~[
```

However, the official MiniMax documentation recommends using `role: "user"` instead. Our proxy implements the transformation approach:

```python
{"role": "tool", "content": "result"}
→
{"role": "user", "content": "Tool Result (function_name):\nresult"}
```

**Why we transform to `role: "user"`**:
1. ✅ Official MiniMax guidance recommends it
2. ✅ Proven working in production (confirmed with codex-tui)
3. ✅ More explicit context with function name prefix
4. ✅ Better compatibility with clients

## Tool Definition Format

The template renders tools in JSONSchema format wrapped in `<tools>` tags (Lines 64-77):

```xml
<tools>
<tool>{"name": "function_name", "parameters": {...}}</tool>
</tools>
```

This matches the official MiniMax specification.

## Verification Commands

To verify the template is loaded and active:

```bash
# Check TabbyAPI logs for template loading
grep -i "chat.*template" /mnt/llm_models/tabbyAPI/server_minimax_m2_180k_tp.out

# View the active template
cat /mnt/llm_models/MiniMaxAI_MiniMax-M2-EXL3/chat_template.jinja

# Check tokenizer config
jq '.chat_template' /mnt/llm_models/MiniMaxAI_MiniMax-M2-EXL3/tokenizer_config.json
```

## Proxy Implementation

Our proxy works with this template by:

1. **Receiving** OpenAI/Anthropic format from clients
2. **Transforming** tool results (`role: "tool"` → `role: "user"`)
3. **Sending** to TabbyAPI in standard OpenAI format
4. **TabbyAPI applies** chat template with message markers
5. **Model generates** XML tool calls with `<think>` blocks
6. **Proxy parses** XML back to JSON and preserves reasoning
7. **Returning** standard OpenAI/Anthropic format to clients

## Summary

✅ All message markers present and correct
✅ Tool call format matches MiniMax XML specification
✅ Think blocks automatically added to generation prompt
✅ Tool results supported (we transform to `role: "user"`)
✅ Template is loaded and active in TabbyAPI
✅ No action required - full compliance verified
