# MiniMax-M2 Chinese Characters & Anthropic API Fix - Investigation Plan

## Problem Statement

**Issue 1**: MiniMax-M2 outputs random Chinese characters mid-text while tool calling and think blocks work correctly
- Example: "common养的", "TUI博物", "internautes养", "penales养"
- Characters appear randomly in otherwise correct English text

**Issue 2**: Anthropic `/v1/messages` endpoint rejects multi-turn tool result messages (422 errors)
- Current implementation converts `tool_result` to `role: "tool"`
- TabbyAPI/MiniMax chat template doesn't support `role: "tool"`
- Need to convert to proper format for MiniMax

## Research Findings

### From Official Documentation

1. **Tokenizer & Sampling** (Hugging Face):
   - Recommended: `temperature=1.0, top_p=0.95, top_k=40`
   - MoE model: 230B total params, 10B active
   - Multilingual training including Chinese (BrowseComp-zh: 48.5%)
   - **CRITICAL**: "you must ensure that the historical content is passed back in its original format"

2. **Tool Result Format** (Tool Calling Guide):
   - Raw format uses `]~b]tool` marker for tool results
   - OpenAI SDK: `{"role": "tool", "tool_call_id": "...", "content": "..."}`
   - Anthropic SDK: `{"role": "user", "content": [{"type": "tool_result", ...}]}`

3. **Chat Template Markers**:
   - `]~b]user` - User messages
   - `]~b]ai` - Assistant messages
   - `]~b]tool` - Tool result messages
   - `]~b]system` - System messages
   - `[e~[` - Message end marker

4. **Thinking Format**:
   - OpenAI: `<think>...</think>` in content when `reasoning_split=False`
   - Anthropic: `type: "thinking"` content blocks
   - Must preserve original format in history to avoid degradation

## Root Cause Hypotheses

### Chinese Characters Issue

**Hypothesis 1: Tokenizer Vocabulary Bleed**
- MiniMax-M2 has multilingual vocab including Chinese
- Certain byte sequences or token IDs map to Chinese characters
- When generating with high temperature, model samples unlikely tokens
- These tokens decode as Chinese instead of intended English tokens

**Hypothesis 2: Chat Template Corruption**
- `add_generation_prompt=True` adds `]~b]ai\n<think>\n` prefix
- This might be shifting token offsets or corrupting generation
- Historical content not being "passed back in its original format"

**Hypothesis 3: Sampling Parameter Mismatch**
- Current settings might not match recommended `temp=1.0, top_p=0.95, top_k=40`
- Incorrect sampling could trigger token generation from Chinese vocab region

**Hypothesis 4: Context Corruption**
- If we're modifying historical messages (wrapping think tags, parsing tools), we might be violating the "original format" requirement
- This could cause model confusion leading to wrong token selection

### Anthropic API 422 Error

**Root Cause: Message Format Mismatch**
- We convert Anthropic `tool_result` → OpenAI `role: "tool"`
- TabbyAPI's chat template sees `role: "tool"` and applies `]~b]tool` marker
- But we're on the Anthropic endpoint which validates against Anthropic schema
- Anthropic schema only allows `role: "user"` or `role: "assistant"`
- Result: 422 validation error

**The Fix**:
- Anthropic endpoint should NOT convert to OpenAI format for TabbyAPI
- Should convert Anthropic to **TabbyAPI's expected format** which uses OpenAI messages
- Then pass to TabbyAPI without going through Anthropic validation

## 10-Step Investigation & Fix Plan

### Step 1: Deep Research ✓
- [x] Fetch all MiniMax-M2 documentation
- [x] Identify key insights about tokenizer, chat template, tool calling
- [x] Document root cause hypotheses

### Step 2: Diagnostic Testing - Chinese Characters
- [ ] Test current TabbyAPI parameters (temp, top_p, top_k)
- [ ] Test with `add_generation_prompt=False` to see if issue persists
- [ ] Test direct TabbyAPI calls vs proxy calls
- [ ] Capture raw TabbyAPI responses to examine token-level output
- [ ] Check if Chinese chars appear in streaming vs non-streaming

### Step 3: Create OpenAI API Regression Tests
- [ ] Test case 1: Single-turn tool call with think blocks
- [ ] Test case 2: Multi-turn with tool result → response
- [ ] Test case 3: Multiple sequential tool calls
- [ ] Test case 4: Streaming with tool calls
- [ ] Test case 5: Non-streaming with tool calls
- [ ] All tests must verify:
  - No Chinese characters in output
  - Think blocks properly preserved
  - Tool calls correctly parsed
  - Multi-turn conversation maintains context

### Step 4: Create Anthropic API Regression Tests
- [ ] Test case 1: Single-turn tool call with thinking blocks
- [ ] Test case 2: Multi-turn with tool_result → response
- [ ] Test case 3: Multiple sequential tool calls
- [ ] Test case 4: Streaming with tool calls
- [ ] Test case 5: Non-streaming with tool calls
- [ ] All tests must verify same criteria as OpenAI tests

### Step 5: Investigate Tokenizer & Sampling Settings
- [ ] Check TabbyAPI config.yml for tokenizer settings
- [ ] Verify sampling parameters match MiniMax recommendations
- [ ] Test different temperature/top_p/top_k combinations
- [ ] Check if TabbyAPI has any vocab filtering options
- [ ] Examine if exl3 quantization affects vocab

### Step 6: Fix Chinese Character Generation
Based on diagnostic findings, implement fixes:
- [ ] Option A: Adjust sampling parameters (temp/top_p/top_k)
- [ ] Option B: Remove `add_generation_prompt` or modify chat template usage
- [ ] Option C: Add vocab filtering or token blocklist
- [ ] Option D: Fix message history formatting to match "original format"
- [ ] Option E: Request TabbyAPI to reload model with different settings

### Step 7: Fix Anthropic Endpoint Tool Result Handling
- [ ] Update `anthropic_messages_to_openai()` in proxy/models.py
- [ ] Keep `tool_result` as `role: "user"` with formatted content
- [ ] Do NOT convert to `role: "tool"`
- [ ] Test that TabbyAPI accepts the new format
- [ ] Verify multi-turn conversations work end-to-end

### Step 8: Run Comprehensive Test Suite
- [ ] Run all OpenAI API regression tests
- [ ] Run all Anthropic API regression tests
- [ ] Verify no Chinese characters in any outputs
- [ ] Verify think blocks preserved correctly
- [ ] Verify multi-turn tool calling works
- [ ] Test both streaming and non-streaming
- [ ] Test edge cases (empty responses, errors, etc.)

### Step 9: End-to-End Test with claude-local
- [ ] Configure claude-local to use proxy
- [ ] Test simple prompt without tools
- [ ] Test prompt with tool calling
- [ ] Test multi-turn conversation with tools
- [ ] Verify no Chinese characters appear
- [ ] Verify think blocks render correctly
- [ ] Verify tool calling works properly

### Step 10: Get Codex Review & Documentation
- [ ] Call codex to review all changes
- [ ] Request codex to generate documentation
- [ ] Have codex verify test coverage
- [ ] Have codex suggest any additional edge cases
- [ ] Document findings in CHINESE_CHAR_FIX.md
- [ ] Document Anthropic API fix in ANTHROPIC_FIX.md
- [ ] Update README with any configuration changes

## Success Criteria

1. **Chinese Characters**: Zero random Chinese characters in any model outputs
2. **Think Blocks**: Preserved correctly in both APIs (1 opening, 1 closing)
3. **Tool Calling**: Works correctly in both APIs with proper XML parsing
4. **Multi-Turn**: 3+ turn conversations with tool usage work flawlessly
5. **Streaming**: Both APIs stream correctly with think blocks and tool calls
6. **claude-local**: Integration works end-to-end without errors
7. **Test Coverage**: 100% pass rate on regression tests for both APIs

## Notes

- Do NOT modify existing working functionality (think blocks, tool parsing)
- Focus ONLY on Chinese character fix and Anthropic API tool results
- All changes must maintain backward compatibility
- Tests must be comprehensive and repeatable
