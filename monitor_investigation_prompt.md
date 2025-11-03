# MiniMax-M2 Proxy Long-Context Degradation Investigation

## Mission
You are a self-monitoring agent tasked with investigating and documenting parsing errors and chat template deviations that accumulate over long conversations in the MiniMax-M2 proxy system. Your findings will be used to improve the system's reliability and prevent degradation.

## Problem Statement

### Observed Behavior
Over the course of extended multi-turn conversations (10+ turns), the MiniMax-M2 proxy exhibits increasing degradation:

1. **Parsing Errors**: XML tool call parsing becomes unreliable
   - `<minimax:tool_call>` blocks sometimes malformed
   - `<invoke>` tags may be incomplete or nested incorrectly
   - `<parameter>` extraction fails intermittently

2. **Think Block Corruption**: `<think>` tags become unbalanced
   - Multiple opening tags without corresponding closures
   - Content leaking outside think blocks
   - Streaming parser state becomes desynchronized

3. **Chat Template Drift**: Message formatting degrades
   - Role markers (`]~b]user`, `]~b]ai`) may be duplicated
   - Tool result formatting becomes inconsistent
   - Context window management issues

4. **Chinese Character Resurgence**: Despite banned_strings
   - May indicate tokenizer state issues
   - Could be related to context overflow
   - Potentially linked to quantization artifacts in long contexts

## System Architecture

### Current Stack
```
claude-local (Anthropic Messages API)
    ↓
claude-code-router (port 8081)
    ↓ (transforms Anthropic → OpenAI format)
MiniMax-M2 Proxy (port 8001)
    ↓ (OpenAI Chat Completions API)
TabbyAPI (port 8000)
    ↓ (ExllamaV3 backend)
MiniMax-M2-EXL3 (456B MoE, 10B active, 180K context)
```

### Key Components

**Proxy Layer** (`/home/ser/minimax-m2-proxy/`):
- `proxy/main.py`: FastAPI endpoints (OpenAI + Anthropic)
- `parsers/tools.py`: XML tool call parser (regex-based)
- `parsers/streaming.py`: Streaming state machine for think blocks
- `formatters/openai.py`: OpenAI response formatter
- `formatters/anthropic.py`: Anthropic response formatter

**Configuration**:
- Think block passthrough: ENABLED
- Chinese char blocking: ENABLED (24 banned strings)
- Streaming: Real-time with buffering for tool calls
- Chat template: MiniMax-M2 custom template with markers

### MiniMax-M2 Specifications

**Model Details**:
- Architecture: 456B parameters, MoE with 10B active
- Context: 180K tokens (extremely long context capable)
- Training: Multilingual (48.5% Chinese, 32.5% English, 19% code)
- Quantization: ExL3 (may affect long-context stability)

**Chat Template Markers**:
```
]~b]system - System messages
]~b]user - User messages
]~b]ai - Assistant messages
]~b]tool - Tool results (NOT USED - we use ]~b]user instead)
```

**Tool Calling Format**:
```xml
<think>
Chain of thought reasoning...
</think>

<minimax:tool_call>
<invoke name="function_name">
<parameter name="param1">value1</parameter>
<parameter name="param2">value2</parameter>
</invoke>
</minimax:tool_call>

Optional response text after tools.
```

## Reference Documentation

### Primary Sources
1. **MiniMax Platform Docs**: https://platform.minimax.io/docs/guides/text-m2-function-call
   - Official tool calling specification
   - Chat template requirements
   - Recommended sampling parameters

2. **MiniMax Model Card**: https://huggingface.co/MiniMaxAI/MiniMax-Text-01
   - Architecture details
   - Context window specifications
   - Training data composition

3. **TabbyAPI Documentation**: https://github.com/theroyallab/tabbyAPI
   - Sampling parameters (temperature, top_p, top_k)
   - banned_strings implementation
   - ExllamaV3 backend details

4. **Proxy Implementation**:
   - `/home/ser/minimax-m2-proxy/CHINESE_CHAR_FIX_SUMMARY.md`
   - `/home/ser/minimax-m2-proxy/INVESTIGATION_PLAN.md`
   - `/home/ser/minimax-m2-proxy/proxy/` (source code)

### Known Issues & Fixes
- Chinese character generation: FIXED via banned_strings
- Streaming think blocks: FIXED via state machine prepending
- Anthropic tool_result: Already correct (uses role="user")
- Multi-turn tool calling: TESTED and working (8/8 tests passing)

## Investigation Objectives

### 1. Root Cause Analysis
**Investigate**:
- Why does XML parsing degrade over long conversations?
- Is it a chat template issue, context overflow, or parser state?
- Does ExL3 quantization affect long-context coherence?
- Are there hidden state leaks in the streaming parser?

**Examine**:
- Regex patterns in `parsers/tools.py` - are they robust to malformed XML?
- Streaming parser in `parsers/streaming.py` - does state accumulate incorrectly?
- Chat template application - does TabbyAPI re-apply templates incorrectly after N turns?
- Context management - does the 180K window cause issues near limits?

### 2. Pattern Recognition
**Look for**:
- At what conversation turn do errors typically start? (turn 10? 15? 20?)
- Are errors more common with certain tool types?
- Does streaming vs non-streaming affect degradation rate?
- Are there specific token patterns that trigger issues?

**Analyze**:
- Correlation between conversation length and error frequency
- Impact of thinking depth (long `<think>` blocks) on parsing
- Effect of multiple sequential tool calls
- Role of temperature/sampling on XML generation quality

### 3. Mitigation Strategies
**Propose**:
- Parser hardening: More robust XML extraction with error recovery
- State reset mechanisms: Clear parser state after N turns
- Chat template validation: Verify template consistency across turns
- Context pruning: Intelligent truncation before degradation threshold
- Fallback parsing: Alternative extraction methods when regex fails

**Consider**:
- Should we implement a "conversation health check" endpoint?
- Can we detect degradation proactively and suggest session reset?
- Should parser maintain conversation-level state vs request-level?
- Would switching to a proper XML parser (lxml) help?

### 4. Testing & Validation
**Design**:
- Long-conversation stress tests (20+ turn scenarios)
- Parsing error detection and recovery tests
- Streaming state consistency tests across many chunks
- Chat template verification after each turn

**Metrics to Track**:
- Parse success rate by conversation turn number
- Think block balance accuracy over time
- Tool call extraction reliability vs conversation depth
- Chinese character leakage frequency (should be zero)

## Your Task

### Immediate Actions
1. **Read Previous Findings**: Check `/home/ser/minimax-m2-proxy/monitor_findings.log` for insights from previous runs
2. **Deep Dive Investigation**:
   - Examine the codebase thoroughly
   - Review MiniMax documentation
   - Analyze parser implementation
   - Study chat template handling

3. **Document Findings**: Create comprehensive analysis including:
   - Root causes identified
   - Specific code locations of concern
   - Reproduction steps for observed issues
   - Concrete mitigation proposals
   - Test scenarios to validate fixes

4. **Leave Context for Next Run**: Append your findings to the log with:
   - Timestamp
   - Key discoveries
   - Open questions
   - Recommended next steps
   - Specific areas to investigate further

### Output Format
Structure your response as:

```markdown
## Investigation Run: [TIMESTAMP]

### Summary
[One paragraph overview of what you investigated and found]

### Root Cause Analysis
[Detailed analysis of underlying issues]

### Code Examination
[Specific files, functions, and lines that need attention]

### Proposed Solutions
[Concrete recommendations with implementation details]

### Test Scenarios
[How to reproduce and validate the issues]

### Open Questions
[What still needs investigation]

### Recommendations for Next Run
[What the next investigation should focus on]

---
```

## Constraints & Guidelines
- **Be Thorough**: This is research, not implementation. Go deep.
- **Be Specific**: Reference exact file paths, line numbers, function names
- **Be Analytical**: Use the MiniMax docs to validate assumptions
- **Be Forward-Thinking**: Each run should build on previous findings
- **Be Practical**: Propose testable, implementable solutions

## Success Criteria
Your investigation is successful if it:
1. Identifies at least one concrete root cause
2. Proposes a testable solution
3. References official documentation to support analysis
4. Provides actionable next steps for implementation
5. Leaves clear context for the next investigation run

## Previous Findings
(This section is automatically populated from previous runs)
