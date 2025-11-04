# MiniMax-M2 Proxy Self-Monitoring Agent

## Overview

This monitoring system uses an autonomous Claude agent to continuously investigate and document long-context degradation issues in the MiniMax-M2 proxy. The agent runs every 2 hours, examines the codebase, reviews documentation, and builds a cumulative understanding of parsing errors and chat template deviations that occur over extended conversations.

## Problem Being Investigated

Over the course of long multi-turn conversations (10+ turns), the proxy experiences:

- **XML Parsing Errors**: Tool call extraction becomes unreliable
- **Think Block Corruption**: `<think>` tags become unbalanced or malformed
- **Chat Template Drift**: Message formatting degrades over time
- **State Accumulation**: Parser state may leak across requests
- **Chinese Character Resurgence**: Despite banned_strings protection

The monitoring agent investigates root causes and proposes solutions.

## Files

### Core Components

- **`monitor_investigation_prompt.md`**: Comprehensive investigation briefing
  - Problem statement with observed degradation patterns
  - System architecture and component details
  - MiniMax-M2 specifications and chat template format
  - Reference documentation links
  - Investigation objectives and success criteria
  - Previous findings automatically appended

- **`run_monitor_agent.sh`**: Main execution script
  - Reads previous findings
  - Builds complete investigation prompt
  - Calls `claude --dangerously-skip-permissions`
  - Saves timestamped outputs
  - Appends findings to cumulative log

- **`setup_monitoring_schedule.sh`**: Scheduling configuration
  - Interactive setup for cron or systemd timer
  - Configures automatic runs every 2 hours
  - Provides management commands

### Generated Files

- **`monitor_findings.log`**: Cumulative findings across all runs
  - Each investigation appends its discoveries
  - Provides continuity between runs
  - Grows over time as understanding deepens

- **`monitor_outputs/investigation_YYYY-MM-DD_HH-MM-SS.md`**: Individual run outputs
  - Timestamped investigation results
  - Detailed analysis and proposals
  - Kept for last 10 runs only (auto-pruned)

- **`monitor_cron.log`**: Cron execution log (if using cron)

## Quick Start

### 1. Run a Manual Investigation (Test)

```bash
cd /home/ser/minimax-m2-proxy
./run_monitor_agent.sh
```

This will:
- Generate an investigation using Claude
- Save results to `monitor_outputs/investigation_[timestamp].md`
- Append findings to `monitor_findings.log`
- Display a summary in the terminal

### 2. Set Up Automatic Scheduling

```bash
./setup_monitoring_schedule.sh
```

Follow the interactive prompts to choose:
1. **Cron** - Traditional, simple
2. **Systemd timer** - Modern, better logging
3. **Both** - Recommended for redundancy
4. **Manual only** - No automatic scheduling

### 3. View Results

```bash
# View cumulative findings
cat monitor_findings.log

# View latest investigation
ls -t monitor_outputs/ | head -1 | xargs -I{} cat monitor_outputs/{}

# View all outputs
ls -lth monitor_outputs/
```

## Scheduling Options

### Option 1: Cron (Simple)

**Advantages:**
- Simple and traditional
- No sudo required
- Works on any Linux system

**Setup:**
```bash
./setup_monitoring_schedule.sh
# Choose option 1
```

**Management:**
```bash
# View schedule
crontab -l

# View logs
tail -f monitor_cron.log

# Disable
crontab -e  # Delete the run_monitor_agent.sh line
```

### Option 2: Systemd Timer (Modern)

**Advantages:**
- Better logging via journald
- More reliable execution
- Can set dependencies and conditions
- Persistent across reboots

**Setup:**
```bash
./setup_monitoring_schedule.sh
# Choose option 2, then 'y' to install
```

**Management:**
```bash
# Check timer status
sudo systemctl status minimax-monitor.timer

# View upcoming runs
sudo systemctl list-timers minimax-monitor.timer

# View logs
sudo journalctl -u minimax-monitor.service -f

# Disable
sudo systemctl stop minimax-monitor.timer
sudo systemctl disable minimax-monitor.timer

# Re-enable
sudo systemctl start minimax-monitor.timer
sudo systemctl enable minimax-monitor.timer
```

## How It Works

### Investigation Cycle

```
Every 2 hours:
  1. Read previous findings from monitor_findings.log
  2. Append findings to investigation prompt
  3. Call claude --dangerously-skip-permissions with full context
  4. Claude investigates:
     - Examines codebase (parsers, formatters, main.py)
     - Reviews MiniMax documentation
     - Analyzes chat template handling
     - Studies streaming parser state machine
     - Proposes root causes and solutions
  5. Save timestamped output
  6. Append findings to cumulative log
  7. Prune old outputs (keep last 10)
```

### Cumulative Learning

Each investigation builds on previous runs:
- **Run 1**: Initial exploration, identifies obvious issues
- **Run 2**: Reads Run 1's findings, dives deeper into specific areas
- **Run 3**: Tests hypotheses from Runs 1-2, proposes concrete solutions
- **Run N**: Comprehensive understanding with implementation roadmap

### Agent Autonomy

The agent runs with `--dangerously-skip-permissions` to:
- Read all project files autonomously
- Search documentation
- Analyze code patterns
- Propose solutions without human intervention

**Note:** The agent does NOT modify code, only investigates and documents.

## Investigation Focus Areas

### 1. Parser Robustness
- **File**: `parsers/tools.py`
- **Question**: Why do regex patterns fail after N turns?
- **Examine**: Error handling, state management, XML malformation tolerance

### 2. Streaming State Machine
- **File**: `parsers/streaming.py`
- **Question**: Does parser state leak across requests?
- **Examine**: Buffer management, think block detection, state reset logic

### 3. Chat Template Application
- **Files**: TabbyAPI config, MiniMax chat template
- **Question**: Does template re-application cause drift?
- **Examine**: Marker duplication, role formatting, context window handling

### 4. Context Window Management
- **Scope**: 180K token context, ExL3 quantization
- **Question**: Does long context cause quality degradation?
- **Examine**: Attention patterns, quantization artifacts, overflow behavior

### 5. Tool Call Generation Quality
- **File**: MiniMax-M2 model behavior
- **Question**: Does model produce malformed XML over time?
- **Examine**: Temperature effects, sampling parameters, token generation patterns

## Expected Outputs

Each investigation produces:

### Analysis Sections
```markdown
## Investigation Run: [TIMESTAMP]

### Summary
One-paragraph overview of discoveries

### Root Cause Analysis
Detailed examination of underlying issues

### Code Examination
Specific files and functions requiring attention

### Proposed Solutions
Concrete recommendations with implementation details

### Test Scenarios
Reproduction steps and validation methods

### Open Questions
Areas requiring further investigation

### Recommendations for Next Run
Focus areas for the next cycle
```

### Example Findings

**Potential Discovery #1:**
> "The streaming parser in `parsers/streaming.py:45-67` maintains a buffer across multiple chunks but never resets the `_pending_content` field when a new request starts. In long conversations with many streaming responses, this buffer accumulates stale data, causing think block tags to become unbalanced."

**Potential Discovery #2:**
> "TabbyAPI applies the chat template on every turn, which includes role markers like `]~b]user`. After turn 15, these markers begin appearing duplicated in the context, causing the model to generate malformed XML because it sees inconsistent examples in its context window."

## Maintenance

### Log Rotation

The system auto-manages disk space:
- **Output files**: Keeps last 10 investigations only
- **Findings log**: Grows indefinitely (monitor manually)

To manually prune findings log:
```bash
# Keep last 1000 lines
tail -1000 monitor_findings.log > monitor_findings.log.tmp
mv monitor_findings.log.tmp monitor_findings.log
```

### Monitoring the Monitor

Check that investigations are running:

**Cron:**
```bash
# Should see entries every 2 hours
grep "Investigation Run Started" monitor_findings.log | tail -5
```

**Systemd:**
```bash
# Should show regular execution
sudo journalctl -u minimax-monitor.service --since "24 hours ago" | grep "Investigation"
```

### Troubleshooting

**Agent not running:**
```bash
# Cron: Check cron logs
tail -100 monitor_cron.log

# Systemd: Check service status
sudo systemctl status minimax-monitor.timer
sudo journalctl -u minimax-monitor.service -n 50
```

**No output files:**
```bash
# Check script permissions
ls -la run_monitor_agent.sh
# Should be -rwxr-xr-x

# Run manually to see errors
./run_monitor_agent.sh
```

**Claude command not found:**
```bash
# Verify claude is in PATH
which claude

# If not, update run_monitor_agent.sh to use full path
# Replace: claude --dangerously-skip-permissions
# With: /full/path/to/claude --dangerously-skip-permissions
```

## Integration with Development

### Acting on Findings

1. **Review Latest Investigation:**
   ```bash
   cat monitor_outputs/investigation_$(ls -t monitor_outputs/ | head -1)
   ```

2. **Identify High-Priority Issues:**
   - Look for "Root Cause Analysis" sections
   - Check "Proposed Solutions" for actionable items

3. **Implement Fixes:**
   - Create new branch for each identified issue
   - Reference investigation timestamp in commit messages
   - Test using scenarios from "Test Scenarios" section

4. **Validate:**
   - Run proposed tests
   - Check if issue persists in next investigation run

### Example Workflow

```bash
# 1. Check latest findings
./run_monitor_agent.sh
cat monitor_outputs/investigation_*.md | tail -100

# 2. Agent identifies: "Parser doesn't handle nested <invoke> tags"

# 3. Create fix branch
git checkout -b fix/nested-invoke-parsing

# 4. Implement solution from agent's proposal

# 5. Test with agent-suggested scenario

# 6. Commit with reference
git commit -m "fix: Handle nested invoke tags in tool parser

Based on investigation from monitoring agent run 2025-01-03_14-00-00
Implements robust XML extraction as proposed in findings.

Ref: monitor_outputs/investigation_2025-01-03_14-00-00.md"

# 7. Wait for next agent run to see if issue is resolved
```

## Security Considerations

### `--dangerously-skip-permissions` Flag

This flag allows Claude to:
- ✅ Read any file in the project
- ✅ Search documentation
- ✅ Analyze code patterns
- ❌ **CANNOT** write files
- ❌ **CANNOT** execute commands
- ❌ **CANNOT** modify system

The agent is read-only for safety.

### Sensitive Data

The monitoring agent has access to:
- Source code
- Configuration files
- Previous investigation logs

**Do NOT** store sensitive data (API keys, credentials) in this repository.

## Advanced Usage

### Custom Investigation Focus

Edit `monitor_investigation_prompt.md` to focus on specific areas:

```markdown
### Priority Investigation for Next 3 Runs

Focus exclusively on streaming parser state management.
Examine every state variable in parsers/streaming.py.
Test hypothesis: Buffer state leaks between requests.
```

### One-Time Deep Dive

Run an extended investigation:

```bash
# Modify prompt for deeper analysis
echo "
SPECIAL INSTRUCTION FOR THIS RUN ONLY:
Conduct an exhaustive analysis of parsers/tools.py.
Examine every regex pattern, error path, and edge case.
Spend maximum tokens on this investigation.
" >> monitor_investigation_prompt.md

# Run investigation
./run_monitor_agent.sh

# Restore prompt (remove special instruction)
git checkout monitor_investigation_prompt.md
```

### Parallel Investigations

Run multiple agents investigating different aspects:

```bash
# Create specialized prompts
cp monitor_investigation_prompt.md parser_investigation.md
cp monitor_investigation_prompt.md template_investigation.md

# Modify each for specific focus

# Run both
./run_monitor_agent.sh  # Uses default prompt
./run_specialized_investigation.sh parser_investigation.md
./run_specialized_investigation.sh template_investigation.md
```

## FAQ

**Q: How much does each run cost?**
A: Depends on Claude model and conversation length. Typically $0.10-0.50 per investigation with Sonnet. Monitor your API usage.

**Q: Will this slow down my system?**
A: No, investigations run in the background and don't affect the proxy. Each investigation takes 2-5 minutes.

**Q: Can I run investigations more frequently?**
A: Yes, edit the cron/timer schedule. However, findings accumulate better with some time between runs to implement and test solutions.

**Q: What if the agent proposes a wrong solution?**
A: The agent is investigative, not authoritative. Always validate proposals through testing before implementing.

**Q: Can I review investigations before they're logged?**
A: Yes, run manually with `./run_monitor_agent.sh` and review the output in `monitor_outputs/` before the next scheduled run appends it.

## Future Enhancements

Potential improvements to the monitoring system:

- **Metrics Dashboard**: Parse findings log to visualize trends
- **Alert System**: Notify when agent identifies critical issues
- **Auto-Testing**: Run regression tests based on agent proposals
- **Multi-Agent**: Specialized agents for parsing, templates, streaming
- **Interactive Mode**: Q&A with findings to drill deeper

## License

Same as parent project (check repository root).

## Support

Issues with the monitoring system:
1. Check logs: `monitor_cron.log` or `journalctl -u minimax-monitor.service`
2. Run manually: `./run_monitor_agent.sh`
3. Review findings: `cat monitor_findings.log | tail -200`

For questions about findings or proposed solutions, consult the investigation outputs directly - they contain detailed explanations.

---

**Last Updated:** 2025-01-03
**Version:** 1.0.0
**Maintainer:** Autonomous monitoring agent + human oversight
