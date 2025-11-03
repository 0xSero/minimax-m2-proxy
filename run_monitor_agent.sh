#!/bin/bash
set -euo pipefail

# MiniMax-M2 Proxy Self-Monitoring Agent
# Runs periodically to investigate long-context degradation issues
# Usage: ./run_monitor_agent.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Configuration
PROMPT_FILE="monitor_investigation_prompt.md"
FINDINGS_LOG="monitor_findings.log"
OUTPUT_DIR="monitor_outputs"
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
OUTPUT_FILE="$OUTPUT_DIR/investigation_$TIMESTAMP.md"

# Ensure directories exist
mkdir -p "$OUTPUT_DIR"

# Initialize findings log if it doesn't exist
if [[ ! -f "$FINDINGS_LOG" ]]; then
    echo "# MiniMax-M2 Proxy Monitoring Agent - Findings Log" > "$FINDINGS_LOG"
    echo "# This file accumulates findings across investigation runs" >> "$FINDINGS_LOG"
    echo "" >> "$FINDINGS_LOG"
    echo "Log initialized: $TIMESTAMP" >> "$FINDINGS_LOG"
    echo "---" >> "$FINDINGS_LOG"
    echo "" >> "$FINDINGS_LOG"
fi

# Build the full prompt
echo "Building investigation prompt with previous findings..."

# Create a temporary prompt file with previous findings appended
TEMP_PROMPT=$(mktemp)
cat "$PROMPT_FILE" > "$TEMP_PROMPT"

# Append previous findings to the prompt
echo "" >> "$TEMP_PROMPT"
echo "---" >> "$TEMP_PROMPT"
echo "" >> "$TEMP_PROMPT"
cat "$FINDINGS_LOG" >> "$TEMP_PROMPT"

# Log the start
echo "" >> "$FINDINGS_LOG"
echo "==============================================================================" >> "$FINDINGS_LOG"
echo "Investigation Run Started: $TIMESTAMP" >> "$FINDINGS_LOG"
echo "==============================================================================" >> "$FINDINGS_LOG"
echo "" >> "$FINDINGS_LOG"

# Run the monitoring agent
echo "Launching claude monitoring agent..."
echo "Prompt file: $TEMP_PROMPT"
echo "Output will be saved to: $OUTPUT_FILE"
echo ""

# Call claude with the investigation prompt
# Using --dangerously-skip-permissions to run autonomously
if claude --dangerously-skip-permissions "$(cat "$TEMP_PROMPT")" > "$OUTPUT_FILE" 2>&1; then
    echo "✅ Investigation completed successfully"

    # Append the findings to the log
    echo "" >> "$FINDINGS_LOG"
    cat "$OUTPUT_FILE" >> "$FINDINGS_LOG"
    echo "" >> "$FINDINGS_LOG"
    echo "---" >> "$FINDINGS_LOG"
    echo "" >> "$FINDINGS_LOG"

    # Create a summary
    echo ""
    echo "📊 Investigation Summary:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    head -50 "$OUTPUT_FILE"
    echo "..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Full output saved to: $OUTPUT_FILE"
    echo "Findings appended to: $FINDINGS_LOG"

    # Keep only last 10 investigation outputs to save space
    cd "$OUTPUT_DIR"
    ls -t investigation_*.md | tail -n +11 | xargs rm -f 2>/dev/null || true
    cd "$SCRIPT_DIR"

else
    echo "❌ Investigation failed"
    echo "Error logged to: $OUTPUT_FILE"

    # Log the failure
    echo "INVESTIGATION FAILED - See $OUTPUT_FILE for details" >> "$FINDINGS_LOG"
    echo "" >> "$FINDINGS_LOG"
fi

# Cleanup
rm -f "$TEMP_PROMPT"

# Show next run info
echo ""
echo "Next scheduled run: In 2 hours (if cron is configured)"
echo ""
echo "To view all findings:"
echo "  cat $FINDINGS_LOG"
echo ""
echo "To view this run's output:"
echo "  cat $OUTPUT_FILE"
echo ""
echo "To manually trigger another investigation:"
echo "  $SCRIPT_DIR/run_monitor_agent.sh"
echo ""

exit 0
