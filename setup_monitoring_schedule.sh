#!/bin/bash
set -euo pipefail

# Setup script for MiniMax-M2 Proxy monitoring agent
# Configures automatic investigation runs every 2 hours

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MONITOR_SCRIPT="$SCRIPT_DIR/run_monitor_agent.sh"

echo "MiniMax-M2 Proxy Monitoring Agent - Schedule Setup"
echo "=================================================="
echo ""

# Verify monitor script exists
if [[ ! -f "$MONITOR_SCRIPT" ]]; then
    echo "❌ Error: Monitor script not found at $MONITOR_SCRIPT"
    exit 1
fi

# Make sure it's executable
chmod +x "$MONITOR_SCRIPT"

echo "Select scheduling method:"
echo "  1) Cron (traditional, simple)"
echo "  2) Systemd timer (modern, better logging)"
echo "  3) Both (recommended)"
echo "  4) Manual only (no automatic scheduling)"
echo ""
read -p "Choice [1-4]: " choice

case "$choice" in
    1|3)
        echo ""
        echo "Setting up cron job..."

        # Create cron entry
        CRON_ENTRY="0 */2 * * * $MONITOR_SCRIPT >> $SCRIPT_DIR/monitor_cron.log 2>&1"

        # Check if entry already exists
        if crontab -l 2>/dev/null | grep -F "$MONITOR_SCRIPT" > /dev/null; then
            echo "⚠️  Cron entry already exists. Removing old entry..."
            crontab -l 2>/dev/null | grep -v "$MONITOR_SCRIPT" | crontab -
        fi

        # Add new entry
        (crontab -l 2>/dev/null; echo "$CRON_ENTRY") | crontab -

        echo "✅ Cron job configured: Runs every 2 hours"
        echo "   Log file: $SCRIPT_DIR/monitor_cron.log"
        echo ""
        echo "To view cron schedule:"
        echo "  crontab -l"
        echo ""
        echo "To disable cron:"
        echo "  crontab -e  # then delete the line with run_monitor_agent.sh"
        echo ""

        if [[ "$choice" != "3" ]]; then
            break
        fi
        ;&  # fallthrough to systemd
    2|3)
        echo ""
        echo "Setting up systemd timer..."

        # Create systemd service file
        SERVICE_FILE="/tmp/minimax-monitor.service"
        cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=MiniMax-M2 Proxy Self-Monitoring Agent
After=network.target

[Service]
Type=oneshot
User=$USER
WorkingDirectory=$SCRIPT_DIR
ExecStart=$MONITOR_SCRIPT
StandardOutput=journal
StandardError=journal
SyslogIdentifier=minimax-monitor

[Install]
WantedBy=multi-user.target
EOF

        # Create systemd timer file
        TIMER_FILE="/tmp/minimax-monitor.timer"
        cat > "$TIMER_FILE" <<EOF
[Unit]
Description=Run MiniMax-M2 monitoring agent every 2 hours
Requires=minimax-monitor.service

[Timer]
OnBootSec=10min
OnUnitActiveSec=2h
Unit=minimax-monitor.service

[Install]
WantedBy=timers.target
EOF

        echo "Created systemd unit files:"
        echo "  $SERVICE_FILE"
        echo "  $TIMER_FILE"
        echo ""
        echo "To install (requires sudo):"
        echo "  sudo cp $SERVICE_FILE /etc/systemd/system/"
        echo "  sudo cp $TIMER_FILE /etc/systemd/system/"
        echo "  sudo systemctl daemon-reload"
        echo "  sudo systemctl enable minimax-monitor.timer"
        echo "  sudo systemctl start minimax-monitor.timer"
        echo ""
        echo "To check status:"
        echo "  sudo systemctl status minimax-monitor.timer"
        echo "  sudo systemctl list-timers minimax-monitor.timer"
        echo ""
        echo "To view logs:"
        echo "  sudo journalctl -u minimax-monitor.service -f"
        echo ""

        read -p "Install systemd timer now? (requires sudo) [y/N]: " install_now
        if [[ "$install_now" == "y" || "$install_now" == "Y" ]]; then
            sudo cp "$SERVICE_FILE" /etc/systemd/system/
            sudo cp "$TIMER_FILE" /etc/systemd/system/
            sudo systemctl daemon-reload
            sudo systemctl enable minimax-monitor.timer
            sudo systemctl start minimax-monitor.timer
            echo "✅ Systemd timer installed and started"
            sudo systemctl status minimax-monitor.timer
        else
            echo "ℹ️  Systemd files created but not installed. Install manually using commands above."
        fi
        echo ""
        ;;
    4)
        echo ""
        echo "Manual mode selected - no automatic scheduling configured"
        echo ""
        echo "To run investigations manually:"
        echo "  $MONITOR_SCRIPT"
        echo ""
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo "Setup complete!"
echo ""
echo "📋 Quick Reference:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Run manually:      $MONITOR_SCRIPT"
echo "View findings:     cat $SCRIPT_DIR/monitor_findings.log"
echo "View outputs:      ls -lth $SCRIPT_DIR/monitor_outputs/"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "The monitoring agent will investigate:"
echo "  • XML parsing degradation over long conversations"
echo "  • Think block corruption patterns"
echo "  • Chat template drift"
echo "  • Context window management issues"
echo ""
echo "Each run builds on previous findings to deepen understanding."
echo ""

exit 0
