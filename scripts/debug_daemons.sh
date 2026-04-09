#!/bin/bash
# L104 Daemon Debug Script
# Run with elevated privileges

set -e

LOG_DIR="/var/log/l104/debug-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$LOG_DIR"

echo "=== L104 Daemon Diagnostic ===" | tee "$LOG_DIR/full_report.log"

# 1. System Overview
echo "1. SYSTEM OVERVIEW" | tee -a "$LOG_DIR/full_report.log"
qsys-stat --detailed >> "$LOG_DIR/full_report.log"
echo "---" >> "$LOG_DIR/full_report.log"

# 2. Daemon Health
echo "2. DAEMON HEALTH CHECK" | tee -a "$LOG_DIR/full_report.log"
for daemon in $(systemctl list-units --type=service --no-legend | grep l104 | awk '{print $1}'); do
    echo "Checking $daemon..." | tee -a "$LOG_DIR/full_report.log"
    systemctl status "$daemon" --no-pager >> "$LOG_DIR/daemon_$daemon.log"
    journalctl -u "$daemon" -n 50 --no-pager >> "$LOG_DIR/journal_$daemon.log"
    
    # Quantum coherence check
    qcoherence-check --service="$daemon" >> "$LOG_DIR/quantum_$daemon.log" 2>&1 || true
done

# 3. Quantum Resources
echo "3. QUANTUM RESOURCE AUDIT" | tee -a "$LOG_DIR/full_report.log"
qresource-usage --by-daemon >> "$LOG_DIR/quantum_resources.log"
cat "$LOG_DIR/quantum_resources.log" >> "$LOG_DIR/full_report.log"

# 4. Temporal Alignment
echo "4. TEMPORAL ALIGNMENT CHECK" | tee -a "$LOG_DIR/full_report.log"
chrono-verify --full >> "$LOG_DIR/temporal.log"
cat "$LOG_DIR/temporal.log" >> "$LOG_DIR/full_report.log"

# 5. Entanglement Network
echo "5. ENTANGLEMENT NETWORK STATUS" | tee -a "$LOG_DIR/full_report.log"
entanglement-map --live >> "$LOG_DIR/entanglement.log"
cat "$LOG_DIR/entanglement.log" >> "$LOG_DIR/full_report.log"

# 6. Generate Summary
echo "=== DIAGNOSTIC SUMMARY ===" > "$LOG_DIR/summary.txt"
grep -i "error\|failed\|corrupt\|decoherence\|drift" "$LOG_DIR"/*.log | head -20 >> "$LOG_DIR/summary.txt"

echo "Debug completed. Reports in: $LOG_DIR"
echo "Summary:"
cat "$LOG_DIR/summary.txt"