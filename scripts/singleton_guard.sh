#!/usr/bin/env bash
# scripts/singleton_guard.sh
#
# Ensures only one instance of EVE Client (exefile.exe) is running.
# If > 1 instance is detected, kills all and restarts via zombie_init.sh.
# Used to prevent HID/Vision confusion from multiple windows.

set -o pipefail

# Log to explicit file or stdout
LOGFILE="/tmp/eve_singleton_guard.log"

log() {
    echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] $1" | tee -a "$LOGFILE"
}

# Only count exefile.exe (EVE Client)
# pgrep -c returns the count
COUNT=$(pgrep -f "exefile.exe" | wc -l)

if [ "$COUNT" -gt 1 ]; then
    log "CRITICAL: Multiple EVE clients detected (Count: $COUNT). Initiating purge."
    
    # Kill all wine/exefile processes
    pkill -f "exefile.exe"
    pkill -f "wineserver"
    
    log "Processes killed. Waiting 10s for teardown..."
    sleep 10
    
    log "Restarting Zombie Client..."
    # Execute init script in background, detaching output to avoid hanging cron
    /home/seb/eve-trader/scripts/zombie_init.sh >> /tmp/eve_zombie_init.log 2>&1 &
    
    log "Restart command issued."
else
    # Optional debugging
    # log "Singleton check passed. Count: $COUNT"
    :
fi
