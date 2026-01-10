#!/bin/bash
set -e

echo "🛡️ Installing EVE Disk Guard System..."

# 1. Install Script
echo "Copying script..."
# Ensure the script is executable
chmod +x /home/seb/nebakineza/eve-trader/scripts/disk_guard.py
# Link it securely
sudo ln -sf /home/seb/nebakineza/eve-trader/scripts/disk_guard.py /usr/local/bin/eve-disk-guard

# 2. Create Service
echo "Creating Systemd Service..."
cat <<SERVICE | sudo tee /etc/systemd/system/eve-disk-guard.service
[Unit]
Description=EVE Trader Disk Space Guardian
After=docker.service

[Service]
Type=oneshot
ExecStart=/usr/local/bin/eve-disk-guard
User=root
# Group=docker
SERVICE

# 3. Create Timer
echo "Creating Systemd Timer..."
cat <<TIMER | sudo tee /etc/systemd/system/eve-disk-guard.timer
[Unit]
Description=Run EVE Disk Guard every 10 minutes

[Timer]
OnBootSec=5min
OnUnitActiveSec=10min
Unit=eve-disk-guard.service

[Install]
WantedBy=timers.target
TIMER

# 4. Enable and Start
echo "Enabling Timer..."
sudo systemctl daemon-reload
sudo systemctl enable eve-disk-guard.timer
sudo systemctl start eve-disk-guard.timer
sudo systemctl start eve-disk-guard.service  # Run once immediately

echo "✅ Disk Guard Active!"
