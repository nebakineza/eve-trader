#!/usr/bin/env bash
set -euo pipefail

# Installs and enables the eve-trader-disk-guard timer on a Debian/Ubuntu host.
# This enforces a 48h retention and a hard guard to keep ./data from growing
# beyond 75% of total disk capacity.

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

need() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "[!] Missing required command: $1" >&2
    exit 1
  }
}

need systemctl
need sudo
need python3

SYSTEMD_DIR="/etc/systemd/system"

echo "[*] Repo: $REPO_DIR"
echo "[*] Installing systemd units to $SYSTEMD_DIR ..."

sudo install -m 0644 "$REPO_DIR/systemd/eve-trader-disk-guard.service" "$SYSTEMD_DIR/eve-trader-disk-guard.service"
sudo install -m 0644 "$REPO_DIR/systemd/eve-trader-disk-guard.timer" "$SYSTEMD_DIR/eve-trader-disk-guard.timer"

echo "[*] Reloading systemd and enabling timer..."
sudo systemctl daemon-reload
sudo systemctl enable --now eve-trader-disk-guard.timer

echo "[*] Status:"
sudo systemctl status eve-trader-disk-guard.timer --no-pager || true

echo "[+] Installed. Check logs with: journalctl -u eve-trader-disk-guard.service -n 200 --no-pager"
