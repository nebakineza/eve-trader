#!/usr/bin/env bash
set -euo pipefail

# Installs and enables the ZombieShot screenshot publisher on a Debian/Ubuntu host.
# This publishes a base64 JPEG to Redis key system:zombie:screenshot for the dashboard.
#
# Usage:
#   ./scripts/install_zombie_shot_systemd.sh
#   ZOMBIE_DISPLAY=:1 ./scripts/install_zombie_shot_systemd.sh
#

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DISPLAY_TO_USE="${ZOMBIE_DISPLAY:-:1}"

need() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "[!] Missing required command: $1" >&2
    exit 1
  }
}

need systemctl
need sudo

echo "[*] Repo: $REPO_DIR"
echo "[*] DISPLAY: $DISPLAY_TO_USE"

echo "[*] Installing host dependencies (scrot + ImageMagick + Xvfb + grim)..."
sudo apt-get update -y
sudo apt-get install -y scrot imagemagick xvfb x11-utils grim

# Install systemd units
SYSTEMD_DIR="/etc/systemd/system"

echo "[*] Installing systemd unit files to $SYSTEMD_DIR ..."
sudo install -m 0644 "$REPO_DIR/systemd/zombie-shot.service" "$SYSTEMD_DIR/zombie-shot.service"
sudo install -m 0644 "$REPO_DIR/systemd/zombie-shot.timer" "$SYSTEMD_DIR/zombie-shot.timer"
sudo install -m 0644 "$REPO_DIR/systemd/zombie-xvfb.service" "$SYSTEMD_DIR/zombie-xvfb.service"

# Override DISPLAY if requested.
if [[ "$DISPLAY_TO_USE" != ":1" ]]; then
  echo "[*] Writing override for zombie-shot.service DISPLAY=$DISPLAY_TO_USE"
  sudo mkdir -p /etc/systemd/system/zombie-shot.service.d
  sudo tee /etc/systemd/system/zombie-shot.service.d/override.conf >/dev/null <<EOF
[Service]
Environment=DISPLAY=$DISPLAY_TO_USE
EOF

  echo "[*] Writing override for zombie-xvfb.service DISPLAY=$DISPLAY_TO_USE"
  sudo mkdir -p /etc/systemd/system/zombie-xvfb.service.d
  sudo tee /etc/systemd/system/zombie-xvfb.service.d/override.conf >/dev/null <<EOF
[Service]
ExecStart=
ExecStart=/usr/bin/Xvfb $DISPLAY_TO_USE -screen 0 1280x720x24 -nolisten tcp -ac
EOF
fi

echo "[*] Reloading systemd and enabling services..."
sudo systemctl daemon-reload
sudo systemctl enable --now zombie-xvfb.service
sudo systemctl enable --now zombie-shot.timer

echo "[*] Status:"
sudo systemctl status zombie-xvfb.service zombie-shot.timer --no-pager || true

echo "[+] Installed. Check logs with: journalctl -u zombie-shot.service -n 100 --no-pager"