#!/bin/bash
# scripts/zombie_init.sh
# Headless "Zombie" Bridge Setup for Debian Host (192.168.14.105)
# Target Hardware: NVIDIA P4000

set -e

echo "[*] Initiating Zombie Node Setup..."

# 1. GPU Drivers & Prereqs
echo "[*] checking NVIDIA drivers..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "[!] NVIDIA driver not found. Installing..."
    # Ensure non-free repositories are available in /etc/apt/sources.list before running this in production
    sudo apt-get update
    sudo apt-get install -y nvidia-driver firmware-misc-nonfree
else
    echo "[ok] NVIDIA driver detected."
    nvidia-smi
fi

# 2. Minimal X11 Container
echo "[*] Installing Headless X11 Stack..."
sudo apt-get install -y xvfb xserver-xorg-core openbox xdotool python3-pip steam

# 3. Environment Configuration
export DISPLAY=:1
export DRI_PRIME=1
export PROTON_NO_ESYNC=1
export PROTON_NO_FSYNC=1

# 4. Virtual Display Launch
echo "[*] Launching Xvfb on :1..."
Xvfb :1 -screen 0 1920x1080x24 &
sleep 2

# 5. Window Manager
echo "[*] Launching Openbox..."
openbox &
sleep 1

# 6. EVE Launcher (Steam/Proton)
# Note: User must have installed EVE Online via Steam previously or use valid path
# This is a placeholder command for the actual launch
echo "[*] Ready to launch EVE Online..."
echo "Run: PROTON_NO_ESYNC=1 PROTON_NO_FSYNC=1 steam steam://rungameid/8500"

# 7. 3D Kill-Switch Daemon
# Waits for EVE window and sends Ctrl+Shift+F9
(
    echo "[*] Waiting for EVE Client to appear..."
    # Loop until window found (adjust window name 'EVE' as needed)
    while true; do
        if xdotool search --name "EVE - "; then
            echo "[*] EVE Client detected. Sleeping 30s for login..."
            sleep 30 
            echo "[*] Sending 3D Kill-Switch (Ctrl+Shift+F9)"
            xdotool search --name "EVE - " windowactivate --sync key --delay 100 ctrl+shift+F9
            echo "[ok] P4000 switched to 2D Compute Mode."
            break
        fi
        sleep 5
    done
) &

echo "[ok] Zombie Node Initialized. Display :1 active."
