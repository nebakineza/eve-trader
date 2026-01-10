#!/usr/bin/env python3
"""launcher_control.py - Neural Interface Edition

Utilizes the iGPU-accelerated Visual Cortex to intelligently manage
the EVE Launcher states (Play, Update, Verify).
"""

import argparse
import sys
import os
import time
import subprocess
import logging

# Path Hygiene: Add project root to path to reach nexus
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

try:
    from nexus.automaton.visual_cortex import VisualCortex
except ImportError:
    print("FATAL: Could not import VisualCortex. Check project structure.")
    sys.exit(1)

try:
    from scripts.arduino_connector import ArduinoBridge
except ImportError:
    try:
        from arduino_connector import ArduinoBridge
    except ImportError:
        ArduinoBridge = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [launcher_control] - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HardwareController:
    def __init__(self):
        self.bridge = None
        if ArduinoBridge:
            try:
                baud_rate = int(os.getenv("ARDUINO_BAUD", "9600"))
                port = os.getenv("ARDUINO_PORT")
                self.bridge = ArduinoBridge(baud_rate=baud_rate, port=port)
                if not self.bridge.is_active():
                    self.bridge = None
            except Exception:
                pass
        
        if self.bridge:
            logger.info("Hardware Bridge ACTIVE (/dev/ttyACM0).")
        else:
            logger.critical("Hardware Bridge DISCONNECTED. TERMINATING TO PREVENT BAN.")
            sys.exit("CRITICAL: HARDWARE DISCONNECTED. TERMINATING TO PREVENT BAN.")

    def click(self, x, y):
        if self.bridge:
            # Protocol: MOV:x,y then CLK
            self.bridge.send_command(f"MOV:{x},{y}")
            # Debounce
            time.sleep(0.1)
            self.bridge.send_command("CLK")
            logger.info(f"HARDWARE CLICK at {x},{y}")
        else:
            sys.exit("CRITICAL: HARDWARE DISCONNECTED.")

    def hotkey(self, cmd: str) -> bool:
        """Send a keyboard macro via GhostHID/Arduino (hardware-only)."""
        if not self.bridge:
            sys.exit("CRITICAL: HARDWARE DISCONNECTED.")
        try:
            ok = bool(self.bridge.send_command(cmd))
            if ok:
                logger.info(f"HARDWARE HOTKEY: {cmd}")
            else:
                logger.warning(f"HARDWARE HOTKEY FAILED: {cmd}")
            return ok
        except Exception as e:
            logger.warning(f"HARDWARE HOTKEY ERROR: {cmd} ({e})")
            return False

    def potato_mode(self) -> None:
        """Apply potato-mode toggles via GhostHID (no software injection)."""
        # NOTE: Command names must match the Arduino firmware mappings.
        # Known supported in hardware/stealth_hid.ino: CTRL+SHIFT+F9
        logger.info("POTATO MODE: sending CTRL+SHIFT+F9")
        self.hotkey("CTRL+SHIFT+F9")
        time.sleep(0.5)
        # Optional: some firmwares support ALT+C for inventory; best-effort.
        logger.info("POTATO MODE: sending ALT+C")
        self.hotkey("ALT+C")

def is_client_running():
    # Safer check for exefile.exe or eve-online.exe
    # Returns 0 (True) if found, 1 (False) if not
    try:
        # pgrep -f matches the full command line
        # We assume if 'exefile.exe' or 'eve-online.exe' is running, the client is up.
        # We suppress output.
        ret = subprocess.call(["pgrep", "-f", "exefile.exe|eve-online.exe"], stdout=subprocess.DEVNULL)
        return ret == 0
    except Exception:
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop", action="store_true")
    # Tighter interval for responsiveness
    parser.add_argument("--interval", type=int, default=5)
    args = parser.parse_args()
    
    controller = HardwareController()
    
    logger.info("Initializing Visual Cortex (iGPU)...")
    vc = VisualCortex(display=":0")
    
    client_was_running = False

    while True:
        try:
            # 1. Check client status
            if is_client_running():
                if not client_was_running:
                    logger.info("Client Launch Detected! Triggering Potato Mode...")
                    # Delay slightly to ensure window mapping
                    time.sleep(5)
                    # Potato Mode via GhostHID (hardware-only)
                    controller.potato_mode()
                    client_was_running = True
                
                logger.info("Game Client Active. Standing by.")
                if not args.loop: break
                time.sleep(args.interval)
                continue
            
            # Client is not running, so we must be in launcher context
            client_was_running = False 
            
            # 2. Analyze Launcher State
            state, coords = vc.analyze_state()
            
            if state == vc.STATE_PLAY_READY and coords:
                cx, cy = coords
                logger.info(f"STATE: PLAY_READY. Engaging at ({cx}, {cy})...")
                controller.click(cx, cy)
                # Wait for response before looping immediately
                time.sleep(15) 
                
            elif state == vc.STATE_UPDATE_REQUIRED:
                # We are purging yellow hallucinations, so we ignore this or log only
                logger.warning("STATE: UPDATE_REQUIRED detected but update-click is DISABLED (Purge Hallucination Mode).")
                
            elif state == vc.STATE_VERIFYING:
                logger.info("STATE: VERIFYING. Waiting for completion...")
                
            else:
                logger.debug("STATE: UNKNOWN/IDLE. Scanning...")

        except KeyboardInterrupt:
            logger.info("Manual Interrupt.")
            break
        except Exception as e:
            logger.error(f"Loop error: {e}")
            time.sleep(5)
        
        if not args.loop:
            break
        time.sleep(args.interval)

if __name__ == "__main__":
    main()
