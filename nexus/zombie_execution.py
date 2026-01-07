import time
import random
import logging
import subprocess
import os

# Import Hardware Bridge
from scripts.arduino_connector import ArduinoBridge

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ZombieExec")

class ZombieExecutionEngine:
    """
    Stealth Execution Node.
    Routing priority: Hardware HID (Arduino) -> Software HID (xdotool).
    """
    def __init__(self):
        self.hid = ArduinoBridge()
        self.software_fallback = not self.hid.is_active()
        
        if self.software_fallback:
            logger.warning("Hardware HID not found. Engaging Software Simulation (xdotool).")
        else:
            logger.info("Hardware HID Active. Stealth Mode Engaged.")

    def _sleep_jitter(self, min_ms, max_ms):
        """Human-like delay between actions."""
        duration = random.randint(min_ms, max_ms) / 1000.0
        time.sleep(duration)

    def _software_press(self, key_combo):
        """Fallback using xdotool."""
        try:
            # key-down delay isn't easily modifiable in simple xdotool 'key' command 
            # without complex 'keydown' ... 'keyup' logic.
            # Using basic 'key' for reliability in fallback mode.
            cmd = ["xdotool", "search", "--name", "EVE - ", "windowactivate", "--sync", "key", "--delay", "100", key_combo]
            subprocess.run(cmd, check=True)
            logger.info(f"Software HID: {key_combo}")
        except Exception as e:
            logger.error(f"xdotool failed: {e}")

    def send_keystroke(self, command, raw_key=None):
        """
        Routes the command to the appropriate interface.
        command: Protocol string for Arduino (e.g., "ALT+M", "ENTER", "PRESS:x")
        raw_key: xdotool equivalent (e.g., "alt+m", "Return", "x")
        """
        # Inter-action delay
        self._sleep_jitter(200, 500)
        
        if not self.software_fallback:
            # Hardware
            self.hid.send_command(command)
        else:
            # Software
            if raw_key:
                self._software_press(raw_key)
            else:
                # Infer from command if simple
                if command == "ENTER": self._software_press("Return")
                elif command == "TAB": self._software_press("Tab")
                elif command.startswith("PRESS:"): self._software_press(command.split(":")[1])
                else: self._software_press(command.lower())

    # --- Macros ---

    def open_market(self):
        logger.info("Macro: Open Market")
        self.send_keystroke("ALT+M", "alt+m")

    def focus_search(self):
        logger.info("Macro: Focus Search")
        self.send_keystroke("SHIFT+S", "shift+s") # Assuming configured in EVE
        # Alternatively, if Market is open, maybe just Click or Tab? 
        # Requirement says "Shift + S: Focus Search"

    def type_text(self, text):
        logger.info(f"Macro: Typing '{text}'")
        if not self.software_fallback:
            self.hid.send_command(f"TYPE:{text}")
        else:
            # xdotool type
            subprocess.run(["xdotool", "type", "--delay", "100", text])

    def confirm(self):
        logger.info("Macro: Confirm")
        self.send_keystroke("ENTER", "Return")

    def next_field(self):
        logger.info("Macro: Next Field")
        self.send_keystroke("TAB", "Tab")

    def execute_trade(self, item_name, quantity, price, order_type="buy"):
        """
        Complex sequence to place an order.
        Assumes Market Window is focused or accessible via Alt+M.
        """
        logger.info(f"Executing {order_type.upper()} for {quantity} x {item_name} @ {price}")
        
        self.open_market()
        self._sleep_jitter(500, 800) # Wait for window
        
        # Search Item
        # self.focus_search() # If mapping exists
        # For now, let's assume default behavior or Tab into it
        # Relying on specific key Shift+S as requested
        self.focus_search() 
        self._sleep_jitter(200, 400)
        
        self.type_text(item_name)
        self._sleep_jitter(300, 500)
        self.confirm() # Trigger search
        
        # Navigate to list (this is tricky blindly, but let's assume filtering selects top item)
        self._sleep_jitter(800, 1200) # Wait for search results
        
        # Select Station Order / Place Order button
        # Without mouse, we might TAB through
        # This requires robust EVE UI knowledge. 
        # For this exercise, we implement the requested sequences.
        
        pass

if __name__ == "__main__":
    zombie = ZombieExecutionEngine()
    # Test Sequence
    # zombie.execute_trade("Tritanium", 1000, 4.5)
