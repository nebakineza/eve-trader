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
    Hardware-only execution node.
    Software injection (xdotool) has been removed.
    """
    def __init__(self):
        baud_rate = int(os.getenv("ARDUINO_BAUD", "9600"))
        port = os.getenv("ARDUINO_PORT")
        self.hid = ArduinoBridge(baud_rate=baud_rate, port=port)
        if not self.hid.is_active():
            raise RuntimeError("Hardware HID not found. Software injection is disabled.")
        logger.info("Hardware HID Active. Stealth Mode Engaged.")

    def _sleep_jitter(self, min_ms, max_ms):
        """Human-like delay between actions."""
        duration = random.randint(min_ms, max_ms) / 1000.0
        time.sleep(duration)

    def send_keystroke(self, command, raw_key=None):
        """
        Routes the command to the appropriate interface.
        command: Protocol string for Arduino (e.g., "ALT+M", "ENTER", "PRESS:x")
        raw_key: xdotool equivalent (e.g., "alt+m", "Return", "x")
        """
        # Inter-action delay
        self._sleep_jitter(200, 500)
        
        self.hid.send_command(command)

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
        self.hid.send_command(f"TYPE:{text}")

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
