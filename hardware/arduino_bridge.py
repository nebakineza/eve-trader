import serial
import serial.tools.list_ports
import time
import sys
import logging
import random

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ArduinoBridge")

class ArduinoBridge:
    def __init__(self, baud_rate=9600):
        self.baud_rate = baud_rate
        self.connection = None
        self.port = self.find_arduino()
        
        if self.port:
            try:
                self.connection = serial.Serial(self.port, self.baud_rate, timeout=1)
                time.sleep(2) # Give bootloader time
                logger.info(f"Connected to Arduino on {self.port}")
            except Exception as e:
                logger.error(f"Failed to connect: {e}")
                self.connection = None
        else:
            logger.warning("No Arduino detected. Falling back to Software/Virtual HID.")

    def find_arduino(self):
        """Scans for an Arduino Leonardo or similar device."""
        ports = list(serial.tools.list_ports.comports())
        for p in ports:
            # Common Arduino VID/PID or Description matching
            if "Arduino" in p.description or "Leonardo" in p.description or "USB Serial" in p.description:
                return p.device
        return None

    def send_command(self, cmd):
        """Sends a text command to the Arduino."""
        if self.connection and self.connection.is_open:
            try:
                msg = f"{cmd}\n"
                self.connection.write(msg.encode('utf-8'))
                logger.debug(f"Hardware HID: Sent {cmd}")
                return True
            except Exception as e:
                logger.error(f"Serial Write Error: {e}")
                return False
        return False

    def is_active(self):
        return self.connection is not None and self.connection.is_open

if __name__ == "__main__":
    bridge = ArduinoBridge()
    if bridge.is_active():
        sys.exit(0) # Success code
    else:
        sys.exit(1) # Failure code
