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
    def __init__(self, baud_rate=9600, port=None):
        self.baud_rate = baud_rate
        self.connection = None
        self.port = port or self.find_arduino()
        
        if self.port:
            try:
                self.connection = serial.Serial(
                    self.port,
                    self.baud_rate,
                    timeout=1,
                    write_timeout=1,
                )
                time.sleep(2) # Give bootloader time
                logger.info(f"Connected to Arduino on {self.port}")
            except Exception as e:
                logger.error(f"Failed to connect: {e}")
                self.connection = None
        else:
            logger.warning("No Arduino detected. Hardware HID is unavailable.")

    def find_arduino(self):
        """Scans for an Arduino Leonardo or similar device."""
        ports = list(serial.tools.list_ports.comports())
        for p in ports:
            try:
                if getattr(p, "vid", None) == 0x2341 and getattr(p, "pid", None) in {0x8036, 0x0036}:
                    return p.device
            except Exception:
                pass
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
                try:
                    self.connection.flush()
                except Exception:
                    pass
                logger.debug(f"Hardware HID: Sent {cmd}")
                return True
            except serial.SerialTimeoutException as e:
                logger.error(f"Serial Write Timeout: {e}")
                return False
            except Exception as e:
                logger.error(f"Serial Write Error: {e}")
                return False
        return False

    def is_active(self):
        return self.connection is not None and self.connection.is_open

    def read_line(self, timeout_s: float = 2.0):
        if not (self.connection and self.connection.is_open):
            return None
        try:
            old_timeout = self.connection.timeout
            self.connection.timeout = max(0.1, float(timeout_s))
            raw = self.connection.readline()
            if not raw:
                return None
            try:
                return raw.decode("utf-8", errors="replace").strip()
            except Exception:
                return str(raw)
        finally:
            try:
                self.connection.timeout = old_timeout
            except Exception:
                pass

    def handshake(self) -> bool:
        if not self.is_active():
            logger.error("Handshake failed: serial inactive")
            return False
        if not self.send_command("PING"):
            logger.error("Handshake failed: could not send PING")
            return False
        line = self.read_line(timeout_s=2.0)
        if line == "PONG":
            logger.info("Handshake Verified: PONG")
            return True
        logger.error(f"Handshake failed: expected PONG, got {line!r}")
        return False

if __name__ == "__main__":
    bridge = ArduinoBridge()
    if bridge.is_active():
        sys.exit(0) # Success code
    else:
        sys.exit(1) # Failure code
