import serial
import time
import logging
import os
import socket
from urllib.parse import urlparse

logger = logging.getLogger("HardwareBridge")


def _parse_redis_url(redis_url: str) -> tuple[str, int, int]:
    parsed = urlparse(redis_url)
    if parsed.scheme not in ("redis", "rediss"):
        raise ValueError(f"Unsupported redis url scheme: {parsed.scheme}")
    host = parsed.hostname or "127.0.0.1"
    port = int(parsed.port or 6379)
    db = 0
    if parsed.path and parsed.path != "/":
        try:
            db = int(parsed.path.lstrip("/"))
        except ValueError:
            db = 0
    return host, port, db


def _read_line(sock: socket.socket) -> bytes:
    data = bytearray()
    while True:
        chunk = sock.recv(1)
        if not chunk:
            raise ConnectionError("Redis connection closed")
        data += chunk
        if data.endswith(b"\r\n"):
            return bytes(data)


def _send_resp_array(sock: socket.socket, parts: list[bytes]) -> None:
    out = bytearray()
    out += f"*{len(parts)}\r\n".encode("ascii")
    for p in parts:
        out += f"${len(p)}\r\n".encode("ascii")
        out += p
        out += b"\r\n"
    sock.sendall(out)


def redis_get(redis_url: str, key: str) -> bytes | None:
    host, port, db = _parse_redis_url(redis_url)
    with socket.create_connection((host, port), timeout=2) as sock:
        if db:
            _send_resp_array(sock, [b"SELECT", str(db).encode("ascii")])
            resp = _read_line(sock)
            if not resp.startswith(b"+"):
                raise RuntimeError(f"Redis SELECT failed: {resp!r}")

        _send_resp_array(sock, [b"GET", key.encode("utf-8")])
        resp = _read_line(sock)
        if resp.startswith(b"$-1"):
            return None
        if not resp.startswith(b"$"):
            raise RuntimeError(f"Redis GET failed: {resp!r}")
        length = int(resp[1:].strip())
        payload = b""
        while len(payload) < length + 2:
            payload += sock.recv(length + 2 - len(payload))
        return payload[:-2]

class ArduinoHID:
    def __init__(self, port='/dev/ttyACM0', baudrate=9600):
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        self.redis_url = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
        self.visual_status_key = os.getenv("VISUAL_STATUS_KEY", "system:visual:status")
        self.required_visual_status = os.getenv("VISUAL_REQUIRED_STATUS", "VERIFIED")
        self._last_block_log = 0.0
        self._connect()

    def _visual_lock_allows(self) -> bool:
        try:
            value = redis_get(self.redis_url, self.visual_status_key)
            status = (value or b"").decode("utf-8", errors="ignore").strip()
        except Exception as e:
            now = time.time()
            if now - self._last_block_log > 2.0:
                logger.warning("Visual Lock check failed; blocking HID (error=%s)", e)
                self._last_block_log = now
            return False

        if status == self.required_visual_status:
            return True

        now = time.time()
        if now - self._last_block_log > 2.0:
            logger.warning(
                "Visual Lock active; blocking HID (key=%s status=%r required=%r)",
                self.visual_status_key,
                status,
                self.required_visual_status,
            )
            self._last_block_log = now
        return False

    def _connect(self):
        try:
            self.serial = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2) # Wait for Arduino reset
            logger.info(f"Connected to Arduino HID at {self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to Arduino: {e}")
            self.serial = None

    def send_key(self, key_spec):
        """
        Sends a keystroke command to the Arduino.
        Protocol: "KEY:<char>" or "SPEC:<code_name>"
        """
        if not self._visual_lock_allows():
            return

        if not self.serial: 
            # Try reconnect?
            self._connect()
            if not self.serial: return

        cmd = f"KEY:{key_spec}\n"
        self.serial.write(cmd.encode('utf-8'))
        logger.debug(f"Sent HID Key: {key_spec}")

    def move_mouse(self, x, y):
        """
        Sends absolute mouse move. 
        Protocol: "MOV:<x>,<y>"
        Assumes Arduino firmware maps screenspace to HID descriptor.
        """
        if not self._visual_lock_allows():
            return
        if not self.serial: return
        cmd = f"MOV:{int(x)},{int(y)}\n"
        self.serial.write(cmd.encode('utf-8'))
    
    def click(self):
        if not self._visual_lock_allows():
            return
        if not self.serial: return
        self.serial.write(b"CLK\n")

    def type_string(self, text):
        if not self._visual_lock_allows():
            return
        for char in text:
            self.send_key(char)
            time.sleep(0.05 + (0.01 * (ord(char) % 5))) # Human jitter
