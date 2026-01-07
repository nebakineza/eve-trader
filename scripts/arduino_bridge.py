import json
import logging
import os
import sys
import time
from datetime import datetime

import redis

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from scripts.arduino_connector import ArduinoBridge

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ArduinoBridgeService")


def _redis_client() -> redis.Redis:
    redis_url = os.getenv("REDIS_URL", "redis://cache:6379/0")
    r = redis.Redis.from_url(redis_url, decode_responses=True)
    r.ping()
    return r


def _set_status(r: redis.Redis, status: str, detail: str | None = None) -> None:
    # Dashboard reads zombie:hid_status (GREEN => online).
    r.set("zombie:hid_status", status)
    r.set("zombie:hid_last_seen", datetime.utcnow().isoformat() + "Z")
    if detail is not None:
        r.set("zombie:hid_detail", detail)


def main() -> int:
    serial_port = os.getenv("ARDUINO_PORT")  # optional; connector also auto-detects
    baud_rate = int(os.getenv("ARDUINO_BAUD", "9600"))
    once = os.getenv("ONCE", "0").strip() in {"1", "true", "yes"}
    retry_seconds = float(os.getenv("RETRY_SECONDS", "5"))

    r = None
    try:
        r = _redis_client()
    except Exception as e:
        logger.error(f"Redis unavailable: {e}")

    def publish(status: str, detail: str | None = None):
        if r is None:
            return
        try:
            _set_status(r, status=status, detail=detail)
        except Exception:
            pass

    while True:
        try:
            bridge = ArduinoBridge(baud_rate=baud_rate)
            # If user specified a fixed port, prefer it.
            if serial_port:
                try:
                    import serial

                    bridge.port = serial_port
                    bridge.connection = serial.Serial(serial_port, baud_rate, timeout=1)
                    time.sleep(2)
                except Exception as e:
                    bridge.connection = None
                    logger.error(f"Failed to open {serial_port}: {e}")

            if bridge.is_active():
                detail = json.dumps({"port": bridge.port, "mode": "SERIAL", "ok": True})
                publish("GREEN", detail=detail)
                logger.info(f"✅ Arduino ONLINE on {bridge.port}")
                if once:
                    return 0
            else:
                publish("AMBER", detail=json.dumps({"port": bridge.port, "mode": "SERIAL", "ok": False}))
                logger.warning("🟠 Arduino OFFLINE (no serial connection)")
                if once:
                    return 1

        except Exception as e:
            publish("AMBER", detail=json.dumps({"error": str(e)}))
            logger.error(f"Arduino bridge loop error: {e}")
            if once:
                return 1

        time.sleep(retry_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
