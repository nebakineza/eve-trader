#!/usr/bin/env python3

import sys
import os

# Keep imports working when executed from repo root or scripts/
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from scripts.arduino_connector import ArduinoBridge


def main() -> int:
    baud_rate = int(os.getenv("ARDUINO_BAUD", "9600"))
    port = os.getenv("ARDUINO_PORT")
    bridge = ArduinoBridge(baud_rate=baud_rate, port=port)
    ok = bridge.handshake()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
