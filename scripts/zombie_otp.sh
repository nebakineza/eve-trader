#!/usr/bin/env bash
# scripts/zombie_otp.sh
#
# OTP Bridge
#
# Mode A (one-shot):
#   ./scripts/zombie_otp.sh 123456
#   -> types the code into the active window on DISPLAY and hits Return
#
# Mode B (bridge loop):
#   ./scripts/zombie_otp.sh --loop
#   -> polls Redis key system:zombie:otp; when set, injects then clears the key
#
# Env:
#   DISPLAY   (default :1)
#   REDIS_URL (default redis://127.0.0.1:6379/0)

set -euo pipefail

DISPLAY_VAL="${DISPLAY:-:1}"
REDIS_URL="${REDIS_URL:-redis://127.0.0.1:6379/0}"
REDIS_KEY="${ZOMBIE_OTP_KEY:-system:zombie:otp}"
POLL_SECONDS="${ZOMBIE_OTP_POLL_SECONDS:-1}"

if ! command -v xdotool >/dev/null 2>&1; then
  echo "xdotool not found" >&2
  exit 2
fi

inject() {
  local code="$1"
  # Best-effort focus: prefer any visible EVE-related window.
  local wid
  wid="$(DISPLAY="$DISPLAY_VAL" xdotool search --name 'EVE Launcher|EVE - |EVE Online|Login|Account' 2>/dev/null | head -n 1 || true)"
  if [[ -n "$wid" ]]; then
    DISPLAY="$DISPLAY_VAL" xdotool windowactivate --sync "$wid" >/dev/null 2>&1 || true
    sleep 0.1
  fi

  DISPLAY="$DISPLAY_VAL" xdotool type --clearmodifiers --delay 30 -- "$code"
  DISPLAY="$DISPLAY_VAL" xdotool key --clearmodifiers Return
}

get_and_clear_otp() {
    REDIS_URL="$REDIS_URL" REDIS_KEY="$REDIS_KEY" python3 - <<'PY'
import os
import socket
from urllib.parse import urlparse

redis_url = os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/0")
key = os.environ.get("REDIS_KEY", "system:zombie:otp")

p = urlparse(redis_url)
host = p.hostname or "127.0.0.1"
port = int(p.port or 6379)
db = 0
if p.path and p.path != "/":
    try:
        db = int(p.path.lstrip("/"))
    except ValueError:
        db = 0

def resp(*parts: str) -> bytes:
    out = f"*{len(parts)}\r\n".encode()
    for part in parts:
        b = part.encode()
        out += f"${len(b)}\r\n".encode() + b + b"\r\n"
    return out

def read_line(s: socket.socket) -> bytes:
    buf = b""
    while not buf.endswith(b"\r\n"):
        chunk = s.recv(1)
        if not chunk:
            raise EOFError
        buf += chunk
    return buf[:-2]

def read_bulk(s: socket.socket):
    first = read_line(s)
    if first[:1] == b"-":
        raise RuntimeError(first.decode(errors="ignore"))
    if first[:1] == b"$":
        n = int(first[1:])
        if n == -1:
            return None
        data = b""
        while len(data) < n + 2:
            data += s.recv(n + 2 - len(data))
        return data[:n]
    if first[:1] == b"+":
        return first[1:]
    if first[:1] == b":":
        return int(first[1:])
    return first

with socket.create_connection((host, port), timeout=2) as s:
    if db:
        s.sendall(resp("SELECT", str(db)))
        read_line(s)

    s.sendall(resp("GET", key))
    val = read_bulk(s)

with socket.create_connection((host, port), timeout=2) as s:
    if db:
        s.sendall(resp("SELECT", str(db)))
        read_line(s)
    # Clear key regardless (idempotent)
    s.sendall(resp("DEL", key))
    read_line(s)

if val is None:
    raise SystemExit(0)

try:
    code = val.decode().strip()
except Exception:
    raise SystemExit(0)

# Only print the OTP to stdout for the bridge script to capture.
# Avoid extra logging here.
print(code)
PY
}

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 <otp-code> | --loop" >&2
  exit 2
fi

if [[ "$1" == "--loop" ]]; then
  while true; do
    code="$(get_and_clear_otp || true)"
    if [[ -n "$code" ]]; then
      inject "$code"
    fi
    sleep "$POLL_SECONDS"
  done
fi

inject "$1"
