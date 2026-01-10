#!/usr/bin/env bash
# scripts/window_scan.sh
# Best-effort window state scan on DISPLAY=:1.
# Writes system:zombie:window_title and system:zombie:client_state to Redis.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LOOP_MODE=0
INTERVAL_SECONDS=5
while [[ $# -gt 0 ]]; do
  case "$1" in
    --loop)
      LOOP_MODE=1
      shift
      ;;
    --interval-seconds)
      INTERVAL_SECONDS="${2:-5}"
      shift 2
      ;;
    *)
      echo "[window_scan] unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

DISPLAY_VAL="${DISPLAY:-:1}"
REDIS_HOST="${REDIS_HOST:-127.0.0.1}"
REDIS_PORT="${REDIS_PORT:-6379}"

if ! command -v xdotool >/dev/null 2>&1; then
  echo "[window_scan] xdotool not found" >&2
  exit 2
fi

scan_once() {
  # Prefer the most meaningful window title.
  PATTERN="EVE Online|Character Selection|EVE - |EVE Launcher|Wine|EULA|License|Agreement"
  WID="$(DISPLAY="$DISPLAY_VAL" xdotool search --name "$PATTERN" 2>/dev/null | head -n 1 || true)"
  TITLE=""
  if [[ -n "$WID" ]]; then
    TITLE="$(DISPLAY="$DISPLAY_VAL" xdotool getwindowname "$WID" 2>/dev/null || true)"
  fi

  STATE="UNKNOWN"
  if [[ "$TITLE" =~ EVE\ Online ]]; then
    STATE="AUTO_ZOMBIE"
  elif [[ "$TITLE" =~ Character\ Selection ]]; then
    STATE="AUTO_ZOMBIE"
  elif [[ "$TITLE" =~ ^EVE\ \-\  ]]; then
    STATE="AUTO_ZOMBIE"
  elif [[ "$TITLE" =~ EULA|License|Agreement ]]; then
    STATE="EULA"
  elif [[ "$TITLE" =~ EVE\ Launcher ]]; then
    STATE="LAUNCHER"
  elif [[ -n "$TITLE" ]]; then
    STATE="OTHER"
  fi

  REDIS_HOST="$REDIS_HOST" REDIS_PORT="$REDIS_PORT" WINDOW_TITLE="$TITLE" CLIENT_STATE="$STATE" python3 - <<'PY'
import os, socket, time

host = os.environ.get('REDIS_HOST', '127.0.0.1')
port = int(os.environ.get('REDIS_PORT', '6379'))

title = os.environ.get('WINDOW_TITLE', '')
state = os.environ.get('CLIENT_STATE', 'UNKNOWN')
now = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())

def send(cmd: bytes) -> None:
    s = socket.create_connection((host, port), timeout=2)
    s.sendall(cmd)
    s.recv(128)
    s.close()

for k, v in [
    ('system:zombie:window_title', title),
    ('system:zombie:client_state', state),
    ('system:zombie:window_scanned_at', now),
]:
    vb = v.encode('utf-8', errors='replace')
    kb = k.encode('utf-8')
    cmd = b"*3\r\n$3\r\nSET\r\n$%d\r\n%b\r\n$%d\r\n%b\r\n" % (len(kb), kb, len(vb), vb)
    send(cmd)
PY
}

if [[ "$LOOP_MODE" -eq 1 ]]; then
  while true; do
    scan_once
    sleep "$INTERVAL_SECONDS"
  done
else
  scan_once
fi
