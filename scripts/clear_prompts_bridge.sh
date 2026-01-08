#!/usr/bin/env bash
# scripts/clear_prompts_bridge.sh
#
# Redis-triggered bridge for the Dialog Sweeper.
# Watches a Redis key and runs clear_prompts.sh when triggered.
#
# This intentionally avoids redis-py / redis-cli dependencies by using Python stdlib sockets.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DISPLAY_VAL="${DISPLAY:-:1}"
REDIS_HOST="${REDIS_HOST:-127.0.0.1}"
REDIS_PORT="${REDIS_PORT:-6379}"
REDIS_KEY="${REDIS_KEY:-system:zombie:eula_accept}"
POLL_SECS="${POLL_SECS:-2}"
SWEEP_SCRIPT="${SWEEP_SCRIPT:-$SCRIPT_DIR/clear_prompts.sh}"

# Optional aggressive EULA force-entry trigger
FORCE_REDIS_KEY="${FORCE_REDIS_KEY:-system:zombie:force_entry}"
FORCE_SCRIPT="${FORCE_SCRIPT:-$SCRIPT_DIR/force_entry.sh}"

MODE="loop"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --once) MODE="once"; shift ;;
    --loop) MODE="loop"; shift ;;
    --key) REDIS_KEY="$2"; shift 2 ;;
    --poll-seconds) POLL_SECS="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ ! -x "$SWEEP_SCRIPT" ]]; then
  echo "sweep script not executable: $SWEEP_SCRIPT" >&2
  exit 2
fi

if [[ -f "$FORCE_SCRIPT" ]] && [[ ! -x "$FORCE_SCRIPT" ]]; then
  echo "force script not executable: $FORCE_SCRIPT" >&2
  exit 2
fi

redis_get() {
  python3 - <<'PY'
import os, socket
host = os.environ.get('REDIS_HOST', '127.0.0.1')
port = int(os.environ.get('REDIS_PORT', '6379'))
key  = os.environ.get('REDIS_KEY', 'system:zombie:eula_accept')

def send(cmd: bytes) -> bytes:
    s = socket.create_connection((host, port), timeout=2)
    s.sendall(cmd)
    data = b""
    s.settimeout(2)
    while True:
        try:
            chunk = s.recv(4096)
            if not chunk:
                break
            data += chunk
            # crude but sufficient: bulk replies include \r\n after payload
            if data.endswith(b"\r\n") and data.startswith((b"$", b"-", b"+", b":")):
                # might still be incomplete for bulk; handle below
                pass
            if data.startswith(b"$-"):
                break
            if data.startswith(b"$") and b"\r\n" in data:
                # if bulk, wait until we've got payload + final CRLF
                first, rest = data.split(b"\r\n", 1)
                try:
                    n = int(first[1:])
                except Exception:
                    break
                if n == -1:
                    break
                if len(rest) >= n + 2:
                    break
            if data.startswith((b"+", b"-", b":")) and b"\r\n" in data:
                break
        except socket.timeout:
            break
    try:
        s.close()
    except Exception:
        pass
    return data

cmd = f"*2\r\n$3\r\nGET\r\n${len(key)}\r\n{key}\r\n".encode('utf-8')
resp = send(cmd)
if resp.startswith(b"$-1"):
    print("")
elif resp.startswith(b"$"):
    # $<n>\r\n<bytes>\r\n
    try:
        header, rest = resp.split(b"\r\n", 1)
        n = int(header[1:])
        val = rest[:n].decode('utf-8', errors='replace')
        print(val)
    except Exception:
        print("")
else:
    print("")
PY
}

redis_del() {
  python3 - <<'PY'
import os, socket
host = os.environ.get('REDIS_HOST', '127.0.0.1')
port = int(os.environ.get('REDIS_PORT', '6379'))
key  = os.environ.get('REDIS_KEY', 'system:zombie:eula_accept')

cmd = f"*2\r\n$3\r\nDEL\r\n${len(key)}\r\n{key}\r\n".encode('utf-8')
try:
    s = socket.create_connection((host, port), timeout=2)
    s.sendall(cmd)
    s.recv(128)
    s.close()
except Exception:
    pass
PY
}

run_script_for_key() {
  local key="$1"
  local script="$2"
  if [[ -z "$key" ]] || [[ -z "$script" ]] || [[ ! -x "$script" ]]; then
    return 0
  fi

  local val
  val="$(REDIS_HOST="$REDIS_HOST" REDIS_PORT="$REDIS_PORT" REDIS_KEY="$key" redis_get)"
  if [[ -n "$val" ]]; then
    REDIS_HOST="$REDIS_HOST" REDIS_PORT="$REDIS_PORT" REDIS_KEY="$key" redis_del
    DISPLAY="$DISPLAY_VAL" "$script" || true
  fi
}

run_sweep_if_triggered() {
  run_script_for_key "$REDIS_KEY" "$SWEEP_SCRIPT"
  run_script_for_key "$FORCE_REDIS_KEY" "$FORCE_SCRIPT"
}

case "$MODE" in
  once)
    run_sweep_if_triggered
    ;;
  loop)
    while true; do
      run_sweep_if_triggered
      sleep "$POLL_SECS"
    done
    ;;
esac
