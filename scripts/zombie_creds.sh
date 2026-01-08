#!/usr/bin/env bash
# scripts/zombie_creds.sh
#
# Template: inject EVE account credentials via xdotool.
#
# Safety:
# - Do NOT hardcode passwords here.
# - Read credentials from environment variables: EVE_USER, EVE_PASS.
# - Avoid running with shell tracing enabled (do not `set -x`).
#
# Usage:
#   export DISPLAY=:1
#   export EVE_USER='you@example.com'
#   export EVE_PASS='your-password'
#   ./scripts/zombie_creds.sh --loop
#
# Notes:
# - The exact window title can vary. Default matches common EVE client windows.
# - The default key sequence is: type user, Tab, type pass, Return.

set -euo pipefail

WINDOW_PATTERN_DEFAULT='EVE - |EVE Online|Login'

MODE='once'
WINDOW_PATTERN="$WINDOW_PATTERN_DEFAULT"
TYPE_DELAY_MS=20
POLL_SECONDS=5

while [[ $# -gt 0 ]]; do
  case "$1" in
    --loop) MODE='loop'; shift ;;
    --once) MODE='once'; shift ;;
    --window-pattern) WINDOW_PATTERN="$2"; shift 2 ;;
    --type-delay-ms) TYPE_DELAY_MS="$2"; shift 2 ;;
    --poll-seconds) POLL_SECONDS="$2"; shift 2 ;;
    -h|--help)
      cat <<EOF
Usage: $0 [--once|--loop] [--window-pattern REGEX] [--type-delay-ms N] [--poll-seconds N]

Env vars (required):
  EVE_USER, EVE_PASS
EOF
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${DISPLAY:-}" ]]; then
  echo "DISPLAY is not set" >&2
  exit 2
fi

if [[ -z "${EVE_USER:-}" || -z "${EVE_PASS:-}" ]]; then
  echo "Missing EVE_USER or EVE_PASS in environment" >&2
  exit 2
fi

if ! command -v xdotool >/dev/null 2>&1; then
  echo "xdotool not found" >&2
  exit 2
fi

inject_once() {
  local wid
  wid="$(xdotool search --name "$WINDOW_PATTERN" 2>/dev/null | head -n 1 || true)"
  if [[ -z "$wid" ]]; then
    echo "[zombie_creds] window not found (pattern=$WINDOW_PATTERN)"
    return 1
  fi

  # Focus the window then type credentials.
  xdotool windowactivate --sync "$wid"
  sleep 0.2

  # Try to land on username field (safe even if already focused)
  xdotool key --clearmodifiers ctrl+l 2>/dev/null || true
  sleep 0.1

  xdotool type --clearmodifiers --delay "$TYPE_DELAY_MS" -- "$EVE_USER"
  xdotool key --clearmodifiers Tab
  xdotool type --clearmodifiers --delay "$TYPE_DELAY_MS" -- "$EVE_PASS"
  xdotool key --clearmodifiers Return

  echo "[zombie_creds] injected credentials into wid=$wid"
  return 0
}

if [[ "$MODE" == 'once' ]]; then
  inject_once
  exit $?
fi

echo "[zombie_creds] loop started (poll=${POLL_SECONDS}s display=$DISPLAY)"
while true; do
  inject_once && exit 0
  sleep "$POLL_SECONDS"
done
