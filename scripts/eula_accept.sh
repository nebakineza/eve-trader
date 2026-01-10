#!/usr/bin/env bash
# scripts/eula_accept.sh
#
# EULA Cleaner: best-effort acceptance of common EULA/license dialogs.
#
# Behavior:
# - Find a window whose title contains: EULA, License, or Agreement
# - Focus it
# - Press End (scroll to bottom)
# - Press Return (accept)
#
# Env:
#   DISPLAY (default :1)

set -euo pipefail

DISPLAY_VAL="${DISPLAY:-:1}"
PATTERN="${EULA_WINDOW_PATTERN:-License|EULA|Agreement}"

if ! command -v xdotool >/dev/null 2>&1; then
  echo "xdotool not found" >&2
  exit 2
fi

WID="$(DISPLAY="$DISPLAY_VAL" xdotool search --name "$PATTERN" 2>/dev/null | head -n 1 || true)"

if [[ -z "$WID" ]]; then
  echo "[eula_accept] no EULA window found (pattern=$PATTERN)"
  exit 0
fi

echo "[eula_accept] found window id=$WID; sending End+Return"
DISPLAY="$DISPLAY_VAL" xdotool windowactivate --sync "$WID"

# Some dialogs only enable Accept once scrolled.
DISPLAY="$DISPLAY_VAL" xdotool key --delay 500 End
DISPLAY="$DISPLAY_VAL" xdotool key --delay 500 Return
