#!/usr/bin/env bash
set -euo pipefail

export DISPLAY=:1

if ! command -v xdotool >/dev/null 2>&1; then
	echo "[force_entry] xdotool not found" >&2
	exit 2
fi

# Find window, bring to front, and send "End" then "Return" 3 times
WID="$(xdotool search --name "EVE|Wine" 2>/dev/null | head -n 1 || true)"
if [[ -z "${WID}" ]]; then
	echo "[force_entry] no matching window found (pattern=EVE|Wine)" >&2
	exit 0
fi

xdotool windowactivate --sync "$WID" || true
sleep 1

for _ in 1 2 3; do
	xdotool key --delay 200 End Return
	sleep 1
done

# One extra Return to clear lingering prompts
xdotool key --delay 200 Return
