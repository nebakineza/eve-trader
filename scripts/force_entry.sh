#!/usr/bin/env bash
# scripts/force_entry.sh
#
# Aggressive "blind-fire" EULA / prompt bypass on the headless host.
#
# Behavior (per spec):
# - Focus any window with "EVE" or "Wine" in title
# - Tab 6 times
# - Space (toggle agree)
# - Return (submit)
# - Alt+A and Alt+Y

set -euo pipefail

DISPLAY_VAL="${DISPLAY:-:1}"
WINDOW_PATTERN="${WINDOW_PATTERN:-EVE|Wine}"
TAB_COUNT="${TAB_COUNT:-6}"
REPEAT="${REPEAT:-2}"
KEY_DELAY_MS="${KEY_DELAY_MS:-120}"

if ! command -v xdotool >/dev/null 2>&1; then
  echo "[force_entry] xdotool not found" >&2
  exit 2
fi

focus_targets() {
  # xdotool uses regex for --name.
  xdotool search --name "$WINDOW_PATTERN" 2>/dev/null || true
}

send_combo() {
  local wid="$1"

  xdotool windowactivate --sync "$wid" 2>/dev/null || return 0
  sleep 0.2

  # TAB navigation
  for _ in $(seq 1 "$TAB_COUNT"); do
    xdotool key --delay "$KEY_DELAY_MS" Tab
  done

  # Toggle checkbox, submit, common accept/yes accelerators.
  xdotool key --delay "$KEY_DELAY_MS" space
  xdotool key --delay "$KEY_DELAY_MS" Return
  xdotool key --delay "$KEY_DELAY_MS" alt+a
  xdotool key --delay "$KEY_DELAY_MS" alt+y
}

main() {
  echo "[force_entry] DISPLAY=$DISPLAY_VAL pattern=$WINDOW_PATTERN tab_count=$TAB_COUNT repeat=$REPEAT"

  local ids
  ids="$(DISPLAY="$DISPLAY_VAL" focus_targets | tr '\n' ' ' | xargs echo || true)"
  if [[ -z "${ids// }" ]]; then
    echo "[force_entry] no matching windows found" >&2
    exit 0
  fi

  local i
  for i in $(seq 1 "$REPEAT"); do
    for wid in $ids; do
      echo "[force_entry] round=$i window=$wid"
      DISPLAY="$DISPLAY_VAL" send_combo "$wid" || true
    done
    sleep 0.5
  done

  echo "[force_entry] done"
}

main "$@"
