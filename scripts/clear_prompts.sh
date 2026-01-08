#!/usr/bin/env bash
# scripts/clear_prompts.sh
#
# Dialog Sweeper: clears common Wine interactive prompts (Mono/Gecko) and EULA dialogs.
#
# Rules:
# - If window title contains: Wine / Mono / Gecko  -> Return
# - If window title contains: License / EULA / Agreement -> End then Return
# - Fallback: send Escape to the currently active window
#
# Env:
#   DISPLAY (default :1)

set -euo pipefail

DISPLAY_VAL="${DISPLAY:-:1}"

if ! command -v xdotool >/dev/null 2>&1; then
  echo "xdotool not found" >&2
  exit 2
fi

# Enumerate all windows. xdotool uses regex; '.*' means any.
WIDS="$(DISPLAY="$DISPLAY_VAL" xdotool search --all --name '.*' 2>/dev/null || true)"

swept=0
while IFS= read -r wid; do
  [[ -z "$wid" ]] && continue

  name="$(DISPLAY="$DISPLAY_VAL" xdotool getwindowname "$wid" 2>/dev/null || true)"
  [[ -z "$name" ]] && continue

  if echo "$name" | grep -Eiq '(Wine|Mono|Gecko)'; then
    echo "[sweep] prompt: wid=$wid name=$name -> Return"
    DISPLAY="$DISPLAY_VAL" xdotool windowactivate --sync "$wid"
    DISPLAY="$DISPLAY_VAL" xdotool key --delay 200 Return
    swept=$((swept + 1))
    continue
  fi

  if echo "$name" | grep -Eiq '(License|EULA|Agreement)'; then
    echo "[sweep] eula: wid=$wid name=$name -> End, Return"
    DISPLAY="$DISPLAY_VAL" xdotool windowactivate --sync "$wid"
    DISPLAY="$DISPLAY_VAL" xdotool key --delay 400 End
    DISPLAY="$DISPLAY_VAL" xdotool key --delay 400 Return
    swept=$((swept + 1))
    continue
  fi

done <<< "$WIDS"

# Pixel-specific fallback for 1024x768 dialogs where window names are unhelpful.
# Standard 'Accept' button location for the EULA prompt.
if [[ "$swept" -eq 0 ]]; then
  echo "[sweep] pixel-fallback -> mousemove 512 600 click 1"
  DISPLAY="$DISPLAY_VAL" xdotool mousemove 512 600 click 1
  swept=$((swept + 1))
fi

# Fallback: if something modal is stealing focus, try Esc once.
active_wid="$(DISPLAY="$DISPLAY_VAL" xdotool getactivewindow 2>/dev/null || true)"
if [[ -n "$active_wid" ]]; then
  active_name="$(DISPLAY="$DISPLAY_VAL" xdotool getwindowname "$active_wid" 2>/dev/null || true)"
  echo "[sweep] fallback -> Escape, Return (active wid=$active_wid name=$active_name)"
  DISPLAY="$DISPLAY_VAL" xdotool key --delay 200 Escape
  DISPLAY="$DISPLAY_VAL" xdotool key --delay 200 Return
fi

echo "[sweep] done (actions=$swept)"
