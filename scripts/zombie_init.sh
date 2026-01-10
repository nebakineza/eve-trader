#!/bin/bash
# scripts/zombie_init.sh
# Headless "Zombie" Bridge Setup for Debian Host (192.168.14.105)
# Target Hardware: NVIDIA P4000

echo "[zombie_init] Direct injection removed from software; automation is disabled." >&2
echo "[zombie_init] This script is intentionally disabled." >&2
exit 2

# 2. Minimal X11 Container
echo "[*] Installing Headless X11 Stack..."
missing_pkgs=()
for cmd in Xvfb openbox python3; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        case "$cmd" in
            Xvfb) missing_pkgs+=(xvfb xserver-xorg-core) ;;
            openbox) missing_pkgs+=(openbox) ;;
            python3) missing_pkgs+=(python3 python3-pip) ;;
        esac
    fi
done

if [ "${#missing_pkgs[@]}" -gt 0 ]; then
    echo "[*] Installing missing packages: ${missing_pkgs[*]}"
    sudo apt-get update
    sudo apt-get install -y "${missing_pkgs[@]}"
else
    echo "[ok] Headless X11 stack already present."
fi

# 3. Environment Configuration
# Prefer an explicitly-provided DISPLAY (e.g. `DISPLAY=:1 ... scripts/zombie_init.sh`).
# Fall back to ZOMBIE_DISPLAY, then :0 for historical dashboard compatibility.
if [ -n "${ZOMBIE_DISPLAY:-}" ]; then
    export DISPLAY="$ZOMBIE_DISPLAY"
elif [ -n "${DISPLAY:-}" ]; then
    export DISPLAY="$DISPLAY"
else
    export DISPLAY=":0"
fi

# Suppress Wine crash dialogs and debug prompts
export WINEDBG=disabled
# Ensure we map to the standard Wine prefix if not set
if [ -z "$WINEPREFIX" ]; then
    export WINEPREFIX="$HOME/.wine"
fi
export DRI_PRIME=1
export PROTON_NO_ESYNC=1
export PROTON_NO_FSYNC=1

# EVE client drop-zone (copied from Skynet via rsync)
EVE_DROPZONE="${EVE_DROPZONE:-$HOME/eve-client}"
EVE_EXE="${EVE_EXE:-$EVE_DROPZONE/tq/bin64/exefile.exe}"

# Prefer launching the CCP/Steam launcher if it exists (required for patching/IncompatibleBuild fixes).
EVE_LAUNCHER_EXE="${EVE_LAUNCHER_EXE:-$EVE_DROPZONE/Launcher/evelauncher.exe}"

EVE_TARGET_EXE="$EVE_EXE"
if [ -f "$EVE_LAUNCHER_EXE" ]; then
    EVE_TARGET_EXE="$EVE_LAUNCHER_EXE"
fi
# Force direct launch for stability
# export EVE_TARGET_EXE="$EVE_EXE"

# Keep the Wine prefix inside the drop-zone so potatofy can watch it directly.
export WINEPREFIX="${ZOMBIE_WINEPREFIX:-$EVE_DROPZONE/wineprefix}"

# Fix White Rectangle Occlusion: prefer desktop OpenGL and a managed Wine virtual desktop.
export QT_OPENGL=desktop

# Software Rendering Fallback (Forced by User Request)
export LIBGL_ALWAYS_SOFTWARE=1
export QT_QUICK_BACKEND=software
export QT_OPENGL=software
export ZOMBIE_SOFTWARE_RENDER=1
export ZOMBIE_WINE_DESKTOP_NAME="EVE"
export ZOMBIE_WINE_DESKTOP_RES="1920x1080"
export ZOMBIE_RESET_X=1

# Start the Potato Guardian early so it can hijack settings immediately when the prefix/settings tree is generated.
POTATOFY_SCRIPT="${POTATOFY_SCRIPT:-$HOME/eve-trader/scripts/potatofy.py}"
POTATOFY_TIMEOUT="${POTATOFY_TIMEOUT:-86400}"
POTATOFY_POLL="${POTATOFY_POLL:-0.5}"

if [ -f "$POTATOFY_SCRIPT" ]; then
    if [ -f /tmp/potatofy.pid ] && ps -p "$(cat /tmp/potatofy.pid 2>/dev/null)" >/dev/null 2>&1; then
        echo "[ok] potatofy already running (pid=$(cat /tmp/potatofy.pid))"
    else
        echo "[*] Starting potatofy (root=$WINEPREFIX, poll=${POTATOFY_POLL}s)..."
        nohup python3 "$POTATOFY_SCRIPT" --root "$WINEPREFIX" --timeout "$POTATOFY_TIMEOUT" --poll "$POTATOFY_POLL" > /tmp/potatofy.log 2>&1 &
        echo $! > /tmp/potatofy.pid
        echo "[ok] potatofy started pid=$(cat /tmp/potatofy.pid)"
    fi
else
    echo "[!] potatofy script not found at $POTATOFY_SCRIPT; skipping guardian start."
fi

# 4. Virtual Display Launch (only if needed)
if [ "${ZOMBIE_RESET_X:-0}" = "1" ]; then
    echo "[*] Forcing X reset on $DISPLAY (ZOMBIE_RESET_X=1)..."
    pids="$(pgrep -f "Xvfb $DISPLAY" 2>/dev/null || true)"
    if [ -n "$pids" ]; then
        kill -9 $pids 2>/dev/null || true
    fi
    lock_num="${DISPLAY#:}"
    rm -f "/tmp/.X${lock_num}-lock" 2>/dev/null || true
fi

if command -v xdpyinfo >/dev/null 2>&1 && xdpyinfo -display "$DISPLAY" >/dev/null 2>&1; then
    echo "[ok] X display $DISPLAY is available."
else
    echo "[*] X display $DISPLAY not available; launching Xvfb..."
    # User requested deep reset params: 24-bit depth + 32 padding, GLX, RENDER, noreset
    XVFB_SCREEN="${ZOMBIE_XVFB_SCREEN:-1920x1080x24+32}"
    Xvfb "$DISPLAY" -screen 0 "$XVFB_SCREEN" -extension GLX +extension RENDER -noreset &
    sleep 2
fi

# 5. Window Manager
echo "[*] Launching Openbox..."
openbox &
sleep 1

# 6. EVE Launcher (Steam/Proton)
# Note: User must have installed EVE Online via Steam previously or use valid path
# This is a placeholder command for the actual launch
echo "[*] Ready to launch EVE Online..."

# Wait until the executable exists before attempting a launch.
ZOMBIE_WAIT_EXE_TIMEOUT="${ZOMBIE_WAIT_EXE_TIMEOUT:-14400}"
ZOMBIE_WAIT_EXE_POLL="${ZOMBIE_WAIT_EXE_POLL:-10}"

start_ts="$(date +%s)"
while [ ! -f "$EVE_EXE" ]; do
    now_ts="$(date +%s)"
    elapsed="$((now_ts - start_ts))"
    if [ "$elapsed" -ge "$ZOMBIE_WAIT_EXE_TIMEOUT" ]; then
        echo "[!] Timed out waiting for exefile.exe: $EVE_EXE"
        exit 1
    fi
    echo "[*] Waiting for client upload... ($elapsed/${ZOMBIE_WAIT_EXE_TIMEOUT}s) missing: $EVE_EXE"
    sleep "$ZOMBIE_WAIT_EXE_POLL"
done

echo "[ok] Found EVE executable: $EVE_EXE"

if [ "$EVE_TARGET_EXE" != "$EVE_EXE" ]; then
    echo "[ok] Found EVE launcher: $EVE_TARGET_EXE"
    echo "[*] Using launcher-first boot path."
else
    echo "[*] No launcher found; using direct exefile.exe boot path."
fi

# Wait until rsync is effectively finished by requiring the drop-zone size and exefile mtime to be stable.
ZOMBIE_SYNC_STABLE_SLEEP="${ZOMBIE_SYNC_STABLE_SLEEP:-30}"
ZOMBIE_SYNC_STABLE_PASSES="${ZOMBIE_SYNC_STABLE_PASSES:-2}"

stable_passes=0
prev_size=""
prev_mtime=""

echo "[*] Waiting for file sync to settle (stable passes: $ZOMBIE_SYNC_STABLE_PASSES)..."
while [ "$stable_passes" -lt "$ZOMBIE_SYNC_STABLE_PASSES" ]; do
    cur_size="$(du -sb "$EVE_DROPZONE" 2>/dev/null | awk '{print $1}')"
    cur_mtime="$(stat -c %Y "$EVE_TARGET_EXE" 2>/dev/null || echo '')"

    if [ -n "$cur_size" ] && [ -n "$cur_mtime" ] && [ "$cur_size" = "$prev_size" ] && [ "$cur_mtime" = "$prev_mtime" ]; then
        stable_passes=$((stable_passes + 1))
    else
        stable_passes=0
    fi

    prev_size="$cur_size"
    prev_mtime="$cur_mtime"
    sleep "$ZOMBIE_SYNC_STABLE_SLEEP"
done

echo "[ok] Client files appear stable; proceeding to First Breath launch."

WINE_BIN=""
if command -v wine64 >/dev/null 2>&1; then
    WINE_BIN="wine64"
elif [ -x /usr/lib/wine/wine64 ]; then
    WINE_BIN="/usr/lib/wine/wine64"
elif command -v wine >/dev/null 2>&1; then
    WINE_BIN="wine"
fi

if [ -n "$WINE_BIN" ]; then
    if file -b "$EVE_TARGET_EXE" 2>/dev/null | grep -q "PE32+"; then
        if [ "$WINE_BIN" = "wine" ]; then
            # Some installs (notably Wine Staging from WineHQ) ship a 64-bit `wine` without a separate `wine64` wrapper.
            wine_path="$(command -v wine 2>/dev/null || true)"
            if [ -n "$wine_path" ] && ! file -b "$wine_path" 2>/dev/null | grep -q "64-bit"; then
                echo "[!] $EVE_TARGET_EXE is 64-bit but the available 'wine' does not appear to be 64-bit. Install wine64 and retry."
                exit 1
            fi
        fi
    fi
    if [ -f /tmp/eve_exefile.pid ] && ps -p "$(cat /tmp/eve_exefile.pid 2>/dev/null)" >/dev/null 2>&1; then
        echo "[ok] EVE already running (pid=$(cat /tmp/eve_exefile.pid))"
    else
        LAUNCH_ARGS=()
        if [ "${ZOMBIE_SOFTWARE_RENDER:-0}" = "1" ]; then
            LAUNCH_ARGS+=(--disable-gpu --disable-gpu-compositing)
        fi

        echo "[*] Launching via managed Wine desktop ($ZOMBIE_WINE_DESKTOP_NAME,$ZOMBIE_WINE_DESKTOP_RES)"
        echo "[*] Command: $WINE_BIN explorer /desktop=$ZOMBIE_WINE_DESKTOP_NAME,$ZOMBIE_WINE_DESKTOP_RES $EVE_TARGET_EXE ${LAUNCH_ARGS[*]}"
        nohup "$WINE_BIN" explorer "/desktop=${ZOMBIE_WINE_DESKTOP_NAME},${ZOMBIE_WINE_DESKTOP_RES}" "$EVE_TARGET_EXE" "${LAUNCH_ARGS[@]}" > /tmp/eve_exefile.log 2>&1 &
        echo $! > /tmp/eve_exefile.pid
        echo "[ok] EVE launched pid=$(cat /tmp/eve_exefile.pid) (log: /tmp/eve_exefile.log)"
    fi
else
    echo "[!] wine is not installed; not launching automatically."
    echo "Install and run: wine \"$EVE_EXE\""
fi


# 7. 3D Kill-Switch Daemon
# Disabled: direct input injection removed from software.

# 8. Identity Phase Automation
# - If launcher is open but client isn't running, nudge the "Play" trigger periodically.
# - If EVE_USER/EVE_PASS are set, inject credentials into the login window.
# - When Character Selection appears, send Return to enter the world.

LAUNCHER_CONTROL_SCRIPT="${LAUNCHER_CONTROL_SCRIPT:-$HOME/eve-trader/scripts/launcher_control.py}"
ZOMBIE_CREDS_SCRIPT="${ZOMBIE_CREDS_SCRIPT:-$HOME/eve-trader/scripts/zombie_creds.sh}"
ZOMBIE_OTP_SCRIPT="${ZOMBIE_OTP_SCRIPT:-$HOME/eve-trader/scripts/zombie_otp.sh}"
ZOMBIE_EULA_SCRIPT="${ZOMBIE_EULA_SCRIPT:-$HOME/eve-trader/scripts/clear_prompts.sh}"
ZOMBIE_EULA_BRIDGE_SCRIPT="${ZOMBIE_EULA_BRIDGE_SCRIPT:-$HOME/eve-trader/scripts/clear_prompts_bridge.sh}"

ZOMBIE_LAUNCHER_POLL_SECONDS="${ZOMBIE_LAUNCHER_POLL_SECONDS:-60}"
ZOMBIE_CREDS_POLL_SECONDS="${ZOMBIE_CREDS_POLL_SECONDS:-5}"

ZOMBIE_CLIENT_PROC_PATTERN="${ZOMBIE_CLIENT_PROC_PATTERN:-exefile.exe}"
ZOMBIE_LAUNCHER_WINDOW_PATTERN="${ZOMBIE_LAUNCHER_WINDOW_PATTERN:-EVE Launcher}"
ZOMBIE_LOGIN_WINDOW_PATTERN="${ZOMBIE_LOGIN_WINDOW_PATTERN:-EVE - |EVE Online|Login}"
ZOMBIE_CHARSEL_WINDOW_PATTERN="${ZOMBIE_CHARSEL_WINDOW_PATTERN:-Character Selection|Select Character}"
ZOMBIE_OTP_BRIDGE="${ZOMBIE_OTP_BRIDGE:-1}"
ZOMBIE_EULA_BRIDGE="${ZOMBIE_EULA_BRIDGE:-1}"
ZOMBIE_EULA_REDIS_KEY="${ZOMBIE_EULA_REDIS_KEY:-system:zombie:eula_accept}"

if [ "${ZOMBIE_DISABLE_LAUNCHER_CONTROL:-0}" = "1" ]; then
    echo "[ok] launcher_control disabled (ZOMBIE_DISABLE_LAUNCHER_CONTROL=1)."
elif [ -f "$LAUNCHER_CONTROL_SCRIPT" ] && [ "$EVE_TARGET_EXE" != "$EVE_EXE" ]; then
    if [ -f /tmp/launcher_control.pid ] && ps -p "$(cat /tmp/launcher_control.pid 2>/dev/null)" >/dev/null 2>&1; then
        echo "[ok] launcher_control already running (pid=$(cat /tmp/launcher_control.pid))"
    else
        echo "[*] Starting launcher_control loop (poll=${ZOMBIE_LAUNCHER_POLL_SECONDS}s)..."
        nohup python3 "$LAUNCHER_CONTROL_SCRIPT" \
            --loop \
            --interval-seconds "$ZOMBIE_LAUNCHER_POLL_SECONDS" \
            --launcher-window-pattern "$ZOMBIE_LAUNCHER_WINDOW_PATTERN" \
            --client-proc-pattern "$ZOMBIE_CLIENT_PROC_PATTERN" \
            --click \
            > /tmp/launcher_control.log 2>&1 &
        echo $! > /tmp/launcher_control.pid
        echo "[ok] launcher_control started pid=$(cat /tmp/launcher_control.pid)"
    fi
else
    echo "[!] launcher_control script not found or launcher not in use; skipping Play automation."
fi

if [ -f "$ZOMBIE_CREDS_SCRIPT" ]; then
    if [ -n "${EVE_USER:-}" ] && [ -n "${EVE_PASS:-}" ]; then
        if [ -f /tmp/zombie_creds.pid ] && ps -p "$(cat /tmp/zombie_creds.pid 2>/dev/null)" >/dev/null 2>&1; then
            echo "[ok] zombie_creds already running (pid=$(cat /tmp/zombie_creds.pid))"
        else
            echo "[*] Starting zombie_creds loop (poll=${ZOMBIE_CREDS_POLL_SECONDS}s)..."
            nohup "$ZOMBIE_CREDS_SCRIPT" \
                --loop \
                --poll-seconds "$ZOMBIE_CREDS_POLL_SECONDS" \
                --window-pattern "$ZOMBIE_LOGIN_WINDOW_PATTERN" \
                > /tmp/zombie_creds.log 2>&1 &
            echo $! > /tmp/zombie_creds.pid
            echo "[ok] zombie_creds started pid=$(cat /tmp/zombie_creds.pid)"
        fi
    else
        echo "[!] EVE_USER/EVE_PASS not set; skipping credential injection."
    fi
else
    echo "[!] zombie_creds script not found; skipping credential injection."
fi

if [ "$ZOMBIE_OTP_BRIDGE" = "1" ] && [ -f "$ZOMBIE_OTP_SCRIPT" ]; then
    if [ -f /tmp/zombie_otp.pid ] && ps -p "$(cat /tmp/zombie_otp.pid 2>/dev/null)" >/dev/null 2>&1; then
        echo "[ok] zombie_otp bridge already running (pid=$(cat /tmp/zombie_otp.pid))"
    else
        echo "[*] Starting OTP bridge loop (Redis key: system:zombie:otp)..."
        nohup "$ZOMBIE_OTP_SCRIPT" --loop > /tmp/zombie_otp.log 2>&1 &
        echo $! > /tmp/zombie_otp.pid
        echo "[ok] zombie_otp bridge started pid=$(cat /tmp/zombie_otp.pid)"
    fi
else
    echo "[!] OTP bridge disabled or script missing; skipping."
fi

if [ "$ZOMBIE_EULA_BRIDGE" = "1" ] && [ -f "$ZOMBIE_EULA_SCRIPT" ] && [ -f "$ZOMBIE_EULA_BRIDGE_SCRIPT" ]; then
        if [ -f /tmp/zombie_eula.pid ] && ps -p "$(cat /tmp/zombie_eula.pid 2>/dev/null)" >/dev/null 2>&1; then
                echo "[ok] zombie_eula bridge already running (pid=$(cat /tmp/zombie_eula.pid))"
        else
                echo "[*] Starting EULA bridge loop (Redis key: $ZOMBIE_EULA_REDIS_KEY)..."
                # Uses stdlib-only Redis I/O (no redis-cli / redis-py needed).
                REDIS_KEY="$ZOMBIE_EULA_REDIS_KEY" DISPLAY="$DISPLAY" SWEEP_SCRIPT="$ZOMBIE_EULA_SCRIPT" \
                    nohup "$ZOMBIE_EULA_BRIDGE_SCRIPT" --loop > /tmp/zombie_eula.log 2>&1 &
                echo $! > /tmp/zombie_eula.pid
                echo "[ok] zombie_eula bridge started pid=$(cat /tmp/zombie_eula.pid)"
        fi
else
        echo "[!] EULA bridge disabled or script missing; skipping."
fi


# Character selection automation disabled: direct input injection removed from software.

echo "[ok] Zombie Node Initialized. Display $DISPLAY active."
