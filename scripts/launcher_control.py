#!/usr/bin/env python3
"""launcher_control.py - Neural Interface Edition

Utilizes the iGPU-accelerated Visual Cortex to intelligently manage
the EVE Launcher states (Play, Update, Verify).
"""

import argparse
import sys
import os
import time
import subprocess
import logging
import re
import socket
from urllib.parse import urlparse

# Path Hygiene: Add project root to path to reach nexus
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

VisualCortex = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [launcher_control] - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _parse_redis_url(redis_url: str) -> tuple[str, int, int]:
    parsed = urlparse(redis_url)
    if parsed.scheme not in ("redis", "rediss"):
        raise ValueError(f"Unsupported redis url scheme: {parsed.scheme}")
    host = parsed.hostname or "127.0.0.1"
    port = int(parsed.port or 6379)
    db = 0
    if parsed.path and parsed.path != "/":
        try:
            db = int(parsed.path.lstrip("/"))
        except ValueError:
            db = 0
    return host, port, db


def _read_line(sock: socket.socket) -> bytes:
    data = bytearray()
    while True:
        chunk = sock.recv(1)
        if not chunk:
            raise ConnectionError("Redis connection closed")
        data += chunk
        if data.endswith(b"\r\n"):
            return bytes(data)


def _send_resp_array(sock: socket.socket, parts: list[bytes]) -> None:
    out = bytearray()
    out += f"*{len(parts)}\r\n".encode("ascii")
    for p in parts:
        out += f"${len(p)}\r\n".encode("ascii")
        out += p
        out += b"\r\n"
    sock.sendall(out)


def redis_set(redis_url: str, key: str, value: str) -> None:
    host, port, db = _parse_redis_url(redis_url)
    with socket.create_connection((host, port), timeout=3) as sock:
        if db:
            _send_resp_array(sock, [b"SELECT", str(db).encode("ascii")])
            resp = _read_line(sock)
            if not resp.startswith(b"+"):
                raise RuntimeError(f"Redis SELECT failed: {resp!r}")

        _send_resp_array(sock, [b"SET", key.encode("utf-8"), value.encode("utf-8")])
        resp = _read_line(sock)
        if not resp.startswith(b"+"):
            raise RuntimeError(f"Redis SET failed: {resp!r}")


def redis_exists(redis_url: str, key: str) -> bool:
    host, port, db = _parse_redis_url(redis_url)
    with socket.create_connection((host, port), timeout=3) as sock:
        if db:
            _send_resp_array(sock, [b"SELECT", str(db).encode("ascii")])
            resp = _read_line(sock)
            if not resp.startswith(b"+"):
                raise RuntimeError(f"Redis SELECT failed: {resp!r}")

        _send_resp_array(sock, [b"EXISTS", key.encode("utf-8")])
        resp = _read_line(sock)
        if not resp.startswith(b":"):
            raise RuntimeError(f"Redis EXISTS failed: {resp!r}")
        try:
            return int(resp[1:].strip()) > 0
        except Exception:
            return False


def redis_ttl(redis_url: str, key: str) -> int:
    host, port, db = _parse_redis_url(redis_url)
    with socket.create_connection((host, port), timeout=3) as sock:
        if db:
            _send_resp_array(sock, [b"SELECT", str(db).encode("ascii")])
            resp = _read_line(sock)
            if not resp.startswith(b"+"):
                raise RuntimeError(f"Redis SELECT failed: {resp!r}")

        _send_resp_array(sock, [b"TTL", key.encode("utf-8")])
        resp = _read_line(sock)
        if not resp.startswith(b":"):
            raise RuntimeError(f"Redis TTL failed: {resp!r}")
        try:
            return int(resp[1:].strip())
        except Exception:
            return -3


def redis_set_nx_ex(redis_url: str, key: str, value: str, ttl_seconds: int) -> bool:
    """SET key value NX EX ttl_seconds.

    Returns True if the key was set, False if it already existed.
    """
    host, port, db = _parse_redis_url(redis_url)
    with socket.create_connection((host, port), timeout=3) as sock:
        if db:
            _send_resp_array(sock, [b"SELECT", str(db).encode("ascii")])
            resp = _read_line(sock)
            if not resp.startswith(b"+"):
                raise RuntimeError(f"Redis SELECT failed: {resp!r}")

        _send_resp_array(
            sock,
            [
                b"SET",
                key.encode("utf-8"),
                value.encode("utf-8"),
                b"NX",
                b"EX",
                str(int(ttl_seconds)).encode("ascii"),
            ],
        )
        resp = _read_line(sock)
        if resp.startswith(b"+OK"):
            return True
        # When NX fails, Redis returns a Null Bulk String in RESP2: $-1\r\n
        if resp.startswith(b"$-1"):
            return False
        raise RuntimeError(f"Redis SET NX EX failed: {resp!r}")


def queue_click(redis_url: str, x: int, y: int) -> bool:
    """Queue a one-shot click, with spam protection.

    - Won't overwrite a pending click if the bridge hasn't consumed it yet.
    - Applies a Redis-backed cooldown so we don't enqueue the same click every loop
      while the Play button remains visible.
    """
    click_key = "system:hardware:click"
    # Default keeps logs sane without changing behavior too aggressively.
    cooldown_s = int(os.getenv("PLAY_CLICK_COOLDOWN_SECONDS", "60") or "60")
    cooldown_key = "system:hardware:click:cooldown"

    # If the bridge hasn't consumed the previous click, don't overwrite it.
    try:
        if redis_exists(redis_url, click_key):
            return False
    except Exception:
        # If EXISTS fails for some reason, fall back to current behavior.
        pass

    if cooldown_s > 0:
        try:
            ok = redis_set_nx_ex(redis_url, cooldown_key, str(int(time.time())), cooldown_s)
            if not ok:
                return False
        except Exception:
            # If cooldown SET fails, still attempt to queue click (better to act than to stall).
            pass

    # One-shot semantics: arduino-bridge consumes by DEL.
    redis_set(redis_url, click_key, f"{int(x)},{int(y)}")
    return True

def is_client_running():
    # Safer check for exefile.exe or eve-online.exe
    # Returns 0 (True) if found, 1 (False) if not
    try:
        # pgrep -f matches the full command line
        # We assume if 'exefile.exe' or 'eve-online.exe' is running, the client is up.
        # We suppress output.
        ret = subprocess.call(["pgrep", "-f", "exefile.exe|eve-online.exe"], stdout=subprocess.DEVNULL)
        return ret == 0
    except Exception:
        return False


def is_client_running_pattern(pattern: str) -> bool:
    try:
        out = subprocess.check_output(["pgrep", "-af", str(pattern)], stderr=subprocess.DEVNULL)
        me = str(os.getpid())
        for line in out.decode("utf-8", errors="ignore").splitlines():
            # pgrep -af format: "<pid> <cmdline>"
            parts = line.strip().split(" ", 1)
            if not parts:
                continue
            pid = parts[0]
            cmdline = parts[1] if len(parts) > 1 else ""
            if pid == me:
                continue
            if "launcher_control.py" in cmdline:
                continue
            return True
        return False
    except subprocess.CalledProcessError:
        return False
    except Exception:
        return False


def is_launcher_detected(*, display: str, pattern: str) -> bool:
    """Best-effort: check X11 window tree for launcher window pattern."""
    try:
        env = os.environ.copy()
        env["DISPLAY"] = display
        out = subprocess.check_output(["xwininfo", "-root", "-tree"], env=env, stderr=subprocess.DEVNULL, timeout=3)
        decoded = out.decode("utf-8", errors="ignore")
        return re.search(pattern, decoded, flags=re.IGNORECASE) is not None
    except Exception:
        return False


def find_launcher_window_id(*, display: str, pattern: str) -> str | None:
    """Return the first matching X11 window id (hex string like 0x3e00007) for the launcher."""
    try:
        env = os.environ.copy()
        env["DISPLAY"] = display
        out = subprocess.check_output(["xwininfo", "-root", "-tree"], env=env, stderr=subprocess.DEVNULL, timeout=3)
        decoded = out.decode("utf-8", errors="ignore")
        for line in decoded.splitlines():
            if re.search(pattern, line, flags=re.IGNORECASE):
                m = re.search(r"\b0x[0-9a-fA-F]+\b", line)
                if m:
                    return m.group(0)
        return None
    except Exception:
        return None


def get_window_region(*, display: str, window_id: str) -> tuple[int, int, int, int] | None:
    """Return (x, y, w, h) for an X11 window id using xwininfo."""
    try:
        env = os.environ.copy()
        env["DISPLAY"] = display
        out = subprocess.check_output(["xwininfo", "-id", str(window_id)], env=env, stderr=subprocess.DEVNULL, timeout=3)
        decoded = out.decode("utf-8", errors="ignore")

        def _find_int(label: str) -> int | None:
            m = re.search(rf"^\s*{re.escape(label)}\s*:\s*(-?\d+)\s*$", decoded, flags=re.MULTILINE)
            return int(m.group(1)) if m else None

        x = _find_int("Absolute upper-left X")
        y = _find_int("Absolute upper-left Y")
        w = _find_int("Width")
        h = _find_int("Height")
        if x is None or y is None or w is None or h is None:
            return None
        if w <= 0 or h <= 0:
            return None
        return int(x), int(y), int(w), int(h)
    except Exception:
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--display", default=os.getenv("DISPLAY", ":0"))
    parser.add_argument("--interval", type=int, default=5, help="Legacy alias for --interval-seconds")
    parser.add_argument("--interval-seconds", type=int, default=None)
    parser.add_argument("--launcher-window-pattern", default=os.getenv("ZOMBIE_LAUNCHER_WINDOW_PATTERN", "EVE Launcher"))
    parser.add_argument("--client-proc-pattern", default=os.getenv("ZOMBIE_CLIENT_PROC_PATTERN", "exefile.exe"))
    parser.add_argument("--click", action="store_true", help="Actually click PLAY when detected (hardware-only).")
    parser.add_argument("--redis-url", default=os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0"))
    args = parser.parse_args()

    interval_seconds = int(args.interval_seconds) if args.interval_seconds is not None else int(args.interval)
    interval_seconds = max(1, int(interval_seconds))
    
    vc = None
    try:
        from nexus.automaton.visual_cortex import VisualCortex as _VisualCortex

        vc = _VisualCortex(display=str(args.display))
        logger.info("VisualCortex initialized (Play Now detection enabled).")
    except Exception as e:
        logger.warning(
            "VisualCortex unavailable (Play Now detection disabled). Reason: %s",
            e,
            exc_info=True,
        )
    
    client_was_running = False

    last_state = None
    last_launcher_present = None
    last_launcher_window_id = None
    last_launcher_region = None

    while True:
        try:
            # 1. Check client status
            if is_client_running_pattern(str(args.client_proc_pattern)):
                if not client_was_running:
                    logger.info("Client Launch Detected!")
                    client_was_running = True
                
                logger.info("Game Client Active. Standing by.")
                if not args.loop: break
                time.sleep(interval_seconds)
                continue
            
            # Client is not running, so we must be in launcher context
            client_was_running = False 

            launcher_present = is_launcher_detected(display=str(args.display), pattern=str(args.launcher_window_pattern))
            if launcher_present != last_launcher_present:
                if launcher_present:
                    logger.info("Launcher detected (pattern=%r). Waiting for Play Now to become available...", args.launcher_window_pattern)
                else:
                    logger.warning("Launcher NOT detected (pattern=%r). Standing by...", args.launcher_window_pattern)
                last_launcher_present = launcher_present

                # Refresh capture region whenever launcher presence flips.
                last_launcher_window_id = None
                last_launcher_region = None

            # If VisualCortex is available and the launcher is present, try to
            # capture the launcher window region (so screenshots include the Play button).
            if vc is not None and launcher_present:
                if last_launcher_window_id is None:
                    last_launcher_window_id = find_launcher_window_id(
                        display=str(args.display),
                        pattern=str(args.launcher_window_pattern),
                    )
                    if last_launcher_window_id:
                        logger.info("Launcher window id=%s", last_launcher_window_id)

                if last_launcher_window_id and last_launcher_region is None:
                    last_launcher_region = get_window_region(
                        display=str(args.display),
                        window_id=str(last_launcher_window_id),
                    )
                    if last_launcher_region:
                        x, y, w, h = last_launcher_region
                        logger.info("Launcher capture region=%d,%d %dx%d", x, y, w, h)
                        try:
                            vc.set_capture_region(last_launcher_region)
                        except Exception:
                            pass
                elif last_launcher_region is None:
                    # Fall back to full-screen capture.
                    try:
                        vc.set_capture_region(None)
                    except Exception:
                        pass
            
            # 2. Analyze Launcher State
            if vc is None:
                if launcher_present:
                    logger.info("STATE: WAITING_FOR_PLAY_NOW. Launcher present; Play Now detection unavailable.")
                else:
                    logger.debug("STATE: UNKNOWN/IDLE. Scanning...")
            else:
                state, coords = vc.analyze_state()

                if state != last_state:
                    logger.info("Launcher VisualCortex state=%s", state)
                    last_state = state

                if state == vc.STATE_PLAY_READY and coords:
                    cx, cy = coords
                    logger.info(f"STATE: PLAY_READY. {'Engaging' if args.click else 'Detected'} at ({cx}, {cy})...")
                    if args.click:
                        try:
                            queued = queue_click(str(args.redis_url), cx, cy)
                            if queued:
                                logger.info("Queued hardware click via Redis (system:hardware:click)")
                            else:
                                # Avoid log spam while still making it debuggable.
                                logger.debug("Click not queued (pending click or cooldown active)")
                        except Exception as e:
                            logger.warning("Failed to queue click via Redis: %s", e)
                    # Wait for response before looping immediately
                    time.sleep(15)

                elif state == vc.STATE_UPDATE_REQUIRED:
                    # We are purging yellow hallucinations, so we ignore this or log only
                    logger.warning(
                        "STATE: UPDATE_REQUIRED detected but update-click is DISABLED (Purge Hallucination Mode)."
                    )

                elif state == vc.STATE_VERIFYING:
                    logger.info("STATE: VERIFYING. Waiting for completion...")

                else:
                    if launcher_present:
                        logger.info("STATE: WAITING_FOR_PLAY_NOW. Launcher present; Play Now not ready yet.")
                    else:
                        logger.debug("STATE: UNKNOWN/IDLE. Scanning...")

        except KeyboardInterrupt:
            logger.info("Manual Interrupt.")
            break
        except Exception as e:
            logger.error(f"Loop error: {e}")
            time.sleep(5)
        
        if not args.loop:
            break
        time.sleep(interval_seconds)

if __name__ == "__main__":
    main()
