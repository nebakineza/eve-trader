#!/usr/bin/env python3
"""launcher_control.py

Automation helper for the EVE Launcher running under Wine on a headless X display.

Goal: keep nudging the launcher "Play" action until the client is actually running.

Design notes:
- The launcher UI is often custom; pixel-perfect detection is unreliable.
- We default to a safe heuristic: focus launcher window, then either:
  - press Return (many launchers map default button to Enter)
  - or click near the center-bottom of the window (common "Play" placement)

This script is intentionally conservative: it only triggers when the client is NOT
running, and the launcher window IS present.

Environment:
- DISPLAY: X display to target (e.g. :1)

Examples:
  DISPLAY=:1 python3 scripts/launcher_control.py --loop
  DISPLAY=:1 python3 scripts/launcher_control.py --click --click-x 0.5 --click-y 0.92 --loop
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class ClickSpec:
    # Relative coordinates in the window (0..1)
    rel_x: float
    rel_y: float


def _run(cmd: list[str], *, env: dict[str, str] | None = None, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=check)


def _which(name: str) -> str:
    p = _run(["bash", "-lc", f"command -v {shlex.quote(name)}"]).stdout.strip()
    if not p:
        raise FileNotFoundError(f"Required tool not found: {name}")
    return p


def _pgrep_any(patterns: list[str]) -> bool:
    # Use pgrep -af so we can match Windows exe strings embedded in Wine.
    for pat in patterns:
        cp = _run(["pgrep", "-af", pat])
        if cp.returncode == 0 and cp.stdout.strip():
            return True
    return False


def _find_window_ids(name_patterns: list[str]) -> list[str]:
    ids: list[str] = []
    for pat in name_patterns:
        cp = _run(["xdotool", "search", "--name", pat])
        if cp.returncode == 0:
            for line in cp.stdout.splitlines():
                line = line.strip()
                if line and line.isdigit():
                    ids.append(line)
    # De-dupe while preserving order
    seen = set()
    out: list[str] = []
    for wid in ids:
        if wid not in seen:
            seen.add(wid)
            out.append(wid)
    return out


def _window_geometry(window_id: str) -> tuple[int, int, int, int] | None:
    # Returns (x, y, width, height)
    cp = _run(["xdotool", "getwindowgeometry", "--shell", window_id])
    if cp.returncode != 0:
        return None
    vals: dict[str, int] = {}
    for line in cp.stdout.splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip()
        if k in {"X", "Y", "WIDTH", "HEIGHT"}:
            try:
                vals[k] = int(v)
            except ValueError:
                pass
    if all(k in vals for k in ("X", "Y", "WIDTH", "HEIGHT")):
        return vals["X"], vals["Y"], vals["WIDTH"], vals["HEIGHT"]
    return None


def _activate(window_id: str) -> None:
    _run(["xdotool", "windowactivate", "--sync", window_id])


def _press_return(window_id: str) -> None:
    _activate(window_id)
    _run(["xdotool", "key", "--clearmodifiers", "Return"])


def _click(window_id: str, click: ClickSpec) -> None:
    geo = _window_geometry(window_id)
    if not geo:
        return
    _, _, w, h = geo
    x = max(0, min(w - 1, int(w * click.rel_x)))
    y = max(0, min(h - 1, int(h * click.rel_y)))
    _activate(window_id)
    _run(["xdotool", "mousemove", "--window", window_id, str(x), str(y)])
    _run(["xdotool", "click", "1"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Automate EVE Launcher Play trigger via xdotool")
    parser.add_argument("--loop", action="store_true", help="Run forever")
    parser.add_argument("--interval-seconds", type=int, default=60)

    parser.add_argument(
        "--launcher-window-pattern",
        action="append",
        default=["EVE Launcher"],
        help="Regex-like pattern passed to xdotool --name (can be repeated)",
    )

    parser.add_argument(
        "--client-proc-pattern",
        action="append",
        default=["exefile.exe", "eve-online.exe"],
        help="Process substring/regex for pgrep -af indicating client is running (can be repeated)",
    )

    parser.add_argument("--press-return", action="store_true", default=True)
    parser.add_argument("--no-press-return", dest="press_return", action="store_false")

    parser.add_argument("--click", action="store_true", help="Also click window at relative coords")
    parser.add_argument("--click-x", type=float, default=0.50)
    parser.add_argument("--click-y", type=float, default=0.92)

    args = parser.parse_args()

    _which("xdotool")
    _which("pgrep")

    interval = max(5, int(args.interval_seconds))
    click_spec = ClickSpec(rel_x=float(args.click_x), rel_y=float(args.click_y))

    def tick() -> None:
        if _pgrep_any(list(args.client_proc_pattern)):
            print("[launcher_control] client appears running; no action")
            return

        wids = _find_window_ids(list(args.launcher_window_pattern))
        if not wids:
            print("[launcher_control] launcher window not found; no action")
            return

        wid = wids[0]
        print(f"[launcher_control] client not running; nudging launcher wid={wid}")
        if args.press_return:
            _press_return(wid)
        if args.click:
            _click(wid, click_spec)

    if not args.loop:
        tick()
        return 0

    print(f"[launcher_control] loop started (interval={interval}s display={os.getenv('DISPLAY','')})")
    while True:
        try:
            tick()
        except Exception as e:
            print(f"[launcher_control] error: {e}")
        time.sleep(interval)


if __name__ == "__main__":
    raise SystemExit(main())
