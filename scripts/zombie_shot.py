import argparse
import base64
import logging
import os
import shutil
import socket
import subprocess
import time
from dataclasses import dataclass
from urllib.parse import urlparse

import re


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ZombieShot")


@dataclass(frozen=True)
class RedisTarget:
    host: str
    port: int
    db: int


def _parse_redis_url(redis_url: str) -> RedisTarget:
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
    return RedisTarget(host=host, port=port, db=db)


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


def redis_set(redis_url: str, key: str, value: bytes) -> None:
    target = _parse_redis_url(redis_url)
    with socket.create_connection((target.host, target.port), timeout=5) as sock:
        if target.db:
            _send_resp_array(sock, [b"SELECT", str(target.db).encode("ascii")])
            resp = _read_line(sock)
            if not resp.startswith(b"+"):
                raise RuntimeError(f"Redis SELECT failed: {resp!r}")

        _send_resp_array(sock, [b"SET", key.encode("utf-8"), value])
        resp = _read_line(sock)
        if not resp.startswith(b"+"):
            raise RuntimeError(f"Redis SET failed: {resp!r}")


def _require_tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise FileNotFoundError(f"Required tool not found in PATH: {name}")
    return path


def _detect_display(*, candidates: list[str], patterns: list[str]) -> str | None:
    xwininfo = shutil.which("xwininfo")
    if not xwininfo:
        logger.info("xwininfo not found; cannot auto-detect DISPLAY")
        return None

    rx = re.compile("|".join(patterns), flags=re.IGNORECASE)
    for display in candidates:
        disp = display.strip()
        if not disp:
            continue
        try:
            out = subprocess.check_output(
                [xwininfo, "-display", disp, "-root", "-tree"],
                stderr=subprocess.DEVNULL,
                timeout=3,
            )
            decoded = out.decode("utf-8", errors="ignore")
            if rx.search(decoded):
                logger.info("Auto-selected DISPLAY=%s (matched window name)", disp)
                return disp
        except Exception:
            continue
    return None


def _wayland_env_for_current_user() -> dict[str, str] | None:
    xdg_runtime = os.environ.get("XDG_RUNTIME_DIR")
    if not xdg_runtime:
        xdg_runtime = f"/run/user/{os.getuid()}"

    if not os.path.isdir(xdg_runtime):
        return None

    # Prefer wayland-0, else any wayland-* socket.
    preferred = os.path.join(xdg_runtime, "wayland-0")
    wayland_display = None
    if os.path.exists(preferred):
        wayland_display = "wayland-0"
    else:
        try:
            for name in os.listdir(xdg_runtime):
                if name.startswith("wayland-") and os.path.exists(os.path.join(xdg_runtime, name)):
                    wayland_display = name
                    break
        except Exception:
            wayland_display = None

    if not wayland_display:
        return None

    return {
        "XDG_RUNTIME_DIR": xdg_runtime,
        "WAYLAND_DISPLAY": wayland_display,
    }


def _capture_x11(*, display: str, png_path: str) -> None:
    scrot = _require_tool("scrot")

    env = os.environ.copy()
    env["DISPLAY"] = display

    # scrot will not overwrite an existing output file unless -o/--overwrite is used.
    subprocess.run([scrot, "-o", png_path], env=env, check=True)


def _capture_wayland(*, png_path: str) -> None:
    grim = _require_tool("grim")

    env = os.environ.copy()
    wl_env = _wayland_env_for_current_user()
    if not wl_env:
        raise RuntimeError("No Wayland runtime detected (missing XDG_RUNTIME_DIR/wayland-* socket)")
    env.update(wl_env)

    # Full-screen capture to PNG.
    subprocess.run([grim, png_path], env=env, check=True)


def _convert_png_to_jpg(*, png_path: str, jpg_path: str, jpg_quality: int) -> None:
    convert = _require_tool("convert")

    # Resize + compress to low-bandwidth JPEG
    # Spec calls for 640x360.
    subprocess.run(
        [
            convert,
            png_path,
            "-resize",
            "640x360!",
            "-quality",
            str(int(jpg_quality)),
            jpg_path,
        ],
        check=True,
    )


def capture_once(
    *,
    display: str,
    out_dir: str,
    redis_url: str,
    redis_key: str,
    jpg_quality: int,
    min_jpg_bytes: int,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, "zombie_live.png")
    jpg_path = os.path.join(out_dir, "zombie_live.jpg")

    last_err: Exception | None = None
    captured_via = None

    # Strategy: if DISPLAY=:0, try Wayland first (most likely for real client).
    # If that fails or produces a tiny frame, fall back to X11.
    # If DISPLAY is not :0, use X11 as primary.
    prefer_wayland_first = str(display).strip() == ":0"

    if prefer_wayland_first:
        try:
            _capture_wayland(png_path=png_path)
            _convert_png_to_jpg(png_path=png_path, jpg_path=jpg_path, jpg_quality=jpg_quality)
            raw = open(jpg_path, "rb").read()
            if min_jpg_bytes > 0 and len(raw) < int(min_jpg_bytes):
                raise RuntimeError(f"Wayland capture too small ({len(raw)} bytes)")
            captured_via = "wayland"
            logger.info("Captured via Wayland")
        except Exception as e:
            last_err = e
            logger.info("Wayland capture failed: %s; trying X11 fallback", e)

    if captured_via is None:
        try:
            if str(display).startswith(":"):
                _capture_x11(display=display, png_path=png_path)
                _convert_png_to_jpg(png_path=png_path, jpg_path=jpg_path, jpg_quality=jpg_quality)
                raw = open(jpg_path, "rb").read()
                if min_jpg_bytes > 0 and len(raw) < int(min_jpg_bytes):
                    raise RuntimeError(f"X11 capture too small ({len(raw)} bytes)")
                captured_via = "x11"
                logger.info("Captured via X11")
            else:
                raise ValueError(f"Unsupported DISPLAY value: {display!r}")
        except Exception as e:
            last_err = e

    # Final fallback: if X11 failed or produced tiny frame, try Wayland once more.
    if captured_via is None and not prefer_wayland_first:
        try:
            _capture_wayland(png_path=png_path)
            _convert_png_to_jpg(png_path=png_path, jpg_path=jpg_path, jpg_quality=jpg_quality)
            raw = open(jpg_path, "rb").read()
            if min_jpg_bytes > 0 and len(raw) < int(min_jpg_bytes):
                raise RuntimeError(f"Wayland capture too small ({len(raw)} bytes)")
            captured_via = "wayland_fallback"
            logger.info("Captured via Wayland (fallback)")
        except Exception as e:
            last_err = last_err or e

    if captured_via is None:
        raise RuntimeError(f"No screenshot captured (last error: {last_err})")

    if not os.path.exists(jpg_path):
        raise RuntimeError(f"Screenshot file missing after capture (via={captured_via})")

    with open(jpg_path, "rb") as f:
        raw = f.read()

    if min_jpg_bytes > 0 and len(raw) < int(min_jpg_bytes):
        raise RuntimeError(f"Screenshot too small ({len(raw)} bytes < {int(min_jpg_bytes)}); likely empty frame")

    # Proof artifact requested by ops: keep a stable filename for quick verification.
    # Write a higher-quality, full-resolution JPEG so it's easier to confirm we're seeing a real frame.
    try:
        proof_path = os.path.join(out_dir, "grim_test.jpg")
        convert = _require_tool("convert")
        subprocess.run([convert, png_path, "-quality", "92", proof_path], check=True)
    except Exception:
        pass

    b64 = base64.b64encode(raw)
    redis_set(redis_url, redis_key, b64)

    logger.info("Published screenshot: %d bytes -> %d b64 bytes (via %s)", len(raw), len(b64), captured_via)


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture DISPLAY framebuffer and publish screenshot to Redis.")
    parser.add_argument("--display", default=os.getenv("DISPLAY", ":0"))
    parser.add_argument("--auto-display", action="store_true")
    parser.add_argument(
        "--display-candidates",
        default=os.getenv("ZOMBIE_SHOT_DISPLAY_CANDIDATES", ":1,:0"),
        help="Comma-separated DISPLAY candidates to probe when --auto-display is set.",
    )
    parser.add_argument("--out-dir", default=os.getenv("ZOMBIE_SHOT_OUT_DIR", "/tmp"))
    parser.add_argument("--redis-url", default=os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0"))
    parser.add_argument("--redis-key", default=os.getenv("ZOMBIE_SHOT_REDIS_KEY", "system:zombie:screenshot"))
    parser.add_argument("--interval-seconds", type=int, default=int(os.getenv("ZOMBIE_SHOT_INTERVAL_SECONDS", "60")))
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--jpg-quality", type=int, default=int(os.getenv("ZOMBIE_SHOT_JPG_QUALITY", "55")))
    parser.add_argument(
        "--min-jpg-bytes",
        type=int,
        default=int(os.getenv("ZOMBIE_SHOT_MIN_JPG_BYTES", "51200")),
        help="Reject screenshots smaller than this many JPEG bytes (default: 51200).",
    )

    args = parser.parse_args()

    if args.auto_display:
        candidates = [c.strip() for c in str(args.display_candidates).split(",") if c.strip()]
        # Added "exefile", "explorer" to patterns to catch the main game window on :0
        detected = _detect_display(candidates=candidates, patterns=["EVE", "Wine", "exefile", "explorer"])
        if detected:
            args.display = detected
    
    # Fallback: if we are on Linux and no display manually enforced, prefer :0 if detection failed
    if args.display == ":0" and not os.path.exists("/tmp/.X11-unix/X0"):
        # This check is weak, but effectively if we defaulted to :0 and it's missing, try :1
        pass


    if not args.loop:
        capture_once(
            display=args.display,
            out_dir=args.out_dir,
            redis_url=args.redis_url,
            redis_key=args.redis_key,
            jpg_quality=args.jpg_quality,
            min_jpg_bytes=args.min_jpg_bytes,
        )
        return 0

    interval = max(10, int(args.interval_seconds))
    logger.info("ZombieShot loop started (display=%s interval=%ss)", args.display, interval)
    while True:
        try:
            capture_once(
                display=args.display,
                out_dir=args.out_dir,
                redis_url=args.redis_url,
                redis_key=args.redis_key,
                jpg_quality=args.jpg_quality,
                min_jpg_bytes=args.min_jpg_bytes,
            )
        except Exception as e:
            logger.error("Capture failed: %s", e)
        time.sleep(interval)


if __name__ == "__main__":
    raise SystemExit(main())
