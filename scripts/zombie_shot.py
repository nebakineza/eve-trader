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


def capture_once(*, display: str, out_dir: str, redis_url: str, redis_key: str, jpg_quality: int) -> None:
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, "zombie_live.png")
    jpg_path = os.path.join(out_dir, "zombie_live.jpg")

    scrot = _require_tool("scrot")
    convert = _require_tool("convert")

    env = os.environ.copy()
    env["DISPLAY"] = display

    # 1) Grab framebuffer
    # scrot will not overwrite an existing output file unless -o/--overwrite is used.
    subprocess.run([scrot, "-o", png_path], env=env, check=True)

    # 2) Resize + compress to low-bandwidth JPEG
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

    with open(jpg_path, "rb") as f:
        raw = f.read()

    b64 = base64.b64encode(raw)
    redis_set(redis_url, redis_key, b64)

    logger.info("Published screenshot: %d bytes -> %d b64 bytes", len(raw), len(b64))


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture DISPLAY framebuffer and publish screenshot to Redis.")
    parser.add_argument("--display", default=os.getenv("DISPLAY", ":0"))
    parser.add_argument("--out-dir", default=os.getenv("ZOMBIE_SHOT_OUT_DIR", "/tmp"))
    parser.add_argument("--redis-url", default=os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0"))
    parser.add_argument("--redis-key", default=os.getenv("ZOMBIE_SHOT_REDIS_KEY", "system:zombie:screenshot"))
    parser.add_argument("--interval-seconds", type=int, default=int(os.getenv("ZOMBIE_SHOT_INTERVAL_SECONDS", "60")))
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--jpg-quality", type=int, default=int(os.getenv("ZOMBIE_SHOT_JPG_QUALITY", "55")))

    args = parser.parse_args()

    if not args.loop:
        capture_once(
            display=args.display,
            out_dir=args.out_dir,
            redis_url=args.redis_url,
            redis_key=args.redis_key,
            jpg_quality=args.jpg_quality,
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
            )
        except Exception as e:
            logger.error("Capture failed: %s", e)
        time.sleep(interval)


if __name__ == "__main__":
    raise SystemExit(main())
