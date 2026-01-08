#!/usr/bin/env python3
"""Push SkyNet (5090) GPU telemetry into Debian Redis.

- Uses nvitop (NVML via nvidia-ml-py) to read GPU metrics.
- Pushes to Redis using raw RESP over TCP (no redis-py dependency).

Keys written (all strings):
- system:skynet:temp   (C)
- system:skynet:load   (percent 0-100)
- system:skynet:vram   (MiB used)
- system:skynet:last_seen (unix ts)

Env:
- REDIS_HOST (default: 192.168.14.105)
- REDIS_PORT (default: 6379)
- GPU_INDEX  (default: 0)
- INTERVAL_SECONDS (default: 2)
"""

from __future__ import annotations

import os
import socket
import time


def _resp(*parts: str) -> bytes:
    out = [f"*{len(parts)}\r\n".encode("utf-8")]
    for p in parts:
        b = p.encode("utf-8")
        out.append(f"${len(b)}\r\n".encode("utf-8"))
        out.append(b)
        out.append(b"\r\n")
    return b"".join(out)


def redis_set(host: str, port: int, key: str, value: str) -> None:
    cmd = _resp("SET", key, value)
    with socket.create_connection((host, port), timeout=2) as s:
        s.sendall(cmd)
        s.recv(128)


def read_gpu_metrics(gpu_index: int) -> tuple[int | None, int | None, int | None]:
    # nvitop exposes NVML via nvidia-ml-py; use its public import path.
    try:
        from nvitop.api import Device  # type: ignore

        devices = Device.all()
        if not devices or gpu_index >= len(devices):
            return None, None, None

        d = devices[gpu_index]
        temp_c = int(round(d.temperature()))
        util = int(round(d.gpu_utilization()))
        mem_used_mib = int(round(d.memory_used() / (1024 * 1024)))
        return temp_c, util, mem_used_mib
    except Exception:
        return None, None, None


def main() -> int:
    host = os.getenv("REDIS_HOST", "192.168.14.105")
    port = int(os.getenv("REDIS_PORT", "6379"))
    gpu_index = int(os.getenv("GPU_INDEX", "0"))
    interval = float(os.getenv("INTERVAL_SECONDS", "2"))

    while True:
        temp_c, util, mem_used_mib = read_gpu_metrics(gpu_index)
        now = str(int(time.time()))

        if temp_c is not None:
            redis_set(host, port, "system:skynet:temp", str(temp_c))
        if util is not None:
            redis_set(host, port, "system:skynet:load", str(util))
        if mem_used_mib is not None:
            redis_set(host, port, "system:skynet:vram", str(mem_used_mib))

        redis_set(host, port, "system:skynet:last_seen", now)
        time.sleep(interval)


if __name__ == "__main__":
    raise SystemExit(main())
