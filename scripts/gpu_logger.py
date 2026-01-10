import os
import subprocess
import time
import logging

import redis


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("P4000Logger")


def _redis_client():
    redis_url = os.getenv("REDIS_URL", "redis://cache:6379/0")
    r = redis.Redis.from_url(redis_url, decode_responses=True)
    r.ping()
    return r


def _read_gpu_metrics() -> tuple[str, str, str]:
    # Required command from spec:
    # nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used --format=csv,noheader,nounits
    cmd = [
        "nvidia-smi",
        "--query-gpu=temperature.gpu,utilization.gpu,memory.used",
        "--format=csv,noheader,nounits",
    ]

    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        raise RuntimeError(f"nvidia-smi failed (rc={proc.returncode}): {stderr or 'no stderr'}")

    out = (proc.stdout or "").strip()
    if not out:
        raise RuntimeError("nvidia-smi returned empty output")

    # If multiple GPUs, take first line
    line = out.splitlines()[0]
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 3:
        raise RuntimeError(f"Unexpected nvidia-smi output: {line}")

    temp, util, mem_used = parts
    return temp, util, mem_used


def main() -> None:
    interval_s = int(os.getenv("GPU_LOGGER_INTERVAL_SECONDS", "30"))
    r = _redis_client()

    logger.info("P4000 telemetry logger started (interval=%ss)", interval_s)

    while True:
        try:
            temp, util, mem_used = _read_gpu_metrics()
            r.set("system:p4000:temp", temp)
            r.set("system:p4000:load", util)
            r.set("system:p4000:vram", mem_used)
            r.set("system:p4000:last_update", str(int(time.time())))
            logger.info("Updated P4000 metrics: temp=%sC util=%s%% vram=%sMiB", temp, util, mem_used)
        except Exception as e:
            # Keep last good Redis values; just report the issue.
            logger.warning("Failed to read P4000 metrics: %s", e)
        time.sleep(max(5, interval_s))


if __name__ == "__main__":
    main()
