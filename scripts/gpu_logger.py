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


def tile_oracle_logs() -> str:
    # Only relevant for Blackwell/SkyNet host
    log_path = "/app/logs/oracle_uvicorn.log"
    if not os.path.exists(log_path):
        return ""
    try:
        cmd = ["tail", "-n", "20", log_path]
        res = subprocess.run(cmd, text=True, capture_output=True)
        return res.stdout
    except:
        return ""

def main() -> None:
    interval_s = int(os.getenv("GPU_LOGGER_INTERVAL_SECONDS", "5"))
    prefix = os.getenv("GPU_METRIC_PREFIX", "system:p4000")
    r = _redis_client()

    logger.info(f"GPU telemetry logger started for {prefix} (interval={interval_s}s)")

    while True:
        # SkyNet Telemetry: Oracle Logs (Priority)
        if "skynet" in prefix:
             try:
                 logs = tile_oracle_logs()
                 if logs:
                     r.set("system:oracle:logs", logs)
             except Exception as e:
                 logger.warning("Failed to tile oracle logs: %s", e)

        # GPU Metrics (Best Effort)
        try:
            temp, util, mem_used = _read_gpu_metrics()
            r.set(f"{prefix}:temp", temp)
            r.set(f"{prefix}:load", util)
            r.set(f"{prefix}:vram", mem_used)
            logger.info("Updated metrics: temp=%sC util=%s%%", temp, util)
        except Exception as e:
            # If GPU is missing or driver failed, just log it but keep the loop alive for logs
            logger.warning("Failed to read GPU metrics: %s", e)

        time.sleep(max(1, interval_s))


if __name__ == "__main__":
    main()
