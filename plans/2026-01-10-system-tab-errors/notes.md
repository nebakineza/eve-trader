# Notes: System tab errors

## Context
- Symptoms:
	- 5090 Temp: N/A
	- 5090 Load: 0%
	- 5090 VRAM: 0 MiB
	- Launcher console: "Log file not found: /app/logs/launcher_stdout.log"
	- Oracle Mind: "Waiting for stream..."

## Findings
- 5090 metrics are read from Redis keys `system:skynet:temp/load/vram`.
- Launcher log panel reads a fixed path in the repo (`logs/launcher_stdout.log`) which often doesn't exist.
- Oracle Mind reads Redis key `system:oracle:logs`.
- `scripts/gpu_logger.py` only published Oracle logs when `GPU_METRIC_PREFIX` contained `skynet`, and it tailed a file that wasn't being written by the Oracle container.

## Fixes
- `nexus/command_center.py`:
	- 5090 defaults changed to `N/A` (avoid misleading 0 values)
	- Show "telemetry last seen" when `system:skynet:last_seen` exists
	- Launcher log panel path can be overridden via `LAUNCHER_STDOUT_LOG_PATH` and missing file is informational
- `docker-compose.yml`:
	- `gpu-logger` defaults to `system:p4000` and includes `PUBLISH_ORACLE_LOGS` env
	- `oracle` now tees stdout/stderr to `/app/logs/oracle_uvicorn.log`
- `scripts/gpu_logger.py`:
	- Publish `system:oracle:logs` when enabled, regardless of prefix

## Next
- Deploy by pulling latest `main` on Debian and recreating affected services (`oracle`, `gpu-logger`, `nexus-dashboard`).
