# Notes: 5090 (SKYNET) telemetry

## Context
- Goal: restore 5090 metrics on Debian dashboard via Redis (`system:skynet:*`).
- Host: SKYNET / 192.168.10.108
- Debian Redis: 192.168.14.105:6379

## Findings
- No eve-trader Docker containers running on SKYNET (good).
- SKYNET has a repo checkout at `/home/seb/nebakineza/eve-trader`.
- Cron jobs present for training + watchdog:
	- `*/4h` `scripts/cron_train.sh`
	- `*/15m` `scripts/induction_watchdog.sh`
- SKYNET had no `nvitop` installed.
- Debian Redis initially had no `system:skynet:*` keys.

## Changes made on SKYNET
- Created venv: `/home/seb/nebakineza/eve-trader/.venv_skynet_telemetry`
- Installed: `nvitop`, `nvidia-ml-py`
- Installed + enabled systemd service:
	- `/etc/systemd/system/eve-trader-skynet-telemetry.service`
	- `systemctl enable --now eve-trader-skynet-telemetry.service`

## Verification
- Debian Redis now shows values, e.g. `system:skynet:temp=22`, `system:skynet:load=0`, `system:skynet:vram≈4000 MiB`, `system:skynet:last_seen=<unix ts>`.
