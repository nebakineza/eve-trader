# Notes: launcher_stdout.log

## Context
- Dashboard reads a host-mounted file at `/app/logs/launcher_stdout.log`.
- `scripts/zombie_init.sh` starts `scripts/launcher_control.py` in a loop.

## Findings
- `zombie_init.sh` previously redirected launcher_control logs to `/tmp/launcher_control.log` (not visible inside the dashboard container).
- It also assumed `$HOME/eve-trader/...` paths, which can be wrong on Debian (`/home/seb/nebakineza/eve-trader`).

## Implementation
- `zombie_init.sh` now computes `REPO_ROOT` and defaults `LAUNCHER_STDOUT_LOG_PATH` to `$REPO_ROOT/logs/launcher_stdout.log`.
- `launcher_control.py` stdout/stderr are appended to that file.

## Deploy/Verify on Debian
- `cd /home/seb/nebakineza/eve-trader && git pull`
- `mkdir -p logs && : > logs/launcher_stdout.log`
- Restart launcher_control (kill PID from `/tmp/launcher_control.pid` if running) or rerun `./scripts/zombie_init.sh`
- Confirm dashboard shows content in the System tab.
