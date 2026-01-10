# Task Plan: Fix dashboard System tab errors

## Goal
Remove confusing System tab errors and ensure:
- 5090 telemetry displays sanely (N/A vs misleading zeros)
- Launcher console log panel is configurable and non-alarming when absent
- Oracle Mind log stream is populated

## Phases
- [x] Phase 1: Plan and setup
- [x] Phase 2: Inspect/research
- [x] Phase 3: Implement
- [x] Phase 4: Verify (local syntax)
- [ ] Phase 5: Deliver (deploy / restart services)

## Key Questions
1. Where do the 5090 metrics come from? (Redis keys `system:skynet:*`)
2. Where does the launcher console log panel read from? (repo path `logs/launcher_stdout.log`)
3. Why does Oracle Mind show "Waiting for stream"? (Redis key `system:oracle:logs` is missing)

## Decisions Made
- Adjust 5090 defaults to `N/A` instead of `0%/0 MiB` and display last-seen age when available.
- Make launcher log path configurable via `LAUNCHER_STDOUT_LOG_PATH` and show an info message if missing.
- Ensure Oracle logs are written to a file inside the container and shipped to Redis.

## Status
**Phase 5 pending** - After pulling latest code on Debian, restart `oracle`, `gpu-logger`, and `nexus-dashboard` to apply changes.
