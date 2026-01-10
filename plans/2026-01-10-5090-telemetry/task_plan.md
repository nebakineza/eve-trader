# Task Plan: Validate 5090 (SKYNET) telemetry

## Goal
Ensure SKYNET (5090 host) continuously publishes GPU telemetry to Debian Redis keys:
- `system:skynet:temp`
- `system:skynet:load`
- `system:skynet:vram`
- `system:skynet:last_seen`

## Phases
- [x] Phase 1: Plan and setup
- [x] Phase 2: Inspect/research
- [x] Phase 3: Implement
- [x] Phase 4: Verify (runtime)
- [ ] Phase 5: Deliver (docs / handoff)

## Key Questions
1. Is SKYNET still running any full eve-trader Docker stack? (should not)
2. Is there an existing telemetry publisher writing `system:skynet:*` into Debian Redis?
3. Does SKYNET have working NVML access for GPU metrics?

## Decisions Made
- Use the existing `scripts/sky_telemetry.py` to publish telemetry directly to Debian Redis.
- Install `nvitop` + `nvidia-ml-py` in a dedicated venv on SKYNET to avoid touching system Python packages.
- Run telemetry as a systemd service (`eve-trader-skynet-telemetry.service`) for persistence and auto-restart.

## Errors Encountered
- `system:skynet:*` keys were absent in Debian Redis: resolved by starting the SKYNET telemetry service.
- `nvitop` missing on SKYNET: resolved by installing into a venv.

## Status
**Running** - SKYNET publishes `system:skynet:*` to Debian Redis and the service is enabled at boot.
