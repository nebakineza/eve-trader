# Task Plan: Fix launcher_stdout.log source

## Goal
Ensure the dashboard's System tab can read a real launcher console log file at `/app/logs/launcher_stdout.log`.

## Phases
- [x] Phase 1: Plan and setup
- [x] Phase 2: Inspect/research
- [x] Phase 3: Implement
- [ ] Phase 4: Verify (runtime)
- [ ] Phase 5: Deliver (deploy / handoff)

## Key Questions
1. What process is expected to write launcher output? (`scripts/zombie_init.sh` starts `scripts/launcher_control.py`)
2. Where does it currently write logs? (`/tmp/launcher_control.log` on the host)
3. What path is visible to the dashboard container? Repo bind-mount path `/app/logs/launcher_stdout.log`

## Decisions Made
- Update `scripts/zombie_init.sh` to write launcher_control stdout/stderr to a repo-relative path (`logs/launcher_stdout.log`) so Docker-mounted dashboard can read it.
- Stop assuming `$HOME/eve-trader` by computing `REPO_ROOT` from the script location.

## Status
**Pending deploy** - Pull latest on Debian and restart launcher_control (or rerun `zombie_init.sh`) so logs stream into `logs/launcher_stdout.log`.
