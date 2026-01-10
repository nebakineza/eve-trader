# Task Plan: Ensure Wayland socket detection is correct

## Goal
ZombieShot should only attempt Wayland capture when a real `wayland-*` socket exists, avoiding misleading "No Wayland runtime" errors caused by non-socket paths or double-check races.

## Phases
- [x] Phase 1: Plan and setup
- [x] Phase 2: Inspect/research
- [x] Phase 3: Implement
- [x] Phase 4: Verify (tests / runtime)
- [ ] Phase 5: Deliver (deploy / handoff)

## Key Questions
1. Is a Wayland compositor actually running (i.e., is there a `wayland-*` socket under `/run/user/$UID/`)?
2. Is ZombieShot treating any `wayland-*` *file* as valid, instead of requiring a socket?

## Decisions Made
- Require real UNIX sockets via `stat.S_ISSOCK()` in Wayland runtime detection.
- Compute Wayland runtime env once per capture attempt and pass it through to the capture routine.

## Errors Encountered
- Host check: `/run/user/$UID` exists but there are no `wayland-*` sockets, which indicates no active Wayland compositor on this machine.

## Status
**Phase 5 pending** - If Wayland capture is required, a compositor/session must be running to create the socket; otherwise ZombieShot will fall back to X11.
