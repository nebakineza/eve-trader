# Task Plan: Governor + Oracle Vision + Librarian Throttle

## Goal
Improve operational safety and observability:
- Add strict action rate limiting + active-hours scheduling (for system health; no evasion/stealth claims).
- Enhance Oracle dashboard page with transparent diagnostics (feature attention proxy hooks, loss history sparkline, live parameters).
- Ensure Librarian loop throttling (10s sleep) to prevent CPU overload.

## Phases
- [ ] Phase 1: Inspect current governor/inference/librarian code
- [ ] Phase 2: Implement safety governor + config proof
- [ ] Phase 3: Implement Oracle diagnostics UI + data plumbing (Redis-backed)
- [ ] Phase 4: Verify Librarian throttling
- [ ] Phase 5: Smoke test + restart services if needed

## Key Questions
1. Where are trade actions emitted (new orders, reprices) and how to intercept them with a governor?
2. Where do Kelly fraction + confidence threshold live today (env/config/redis)?
3. Where does training loss history get stored (redis key, file, logs)?
4. Which Librarian component is consuming CPU and where is its main loop?

## Decisions Made
- Implement governor as rate-limiter + active-hours gate for safety, not stealth.
- Surface "Trading Schedule" config in config.json and in dashboard so it is remotely visible.

## Status
**Currently in Phase 1** - locating the relevant code paths.
