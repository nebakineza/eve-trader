# Task Plan: Consolidate Production Venv

## Goal
Use one repo-wide `.venv/` for production host scripts instead of multiple one-off venvs.

## Phases
- [x] Phase 1: Plan and setup
- [x] Phase 2: Inspect/research
- [x] Phase 3: Implement
- [x] Phase 4: Verify (tests / runtime)
- [x] Phase 5: Deliver (docs / handoff)

## Key Questions
1. Which entrypoints need a stable interpreter path? (scripts + systemd units)
2. Which OpenCV version is compatible with pinned `numpy==1.26.2`?

## Decisions Made
- Pin `opencv-python-headless==4.11.0.86`: OpenCV 4.12+ requires `numpy>=2`, which conflicts with pinned numpy.
- Keep a single install entrypoint `requirements.venv.prod.txt` that includes `requirements.txt`.
- Add `scripts/bootstrap_prod_venv.sh` to create/update `.venv` and verify `cv2` import.
- Update `scripts/zombie_init.sh` to prefer `.venv/bin/python` (fallback to `.venv_launcher_control/bin/python`, then system python).

## Errors Encountered
- `pip install` was interrupted mid-run: resolved by re-running install (pip is idempotent).

## Status
**Done** - Unified `.venv` is created/installed/validated; legacy venvs remain but are no longer required.
