# Task Plan: Fix Play Now detection (VisualCortex init)

## Goal
`scripts/launcher_control.py` should initialize VisualCortex reliably so Play Now detection is enabled (no more "detection unavailable" state due to OpenCL init crashes).

## Phases
- [x] Phase 1: Plan and setup
- [x] Phase 2: Inspect/research
- [x] Phase 3: Implement
- [x] Phase 4: Verify (tests / runtime)
- [x] Phase 5: Deliver (docs / handoff)

## Key Questions
1. Why is VisualCortex initialization failing (making `vc=None`)?
2. Can VisualCortex recover if OpenCL probing throws?

## Decisions Made
- Make OpenCL enablement best-effort: OpenCL probing can raise on some hosts/drivers; never let that disable detection.
- Add `exc_info=True` to the VisualCortex init failure log to capture the real root cause in journald.

## Errors Encountered
- VisualCortex init could throw during OpenCL setup (e.g., `cv2.ocl.Device.getDefault()`): caught and degraded to CPU.

## Status
**Completed** - VisualCortex init is resilient; launcher_control now enables Play Now detection under `.venv`.
