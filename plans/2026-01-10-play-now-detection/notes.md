# Notes: Play Now detection

## Context
- Goal: Fix "STATE: WAITING_FOR_PLAY_NOW... Play Now detection unavailable" by ensuring VisualCortex initializes reliably.
- Constraints: Minimal, surgical change; do not add new UX/features.
- Environment: Unified repo `.venv` (PEP 668 host), OpenCV headless.

## Findings
- The "detection unavailable" branch is taken when `vc is None` in `scripts/launcher_control.py`.
- `vc` becomes `None` when `VisualCortex(display=...)` raises during init; OpenCL probing can raise on some systems.
- Host dependencies confirmed installed: `scrot 1.8.1`, `tesseract 5.3.0`.

## Commands / Experiments
- `./.venv/bin/python -c "from nexus.automaton.visual_cortex import VisualCortex; VisualCortex(display=':0'); print('VisualCortex init OK')"`
- `./.venv/bin/python scripts/launcher_control.py --display ":0" --interval-seconds 1`
- Debug run: `VISUAL_CORTEX_DEBUG=1 ./.venv/bin/python scripts/launcher_control.py --display ":0" --interval-seconds 1`

## Debug Artifacts
- `logs/visual_cortex_frame.jpg`
- `logs/visual_cortex_roi.jpg`
- `logs/visual_cortex_teal_mask.png`
- `logs/visual_cortex_ocr_window.jpg`
- `logs/visual_cortex_ocr_processed.png`

## Links
- 

## Open Questions
- If Play Now is still not detected, check that `scrot` and `tesseract` binaries exist on the host (capture/OCR are runtime dependencies).
