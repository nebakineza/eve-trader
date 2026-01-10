# Task Plan: Play click not firing

## Goal
When `launcher_control.py` detects PLAY NOW and queues a click, the system performs a real click that actually activates the launcher (no silent no-op).

## Phases
- [x] Phase 1: Plan and setup
- [x] Phase 2: Inspect/research
- [x] Phase 3: Implement
- [x] Phase 4: Verify (tests / runtime)
- [x] Phase 5: Deliver (docs / handoff)

## Key Questions
1. Are queued clicks being consumed by the bridge and gated correctly?
2. Does the Arduino firmware actually implement mouse commands (`MOV:`/`CLK`) or are we only confirming serial writes?

## Decisions Made
- Add explicit firmware capability detection (`CAPS`) and refuse click requests when mouse is unsupported: avoids silent failures.
- Update the Stealth HID firmware to support `MOV:x,y` + `CLK`: required for Play Now automation when using HID mouse.

## Errors Encountered
- Arduino bridge crash: `mouse_supported` referenced before assignment. Resolution: fixed indentation and ensured `mouse_supported` is initialized each loop.

## Status
**Completed** - Firmware flashed to Arduino Leonardo; serial probe confirms `CAPS:KEYBOARD,MOUSE,MOVCLK_V1` and `MOV`/`CLK` return `OK`. Docker `arduino-bridge` connects on `/dev/ttyACM0` and executes Redis click requests successfully.
