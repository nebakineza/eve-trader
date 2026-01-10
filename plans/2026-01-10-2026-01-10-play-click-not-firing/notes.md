# Notes: Play click not firing

## Context
- Goal:
- Make PLAY NOW automation actually click (not just queue/log).
- Constraints:
- Keep changes minimal; don't refactor unrelated systems.
- Environment:
- Host runs Xvfb `:0` and `launcher_control.py` captures via `scrot` on `:0`.
- Arduino bridge runs in Docker and consumes Redis DB0 keys.

## Findings
- Redis gates are open: `system:visual:status=VERIFIED`, `system:hardware_lock=false`.
- `launcher_control.py` repeatedly queued `system:hardware:click`.
- `arduino:command_log` previously showed `CLICK ok:true` but that was only confirming serial write success.
- Direct serial probe of `/dev/ttyACM1` showed firmware responds to `PING` with `PONG` but ignores `MOV:`/`CLK` entirely.
	- Root cause: Arduino firmware did not implement mouse commands, so clicks were no-ops.

## Resolution
- Flashed updated `hardware/stealth_hid.ino` to the Arduino Leonardo.
- Serial probe confirms:
	- `PING -> PONG`
	- `CAPS -> CAPS:KEYBOARD,MOUSE,MOVCLK_V1`
	- `MOV:x,y -> OK`
	- `CLK -> OK`
- Updated Docker compose to use `/dev/ttyACM0` for the `arduino-bridge` device + `ARDUINO_PORT`.
- Verified end-to-end: `arduino:command_log` now records `CLICK ok:true` with `sequence: [MOV:x,y, CLK]`.

## Commands / Experiments
- Probed firmware capabilities by writing to `/dev/ttyACM1` and reading responses.
- Confirmed Docker bind-mount for `eve-trader-arduino-bridge` is `/home/seb/nebakineza/eve-trader -> /app`.

## Links
- Firmware: `hardware/stealth_hid.ino`
- Bridge: `scripts/arduino_bridge.py`, `scripts/arduino_connector.py`

## Open Questions
- After flashing updated firmware, does MOV land accurately enough under current pointer acceleration?
