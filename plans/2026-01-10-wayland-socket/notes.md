# Notes: Wayland socket

## Context
- Goal: avoid Wayland attempts unless a real `wayland-*` socket exists; reduce noisy "No Wayland runtime" errors.
- Environment: Debian host, ZombieShot run via systemd.

## Findings
- Current user runtime path exists (`/run/user/1002`) but contains **no** `wayland-*` sockets.
- This typically means there is no active Wayland compositor/session on the host.

## Commands / Experiments
- `ls -la /run/user/$(id -u)/wayland-*`
- `loginctl show-user $(id -un) -p RuntimePath -p Linger`

## Implementation
- ZombieShot Wayland runtime detection now requires the path to be an actual UNIX socket (`stat.S_ISSOCK`).
- Capture flow computes `wl_env` once per run and passes it into the Wayland capture function.
