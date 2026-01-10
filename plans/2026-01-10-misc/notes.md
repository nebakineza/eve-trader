# Notes: ZombieShot GIF refresh

## Context
- Goal: Verify ZombieShot media is valid and dashboard refreshes it via Streamlit fragment.
- Constraints: Minimal changes; Debian host runs docker-compose stack; avoid unrelated refactors.
- Environment: Redis in compose (`cache`), dashboard container (`nexus-dashboard`).

## Findings
- `system:zombie:screenshot` in Redis is base64; sometimes a large GIF (~422KB raw / ~562KB base64), sometimes an OFFLINE placeholder JPG (~18KB raw / ~23.9KB base64).
- Running dashboard container had Streamlit `1.28.2` and no `st.fragment` / `st.experimental_fragment`, so the code’s fragment path cannot execute until rebuild.
- Using redis-py inside container confirms payload type and avoids partial socket reads.
- ZombieShot now preserves an existing last-known-good GIF when going OFFLINE (env `ZOMBIE_SHOT_PRESERVE_LAST_GOOD=1` default); set it to `0` to revert to overwriting with the OFFLINE placeholder.

## Commands / Experiments
- `docker compose exec -T cache redis-cli STRLEN system:zombie:screenshot`
- `docker compose exec -T cache redis-cli GET system:zombie:screenshot | head -c 200`
- `docker compose exec -T nexus-dashboard python -c "import streamlit as st; print(st.__version__, hasattr(st,'fragment'), hasattr(st,'experimental_fragment'))"`
- redis-py polling to validate decode + type (gif/jpg) + change detection.

## Links
- 

## Open Questions
- Do we want OFFLINE placeholder to overwrite the last good GIF, or preserve last-known-good media while setting status keys?
