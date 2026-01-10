# Task Plan: ZombieShot GIF stream + Streamlit fragment refresh

## Goal
Ensure the dashboard’s “Visual Verification” panel refreshes the ZombieShot media via Streamlit fragments (partial refresh), and confirm the Redis payload is valid and updating.

## Phases
- [x] Phase 1: Plan and setup
- [x] Phase 2: Inspect/research
- [x] Phase 3: Implement
- [x] Phase 4: Verify (tests / runtime)
- [ ] Phase 5: Deliver (docs / handoff)

## Key Questions
1. Is ZombieShot publishing a valid base64 payload to the Redis instance used by the dashboard?
2. Is the running dashboard container using a Streamlit version that supports `st.fragment` / `st.experimental_fragment`?
3. If the media sometimes flips to OFFLINE JPG, is that expected (variance gate) or a reliability bug?

## Decisions Made
- Prefer verifying Redis payloads via `redis` client library (avoids partial RESP reads).
- Enable fragment refresh by rebuilding the dashboard container with the pinned Streamlit version from `requirements.txt`.

## Errors Encountered
- Redis polling script saw `binascii.Error: Incorrect padding`: caused by incomplete socket reads (single `recv`); use redis-py or a looped RESP reader.
- Dashboard container Streamlit `1.28.2` lacked fragment support: requires rebuild to pick up pinned Streamlit version.

## Status
**Currently in Phase 5** - Handoff: dashboard container now supports fragment refresh; optional follow-up is deciding whether OFFLINE placeholders should overwrite last-good media.
