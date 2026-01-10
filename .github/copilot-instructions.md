# Copilot Chat Instructions (EVE Trader)

## Default workflow: Planning with Files
For any non-trivial task (3+ steps, research, multi-file changes, long debugging):
1) Create a per-task folder: `plans/<YYYY-MM-DD>-<topic>/`.
2) Create/update `plans/<YYYY-MM-DD>-<topic>/task_plan.md` using `templates/task_plan.md`.
3) Create/update `plans/<YYYY-MM-DD>-<topic>/notes.md` using `templates/notes.md`.
3) Read `task_plan.md` before major decisions; update it after each phase.
4) Log errors and resolutions in `task_plan.md` (don’t hide failures).

Tip: In VS Code, run the task “Planning: Init per-task plan + notes” to generate these files.

Reference: see `AGENT_WORKFLOW.md` and `third_party/planning-with-files/`.

## Repo-specific operating constraints
- Production host is Debian `192.168.14.105`. Do **not** run production Docker commands locally; use `ssh seb@192.168.14.105`.
- Prefer minimal, surgical changes; don’t refactor unrelated code.
- When adding third-party code, keep license text and attribution.

## Verification
- If code is changed, run the narrowest checks possible (lint/test/quick runtime) before claiming success.
