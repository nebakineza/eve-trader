# Agent Workflow (Planning With Files)

This repo benefits from a lightweight, file-based “working memory” pattern for multi-step work.

## When to use
Use this when the task is:
- 3+ steps
- involves research / exploration
- spans many tool calls
- needs progress tracking or error breadcrumbs

Skip it for quick questions or one-file edits.

## The 3-file pattern
For any non-trivial task, create three files:
- `plans/<YYYY-MM-DD>-<topic>/task_plan.md` — phases + status + decisions + errors (update as you go)
- `plans/<YYYY-MM-DD>-<topic>/notes.md` — findings, commands run, links, experiments (append, don’t overwrite)
- a deliverable — e.g. `design.md`, `runbook.md`, `pr_description.md`, etc.

Templates are in:
- `templates/task_plan.md`
- `templates/notes.md`

VS Code helper:
- Run the task “Planning: Init per-task plan + notes” to create the folder and files.

## Third-party reference
We vendor the original “planning-with-files” markdown skill (MIT licensed) here:
- `third_party/planning-with-files/`

If you’re using Claude Code skills locally, you can install that folder into your Claude skills directory; for this repo, we mainly use it as a reference and keep our actual plans/notes as repo files.
