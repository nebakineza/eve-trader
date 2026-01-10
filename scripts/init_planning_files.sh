#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/init_planning_files.sh --topic "<short topic>"

Creates per-task planning files:
  plans/<YYYY-MM-DD>-<topic>/task_plan.md
  plans/<YYYY-MM-DD>-<topic>/notes.md
EOF
}

topic=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --topic)
      topic="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$topic" ]]; then
  echo "Missing required --topic" >&2
  usage >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

plan_src="$repo_root/templates/task_plan.md"
notes_src="$repo_root/templates/notes.md"

if [[ ! -f "$plan_src" ]]; then
  echo "Missing template: $plan_src" >&2
  exit 1
fi
if [[ ! -f "$notes_src" ]]; then
  echo "Missing template: $notes_src" >&2
  exit 1
fi

date_prefix="$(date +%F)"

# Lowercase, replace whitespace with '-', drop unsafe chars.
topic_slug="$(printf '%s' "$topic" | tr '[:upper:]' '[:lower:]' | tr -s '[:space:]' '-' | tr -cd 'a-z0-9._-')"
topic_slug="${topic_slug#-}"
topic_slug="${topic_slug%-}"

if [[ -z "$topic_slug" ]]; then
  echo "Topic became empty after sanitization. Pick a simpler topic." >&2
  exit 2
fi

task_dir="$repo_root/plans/${date_prefix}-${topic_slug}"
mkdir -p "$task_dir"

plan_dst="$task_dir/task_plan.md"
notes_dst="$task_dir/notes.md"

if [[ -f "$plan_dst" ]]; then
  echo "Exists: $plan_dst (leaving as-is)"
else
  cp "$plan_src" "$plan_dst"
  echo "Created: $plan_dst"
fi

if [[ -f "$notes_dst" ]]; then
  echo "Exists: $notes_dst (leaving as-is)"
else
  cp "$notes_src" "$notes_dst"
  echo "Created: $notes_dst"
fi

echo "Task directory: $task_dir"
