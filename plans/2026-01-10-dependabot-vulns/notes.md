# Notes: Dependabot vulnerabilities

## Context
- Goal: clear reported Python dependency vulnerabilities.
- Constraints: Debian system Python is externally managed; use a virtualenv for security tooling.
- Environment: Debian 12, Python 3.11.

## Findings
- FastAPI versions up through ~0.120 pinned Starlette too low for patched releases.
- FastAPI `0.128.0` is compatible with patched Starlette (`0.47.2+`, tested with `0.49.3` and `0.50.0`).
- FastAPI `0.128.0` requires `pydantic>=2.7.0`.

## Commands / Experiments
- `python -m pip index versions fastapi`
- `python -m pip index versions starlette`
- `python -m pip install --dry-run "fastapi==0.128.0" "starlette==0.49.3"`
- `pip-audit -r requirements.txt -r requirements.oracle.txt -r nexus/requirements.txt`

## Open Questions
- None.
