# Notes: venv-tidy

## Context
- Goal: consolidate multiple venvs into a single repo-wide production `.venv/`.
- Constraints: Debian host uses externally-managed system Python (PEP 668) → use venv.
- Environment: Python 3.11, host scripts + systemd unit templates in repo.

## Findings
- Found legacy venvs: `.venv_launcher_control`, `.venv_audit`, `.venv_recovery`.
- `nexus/automaton/visual_cortex.py` needs `cv2` + `numpy`.
- `requirements.txt` pins `numpy==1.26.2`; OpenCV 4.12+ requires `numpy>=2`.

## Commands / Experiments
- Create + install: `python3 -m venv .venv && . .venv/bin/activate && python -m pip install -U pip wheel && python -m pip install -r requirements.venv.prod.txt`
- Verify: `. .venv/bin/activate && python -c "import cv2, numpy as np; print(cv2.__version__, np.__version__)"`

## Open Questions
- None (repo templates updated; production checkout may need the same `.venv` created if it’s a separate directory).
