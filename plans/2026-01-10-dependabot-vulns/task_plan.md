# Task Plan: Fix Dependabot vulnerability findings

## Goal
Update pinned dependencies so local vulnerability auditing reports no known vulnerabilities.

## Phases
- [x] Phase 1: Plan and setup
- [x] Phase 2: Inspect/research
- [x] Phase 3: Implement
- [x] Phase 4: Verify (tests / runtime)
- [x] Phase 5: Deliver (docs / handoff)

## Key Questions
1. Which FastAPI versions allow patched Starlette versions?
2. Are our pins internally consistent for pip resolution?

## Decisions Made
- Use `pip-audit` in `.venv_audit`: Debian uses an externally-managed Python environment (PEP 668).
- Pin `fastapi==0.128.0`: earliest tested version that allows Starlette >= 0.47.2.
- Pin `starlette==0.49.3`: patched version satisfying audit findings while staying close to prior constraints.
- Pin `pydantic==2.12.5`: required because FastAPI 0.128.0 depends on `pydantic>=2.7.0`.

## Errors Encountered
- `pip-audit` install failure (externally-managed-environment): resolved by creating and using `.venv_audit`.
- Dependency conflict: `fastapi==0.128.0` requires `pydantic>=2.7.0` but `requirements.txt` pinned `pydantic==2.5.2`; resolved by updating Pydantic pin.

## Status
**Done** - `pip-audit` reports no known vulnerabilities for `requirements.txt`, `requirements.oracle.txt`, and `nexus/requirements.txt`.
