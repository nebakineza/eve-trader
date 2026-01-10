# Task Plan: GitHub auth + remote sync (eve-trader)

## Goal
Authenticate this machine with GitHub and sync/push this local repo to the remote `eve-trader` repository.

## Phases
- [ ] Phase 1: Plan and setup
- [ ] Phase 2: Inspect/research
- [ ] Phase 3: Implement
- [ ] Phase 4: Verify (tests / runtime)
- [ ] Phase 5: Deliver (docs / handoff)

## Key Questions
1. Is the remote repo `eve-trader` under a user or an org (and what is the exact URL)?
2. Do you prefer SSH or HTTPS (GitHub CLI) for auth on this machine?

## Decisions Made
- Use the minimal authentication method available (prefer `gh auth login` if installed; otherwise SSH key).

## Errors Encountered
- 

## Status
**Currently in Phase 1** - Creating plan/notes and inspecting git state
