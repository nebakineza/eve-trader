# Task Plan: Dashboard winners + ledger restructure

## Goal
Dashboard shows (1) top biggest winners for today (server timezone) across BUY+SELL, and (2) Market Maker Ledger shows latest completed BUY orders and latest completed SELL orders.

## Phases
- [x] Phase 1: Plan and setup
- [x] Phase 2: Inspect/research
- [x] Phase 3: Implement
- [x] Phase 4: Verify (tests / runtime)
- [x] Phase 5: Deliver (docs / handoff)

## Key Questions
1. Which table/query is the source of trade outcomes in the dashboard?
2. What exactly counts as "COMPLETED" in the current schema (virtual_outcome/status vs. exit_price present)?
3. Is trade timestamp stored as UTC (ISO with Z) and how should it be bucketed by server local day?

## Decisions Made
- Use server local timezone day-bucketing by converting timestamps to server tz and filtering [midnight, midnight+1d).
- Treat "COMPLETED" as trades with non-null exit_price/profit and status != PENDING.

## Errors Encountered
- 

## Status
**Completed** - Dashboard now shows top winners for today (server timezone) and Market Maker Ledger shows latest completed BUY and SELL orders.
