# Notes: Ledger winners + BUY/SELL restructure

## Context
- Goal: Fix dashboard panels for top winners today; restructure Market Maker Ledger tabs to BUY ORDERS / SELL ORDERS.
- Constraints: Minimal UI changes; keep safety gates; server timezone day definition.
- Environment: Debian host 192.168.14.105 running docker compose services.

## Findings
- Dashboard "Visual status" mismatch root cause: Redis client uses decode_responses=True (str values), while visual-status parsing assumed bytes; fixed by normalizing values.
- Trade performance section was sorting recent trades by profit, but did not filter to "today" nor restrict to completed trades.
- Market Maker Ledger tabs were Winners/Losers; requirement is now BUY ORDERS and SELL ORDERS showing most recent completed.
- Important: "recent trades" queries often return only very fresh shadow_trades with no horizon exit yet; ledger needs a query that selects trades older than the horizon and with a resolved exit_price.

## Commands / Experiments
- Recreated dashboard container: `docker compose up -d --force-recreate nexus-dashboard`

## Links
- 

## Open Questions
- "COMPLETED" implemented as: profit_net present AND status != PENDING (status derived from virtual_outcome when present).
- Display outcome is now based on computed profit_net (WIN if profit_net>0 else LOSS); DB virtual_outcome is not used for winner selection.
