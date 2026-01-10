# Agent Instructions (MCP-First Architecture)

All future development MUST utilize the kongyo2 MCP ecosystem (Market, Traffic, OSINT) as the primary data source.

- Direct ESI calls are deprecated.
- Logic must prioritize **Military Expenditure (OSINT)** and **Player Demand (Status)** as the primary predictive features.

If an MCP source is unavailable at runtime, fall back to legacy sources only as a temporary best-effort measure, and surface a clear operator warning.
