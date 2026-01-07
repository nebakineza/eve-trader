# EVE Online Station Trading Bot - Agentic Microservices Architecture

## Overview
This project focuses on the Data Acquisition and Decision/Thinking layers for a Station Trading bot. The logic is strictly confined to the market interface and station-bound operations.

The architecture follows an Agentic Microservices pattern. Each module is a self-contained script or service that communicates via a central message bus or API.

## Phase 1: The Data Pipeline & Intelligence Layer
This layer handles the "Librarian" and "Analyst" roles, adhering to MLOps practices for clean, versioned data.

### 1. Module: Librarian-ESI-Scraper
**Purpose:** High-frequency raw data ingestion.

**Functions:**
- `fetch_market_snapshot()`: Pulls full order books for the "Top 500" whitelist.
- `fetch_historical_logs()`: Gathers 1-year daily OHLCV (Open, High, Low, Close, Volume) data.
- `validator_cleaner()`: Removes "outlier" spikes caused by market manipulation or fat-finger trades.

**Tech Stack:** Python 3.13, Aiohttp (asynchronous ESI calls), PostgreSQL (TimescaleDB).

### 2. Module: Analyst-Feature-Factory
**Purpose:** Transform raw numbers into "signals" for the AI.

**Functions:**
- `compute_market_velocity()`: Measures order fill rate vs. creation rate.
- `calculate_spread_efficiency()`: Determines net profit after tax/broker fees.
- `order_book_imbalance()`: Analyzes buy-wall vs. sell-wall depth ratio.

**Tech Stack:** Pandas, Dask, Redis (real-time features).

## Phase 2: The Decision & Thinking Model
Uses Temporal Transformers and Reinforcement Learning (RL) with "Train-Test-Validate" splits.

### 3. Module: Oracle-Temporal-Transformer
**Purpose:** Predicting the "Next-Hour" price distribution.

**Functions:**
- `encode_temporal_patterns()`: Identifies weekly/daily cycles via Multi-Head Attention.
- `predict_regime()`: Classifies market state (Stable, Trending, Volatile War).
- `confidence_scoring()`: Outputs probability score; <40% triggers "Wait" state.

**Tech Stack:** PyTorch, Temporal Fusion Transformer (TFT).

### 4. Module: Strategist-RL-Agent
**Purpose:** The "Brain" deciding trade actions.

**Functions:**
- `evaluate_action_space()`: Post New Buy, Update Sell, Cancel Order, or Hold.
- `reward_function_optimizer()`: Calculates Quality of Move (Profit - Time-Weighted Broker Fee).
- `risk_governor()`: Hard-coded safety layer vetoing high-risk trades.

**Tech Stack:** Stable Baselines3, PPO.

## Phase 3: Orchestration & Inter-Module Communication
Uses a Unified Message Bus for distributed operation.

### 5. Module: Nexus-Coordinator (The App)
**Purpose:** Central nervous system syncing all modules.

**Functions:**
- `state_synchronizer()`: Ensures Oracle uses fresh data.
- `command_dispatcher()`: Packages decisions into standardized JSON for execution.
- `performance_monitor()`: Tracks "Paper Trading" vs. "Real Trading" accuracy.

**Tech Stack:** FastAPI, RabbitMQ.

## Proper LM & Data Practices Verification
- **Data Versioning:** DVC (Data Version Control) for pinpointing error sources.
- **Model Evaluation:** Sharpe Ratio and Maximum Drawdown.
- **Agentic Oversight:** Strategist requests predictions; requires high-confidence forecasts.

## System Flow Summary
1. **Librarian** pulls Jita data → Stores in PostgreSQL.
2. **Analyst** reads DB → Generates features → Updates Redis.
3. **Oracle** reads Redis → Predicts price trend → Sends to Nexus.
4. **Strategist** evaluates Nexus data → Picks highest-ROI action.
5. **Nexus** verifies action against Risk Governor → Triggers Execution.
