-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- 1. Market History (OHLCV)
-- Storing 1-year daily/hourly logs for items
CREATE TABLE market_history (
    timestamp TIMESTAMPTZ NOT NULL,
    region_id INT NOT NULL,
    type_id INT NOT NULL,
    open DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    close DOUBLE PRECISION,
    volume BIGINT,
    PRIMARY KEY (timestamp, region_id, type_id)
);

-- Convert to Hypertable partitioned by timestamp
SELECT create_hypertable('market_history', 'timestamp', if_not_exists => TRUE);
CREATE INDEX idx_item_time ON market_history (type_id, timestamp DESC);

-- 2. Market Orders (Snapshots)
-- High-volume table for order book depths
CREATE TABLE market_orders (
    timestamp TIMESTAMPTZ NOT NULL,
    order_id BIGINT NOT NULL,
    type_id INT NOT NULL,
    region_id INT NOT NULL,
    location_id BIGINT,
    price DOUBLE PRECISION,
    volume_remaining BIGINT,
    is_buy_order BOOLEAN,
    issue_date TIMESTAMPTZ,
    duration INT
);

SELECT create_hypertable('market_orders', 'timestamp', if_not_exists => TRUE);

-- 3. Trade Logs (Decisions)
-- Stores both real and shadow trades
CREATE TABLE trade_logs (
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    trade_id UUID DEFAULT gen_random_uuid(),
    action_type INT, -- 1=Buy, 2=Sell, 0=Hold
    type_id INT,
    price DOUBLE PRECISION,
    quantity BIGINT,
    simulated BOOLEAN DEFAULT FALSE, -- TRUE for Shadow Mode/Backtest
    profit_loss DOUBLE PRECISION, -- Estimated PnT for closed positions
    status TEXT -- 'FILLED', 'VIRTUAL_FILL', 'PENDING'
);

SELECT create_hypertable('trade_logs', 'timestamp', if_not_exists => TRUE);

-- 4. Pending Trades (Active Orders for Shadow Mode)
-- Tracks the lifecycle of a virtual order from placement to fill/cancel
CREATE TABLE pending_trades (
    order_id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    type_id INT NOT NULL,
    action_type INT NOT NULL, -- 1=Buy, 2=Sell
    price DOUBLE PRECISION NOT NULL,
    quantity BIGINT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    status TEXT DEFAULT 'OPEN', -- 'OPEN', 'FILLED', 'CANCELLED', 'STALE'
    simulated BOOLEAN DEFAULT TRUE
);

-- 5. Market Radar (Opportunity Scanner)
-- Stores candidates identified by Oracle/Strategist for visualization
CREATE TABLE market_radar (
    type_id INT NOT NULL,
    signal_strength DOUBLE PRECISION, -- Oracle confidence or Rule score
    predicted_target DOUBLE PRECISION,
    current_price DOUBLE PRECISION,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    status TEXT DEFAULT 'WATCH', -- 'WATCH', 'ENTER', 'IGNORE'
    PRIMARY KEY (type_id)
);
);

SELECT create_hypertable('trade_logs', 'timestamp', if_not_exists => TRUE);

-- =========================================================
-- Data Lifecycle Management (Compression & Retention)
-- =========================================================

-- 1. Enable Native Compression for Market Orders
-- Grouping by type_id ensures efficient queries for specific items
ALTER TABLE market_orders SET (
  timescaledb.compress,
  timescaledb.compress_segmentby = 'type_id',
  timescaledb.compress_orderby = 'timestamp DESC'
);

-- 2. Add Policies

-- Compression Policy: Compress data older than 7 days
-- This reduces storage footprint by ~90% for historical depth data
SELECT add_compression_policy('market_orders', INTERVAL '7 days');

-- Retention Policy (Cleanup): Drop granular order snapshots older than 30 days
-- We don't need tick-level order book depth beyond a month for training
SELECT add_retention_policy('market_orders', INTERVAL '30 days');

-- Retention Policy (Logs): Drop trade logs older than 90 days
-- Virtual fills are useful for quarterly performance review, then discarded
SELECT add_retention_policy('trade_logs', INTERVAL '90 days');

-- Indexes for Dashboard performance
CREATE INDEX idx_trade_logs_simulated ON trade_logs (simulated, timestamp DESC);
