-- Shadow Ledger: stores intended (simulated) trades from SHADOW mode

CREATE TABLE IF NOT EXISTS shadow_trades (
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    type_id BIGINT NOT NULL,
    signal_type TEXT NOT NULL CHECK (signal_type IN ('BUY', 'SELL')),
    predicted_price DOUBLE PRECISION,
    actual_price_at_time DOUBLE PRECISION,
    reasoning TEXT,
    virtual_outcome TEXT NOT NULL DEFAULT 'PENDING' CHECK (virtual_outcome IN ('PENDING', 'WIN', 'LOSS')),
    forced_exit BOOLEAN NOT NULL DEFAULT FALSE
);

-- Forward-compatible schema update for existing installs.
ALTER TABLE shadow_trades ADD COLUMN IF NOT EXISTS forced_exit BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE shadow_trades ADD COLUMN IF NOT EXISTS reasoning TEXT;

CREATE INDEX IF NOT EXISTS idx_shadow_trades_type_time ON shadow_trades (type_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_shadow_trades_outcome_time ON shadow_trades (virtual_outcome, timestamp DESC);
