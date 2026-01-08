import asyncio
import asyncpg
import os

# Credentials from .env
DSN = os.getenv(
    "DATABASE_URL",
    "postgresql://eve_user:eve_pass@localhost:5432/eve_market_data",
)

async def init_db(dsn):
    print(f"Connecting to {dsn}...")
    try:
        conn = await asyncpg.connect(dsn)
        
        print("Creating table: market_radar")
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS market_radar (
                id SERIAL PRIMARY KEY,
                type_id INTEGER,
                region_id INTEGER,
                metric_score FLOAT,
                opportunity_type TEXT,
                detected_at TIMESTAMP DEFAULT NOW()
            );
        """)
        
        print("Creating table: pending_trades")
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS pending_trades (
                id SERIAL PRIMARY KEY,
                type_id INTEGER,
                action TEXT, -- 'BUY', 'SELL'
                quantity INTEGER,
                price FLOAT,
                status TEXT DEFAULT 'PENDING',
                created_at TIMESTAMP DEFAULT NOW()
            );
        """)

        print("Re-creating table: market_orders (DROP & CREATE)")
        await conn.execute("DROP TABLE IF EXISTS market_orders;")
        await conn.execute("""
            CREATE TABLE market_orders (
                order_id BIGINT PRIMARY KEY,
                type_id INTEGER,
                region_id INTEGER,
                location_id BIGINT,
                price DOUBLE PRECISION,
                volume_remaining INTEGER,
                is_buy_order BOOLEAN,
                issue_date TIMESTAMP,
                duration INTEGER,
                timestamp TIMESTAMP DEFAULT NOW()
            );
        """)
        
        # TimescaleDB hypertable conversion (optional but recommended for time-series)
        # try:
        #     await conn.execute("SELECT create_hypertable('market_orders', 'timestamp', if_not_exists => TRUE);")
        # except Exception as e:
        #     print(f"Hypertable create failed (ignore if not timescale): {e}")

        print("Database sync complete.")
        await conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print(f"Using DSN: {DSN}")
    asyncio.run(init_db(DSN))
