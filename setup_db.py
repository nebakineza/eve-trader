import asyncio
import asyncpg
import os

# Credentials from .env
DSN = "postgres://eve_user:afterburn118921@localhost:5432/eve_market_data"

async def init_db():
    print(f"Connecting to {DSN}...")
    try:
        conn = await asyncpg.connect(DSN)
        
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
        
        print("Database sync complete.")
        await conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(init_db())
