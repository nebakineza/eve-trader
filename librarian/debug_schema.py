import asyncio
import asyncpg

DSN = "postgres://eve_user:afterburn118921@localhost:5432/eve_market_data"

async def check_schema():
    try:
        conn = await asyncpg.connect(DSN)
        print("Connected.")
        
        tables = await conn.fetch("SELECT table_name FROM information_schema.tables WHERE table_schema='public';")
        print("Tables:", [t['table_name'] for t in tables])

        rows = await conn.fetch("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'market_orders';
        """)
        print("Schema for market_orders:")
        for r in rows:
            print(f"{r['column_name']}: {r['data_type']}")
        await conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(check_schema())
