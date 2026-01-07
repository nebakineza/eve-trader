import psycopg2
import pandas as pd
import os
import logging
from datetime import datetime, timedelta

# Configuration
DB_HOST = os.getenv("DB_HOST", "192.168.14.105") # Accessing Body from Brain
DB_PORT = 5432
DB_NAME = "eve_market_data"
DB_USER = "eve_user"
DB_PASS = os.getenv("POSTGRES_PASSWORD", "afterburn118921") # Use verified pass
OUTPUT_FILE = "training_data_24h.parquet"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DatasetGen")

def main():
    logger.info("🛠️ Preparation: Training Dataset Generator (Last 24h)")
    
    conn_str = f"host={DB_HOST} port={DB_PORT} dbname={DB_NAME} user={DB_USER} password={DB_PASS}"
    
    query = """
    SELECT 
        timestamp, region_id, type_id, open, high, low, close, volume 
    FROM market_history 
    WHERE timestamp >= NOW() - INTERVAL '24 HOURS'
    ORDER BY timestamp ASC, type_id
    """
    
    try:
        logger.info(f"🔌 Connecting to DB at {DB_HOST}...")
        conn = psycopg2.connect(conn_str)
        
        logger.info("📥 Fetching Data...")
        df = pd.read_sql(query, conn)
        
        if df.empty:
            logger.warning("Construction Empty: No data found in the last 24h.")
            logger.info("   (This is expected if the first scrape just happened or hasn't covered enough history yet)")
        else:
            logger.info(f"✅ Data Retrieved: {len(df)} rows.")
            
            # Save to Parquet
            df.to_parquet(OUTPUT_FILE, index=False)
            logger.info(f"💾 Saved to {OUTPUT_FILE}")
            
        conn.close()
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
