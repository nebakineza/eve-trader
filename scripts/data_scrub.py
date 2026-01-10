import os
import logging
from pathlib import Path

import pandas as pd
import psycopg2
import sqlalchemy

# Data Scrubber Script
# Exports a time-series dataset from market_history (the "movie") suitable for
# sequence training. Also removes obvious outliers and low-volume noise.

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DataScrubber")

DATABASE_URL = os.getenv("DATABASE_URL")

# Fallbacks (only used if DATABASE_URL is not set)
DB_HOST = os.getenv("DB_HOST", "db")
DB_USER = os.getenv("DB_USER", "eve_user")
DB_PASS = os.getenv("DB_PASS", "eve_pass")
DB_NAME = os.getenv("DB_NAME", "eve_market_data")

OUTPUT_PATH = Path(os.getenv("OUTPUT_PATH", "/app/data/training_data_cleaned.parquet"))

def scrub_data():
    try:
        # Connect (prefer the same DSN used by the running stack)
        if DATABASE_URL:
            conn = psycopg2.connect(DATABASE_URL)
        else:
            # Note: If running on host, map 5432 is exposed.
            conn = psycopg2.connect(
                host=DB_HOST,
                user=DB_USER,
                password=DB_PASS,
                dbname=DB_NAME,
                port=5432,
            )
        conn.autocommit = True
        cur = conn.cursor()

        # Ensure expected optional columns exist (safe no-op if already present)
        try:
            cur.execute("ALTER TABLE market_history ADD COLUMN IF NOT EXISTS avg_price DOUBLE PRECISION;")
            cur.execute("ALTER TABLE market_history ADD COLUMN IF NOT EXISTS max_velocity DOUBLE PRECISION;")
            cur.execute("ALTER TABLE market_history ADD COLUMN IF NOT EXISTS median_spread DOUBLE PRECISION;")
            cur.execute("ALTER TABLE market_history ADD COLUMN IF NOT EXISTS order_book_imbalance DOUBLE PRECISION;")
        except Exception:
            # If market_history doesn't exist yet, the later queries will surface a clear error.
            pass
        
        # 1. Count before
        cur.execute("SELECT count(*) FROM market_history;")
        before = cur.fetchone()[0]
        logger.info(f"Rows before scrub: {before}")
        
        # 2. Filter High Price Outliers (SKINs, Keepstars, errors) > 49B
        # Prefer avg_price if present, otherwise fall back to close.
        logger.info("Scrubbing > 49B ISK history rows...")
        cur.execute("DELETE FROM market_history WHERE COALESCE(avg_price, close) > 49000000000;")
        deleted_high = cur.rowcount
        logger.info(f"Deleted {deleted_high} high-value outliers.")
        
        # 3. Filter Low Volume Noise (< 5)
        # Note: market_history.volume is a derived depth/volume proxy.
        logger.info("Scrubbing < 5 volume history rows...")
        cur.execute("DELETE FROM market_history WHERE COALESCE(volume, 0) < 5;")
        deleted_low = cur.rowcount
        logger.info(f"Deleted {deleted_low} low-volume items.")
        
        # 4. Count after
        cur.execute("SELECT count(*) FROM market_history;")
        after = cur.fetchone()[0]
        logger.info(f"Rows after scrub: {after}")
        logger.info(f"Total Removed: {before - after}")

        # 5. Export cleaned dataset to Parquet
        logger.info("Exporting cleaned sequence dataset to Parquet...")
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        if DATABASE_URL:
            engine = sqlalchemy.create_engine(DATABASE_URL)
            df = pd.read_sql_query(
                "SELECT * FROM market_history ORDER BY timestamp ASC, type_id",
                engine,
            )
        else:
            df = pd.read_sql_query(
                "SELECT * FROM market_history ORDER BY timestamp ASC, type_id",
                conn,
            )
        df.to_parquet(OUTPUT_PATH, index=False)
        logger.info(f"✅ Wrote: {OUTPUT_PATH} (rows={len(df)})")
        
        cur.close()
        conn.close()
        
    except Exception as e:
        logger.error(f"Scrub failed: {e}")

if __name__ == "__main__":
    scrub_data()
