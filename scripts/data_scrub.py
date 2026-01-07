import psycopg2
import os
import logging

# Data Scrubber Script
# Removes Outliers (>49B ISK) and Low Volume (<5 units) items
# To ensure quality of the 1.1M row dataset.

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DataScrubber")

# Env vars usually available in container, but for manual run:
DB_HOST = os.getenv("DB_HOST", "db") 
DB_USER = "eve_user"

DB_PASS = "afterburn118921" # From earlier messages/env
DB_NAME = "eve_market_data"

def scrub_data():
    try:
        # Connect
        # Note: If running on host, map 5432 is exposed.
        conn = psycopg2.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASS,
            dbname=DB_NAME,
            port=5432
        )
        conn.autocommit = True
        cur = conn.cursor()
        
        # 1. Count before
        cur.execute("SELECT count(*) FROM market_orders;")
        before = cur.fetchone()[0]
        logger.info(f"Rows before scrub: {before}")
        
        # 2. Filter High Price Outliers (SKINs, Keepstars, errors) > 49B
        logger.info("Scrubbing > 49B ISK items...")
        cur.execute("DELETE FROM market_orders WHERE price > 49000000000;")
        deleted_high = cur.rowcount
        logger.info(f"Deleted {deleted_high} high-value outliers.")
        
        # 3. Filter Low Volume Noise (< 5 units remaining)
        # Note: 'volume_remaining'
        logger.info("Scrubbing < 5 volume items...")
        cur.execute("DELETE FROM market_orders WHERE volume_remaining < 5;")
        deleted_low = cur.rowcount
        logger.info(f"Deleted {deleted_low} low-volume items.")
        
        # 4. Count after
        cur.execute("SELECT count(*) FROM market_orders;")
        after = cur.fetchone()[0]
        logger.info(f"Rows after scrub: {after}")
        logger.info(f"Total Removed: {before - after}")
        
        cur.close()
        conn.close()
        
    except Exception as e:
        logger.error(f"Scrub failed: {e}")

if __name__ == "__main__":
    scrub_data()
