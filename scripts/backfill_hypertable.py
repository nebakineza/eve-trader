import sys
import os
import argparse
import logging
import hashlib
from sqlalchemy import create_engine, text
import pandas as pd
import numpy as np

# Increase logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# TARGET: Market Orders
TARGET_TABLE = 'market_orders'

def generate_order_id(row):
    # Create a deterministic 64-bit integer ID from unique constraints
    s = f"{row.timestamp}-{row.region_id}-{row.type_id}"
    return int(hashlib.sha256(s.encode('utf-8')).hexdigest()[:15], 16)

def backfill(parquet_file):
    db_url = os.getenv("DATABASE_URL", "postgresql://eve_user:eve_pass@localhost:5432/eve_market_data")
    safe_url = db_url.split('@')[-1] if '@' in db_url else 'db'
    logger.info(f"Connecting to DB... {safe_url}")
    
    engine = create_engine(db_url)
    
    logger.info(f"Reading parquet: {parquet_file}")
    df = pd.read_parquet(parquet_file)
    
    # Synthesize Market Orders from History
    logger.info("Synthesizing Market Orders from History Data...")
    
    orders_df = pd.DataFrame()
    
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    orders_df['order_id'] = df.apply(generate_order_id, axis=1) # Generate FIRST to drop dups
    orders_df['timestamp'] = df['timestamp']
    orders_df['issue_date'] = df['timestamp']
    orders_df['type_id'] = df['type_id']
    orders_df['region_id'] = df['region_id']
    orders_df['price'] = df['avg_price']
    orders_df['volume_remaining'] = df['volume']
    orders_df['is_buy_order'] = False 
    orders_df['duration'] = 90
    orders_df['location_id'] = 60003760 

    # SAFETY: Cap INT32 columns
    logger.info("Capping INT32 columns...")
    orders_df['volume_remaining'] = orders_df['volume_remaining'].clip(upper=2147483647)
    orders_df['type_id'] = orders_df['type_id'].clip(upper=2147483647)
    orders_df['region_id'] = orders_df['region_id'].clip(upper=2147483647)
    orders_df['duration'] = orders_df['duration'].astype(int)

    # SAFETY: Ensure Float
    orders_df['price'] = orders_df['price'].astype(float)
    
    orders_df = orders_df.drop_duplicates(subset=['order_id'])
    
    logger.info(f"Orders shape: {orders_df.shape}")

    # Write to DB
    chunk_size = 5000 
    logger.info(f"Truncating and Writing to table '{TARGET_TABLE}'...")
    
    try:
        with engine.begin() as conn:
            conn.execute(text(f"TRUNCATE TABLE {TARGET_TABLE};"))
            logger.info("Table Truncated.")
            
        orders_df.to_sql(
            TARGET_TABLE,
            engine,
            if_exists='append',
            index=False,
            chunksize=chunk_size
        )
        logger.info("Success! Market Orders Backfill complete.")
        
        # Verify
        with engine.connect() as conn:
            cnt = conn.execute(text(f"SELECT count(*) FROM {TARGET_TABLE}")).scalar()
            logger.info(f"Final Table Count: {cnt}")
            
    except Exception as e:
        logger.error("Failed to write to DB")
        logger.error(str(e).split('[SQL:')[0][:500])
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Path to parquet file")
    args = parser.parse_args()
    
    backfill(args.source)
