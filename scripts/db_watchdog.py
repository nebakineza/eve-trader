import time
import os
import redis
import sqlalchemy
from sqlalchemy import text
import json
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [WATCHDOG] - %(message)s')
logger = logging.getLogger()

# Connections
REDIS_URL = os.getenv('REDIS_URL', 'redis://cache:6379/0')
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://eve_user:eve_password@db:5432/eve_market_data')

def get_redis():
    return redis.Redis.from_url(REDIS_URL, decode_responses=True)

def get_db_count():
    try:
        engine = sqlalchemy.create_engine(DATABASE_URL)
        with engine.connect() as conn:
            res = conn.execute(text("SELECT count(*) FROM market_orders")).fetchone()
            return res[0]
    except Exception as e:
        logger.error(f"DB Error: {e}")
        return None

def main():
    r = get_redis()
    logger.info("Watchdog started. Monitoring table 'market_orders'...")
    
    # History tracking: list of {'ts': int, 'count': int}
    
    while True:
        count = get_db_count()
        if count is not None:
            ts = int(time.time())
            
            # 1. Publish current state
            r.set("system:db_row_count", count)
            r.set("system:db_last_check", ts)
            
            # 2. Maintain history for velocity calculation (last 10 mins)
            entry = json.dumps({"ts": ts, "count": count})
            r.rpush("system:db_history", entry)
            r.ltrim("system:db_history", -10, -1) # Keep last 10 entries (approx 10 mins)
            
            logger.info(f"Logged count: {count}")
            
        time.sleep(60)

if __name__ == "__main__":
    time.sleep(10) # Wait for services
    main()
