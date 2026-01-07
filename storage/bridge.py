import os
import json
import logging
import asyncio
import redis
from datetime import datetime
import asyncpg

class DataBridge:
    def __init__(self, postgres_dsn: str, redis_url: str):
        self.pg_dsn = postgres_dsn
        self.redis_url = redis_url
        self.logger = logging.getLogger("DataBridge")
        self.redis_client = redis.Redis.from_url(self.redis_url, decode_responses=True)

    def get_from_redis(self, key: str):
        try:
            return self.redis_client.get(key)
        except Exception as e:
            self.logger.error(f"Redis Read Error ({key}): {e}")
            return None

    async def clear_table(self, table):
        try:
            conn = await asyncpg.connect(self.pg_dsn)
            await conn.execute(f"TRUNCATE TABLE {table};")
            await conn.close()
            self.logger.info(f"✅ [DataBridge] Cleared table {table}.")
        except Exception as e:
            self.logger.error(f"Failed to clear table {table}: {e}")

    async def push_to_postgres(self, table, data):
        if not data: return
        try:
            conn = await asyncpg.connect(self.pg_dsn)
            ts = datetime.utcnow()
            timestamp = ts
            
            if table == "market_orders":
                records = []
                columns = ['timestamp', 'order_id', 'type_id', 'region_id', 'location_id', 'price', 'volume_remaining', 'is_buy_order', 'issue_date', 'duration']
                
                for d in data:
                    issued = d.get('issued')
                    if isinstance(issued, str):
                        # Use naive UTC to match TIMESTAMP WITHOUT TIME ZONE
                        issued = datetime.fromisoformat(issued.replace('Z', ''))
                    
                    records.append((
                        timestamp,  
                        int(d.get('order_id', 0)), 
                        int(d.get('type_id', 0)), 
                        int(d.get('region_id', 10000002)), 
                        int(d.get('location_id', 0)), 
                        float(d.get('price', 0)), 
                        int(d.get('volume_remain', 0)), 
                        bool(d.get('is_buy_order', False)), 
                        issued, 
                        int(d.get('duration', 0))
                    ))
                
                # self.logger.info(f"Preparing to copy {len(records)} records to {table}")
                async with conn.transaction():
                    await conn.copy_records_to_table(table, records=records, columns=columns)
                self.logger.info(f"✅ [DataBridge] Streamed {len(records)} orders to DB.")
            await conn.close()
        except Exception as e:
            self.logger.error(f"Postgres Write Error: {e}")
            import traceback
            traceback.print_exc()

    async def stream_to_db(self, data, table="market_orders"):
        return await self.push_to_postgres(table, data)

    def push_to_redis(self, key, value, ttl=None):
        try:
            if isinstance(value, dict) or isinstance(value, list):
                value = json.dumps(value)
            self.redis_client.set(key, value)
            if ttl:
                self.redis_client.expire(key, ttl)
        except Exception as e:
            self.logger.error(f"Redis Write Error ({key}): {e}")
