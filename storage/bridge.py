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

    async def stream_to_db(self, data, table="market_orders"):
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
                        issued = datetime.fromisoformat(issued.replace('Z', '+00:00'))
                    records.append((
                        timestamp, d.get('order_id'), d.get('type_id'), d.get('region_id', 10000002), 
                        d.get('location_id'), float(d.get('price', 0)), int(d.get('volume_remain', 0)), 
                        d.get('is_buy_order'), issued, d.get('duration')
                    ))
                
                async with conn.transaction():
                    await conn.copy_records_to_table('market_orders', records=records, columns=columns)
                self.logger.info(f"✅ [DataBridge] Streamed {len(records)} orders to DB.")
            await conn.close()
        except Exception as e:
            self.logger.error(f"Postgres Write Error: {e}")
