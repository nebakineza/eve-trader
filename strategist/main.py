import asyncio
import logging
import os
import sys
import json
import aiohttp
import sqlalchemy
from sqlalchemy import text
from datetime import datetime

sys.path.append(os.getcwd())

from storage.bridge import DataBridge
from strategist.agent import StrategistRLAgent
from strategist.logic import update_market_radar
from scripts.notifier import send_alert

async def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("StrategistMain")
    
    db_url = os.getenv("DATABASE_URL", "postgresql://eve_user:eve_password@localhost:5432/eve_market_data")
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    
    # DB Engine for Direct Queries
    db_engine = sqlalchemy.create_engine(db_url.replace("asyncpg", "psycopg2")) if "postgresql" in db_url else None
    
    # Wait for DB/Redis capabilities
    bridge = DataBridge(db_url, redis_url)
    agent = StrategistRLAgent(system_mode=os.getenv("SYSTEM_MODE", "SHADOW"), db_bridge=bridge)
    
    # Dynamic Watchlist (User Request: Full 500)
    # WATCHLIST = [34, 35, 36, 37, 38, 39, 40, 29668, 44992] 

    logger.info("Strategist Main Loop Started (Rate: 5 mins)")

    while True:
        try:
            logger.info("🔍 Scanning Market Opportunities...")
            
            # Fetch ALL active items from Redis keys
            all_keys = bridge.redis_client.keys("features:*")
            # Extract IDs (features:123 -> 123)
            WATCHLIST = []
            for k in all_keys:
                if len(k.split(':')) == 2: # Ignore sub-properties
                   try:
                       WATCHLIST.append(int(k.split(':')[1]))
                   except: 
                       pass
                       
            logger.info(f"Strategist targets found: {len(WATCHLIST)}")

            for type_id in WATCHLIST:
                # 1. Fetch Features from Redis (Live Order Book Data)
                features_raw = bridge.get_from_redis(f"features:{type_id}")
                features = {}
                
                if features_raw:
                    features = json.loads(features_raw) if isinstance(features_raw, str) else features_raw
                
                # 2. Enrich with History (Resilience Fix)
                # If Redis is empty or just to add trend data, we pull the last history row.
                if db_engine:
                    try:
                        with db_engine.connect() as conn:
                            query = text("""
                                SELECT close, volume, high, low 
                                FROM market_history 
                                WHERE type_id = :tid 
                                ORDER BY timestamp DESC 
                                LIMIT 1
                            """)
                            result = conn.execute(query, {"tid": type_id}).fetchone()
                            if result:
                                features['last_close'] = result[0]
                                features['last_volume'] = result[1]
                                # Fallback if live price is missing
                                if 'price' not in features:
                                    features['price'] = result[0]
                    except Exception as e:
                        logger.warning(f"History fetch failed for {type_id}: {e}")

                # Ensure we have minimum data for Oracle
                if not features or 'price' not in features:
                    # Logic: If we have absolutely no data (no Redis, no History), skip
                    continue
                    
                current_price = features.get('price', 0.0)
                    
                # Prepare Payload for Oracle
                    # (Mocking dimensions to match Oracle's expectation if simple)
                    # Real features would serve just the raw dict if Oracle handles it.
                    
                    # 2. Call Oracle
                    host = os.getenv("ORACLE_HOST", "http://oracle:8000")
                    oracle_response = None
                    
                    try:
                        async with aiohttp.ClientSession() as session:
                             async with session.post(f"{host}/predict", json={"features": features}) as resp:
                                if resp.status == 200:
                                    oracle_response = await resp.json()
                    except Exception as e:
                        logger.warning(f"Oracle offline for {type_id}: {e}")

                    # Fallback / Mock if Oracle not ready yet (during bootstrap)
                    if not oracle_response:
                        continue

                    # 3. Analyze Prediction
                    predicted_price = oracle_response.get("predicted_price", current_price)
                    confidence = oracle_response.get("confidence", 0.0)
                    
                    # Calculate Projected Profit (Unit Profit * Volume assumed or Lot Size)
                    # For filtering, let's assume a standard lot of 1,000 units for cheap items, 1 for expensive?
                    # Using a simplified 'Potenial' metric: (Predicted - Current) * 10000 units
                    potential_unit_profit = predicted_price - current_price
                    # Standard batch for calculation
                    batch_size = 10000 
                    predicted_profit = potential_unit_profit * batch_size
                    
                    # 4. Trigger Signals
                    signal_strength = confidence
                    status = "WATCH"
                    
                    # --- CACHE REFRESH FIX ---
                    # Explicitly update status in Redis to avoid stale keys
                    status_key = f"market_radar:{type_id}"
                    bridge.redis_client.set(status_key, json.dumps({
                        "type_id": type_id,
                        "status": "WATCH", # Reset to WATCH in loop to re-evaluate
                        "timestamp": datetime.utcnow().isoformat(),
                        "price": current_price,
                        "predicted": predicted_price,
                        "confidence": confidence
                    }), ex=300) # 5 min expiry matches loop

                    
                    # STRONG BUY Condition
                    if confidence > 0.6 and predicted_profit > 50_000_000:
                        status = "ENTER"
                        logger.info(f"🚀 STRONG BUY SIGNAL: {type_id} | Conf: {confidence:.2f}")
                        
                        # Send Alert
                        # Item Name map (Mock)
                        item_names = {34: "Tritanium", 35: "Pyerite", 36: "Mexallon", 37: "Isogen", 38: "Nocxium", 39: "Zydrine", 40: "Megacyte"}
                        name = item_names.get(type_id, f"TypeID {type_id}")
                        
                        send_alert(name, predicted_price, confidence, predicted_profit)

                    # 5. Persist to Market Radar
                    # Need a connection pool. DataBridge has one or we create one?
                    # DataBridge uses asyncpg but we need to access the pool.
                    # Assuming bridge.db_pool is accessible or we use a fresh connection.
                    # For now, let's assume bridge handles it or we instantiate a pool here if needed.
                    # To keep it simple, we'll access the private _pool if available or create one.
                    if hasattr(bridge, 'pool'):
                        await update_market_radar(bridge.pool, type_id, signal_strength, predicted_price, current_price, status)
                
        except Exception as e:
            logger.error(f"Strategist Loop Error: {e}")
            
        # Wait 5 minutes for next tick
        await asyncio.sleep(300) 


if __name__ == "__main__":
    asyncio.run(main())
