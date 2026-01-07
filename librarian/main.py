import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.getcwd())

from librarian.scraper import LibrarianESIScraper
from storage.bridge import DataBridge
from analyst.simple_feature import AnalystFeatureFactory as AFF_V2

# Configuration
REGION_ID = 10000002 # The Forge
JITA_ID = 60003760
# Top 500 Type IDs - In prod this would be loaded from a file/DB
# Mocking a few for demonstration
# FORCE REBUILD 12345
WATCHLIST_TYPE_IDS = [34, 35, 36, 37, 38, 39, 40, 29668, 44992] 

async def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("LibrarianMain")

    db_url = os.getenv("DATABASE_URL", "postgresql://eve_user:eve_password@localhost:5432/eve_market_data")
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    scraper = LibrarianESIScraper(db_url)
    bridge = DataBridge(db_url, redis_url)
    analyst = AFF_V2(redis_client=bridge.redis_client)

    logger.info("Librarian Service Started.")
    logger.info("Signal Shield Active (orders): window=24h z>3.0 -> drop per-type tick")

    last_history_fetch = datetime.min
    
    # Run immediately on startup (Initial Burst)
    first_run = True

    while True:
        try:
            if not first_run:
                # Wait 5 minutes for next tick
                await asyncio.sleep(300) 
            
            first_run = False
            
            now = datetime.utcnow()
            
            # --- 1. Snapshot Loop (Every 5 mins) ---
            logger.info("Fetching Order Snapshot (Streaming)...")
            
            full_snapshot = []
            batch_count = 0
            
            # Simple TypeID Map for Verbose Logging
            type_names = {
                34: "Tritanium", 35: "Pyerite", 36: "Mexallon", 37: "Isogen", 
                38: "Nocxium", 39: "Zydrine", 40: "Megacyte", 
                29668: "Skill Injector", 44992: "PLEX"
            }

            # Stream Ingestion Loop
            async for batch in scraper.fetch_market_snapshot_stream(REGION_ID, WATCHLIST_TYPE_IDS):
                if not batch:
                    continue
                
                # Validator/Cleaner
                clean_batch = scraper.validator_cleaner(batch, data_type="orders")

                # Signal Shield: drop manipulated per-type ticks (24h rolling z-score on mid)
                clean_batch = scraper.signal_shield_filter_orders(
                    clean_batch,
                    redis_client=bridge.redis_client,
                    window_hours=24,
                    z_threshold=3.0,
                )
                
                # Push Batch to DB immediately
                await bridge.push_to_postgres("market_orders", clean_batch)
                
                # Accumulate for Analyst
                full_snapshot.extend(clean_batch)
                batch_count += 1
                
                # Logging heartbeat
                last_item = clean_batch[-1] if clean_batch else {}
                t_id = last_item.get('type_id', 'Unknown')
                t_name = type_names.get(t_id, 'Unknown')
                logger.debug(f"Streamed Batch #{batch_count}: {len(clean_batch)} items. Last: {t_name}")

            logger.info(f"[Librarian] Total Snapshot size: {len(full_snapshot)} items.")
            
            # --- 1b. Force-Populate History ---
            # Close the loop -> Turn orders into History Candles
            await bridge.generate_history_tick()

            # --- 2. Bridge to Redis & DB (Analyst Phase) ---
            # We already pushed raw orders to DB in stream.
            # Now we run Analyst on the full memory view.
            
            if full_snapshot:
                # 1b. Run Analyst Pipeline
                # Convert list of dicts to DataFrame for Analyst
                import pandas as pd
                df_orders = pd.DataFrame(full_snapshot)
                
                # Generate Features per Type ID
                for type_id in WATCHLIST_TYPE_IDS:
                    type_orders = df_orders[df_orders['type_id'] == type_id]
                    if not type_orders.empty:
                        # Calculate features
                        imbalance = analyst.order_book_imbalance(type_orders)
                        
                        # Pack generic features
                        # In a real app we'd aggregate history here too for full features
                        features = {
                            "type_id": type_id,
                            "price": float(type_orders['price'].min()), # Best Ask approx
                            "imbalance": imbalance,
                            "timestamp": now.timestamp()
                        }
                        
                        # Push to Redis for Strategist/Oracle
                        bridge.push_to_redis(f"features:{type_id}", features, ttl=300)
                        
                logger.info(f"Refreshed features for {len(WATCHLIST_TYPE_IDS)} items in Redis.")

            # --- 2. History Loop (Every 24 hours) ---
            if now - last_history_fetch > timedelta(hours=24):
                logger.info("Fetching Historical Logs...")
                history = await scraper.fetch_historical_logs(REGION_ID, WATCHLIST_TYPE_IDS)
                clean_history = scraper.validator_cleaner(history, data_type="history")
                
                if clean_history:
                    await bridge.push_to_postgres("market_history", clean_history)
                    last_history_fetch = now
        
        except Exception as e:
            logger.error(f"Error in Librarian Loop: {e}")
        
        # Wait 5 minutes
        await asyncio.sleep(300)

if __name__ == "__main__":
    asyncio.run(main())
