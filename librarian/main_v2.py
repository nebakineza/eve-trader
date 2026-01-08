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
# WATCHLIST_TYPE_IDS = [34, 35, 36, 37, 38, 39, 40, 29668, 44992] 
# Processing all items in stream to capture Top Volume

async def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("LibrarianMainV2")

    db_url = os.getenv("DATABASE_URL", "postgresql://eve_user:eve_password@localhost:5432/eve_market_data")
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    scraper = LibrarianESIScraper(db_url)
    bridge = DataBridge(db_url, redis_url)
    analyst = AFF_V2(redis_client=bridge.redis_client)

    logger.info("Librarian Service V2 Started.")
    logger.info("Signal Shield Active (orders): window=24h z>3.0 -> drop per-type tick")

    last_history_fetch = datetime.min
    first_run = True

    while True:
        try:
            if not first_run:
                # 5 minute cycle (Temporary: 60s for Recovery)
                await asyncio.sleep(60) 
            first_run = False
            now = datetime.utcnow()
            
            # Clear old snapshot before ingestion
            await bridge.clear_table("market_orders")

            logger.info("Fetching Order Snapshot (Streaming)...")
            full_snapshot = []
            batch_count = 0
            
            # Pass empty list to fetch all items in region
            async for batch in scraper.fetch_market_snapshot_stream(REGION_ID, []):
                if not batch: continue
                clean_batch = scraper.validator_cleaner(batch, data_type="orders")

                # Signal Shield: drop manipulated per-type ticks (24h rolling z-score on mid)
                clean_batch = scraper.signal_shield_filter_orders(
                    clean_batch,
                    redis_client=bridge.redis_client,
                    window_hours=24,
                    z_threshold=3.0,
                )
                await bridge.push_to_postgres("market_orders", clean_batch)
                full_snapshot.extend(clean_batch)
                batch_count += 1
                logger.debug(f"Streamed Batch #{batch_count}: {len(clean_batch)} items.")

            logger.info(f"[Librarian] Total Snapshot size: {len(full_snapshot)} items.")
            # await bridge.generate_history_tick()

            if full_snapshot:
                import pandas as pd
                df_orders = pd.DataFrame(full_snapshot)
                
                # Dynamic Type Discovery
                unique_types = df_orders['type_id'].unique()
                logger.info(f"Analyst found {len(unique_types)} unique active items. Processing features...")
                
                # Limit to Top 500 by Volume to prevent CPU choke if scaling too fast
                # (Simple heuristic: most rows = most active?) -> No, volume quantity.
                # Just processing first 500 unique found for now to ensure coverage
                count = 0
                max_items = 500
                
                for type_id in unique_types:
                    if count >= max_items: break
                    
                    type_orders = df_orders[df_orders['type_id'] == type_id]
                    if not type_orders.empty:
                        try:
                            # Calculate Spread & Prices
                            bids = type_orders[type_orders['is_buy_order'] == True]
                            asks = type_orders[type_orders['is_buy_order'] == False]
                            
                            best_bid = float(bids['price'].max()) if not bids.empty else 0.0
                            best_ask = float(asks['price'].min()) if not asks.empty else 0.0
                            
                            # Safe Market Price (Mid or Ask)
                            market_price = best_ask if best_ask > 0 else (best_bid if best_bid > 0 else 0)
                            
                            # Calculate Imbalance
                            imbalance = analyst.order_book_imbalance(type_orders)
                            
                            # --- Price Velocity / Spread Logic ---
                            # Try to fetch previous feature to calculate velocity
                            # If unavailable, fallback to Bid/Ask Spread
                            
                            spread_val = 0.0
                            use_velocity = False
                            
                            # 1. Try Velocity (Previous Price vs Current)
                            prev_feature_json = bridge.redis_client.get(f"features:{type_id}")
                            if prev_feature_json:
                                import json
                                try:
                                    prev_data = json.loads(prev_feature_json)
                                    prev_price = float(prev_data.get("price", 0))
                                    if prev_price > 0 and market_price > 0:
                                        # Velocity: (Current - Prev) / Prev
                                        spread_val = (market_price - prev_price) / prev_price
                                        use_velocity = True
                                except:
                                    pass

                            # 2. Fallback to Bid/Ask Spread if Velocity not possible or effectively zero (same tick)
                            # Or if the user strictly wanted "Previous missing -> fallback"
                            if not use_velocity or spread_val == 0.0:
                                if best_bid > 0 and best_ask > 0:
                                    spread_val = (best_ask - best_bid) / best_bid

                            features = {
                                "type_id": int(type_id),
                                "price": market_price,
                                "best_bid": best_bid,
                                "best_ask": best_ask,
                                "spread": spread_val,
                                "imbalance": imbalance,
                                "timestamp": now.timestamp()
                            }
                            # Use new key format: features:{type_id}
                            
                            # --- Integrity Filter (User Request) ---
                            # Logic: If spread > 500% (5.0) or (implied) low volume, do not pass to "Alpha Pulse"
                            
                            # Add 'volume' to features for validator (Current Market Depth Volume as proxy)
                            current_vol = int(type_orders['volume_remain'].sum())
                            
                            features = {
                                "type_id": int(type_id),
                                "price": market_price,
                                "best_bid": best_bid,
                                "best_ask": best_ask,
                                "spread": spread_val,
                                "imbalance": imbalance,
                                "timestamp": now.timestamp(),
                                "volume": current_vol
                            }
                            
                            valid_pulse = analyst.validate_market_integrity(features)
                            features["valid_pulse"] = valid_pulse

                            # Keep features alive across long snapshot ingestion; strategist relies on these keys.
                            bridge.push_to_redis(f"features:{type_id}", features, ttl=3600)
                            count += 1
                        except Exception as ex:
                            logger.error(f"Feature Gen Error {type_id}: {ex}")
                            
                logger.info(f"Refreshed features for {count} items in Redis.")
                
            # Heartbeat Update (Confirmed End of Loop)
            bridge.redis_client.set("system:heartbeat:latest", str(now.timestamp()))
            bridge.redis_client.set("system:heartbeat:last_update_minutes", "0.0")
            logger.info("💓 System Heartbeat Updated.")

            if now - last_history_fetch > timedelta(hours=24):
                # For history, we might still want a restricted list or we fetch all?
                # Fetching history for 500 items takes time. Let's stick to the core watchlist for deep history for now.
                core_watchlist = [34, 35, 36, 37, 38, 39, 40, 29668, 44992]
                history = await scraper.fetch_historical_logs(REGION_ID, core_watchlist)
                clean_history = scraper.validator_cleaner(history, data_type="history")
                if clean_history:
                    await bridge.push_to_postgres("market_history", clean_history)
                    last_history_fetch = now
        
        except Exception as e:
            logger.error(f"Error in Librarian Loop: {e}")
            import traceback
            traceback.print_exc()
        
        logger.info("Cycle complete. Sleeping 60s...")
        await asyncio.sleep(60)

if __name__ == "__main__":
    asyncio.run(main())
