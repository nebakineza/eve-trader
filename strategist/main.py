import asyncio
import logging
import os
import sys
import json
import aiohttp
import sqlalchemy
from sqlalchemy import text
from datetime import datetime
from urllib.parse import urlparse

sys.path.append(os.getcwd())

from storage.bridge import DataBridge
from strategist.agent import StrategistRLAgent
from strategist.logic import update_market_radar
from scripts.notifier import send_alert
from strategist.shadow_manager import ShadowManager
from strategist.shadow_manager import ShadowConfig


def _read_shadow_threshold(*, bridge: DataBridge, fallback: float) -> float:
    # Primary key per spec
    keys = [
        "system:shadow:confidence_threshold",
        # Back-compat aliases (safe if operators already wrote a different key)
        "system:shadow:threshold",
        "system:shadow:confidence",
    ]
    for k in keys:
        try:
            v = bridge.redis_client.get(k)
            if v is None:
                continue
            val = float(v)
            if 0.0 <= val <= 1.0:
                return val
        except Exception:
            continue
    return float(fallback)

async def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("StrategistMain")
    
    db_url = os.getenv("DATABASE_URL", "postgresql://eve_user:eve_password@localhost:5432/eve_market_data")
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    def _redact_db_url(url: str) -> str:
        try:
            if "://" not in url or "@" not in url:
                return url
            scheme, rest = url.split("://", 1)
            creds, hostpart = rest.split("@", 1)
            if ":" in creds:
                user, _pwd = creds.split(":", 1)
                creds = f"{user}:***"
            return f"{scheme}://{creds}@{hostpart}"
        except Exception:
            return "<redacted>"

    logger.info(f"DB URL: {_redact_db_url(db_url)}")

    oracle_host = os.getenv("ORACLE_HOST", "http://oracle:8000")
    logger.info(f"Oracle host: {oracle_host}")
    
    # DB Engine for Direct Queries
    db_engine = sqlalchemy.create_engine(db_url.replace("asyncpg", "psycopg2")) if "postgresql" in db_url else None
    
    # Wait for DB/Redis capabilities
    bridge = DataBridge(db_url, redis_url)
    agent = StrategistRLAgent(system_mode=os.getenv("SYSTEM_MODE", "SHADOW"), db_bridge=bridge)

    shadow = None
    if db_url and "postgresql" in db_url:
        try:
            shadow = ShadowManager(db_url)
            shadow.ensure_schema()
        except Exception as e:
            logger.exception("Shadow ledger unavailable")
    
    # Dynamic Watchlist (User Request: Full 500)
    # WATCHLIST = [34, 35, 36, 37, 38, 39, 40, 29668, 44992] 

    logger.info(f"Listening for Oracle Signals (SYSTEM_MODE={os.getenv('SYSTEM_MODE', 'SHADOW')})")
    logger.info("Strategist Main Loop Started (Rate: 5 mins)")

    oracle_is_up = False

    while True:
        try:
            oracle_success_this_tick = False
            oracle_offline_logged = False
            oracle_non_200_logged = False

            # Dynamic thresholding (Bayesian tuner can update Redis continuously)
            if shadow is not None:
                try:
                    env_fallback = float(os.getenv("SHADOW_CONFIDENCE_THRESHOLD", "0.6"))
                except Exception:
                    env_fallback = 0.6

                thr = _read_shadow_threshold(bridge=bridge, fallback=env_fallback)
                if thr != shadow.config.confidence_threshold:
                    shadow.config = ShadowConfig(
                        horizon_minutes=shadow.config.horizon_minutes,
                        confidence_threshold=float(thr),
                        fee_rate=shadow.config.fee_rate,
                    )
                    logger.info(f"Shadow threshold updated from Redis: {thr:.3f}")

            if shadow is not None:
                try:
                    # Traffic Guardrails: widen stop-loss during high-volatility warfare.
                    warfare_isk = 0.0
                    try:
                        v = bridge.redis_client.get("system:macro:warfare")
                        if v is not None:
                            warfare_isk = float(v)
                    except Exception:
                        warfare_isk = 0.0

                    warfare_threshold = float(os.getenv("SHADOW_WARFARE_THRESHOLD_ISK", "0"))
                    war_stop_loss_pct = float(os.getenv("SHADOW_WAR_STOP_LOSS_PCT", "0.03"))
                    stop_loss_override = None
                    if warfare_threshold > 0 and warfare_isk > warfare_threshold:
                        stop_loss_override = war_stop_loss_pct
                        logger.warning(
                            "Traffic Guardrail ACTIVE: warfare_isk=%.2f > threshold=%.2f -> stop_loss=%.3f",
                            warfare_isk,
                            warfare_threshold,
                            war_stop_loss_pct,
                        )

                    settled = shadow.settle_pending_trades(stop_loss_pct_override=stop_loss_override)
                    if settled:
                        logger.info(f"Shadow ledger: settled {settled} pending trades")
                except Exception as e:
                    logger.warning(f"Shadow settlement error: {e}")

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

            # If Redis feature keys are absent (e.g., long ingestion window + TTL expiry),
            # fall back to DB-derived watchlist so shadow trading can continue.
            if not WATCHLIST and db_engine:
                try:
                    with db_engine.connect() as conn:
                        rows = conn.execute(
                            text(
                                """
                                SELECT DISTINCT type_id
                                FROM market_orders
                                WHERE type_id IS NOT NULL
                                LIMIT 500
                                """
                            )
                        ).fetchall()
                    WATCHLIST = [int(r[0]) for r in rows if r and r[0] is not None]
                    logger.info(f"Strategist watchlist fallback: {len(WATCHLIST)} type_ids from market_orders")
                except Exception as e:
                    logger.warning(f"Strategist watchlist fallback failed: {e}")
                       
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

                # If we still don't have a price, derive a basic quote from the current order book snapshot.
                if db_engine and (not features or 'price' not in features):
                    try:
                        with db_engine.connect() as conn:
                            row = conn.execute(
                                text(
                                    """
                                    SELECT
                                        MAX(price) FILTER (WHERE is_buy_order)  AS best_bid,
                                        MIN(price) FILTER (WHERE NOT is_buy_order) AS best_ask,
                                        SUM(volume_remaining) FILTER (WHERE is_buy_order) AS buy_vol,
                                        SUM(volume_remaining) FILTER (WHERE NOT is_buy_order) AS sell_vol
                                    FROM market_orders
                                    WHERE type_id = :tid
                                    """
                                ),
                                {"tid": type_id},
                            ).fetchone()

                        if row:
                            best_bid, best_ask, buy_vol, sell_vol = row
                            best_bid = float(best_bid) if best_bid is not None else 0.0
                            best_ask = float(best_ask) if best_ask is not None else 0.0
                            buy_vol = float(buy_vol) if buy_vol is not None else 0.0
                            sell_vol = float(sell_vol) if sell_vol is not None else 0.0
                            market_price = best_ask if best_ask > 0 else (best_bid if best_bid > 0 else 0.0)
                            if market_price > 0:
                                features.setdefault('best_bid', best_bid)
                                features.setdefault('best_ask', best_ask)
                                features.setdefault('price', market_price)
                                features.setdefault('imbalance', (buy_vol + 1.0) / (sell_vol + 1.0))
                    except Exception as e:
                        logger.warning(f"Order-book fallback failed for {type_id}: {e}")

                # Ensure we have minimum data for Oracle
                if not features or 'price' not in features:
                    # Logic: If we have absolutely no data (no Redis, no History), skip
                    continue
                    
                current_price = features.get('price', 0.0)
                # Prepare Payload for Oracle
                # (Mocking dimensions to match Oracle's expectation if simple)
                # Real features would serve just the raw dict if Oracle handles it.

                # 2. Call Oracle
                host = oracle_host
                oracle_response = None

                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(f"{host}/predict", json={"features": features}) as resp:
                            if resp.status == 200:
                                oracle_response = await resp.json()
                                oracle_success_this_tick = True

                                if not oracle_is_up:
                                    try:
                                        parsed = urlparse(host)
                                        origin = parsed.hostname or host
                                    except Exception:
                                        origin = host
                                    logger.info(f"Inference Received from {origin}")
                                    oracle_is_up = True
                            else:
                                try:
                                    body = await resp.text()
                                except Exception:
                                    body = ""
                                if not oracle_non_200_logged:
                                    logger.warning(
                                        "Oracle non-200: status=%s host=%s body=%s",
                                        resp.status,
                                        host,
                                        (body[:200] if body else ""),
                                    )
                                    oracle_non_200_logged = True
                except Exception as e:
                    if not oracle_offline_logged:
                        logger.warning(f"Oracle offline (host={host}): {e}")
                        oracle_offline_logged = True

                # Fallback / Mock if Oracle not ready yet (during bootstrap)
                if not oracle_response:
                    continue

                # 3. Analyze Prediction
                predicted_price = oracle_response.get("predicted_price", current_price)
                confidence = oracle_response.get("confidence", 0.0)

                if shadow is not None:
                    try:
                        warfare_isk = None
                        global_cap_isk = None
                        try:
                            v = bridge.redis_client.get("system:macro:warfare")
                            if v is not None:
                                warfare_isk = float(v)
                        except Exception:
                            warfare_isk = None

                        try:
                            v = bridge.redis_client.get("system:macro:global_market_cap")
                            if v is not None:
                                global_cap_isk = float(v)
                        except Exception:
                            global_cap_isk = None

                        shadow.maybe_record_intended_trade(
                            type_id=type_id,
                            predicted_price=float(predicted_price) if predicted_price is not None else None,
                            current_price=float(current_price) if current_price is not None else None,
                            confidence=float(confidence) if confidence is not None else 0.0,
                            warfare_isk_60m=warfare_isk,
                            global_market_cap_isk=global_cap_isk,
                        )
                    except Exception as e:
                        logger.warning(f"Shadow record error for {type_id}: {e}")

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

            if oracle_is_up and not oracle_success_this_tick:
                logger.warning(f"Oracle connection lost (host={oracle_host})")
                oracle_is_up = False
                
        except Exception as e:
            logger.error(f"Strategist Loop Error: {e}")
            
        # Wait 5 minutes for next tick
        await asyncio.sleep(300) 


if __name__ == "__main__":
    asyncio.run(main())
