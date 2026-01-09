import asyncio
import logging
import os
import sys
import json
import time
from datetime import datetime, timedelta, timezone
from collections import defaultdict, deque

# Add project root to path (ensure it wins over sibling modules like librarian/analyst.py)
sys.path.insert(0, os.getcwd())

if os.getenv("DEBUG_IMPORTS", "0") == "1":
    import importlib.util

    spec = importlib.util.find_spec("analyst")
    print("DEBUG_IMPORTS cwd=", os.getcwd())
    print("DEBUG_IMPORTS sys.path[0:5]=", sys.path[0:5])
    print(
        "DEBUG_IMPORTS analyst spec=",
        getattr(spec, "origin", None),
        getattr(spec, "submodule_search_locations", None),
    )
    raise SystemExit(0)

import aiohttp
import re

from librarian.scraper import LibrarianESIScraper
from storage.bridge import DataBridge
from analyst.simple_feature import AnalystFeatureFactory as AFF_V2
from oracle.indicators import latest_indicators_from_prices

# Configuration
REGION_ID = 10000002 # The Forge
JITA_ID = 60003760
# WATCHLIST_TYPE_IDS = [34, 35, 36, 37, 38, 39, 40, 29668, 44992] 
# Processing all items in stream to capture Top Volume

async def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("LibrarianMainV2")

    db_url = os.getenv("DATABASE_URL", "postgresql://eve_user:eve_pass@localhost:5432/eve_market_data")
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    scraper = LibrarianESIScraper(db_url)
    bridge = DataBridge(db_url, redis_url)
    analyst = AFF_V2(redis_client=bridge.redis_client)

    logger.info("Librarian Service V2 Started.")
    logger.info("Signal Shield Active (orders): window=24h z>3.0 -> drop per-type tick")

    last_history_fetch = datetime.min
    last_players_poll = datetime.min
    last_warfare_poll = datetime.min
    first_run = True

    # Rolling indicator state (online). We intentionally keep this in-process to avoid DB fanout.
    price_hist: dict[int, deque[float]] = defaultdict(lambda: deque(maxlen=256))

    last_macro_history_error_log_at = 0.0

    def _append_macro_history(*, key: str, value: float | int, ts: float, maxlen: int = 100) -> None:
        nonlocal last_macro_history_error_log_at
        try:
            payload = json.dumps({"ts": float(ts), "v": value}, separators=(",", ":"))
            pipe = bridge.redis_client.pipeline()
            pipe.rpush(key, payload)
            pipe.ltrim(key, -int(maxlen), -1)
            # Keep a modest TTL so stale history doesn't linger forever.
            pipe.expire(key, int(os.getenv("MACRO_HISTORY_TTL_SECONDS", str(86400 * 7))))
            pipe.execute()
        except Exception:
            # History is best-effort; never fail the main loop.
            # But do log (rate-limited) so failures don't go unnoticed.
            now_s = time.time()
            if now_s - last_macro_history_error_log_at >= 60:
                last_macro_history_error_log_at = now_s
                logger.warning(
                    "Macro-Librarian: failed to append history key=%s",
                    key,
                    exc_info=True,
                )
            return

    async def _poll_players_online() -> int | None:
        timeout = aiohttp.ClientTimeout(total=float(os.getenv("ESI_STATUS_TIMEOUT_SECONDS", "10")))

        # MCP-first: if a kongyo2 OSINT MCP endpoint is configured, prefer it.
        # Expected JSON shape (best-effort):
        # - {"players": <int>} or {"players_online": <int>} or {"system:macro:players": <int>}
        mcp_endpoint = os.getenv("KONGYO2_OSINT_MCP_ENDPOINT", "").strip()
        if mcp_endpoint:
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(mcp_endpoint, headers={"Accept": "application/json"}) as resp:
                        if resp.status == 200:
                            payload = await resp.json()
                            if isinstance(payload, dict):
                                players = payload.get("players")
                                if players is None:
                                    players = payload.get("players_online")
                                if players is None:
                                    players = payload.get("system:macro:players")
                                if players is not None:
                                    players_i = int(float(players))
                                    if players_i >= 0:
                                        return players_i
            except Exception:
                # Fall back to ESI status.
                pass

        # Fallback (deprecated): ESI status endpoint.
        url = os.getenv("ESI_STATUS_URL", "https://esi.evetech.net/latest/status/")
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        return None
                    payload = await resp.json()
        except Exception:
            return None

        # ESI status JSON typically contains: {"players": <int>, ...}
        try:
            if isinstance(payload, dict):
                players = payload.get("players")
                if players is not None:
                    players_i = int(float(players))
                    if players_i >= 0:
                        return players_i
        except Exception:
            pass

        # Final fallback: scrape eve-offline.net (HTML). This is only used when ESI
        # doesn't provide a player count.
        eve_offline_url = os.getenv("EVE_OFFLINE_URL", "https://eve-offline.net/").strip() or "https://eve-offline.net/"
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(eve_offline_url, headers={"Accept": "text/html"}) as resp:
                    if resp.status != 200:
                        return None
                    html = await resp.text()
            # Example snippet: "Tranquility; ok (32,347 players)"
            m = re.search(r"Tranquility\s*;[^\(]*\((?P<n>[0-9][0-9,\.]*)\s+players\)", html, flags=re.IGNORECASE)
            if not m:
                return None
            raw = m.group("n")
            raw = raw.replace(",", "")
            players_i = int(float(raw))
            if players_i >= 0:
                return players_i
        except Exception:
            return None

    def _parse_kill_time(raw) -> datetime | None:
        try:
            if raw is None:
                return None
            s = str(raw).strip()
            if not s:
                return None
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            return None

    async def _poll_warfare_isk_destroyed() -> tuple[float, str] | None:
        timeout = aiohttp.ClientTimeout(total=float(os.getenv("WAR_TIMEOUT_SECONDS", "20")))

        # MCP-first: best-effort keys for "military expenditure" / ISK destroyed
        mcp_endpoint = os.getenv("KONGYO2_OSINT_MCP_ENDPOINT", "").strip()
        if mcp_endpoint:
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(mcp_endpoint, headers={"Accept": "application/json"}) as resp:
                        if resp.status == 200:
                            payload = await resp.json()
                            if isinstance(payload, dict):
                                v = payload.get("isk_destroyed")
                                if v is None:
                                    v = payload.get("warfare_isk_destroyed")
                                if v is None:
                                    v = payload.get("military_expenditure")
                                if v is None:
                                    v = payload.get("system:macro:warfare")
                                if v is not None:
                                    val = float(v)
                                    if val >= 0:
                                        return float(val), "kongyo2_osint_mcp"
            except Exception:
                # Fall back to zKillboard.
                pass

        # Fallback: zKillboard kills feed, sum zkb.totalValue in lookback window.
        base_url = os.getenv("WAR_ZKILL_URL", "https://zkillboard.com/api/kills/").rstrip("/") + "/"
        lookback_minutes = int(os.getenv("WAR_LOOKBACK_MINUTES", "60"))
        max_pages = int(os.getenv("WAR_ZKILL_MAX_PAGES", "3"))
        sleep_s = float(os.getenv("WAR_ZKILL_SLEEP_SECONDS", "1.1"))
        user_agent = os.getenv("WAR_USER_AGENT", "eve-trader/librarian (contact: ops)")
        # Some zKillboard responses omit killmail_time. When enabled, we resolve
        # killmail_time via ESI using killmail_id + zkb.hash (public endpoint).
        esi_time_lookup = os.getenv("WAR_ZKILL_ESI_TIME_LOOKUP", "1").strip().lower() in {"1", "true", "yes"}
        esi_time_lookup_max = int(os.getenv("WAR_ZKILL_ESI_TIME_LOOKUP_MAX", "30"))
        esi_base = os.getenv("ESI_BASE_URL", "https://esi.evetech.net/latest").rstrip("/")
        cutoff = datetime.now(tz=timezone.utc) - timedelta(minutes=int(lookback_minutes))

        total = 0.0
        esi_lookups = 0
        parsed_times = 0
        try:
            async with aiohttp.ClientSession(timeout=timeout, headers={"User-Agent": user_agent, "Accept": "application/json"}) as session:
                for page in range(1, max_pages + 1):
                    url = base_url if page == 1 else (base_url + f"page/{page}/")
                    try:
                        async with session.get(url) as resp:
                            if resp.status != 200:
                                break
                            data = await resp.json()
                    except Exception:
                        break

                    if not isinstance(data, list) or not data:
                        break

                    async def _killmail_time_via_esi(killmail_id: int, km_hash: str) -> datetime | None:
                        nonlocal esi_lookups
                        if not esi_time_lookup:
                            return None
                        if esi_time_lookup_max > 0 and esi_lookups >= esi_time_lookup_max:
                            return None
                        if not km_hash:
                            return None
                        esi_lookups += 1
                        url = f"{esi_base}/killmails/{int(killmail_id)}/{str(km_hash).strip()}/"
                        try:
                            async with session.get(url) as r2:
                                if r2.status != 200:
                                    return None
                                payload = await r2.json()
                            if isinstance(payload, dict):
                                return _parse_kill_time(payload.get("killmail_time") or payload.get("killmailTime"))
                        except Exception:
                            return None
                        return None

                    oldest_in_page = None
                    for item in data:
                        if not isinstance(item, dict):
                            continue
                        dt = _parse_kill_time(item.get("killmail_time") or item.get("killmailTime"))
                        if dt is None:
                            # zKillboard schema variant: only killmail_id + zkb.hash.
                            try:
                                killmail_id = int(item.get("killmail_id") or item.get("killmailID") or 0)
                            except Exception:
                                killmail_id = 0
                            km_hash = ((item.get("zkb") or {}) if isinstance(item.get("zkb"), dict) or item.get("zkb") is None else {}).get("hash")
                            if killmail_id and km_hash:
                                dt = await _killmail_time_via_esi(killmail_id, str(km_hash))
                        if dt is None:
                            continue
                        parsed_times += 1
                        if oldest_in_page is None or dt < oldest_in_page:
                            oldest_in_page = dt
                        if dt < cutoff:
                            continue

                        zkb = item.get("zkb") or {}
                        val = zkb.get("totalValue")
                        if val is None:
                            val = zkb.get("total_value")
                        try:
                            v = float(val)
                        except Exception:
                            continue
                        if v >= 0:
                            total += v

                    if oldest_in_page is not None and oldest_in_page < cutoff:
                        break
                    await asyncio.sleep(max(0.0, sleep_s))

            # If we couldn't parse ANY timestamps at all, treat this as a fetch/parse failure
            # rather than "0 ISK destroyed"; do not clobber the last known good value.
            if parsed_times == 0:
                return None

            return float(total), ("zkillboard" if esi_lookups == 0 else "zkillboard+esi")
        except Exception:
            return None

    async def _poll_jita_activity() -> dict[str, int] | None:
        """Proxy for market density: ship jumps + kills in Jita system (30000142)."""
        timeout = aiohttp.ClientTimeout(total=float(os.getenv("JITA_ACTIVITY_TIMEOUT_SECONDS", "10")))
        jita_system_id = int(os.getenv("JITA_SYSTEM_ID", "30000142"))
        base = os.getenv("ESI_BASE_URL", "https://esi.evetech.net/latest").rstrip("/")
        url_jumps = f"{base}/universe/system_jumps/"
        url_kills = f"{base}/universe/system_kills/"

        try:
            async with aiohttp.ClientSession(timeout=timeout, headers={"Accept": "application/json"}) as session:
                async with session.get(url_jumps) as resp:
                    if resp.status != 200:
                        return None
                    jumps = await resp.json()
                async with session.get(url_kills) as resp:
                    if resp.status != 200:
                        return None
                    kills = await resp.json()

            ship_jumps = 0
            if isinstance(jumps, list):
                for row in jumps:
                    if isinstance(row, dict) and int(row.get("system_id") or 0) == jita_system_id:
                        ship_jumps = int(row.get("ship_jumps") or 0)
                        break

            ship_kills = 0
            npc_kills = 0
            pod_kills = 0
            if isinstance(kills, list):
                for row in kills:
                    if isinstance(row, dict) and int(row.get("system_id") or 0) == jita_system_id:
                        ship_kills = int(row.get("ship_kills") or 0)
                        npc_kills = int(row.get("npc_kills") or 0)
                        pod_kills = int(row.get("pod_kills") or 0)
                        break

            return {
                "ship_jumps": max(0, ship_jumps),
                "ship_kills": max(0, ship_kills),
                "npc_kills": max(0, npc_kills),
                "pod_kills": max(0, pod_kills),
            }
        except Exception:
            return None

    while True:
        try:
            if not first_run:
                # 5 minute cycle (Temporary: 60s for Recovery)
                await asyncio.sleep(60) 
            first_run = False
            now = datetime.utcnow()

            # Macro-Librarian: poll ESI status for player population every 5 minutes.
            if now - last_players_poll >= timedelta(minutes=5):
                raw_poll = await _poll_players_online()
                players_online = None
                
                # Check for 0 or None -> Connection Hiccup -> Fallback to previous (LKG)
                if raw_poll is None or raw_poll == 0:
                    try:
                        prev_s = bridge.redis_client.get("system:macro:players")
                        if prev_s:
                            players_online = int(float(prev_s))
                            bridge.redis_client.set("system:macro:players:stale", "1", ex=600)
                            logger.warning(f"LKG Pulse Active: using stale players={players_online}")
                    except Exception:
                        pass
                else:
                    players_online = raw_poll
                    bridge.redis_client.delete("system:macro:players:stale")

                if players_online is not None and players_online > 0:
                    # Keep TTL modest so consumers can detect staleness.
                    # Update: Using longer TTL (1h) to support LKG during prolonged outages.
                    bridge.redis_client.set("system:macro:players", str(players_online), ex=3600)
                    bridge.redis_client.set("system:macro:players:updated_at", str(now.timestamp()), ex=3600)
                    # Best-effort provenance
                    source = "kongyo2_osint_mcp" if os.getenv("KONGYO2_OSINT_MCP_ENDPOINT", "").strip() else "esi_or_eve_offline"
                    bridge.redis_client.set("system:macro:players:source", str(source), ex=3600)

                    # Prime history if empty so Fundamentals charts don't read as zero.
                    # Seed two points to give the chart a stable baseline line.
                    try:
                        maxlen = int(os.getenv("MACRO_HISTORY_MAXLEN", "100"))
                        if int(bridge.redis_client.llen("system:macro:players:history") or 0) == 0:
                            _append_macro_history(
                                key="system:macro:players:history",
                                value=int(players_online),
                                ts=now.timestamp() - 300.0,
                                maxlen=maxlen,
                            )
                    except Exception:
                        pass
                    _append_macro_history(
                        key="system:macro:players:history",
                        value=int(players_online),
                        ts=now.timestamp(),
                        maxlen=int(os.getenv("MACRO_HISTORY_MAXLEN", "100")),
                    )
                    last_players_poll = now
                    logger.info("Macro-Librarian: players_online=%s", players_online)

            # Macro-Librarian: poll warfare / military expenditure signal periodically.
            warfare_poll_seconds = int(os.getenv("WAR_POLL_SECONDS", "300"))
            if now - last_warfare_poll >= timedelta(seconds=max(30, warfare_poll_seconds)):
                warfare = await _poll_warfare_isk_destroyed()
                if warfare is not None:
                    warfare_isk, warfare_source = warfare
                    bridge.redis_client.set("system:macro:warfare", str(float(warfare_isk)), ex=600)
                    bridge.redis_client.set("system:macro:warfare:updated_at", str(now.timestamp()), ex=600)
                    bridge.redis_client.set("system:macro:warfare:source", str(warfare_source), ex=600)

                    # Prime history if empty so Fundamentals charts don't read as zero.
                    try:
                        maxlen = int(os.getenv("MACRO_HISTORY_MAXLEN", "100"))
                        if int(bridge.redis_client.llen("system:macro:warfare:history") or 0) == 0:
                            _append_macro_history(
                                key="system:macro:warfare:history",
                                value=float(warfare_isk),
                                ts=now.timestamp() - 300.0,
                                maxlen=maxlen,
                            )
                    except Exception:
                        pass
                    _append_macro_history(
                        key="system:macro:warfare:history",
                        value=float(warfare_isk),
                        ts=now.timestamp(),
                        maxlen=int(os.getenv("MACRO_HISTORY_MAXLEN", "100")),
                    )
                    last_warfare_poll = now
                    logger.info(
                        "Macro-Librarian: warfare_isk_destroyed=%.2f (source=%s)",
                        float(warfare_isk),
                        str(warfare_source),
                    )

            # Macro-Librarian: Jita activity proxy (system jumps + kills)
            jita_poll_seconds = int(os.getenv("JITA_POLL_SECONDS", "300"))
            last_jita_poll_raw = bridge.redis_client.get("system:macro:jita:updated_at")
            last_jita_poll = None
            try:
                if last_jita_poll_raw:
                    last_jita_poll = datetime.fromtimestamp(float(last_jita_poll_raw), tz=timezone.utc)
            except Exception:
                last_jita_poll = None
            if last_jita_poll is None or (now.replace(tzinfo=timezone.utc) - last_jita_poll) >= timedelta(seconds=max(30, jita_poll_seconds)):
                jita = await _poll_jita_activity()
                if jita is not None:
                    bridge.redis_client.set("system:macro:jita:ship_jumps", str(int(jita.get("ship_jumps", 0))), ex=600)
                    bridge.redis_client.set("system:macro:jita:ship_kills", str(int(jita.get("ship_kills", 0))), ex=600)
                    bridge.redis_client.set("system:macro:jita:npc_kills", str(int(jita.get("npc_kills", 0))), ex=600)
                    bridge.redis_client.set("system:macro:jita:pod_kills", str(int(jita.get("pod_kills", 0))), ex=600)
                    bridge.redis_client.set("system:macro:jita:updated_at", str(now.replace(tzinfo=timezone.utc).timestamp()), ex=600)
                    bridge.redis_client.set("system:macro:jita:source", "esi_universe", ex=600)
                    logger.info(
                        "Macro-Librarian: jita_activity jumps=%d ship_kills=%d pod_kills=%d",
                        int(jita.get("ship_jumps", 0)),
                        int(jita.get("ship_kills", 0)),
                        int(jita.get("pod_kills", 0)),
                    )
            
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

                # Read macro once per cycle (avoid a Redis get per type_id).
                players_online = 0
                try:
                    v = bridge.redis_client.get("system:macro:players")
                    if v is not None:
                        players_online = int(float(v))
                except Exception:
                    players_online = 0

                warfare_isk_destroyed = 0.0
                try:
                    w = bridge.redis_client.get("system:macro:warfare")
                    if w is not None:
                        warfare_isk_destroyed = float(w)
                except Exception:
                    warfare_isk_destroyed = 0.0
                
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
                                "volume": current_vol,
                                "players_online": int(players_online),
                                "warfare_isk_destroyed": float(warfare_isk_destroyed),
                            }

                            # Online technical indicators (requires rolling price history)
                            if market_price and market_price > 0:
                                price_hist[int(type_id)].append(float(market_price))
                                ind = latest_indicators_from_prices(price_hist[int(type_id)])
                                # Only include when computed; warmup windows may be None.
                                for k, v in ind.items():
                                    if v is not None:
                                        features[k] = float(v)
                            
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
