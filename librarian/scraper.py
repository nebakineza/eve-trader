import asyncio
import aiohttp
import logging
import pandas as pd
import numpy as np
import time
from typing import List, Dict, Any

class LibrarianESIScraper:
    """
    Librarian-ESI-Scraper Module
    Purpose: High-frequency raw data ingestion.
    """
    def __init__(self, database_url: str):
        self.db_url = database_url
        self.logger = logging.getLogger("Librarian")

    async def fetch_market_snapshot_stream(self, region_id: int, type_ids: List[int]):
        """
        Yields pages of market orders as they arrive to allow streaming ingestion.
        """
        base_url = "https://esi.evetech.net/latest"
        headers = {
            "User-Agent": "EveTrader/1.0 (internal; debug) maintainer@eve-trader.local",
            "Accept": "application/json"
        }
        
        async with aiohttp.ClientSession(headers=headers) as session:
            url = f"{base_url}/markets/{region_id}/orders/?order_type=all&page=1"
            
            x_pages = 1
            try:
                # Fetch Page 1
                async with session.get(url) as response:
                    if response.status == 200:
                        esi_limit = int(response.headers.get("X-Esi-Error-Limit-Remain", 100))
                        if esi_limit < 50:
                            self.logger.warning(f"ESI Error Budget Critical ({esi_limit}). Pausing Scraper for 60s.")
                            await asyncio.sleep(60)

                        x_pages = int(response.headers.get("X-Pages", 1))
                        self.logger.info(f"Fetching market snapshot for Region {region_id} (Pages: {x_pages}) - STREAMING MODE")
                        
                        data = await response.json()
                        yield self._filter_orders(data, type_ids)
                    else:
                        self.logger.error(f"Failed to fetch market data (Page 1): {response.status}")
                        return

                if x_pages > 1:
                    # Helper function for concurrent fetching
                    async def fetch_page(page_num):
                         p_url = f"{base_url}/markets/{region_id}/orders/?order_type=all&page={page_num}"
                         try:
                             async with session.get(p_url) as resp:
                                 if resp.status == 200:
                                     return await resp.json()
                                 else:
                                     self.logger.error(f"Failed to fetch page {page_num}: {resp.status}")
                                     return []
                         except Exception as e:
                             self.logger.error(f"Page {page_num} Fetch Error: {e}")
                             return []

                    # Create tasks for remaining pages
                    tasks = [fetch_page(p) for p in range(2, x_pages + 1)]
                    
                    # Stream results as they complete
                    for future in asyncio.as_completed(tasks):
                        try:
                            page_data = await future
                            if page_data:
                                yield self._filter_orders(page_data, type_ids)
                        except Exception as e:
                            self.logger.error(f"Page Processing Error: {str(e)}")
                        
            except Exception as e:
                self.logger.error(f"ESI Connection Error: {str(e)}")

    def _filter_orders(self, orders: List[Dict[str, Any]], type_ids: List[int]) -> List[Dict[str, Any]]:
        if not type_ids:
            return orders
        whitelist_set = set(type_ids)
        return [o for o in orders if o['type_id'] in whitelist_set]

    async def fetch_historical_logs(self, region_id: int, type_ids: List[int]) -> List[Dict[str, Any]]:
        """
        Gathers 1-year daily OHLCV (Open, High, Low, Close, Volume) data.
        """
        base_url = "https://esi.evetech.net/latest"
        history_data = []
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            for type_id in type_ids:
                url = f"{base_url}/markets/{region_id}/history/?type_id={type_id}"
                tasks.append(self._fetch_single_history(session, url, type_id))
            
            results = await asyncio.gather(*tasks)
            # Flatten or structure results
            for res in results:
                if res:
                    history_data.extend(res)
                    
        return history_data

    async def _fetch_single_history(self, session, url: str, type_id: int) -> List[Dict[str, Any]]:
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    # Inject type_id since history endpoint doesn't return it in the body
                    for entry in data:
                        entry['type_id'] = type_id
                    return data
                else:
                    self.logger.warning(f"Failed to fetch history for type {type_id}: {response.status}")
                    return []
        except Exception as e:
            self.logger.error(f"Error fetching history for {type_id}: {e}")
            return []

    def validator_cleaner(self, raw_data: List[Dict[str, Any]], data_type: str = "orders") -> List[Dict[str, Any]]:
        """
        Removes "outlier" spikes and enforces Station Lockdown.
        
        Args:
            raw_data: List of dicts (orders or history entries).
            data_type: 'orders' or 'history'.
        """
        if not raw_data:
            return []

        df = pd.DataFrame(raw_data)

        # 1. Station Lockdown (For Orders Snapshot)
        # Jita IV - Moon 4 - Caldari Navy Assembly Plant = 60003760
        if data_type == "orders":
            if 'location_id' in df.columns:
                df = df[df['location_id'] == 60003760]
            
            # Return cleaned orders as list of dicts
            return df.to_dict('records')

        # 2. Z-Score Outlier Detection (For History/Time-Series)
        elif data_type == "history":
            # Assuming 'average' or 'close' price is the target. Using 'average' for history.
            target_col = 'average' if 'average' in df.columns else 'price'
            
            if target_col in df.columns:
                # Calculate Z-Score
                window_mean = df[target_col].rolling(window=24, min_periods=1).mean()
                window_std = df[target_col].rolling(window=24, min_periods=1).std()
                
                # Avoid division by zero
                window_std = window_std.replace(0, 1)
                
                z_scores = (df[target_col] - window_mean) / window_std
                
                # Identify outliers (> 3 std dev)
                outliers = np.abs(z_scores) > 3
                
                if outliers.any():
                    self.logger.warning(f"Detected {outliers.sum()} outliers. Skipping smoothing to prevent hangs.")
                    # FIXME: Smoothing on mixed-type dataframe is dangerous/wrong.
                    # df.loc[outliers, target_col] = window_mean[outliers]
            
            return df.to_dict('records')

        return raw_data

    def signal_shield_filter_orders(
        self,
        orders: List[Dict[str, Any]],
        *,
        redis_client,
        window_hours: int = 24,
        z_threshold: float = 3.0,
        min_samples: int = 30,
        samples_per_hour: int = 60,
    ) -> List[Dict[str, Any]]:
        """Drop manipulated per-type ticks using a rolling z-score over ~24h.

        Implementation notes:
        - We compute a per-type mid price from the current order snapshot.
        - We keep a Redis list of recent mids per type_id.
        - If the current mid deviates by > z_threshold, we drop ALL orders for that type_id
          from this ingest tick (so Historian/training only sees clean points).
        """

        if not orders or redis_client is None:
            return orders

        try:
            df = pd.DataFrame(orders)
        except Exception:
            return orders

        required = {"type_id", "is_buy_order", "price"}
        if not required.issubset(set(df.columns)):
            return orders

        df = df.copy()
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df = df.dropna(subset=["price", "type_id", "is_buy_order"])
        if df.empty:
            return []

        max_len = int(max(1, window_hours * samples_per_hour))

        # Precompute per-type mid prices
        mids: dict[int, float] = {}
        for type_id, g in df.groupby("type_id", sort=False):
            bids = g[g["is_buy_order"] == True]
            asks = g[g["is_buy_order"] == False]
            best_bid = float(bids["price"].max()) if not bids.empty else 0.0
            best_ask = float(asks["price"].min()) if not asks.empty else 0.0

            if best_bid > 0.0 and best_ask > 0.0:
                mid = (best_bid + best_ask) / 2.0
            else:
                mid = best_ask if best_ask > 0.0 else (best_bid if best_bid > 0.0 else float("nan"))

            if not np.isfinite(mid) or mid <= 0.0:
                continue
            mids[int(type_id)] = float(mid)

        if not mids:
            return orders

        allowed_type_ids: set[int] = set(mids.keys())
        now_ts = time.time()
        for type_id, mid in mids.items():
            key = f"signal_shield:mid:{type_id}"
            try:
                # lrange returns newest->oldest (because we lpush)
                raw_vals = redis_client.lrange(key, 0, max_len - 1)
                vals = []
                for v in raw_vals:
                    try:
                        fv = float(v)
                        if np.isfinite(fv) and fv > 0.0:
                            vals.append(fv)
                    except Exception:
                        continue

                if len(vals) >= min_samples:
                    mu = float(np.mean(vals))
                    sigma = float(np.std(vals, ddof=0))
                    if sigma > 0.0 and np.isfinite(mu) and np.isfinite(sigma):
                        z = abs((mid - mu) / sigma)
                        if z > float(z_threshold):
                            allowed_type_ids.discard(type_id)
                            self.logger.warning(
                                "Signal Shield: dropped type_id=%s mid=%.4f (z=%.2f, mu=%.4f, sigma=%.4f, n=%d)",
                                type_id,
                                mid,
                                z,
                                mu,
                                sigma,
                                len(vals),
                            )
                            continue

                # Only record non-outliers to keep the rolling window clean.
                redis_client.lpush(key, str(mid))
                redis_client.ltrim(key, 0, max_len - 1)
                # Optional: keep a last-update heartbeat for debugging.
                redis_client.setex(f"signal_shield:last_ok_ts:{type_id}", 60 * 60 * 48, str(now_ts))
            except Exception as e:
                # Never block ingestion on Redis issues.
                self.logger.error("Signal Shield error for type_id=%s: %s", type_id, e)
                continue

        if not allowed_type_ids:
            return []

        # Filter original orders list (keep non-numeric rows out already)
        kept = [o for o in orders if int(o.get("type_id", -1)) in allowed_type_ids]
        return kept
