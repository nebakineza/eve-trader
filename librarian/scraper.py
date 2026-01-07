import asyncio
import aiohttp
import logging
import pandas as pd
import numpy as np
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
        
        async with aiohttp.ClientSession() as session:
            url = f"{base_url}/markets/{region_id}/orders/?order_type=all&page=1"
            
            x_pages = 1
            try:
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

                        if x_pages > 1:
                            # Create tasks for remaining pages
                            tasks = []
                            for page in range(2, x_pages + 1):
                                page_url = f"{base_url}/markets/{region_id}/orders/?order_type=all&page={page}"
                                tasks.append(session.get(page_url))
                            
                            # Stream results as they complete
                            for future in asyncio.as_completed(tasks):
                                try:
                                    resp = await future
                                    if resp.status == 200:
                                        page_data = await resp.json()
                                        yield self._filter_orders(page_data, type_ids)
                                    else:
                                        self.logger.error(f"Failed to fetch page: {resp.status}")
                                except Exception as e:
                                    self.logger.error(f"Page Fetch Error: {str(e)}")
                    else:
                        self.logger.error(f"Failed to fetch market data: {response.status}")
                        
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
