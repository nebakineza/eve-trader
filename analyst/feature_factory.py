import pandas as pd
from typing import Dict, Any

class AnalystFeatureFactory:
    def __init__(self, redis_client):
        self.redis = redis_client

    def compute_market_velocity(self, order_book_history: pd.DataFrame) -> Dict[str, float]:
        if order_book_history.empty or 'volume_remaining' not in order_book_history.columns:
            return {'v_5m': 0.0, 'v_15m': 0.0, 'v_60m': 0.0}
        return {'v_5m': 0.0}
