import pandas as pd
from typing import Dict, Any

class AnalystFeatureFactory:
    def __init__(self, redis_client):
        self.redis = redis_client

    def compute_market_velocity(self, order_book_history: pd.DataFrame) -> Dict[str, float]:
        if order_book_history.empty or 'volume_remaining' not in order_book_history.columns:
            return {'v_5m': 0.0, 'v_15m': 0.0, 'v_60m': 0.0}
        return {'v_5m': 0.0}

    def calculate_jita_tax(self, price: float, is_sell: bool = True) -> float:
        """
        Calculates Jita 4-4 taxes.
        Broker Fee: 0.5%
        Sales Tax: 2.0% (Sell only)
        """
        BROKER_RATE = 0.005
        SALES_TAX_RATE = 0.020
        
        fee = price * BROKER_RATE
        if is_sell:
            fee += price * SALES_TAX_RATE
            
        return fee

    def apply_tax_to_profit(self, buy_price: float, sell_price: float) -> float:
        """ Returns net profit after buying and selling fees. """
        buy_cost = buy_price + self.calculate_jita_tax(buy_price, is_sell=False)
        sell_revenue = sell_price - self.calculate_jita_tax(sell_price, is_sell=True)
        return sell_revenue - buy_cost
