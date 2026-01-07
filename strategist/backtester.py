import logging
import random
from typing import List, Dict, Any, Tuple

class BacktestEngine:
    """
    BacktestEngine: Simulates trading logic against historical data.
    """
    def __init__(self, initial_capital: float = 1_000_000_000.0):
        self.capital = initial_capital
        self.inventory: Dict[int, int] = {} # type_id -> quantity
        self.transaction_log: List[Dict[str, Any]] = []
        self.logger = logging.getLogger("Backtester")
        
        # 2026 EVE Tax Structure
        self.BROKER_FEE = 0.005 # 0.5%
        self.SALES_TAX = 0.02   # 2.0%

    def _simulate_fill_probability(self, volume: float) -> bool:
        """
        Physics of EVE: If daily_volume is low, the order has a higher chance 
        of being outbid/undercut before it fills.
        """
        if volume <= 0:
            return False
            
        # Simplified probability function:
        # High volume (e.g. 10,000 units) -> ~100% chance for competitive price
        # Low volume (e.g. 10 units) -> Low chance
        # Using a sigmoid-like or logarithmic ramp
        
        # Threshold where we consider it "liquid enough"
        LIQUIDITY_THRESHOLD = 5000.0 
        
        prob = min(0.95, volume / LIQUIDITY_THRESHOLD)
        
        # Random roll
        return random.random() < prob

    def step(self, candle: Dict[str, Any], action: int, price: float, quantity: int = 100):
        """
        Process a single trading step based on a historical candle.
        
        Args:
            candle: Dict containing 'low', 'high', 'volume', 'type_id', 'date'
            action: 1=Buy, 2=Sell, 0=Hold
            price: The Limit Price we are trying to trade at
            quantity: Number of units
        """
        if action == 0:
            return

        low = candle.get('low', 0.0)
        high = candle.get('high', 0.0)
        volume = candle.get('volume', 0.0)
        timestamp = candle.get('date')
        type_id = candle.get('type_id', 0)

        # BUY LOGIC
        if action == 1: 
            cost_basis = price * quantity * (1 + self.BROKER_FEE)
            
            if self.capital < cost_basis:
                self.logger.warning("Backtest: Insufficient Capital for BUY")
                return

            # EVE Market Logic: To buy as a Maker, market price must drop to hit our bid (or someone sells to us)
            # We assume 'low' captures the lowest sell/highest buy interaction of the timeframe
            if price >= low: 
                if self._simulate_fill_probability(volume):
                    self.capital -= cost_basis
                    self.inventory[type_id] = self.inventory.get(type_id, 0) + quantity
                    self._log_trade(timestamp, 'BUY', price, quantity, cost_basis)

        # SELL LOGIC
        elif action == 2:
            if self.inventory.get(type_id, 0) < quantity:
                self.logger.warning("Backtest: Insufficient Inventory for SELL")
                return

            # EVE Market Logic: To sell as Maker, market must rise to hit our ask
            if price <= high:
                # Revenue: Gross - Broker - Tax
                gross_proceeds = price * quantity
                net_revenue = gross_proceeds * (1 - self.BROKER_FEE - self.SALES_TAX)
                
                if self._simulate_fill_probability(volume):
                    self.inventory[type_id] -= quantity
                    self.capital += net_revenue
                    self._log_trade(timestamp, 'SELL', price, quantity, net_revenue)

    def _log_trade(self, time, side, price, qty, impact):
        entry = {
            'time': time,
            'side': side,
            'price': price,
            'qty': qty,
            'impact_isk': impact,
            'rolling_capital': self.capital,
            'inventory_val': len(self.inventory) # Simplified tracking
        }
        self.transaction_log.append(entry)

    def generate_stats(self):
        """Return simple stats for the run."""
        return {
            "final_capital": self.capital,
            "total_trades": len(self.transaction_log)
        }
