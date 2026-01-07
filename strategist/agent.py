import logging
from typing import Dict, Any, Tuple
import numpy as np

# Assuming Stable Baselines3 usage
# from stable_baselines3 import PPO

class StrategistRLAgent:
    """
    Strategist-RL-Agent Module
    Purpose: Acting as the "Brain" that decides the actual trade action.
    """
    def __init__(self, model_path: str = None, system_mode: str = "SIMULATION", db_bridge = None):
        self.model = None 
        self.min_confidence_threshold = 0.40 
        self.max_wallet_risk_percent = 0.05 
        
        self.system_mode = system_mode
        self.db = db_bridge # Instance of DataBridge for logging
        self.logger = logging.getLogger("Strategist")

    def evaluate_action_space(self, observation: Dict[str, Any]) -> Tuple[int, float]:
        """
        Determines the optimal action and price.
        
        Outputs:
        - Action ID (0=Hold, 1=Buy, 2=Sell, 3=Cancel)
        - Price or Parameter (e.g., Undercut Amount)
        
        Logic:
        Calculate 'Optimal Undercut' rather than simple -0.01 ISK.
        Looks at the 'spread_efficiency' and 'imbalance' from Analyst.
        """
        # 1. Parse Observation
        # Assuming observation contains:
        # 'price_prediction': float
        # 'spread': float (difference between Best Bid/Ask)
        # 'volatility': float
        # 'imbalance': float (>1 Bullish, <1 Bearish)
        
        spread = observation.get('spread', 0.0)
        imbalance = observation.get('imbalance', 1.0)
        
        # 2. Logic: The "Scare" Tactic
        # If the spread is wide (healthy profit margin) AND competition is tight,
        # we might undercut by a larger amount (e.g., 5-10% of spread) 
        # to demoralize 0.01 ISK bots, securing the volume.
        
        optimal_undercut = 0.01
        
        if spread > 1000: # High margin item
            # If we are selling and market is Bearish (imbalance < 1), dump harder
            if imbalance < 0.8: 
                optimal_undercut = spread * 0.10 # Sacrifice 10% of spread to exit fast
            else:
                 # Standard aggressive undercut
                optimal_undercut = spread * 0.01 
        
        # 3. Model Decision (Mocking PPO inference for now)
        # action = self.model.predict(observation)
        # For this stage, we assume a simple heuristic if model is not trained:
        action = 0 # Default Hold
        
        # Heuristic override for testing logic
        if observation.get('price_prediction') > observation.get('current_price', 0):
            action = 1 # Signal Buy
            
        return action, optimal_undercut

    def calculate_relist_fee(self, new_value: float, broker_fee_pct: float = 0.005, adv_broker_lvl: int = 5) -> float:
        """
        Calculates the 2026 Relist Fee friction.
        Formula: (1.0 - (0.5 + 0.06 * AdvBrokerLvl)) * BrokerPct * NewValue
        """
        discount_factor = 0.5 + (0.06 * adv_broker_lvl)
        relist_fee = (1.0 - discount_factor) * broker_fee_pct * new_value
        return max(relist_fee, 0.0)

    def reward_function_optimizer(self, profit_isk: float, duration_hours: float, was_update: bool = False, new_value: float = 0.0) -> float:
        """
        Calculates the "Quality of Move" based on ISK profit minus the "Time-Weighted Broker Fee."
        Now includes Relist Fee Friction to discourage spamming updates.
        """
        # Standard time decay
        time_penalty = duration_hours * 0.01 
        
        net_reward = profit_isk - time_penalty
        
        # Subtract Relist Fee Friction if this was an Order Update
        if was_update:
            friction_cost = self.calculate_relist_fee(new_value)
            net_reward -= friction_cost
            
        return net_reward

    def risk_governor(self, proposed_action: int, wallet_balance: float, trade_value: float) -> bool:
        """
        A hard-coded safety layer that vetoes any trade that risks more than X% of the total wallet balance.
        Returns True if action is safe, False if vetoed.
        """
        if trade_value > (wallet_balance * self.max_wallet_risk_percent):
            return False
        return True

    async def step(self, observation: Dict[str, Any], oracle_output: Dict[str, Any]) -> Tuple[int, float]:
        """
        Main decision step.
        """
        # 1. Confidence Veto
        if oracle_output.get('status') == "INSUFFICIENT_DATA":
            print("Strategist: Holding due to Low Oracle Confidence.")
            return 0, 0.0 # Force HOLD
        
        # 2. Evaluate
        action, parameter = self.evaluate_action_space(observation)
        
        # 3. Shadow Mode Logic
        if self.system_mode == 'SHADOW' and action != 0:
            import asyncio
            import random
            from datetime import datetime
            
            # Simulate "Virtual Processing" delay
            delay = random.uniform(5.0, 15.0)
            self.logger.info(f"Shadow Mode: Simulating Execution Delay of {delay:.2f}s...")
            await asyncio.sleep(delay)
            
            # Log Virtual Fill
            if self.db:
                record = {
                    "time": datetime.utcnow(),
                    "action_type": action,
                    "type_id": observation.get('type_id', 0),
                    "price": observation.get('current_price', 0), 
                    "quantity": 100, 
                    "simulated": True,
                    "status": "VIRTUAL_FILL"
                }
                # Using the bridge to push 
                await self.db.push_to_postgres("trade_logs", [record])
                
            self.logger.info(f"Shadow Mode: Virtual Fill Logged. Action: {action} @ {parameter}")
        
        # 3. Risk Check (in Nexus or here)
        # Handled by Nexus typically, but good for internal policy
        
        return action, parameter
