def calculate_weighted_reward(net_profit_isk, duration_hours, volatility_index):
    """
    Calculates the reward score for the RL Agent.
    
    Parameters:
        net_profit_isk (float): The actual realized profit after taxes/fees.
        duration_hours (float): Time from Buy order fill to Sell order fill.
        volatility_index (float): A normalized measure of price variance (0.0 to 1.0).
        
    Returns:
        float: The weighted reward score.
    """
    
    # --- Alpha: Profit Component ---
    # Base reward is the log of profit to handle scaling (avoid exploding gradients with billions of ISK)
    # Adding a constant to handle 0 or negative profit slightly gracefully likely handled upstream, 
    # but here assuming profit is > 0 for a win, < 0 for loss.
    if net_profit_isk > 0:
        alpha = net_profit_isk / 1_000_000.0 # Scale to "Millions of ISK"
    else:
        # Penalize losses more heavily
        alpha = (net_profit_isk / 1_000_000.0) * 1.5

    # --- Beta: Velocity Component (Time Decay) ---
    # We want to turn capital over quickly.
    # If duration > 4 hours, start penalizing.
    # Standard sigmoid-like decay or simple linear penalty?
    # Let's use a "Holding Cost" multiplier.
    LIMIT_HOURS = 4.0
    if duration_hours > LIMIT_HOURS:
        # Decay factor: reduces reward by 10% for every hour over the limit
        overage = duration_hours - LIMIT_HOURS
        beta_multiplier = max(0.1, 1.0 - (overage * 0.1))
    else:
        # Slight bonus for very fast trades (< 1 hour)
        beta_multiplier = 1.0 + max(0.0, (1.0 - duration_hours) * 0.1)

    # --- Gamma: Stability Component (Risk Adjustment) ---
    # We prefer low volatility items to avoid Pump & Dump traps.
    # volatility_index assumed 0 (stable) to 1 (crypto-like).
    # Penalty increases as volatility increases.
    gamma_penalty = volatility_index * 2.0 # 0.0 -> 0 penalty, 0.5 -> 1.0M ISK equivalent penalty? 
    # Actually, let's make it a multiplier on the Alpha.
    # High volatility reduces the effective value of the profit because it was "risky".
    gamma_multiplier = 1.0 - (volatility_index * 0.5) # Max 50% penalty for super volatile

    # Combine
    reward = (alpha * beta_multiplier * gamma_multiplier)
    
    return float(reward)

# --- DB Helper Functions for Shadow Mode ---

async def log_pending_trade(pool, type_id, action_type, price, quantity):
    """
    Logs a new virtual order into pending_trades.
    """
    query = """
    INSERT INTO pending_trades (type_id, action_type, price, quantity, status, simulated)
    VALUES ($1, $2, $3, $4, 'OPEN', TRUE)
    RETURNING order_id
    """
    async with pool.acquire() as conn:
        order_id = await conn.fetchval(query, type_id, action_type, price, quantity)
        return order_id

async def update_market_radar(pool, type_id, signal_strength, predicted_target, current_price, status="WATCH"):
    """
    Updates the market radar with the latest opportunity signal.
    """
    query = """
    INSERT INTO market_radar (type_id, signal_strength, predicted_target, current_price, timestamp, status)
    VALUES ($1, $2, $3, $4, NOW(), $5)
    ON CONFLICT (type_id) 
    DO UPDATE SET 
        signal_strength = EXCLUDED.signal_strength,
        predicted_target = EXCLUDED.predicted_target,
        current_price = EXCLUDED.current_price,
        timestamp = NOW(),
        status = EXCLUDED.status
    """
    async with pool.acquire() as conn:
        await conn.execute(query, type_id, signal_strength, predicted_target, current_price, status)

async def execute_virtual_order(pool, type_id, action_type, current_price, quantity, slippage=0.001):
    """
    Executes a virtual trade with realistic market physics (Taxes + Slippage).
    
    Args:
        slippage: 0.1% default impact on price.
    """
    # 1. Apply Slippage (Buy Higher, Sell Lower)
    if action_type == 1: # BUY
        effective_price = current_price * (1 + slippage)
    else: # SELL
        effective_price = current_price * (1 - slippage)
        
    # 2. Apply Tax (Sales Tax + Broker Fee approx 3.0% for immediate sell orders/market takers)
    # Note: Buying usually has broker fee, Selling has broker fee + tax.
    # For simplicity in Shadow Mode: 
    # Buy Cost = Price * 1.00 (Fees handled in P&L mostly, or add 2%)
    # Sell Revenue = Price * 0.97 (3% loss to system)
    
    # We log the 'Execution Price' as the effective price, but P&L calculation handles the tax.
    # Or we can store 'net_price'. Let's store effective market price.
    
    query = """
    INSERT INTO trade_logs (type_id, action_type, price, quantity, status, simulated, profit_loss)
    VALUES ($1, $2, $3, $4, 'FILLED', TRUE, 0.0)
    RETURNING trade_id
    """
    async with pool.acquire() as conn:
        tid = await conn.fetchval(query, type_id, action_type, effective_price, quantity)
        return tid


