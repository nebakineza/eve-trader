import json
import os
import sys

def generate_report(wallet_file="paper_wallet.json"):
    """
    Reads the paper_wallet.json and calculates Cumulative Profit 
    and Maximum Drawdown.
    """
    if not os.path.exists(wallet_file):
        print(f"Error: {wallet_file} not found.")
        return

    try:
        with open(wallet_file, 'r') as f:
            transactions = json.load(f)
    except json.JSONDecodeError:
        print("Error: Invalid JSON format.")
        return

    if not transactions:
        print("No transactions found.")
        return

    # Simulation Parameters
    INITIAL_CAPITAL = 1_000_000_000.0 # 1B ISK default
    capital = INITIAL_CAPITAL
    
    # We must simulate the balance change here because Nexus only logged the *Order* creation, 
    # not the result (unless we assume instant fills for this report).
    # Assumption for this Simple Report: 
    # All logged orders executed successfully at the requested price.
    # Fixed Quantity = 1 (or we need to check if 'qty' is in log, if not assume 1 or 100).
    
    equity_curve = [INITIAL_CAPITAL]
    
    print(f"--- Performance Report ---")
    print(f"Transactions: {len(transactions)}")
    
    for tx in transactions:
        action = tx.get("action") # 1=Buy, 2=Sell
        price = tx.get("price", 0)
        qty = tx.get("quantity", 100) # Default lot size if missing
        
        # 2026 EVE Tax Structure
        BROKER_FEE = 0.005
        SALES_TAX = 0.02

        if action == 1: # BUY
            # Cost = Price * Qty * (1 + Broker)
            cost = price * qty * (1 + BROKER_FEE)
            capital -= cost
            
        elif action == 2: # SELL
            # Revenue = Price * Qty * (1 - Broker - Tax)
            revenue = price * qty * (1 - BROKER_FEE - SALES_TAX)
            capital += revenue
            
        equity_curve.append(capital)

    # 1. Cumulative Profit
    final_equity = equity_curve[-1]
    cumulative_profit = final_equity - INITIAL_CAPITAL
    
    # 2. Max Drawdown
    peak = INITIAL_CAPITAL
    max_drawdown_amount = 0.0
    max_drawdown_pct = 0.0
    
    for equity in equity_curve:
        if equity > peak:
            peak = equity
        
        drawdown = peak - equity
        if drawdown > 0:
            pct = drawdown / peak
            if pct > max_drawdown_pct:
                max_drawdown_pct = pct
                max_drawdown_amount = drawdown

    print(f"Initial Capital: {INITIAL_CAPITAL:,.2f} ISK")
    print(f"Final Capital:   {final_equity:,.2f} ISK")
    print(f"Total PnL:       {cumulative_profit:,.2f} ISK")
    print(f"Max Drawdown:    {max_drawdown_pct*100:.2f}% ({max_drawdown_amount:,.2f} ISK)")
    
    if cumulative_profit > 0:
        print("Result: PROFITABLE")
    else:
        print("Result: UNPROFITABLE")

if __name__ == "__main__":
    target_file = sys.argv[1] if len(sys.argv) > 1 else "paper_wallet.json"
    generate_report(target_file)
