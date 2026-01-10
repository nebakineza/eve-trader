import sqlalchemy
from sqlalchemy import text
import pandas as pd
import sys
import time
import os
import argparse

# Configuration
DB_HOST = os.getenv("POSTGRES_HOST", "192.168.14.105")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DB_USER = os.getenv("POSTGRES_USER", "eve_user")
DB_PASS = os.getenv("POSTGRES_PASSWORD", "eve_pass")
DB_NAME = os.getenv("POSTGRES_DB", "eve_market_data")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

def validate_data_integrity(show_all=False):
    print("==========================================")
    print("   🛡️  EVE Trader - Data Integrity Validator")
    print("==========================================")
    
    try:
        engine = sqlalchemy.create_engine(DATABASE_URL)
        print("🔗 Connecting to Database...")
        
        while True:
            # 1. Fetch Last 100 Market History Ticks
            query = text("""
                SELECT * FROM market_history 
                ORDER BY timestamp DESC 
                LIMIT 100
            """)
            
            with engine.connect() as conn:
                df = pd.read_sql(query, conn)
            
            row_count = len(df)
            
            if show_all:
                print(f"✅ [market_history] Found {row_count} rows.")
                if not df.empty:
                    # Simple Map (Duplicate of Librarian logic, ideally shared config)
                    type_names = {
                        34: "Tritanium", 35: "Pyerite", 36: "Mexallon", 37: "Isogen", 
                        38: "Nocxium", 39: "Zydrine", 40: "Megacyte", 
                        29668: "Skill Injector", 44992: "PLEX"
                    }
                    df['type_name'] = df['type_id'].map(type_names).fillna("Unknown")
                    cols = ['timestamp', 'type_name', 'type_id', 'volume', 'close'] # Select relevant columns
                    print(df[cols].head(10).to_string(index=False))
                
                # Also check market_orders
                query_orders = text("SELECT * FROM market_orders LIMIT 10")
                with engine.connect() as conn:
                    df_orders = pd.read_sql(query_orders, conn)
                print(f"✅ [market_orders] Found {len(df_orders)} sample rows (Limit 10).")
                if not df_orders.empty:
                    print(df_orders.to_string())
                
                return

            if row_count < 10:
                print(f"⏳ Waiting for data... (Found {row_count}/10 ticks) - Sleeping 60s")
                time.sleep(60)
                continue

            print(f"✅ Retrieved {row_count} rows. Sufficient for initial validation.")
            break
            
        
        # 2. Check for NaN/Inf values (Z-Score validation placeholder)
        # Note: If Analyst writes pre-computed features, check those. 
        # Here we check raw OHLCV for integrity first.
        if df.isnull().values.any():
            print("❌ Error: NaN values detected in raw data.")
            print(df[df.isnull().any(axis=1)])
        else:
            print("✅ Data Integrity: No NaN values found.")

        # 3. Calculate Sampling Heartbeat
        # Check gap between most recent timestamp and 'now'
        # And average gap between rows (assuming single item focus or aggregate)
        
        latest_ts = pd.to_datetime(df['timestamp'].max())
        now_ts = pd.Timestamp.now(tz=latest_ts.tz)
        lag = (now_ts - latest_ts).total_seconds()
        
        print(f"⏱️  System Lag: {lag:.1f} seconds")
        
        if lag > 600: # 10 Minutes tolerance
            print("❌ Error: Sampling Lag > 10 minutes.")
            print("   Librarian might be paused or ESI is down.")
        elif lag > 300:
            print("⚠️  Warning: Moderate Lag (> 5 minutes).")
        else:
            print("✅ Sampling Heartbeat: Healthy.")
            
        print("------------------------------------------")
        print("🚀 SYSTEM READY FOR TRAINING")
        print("------------------------------------------")

    except Exception as e:
        print(f"❌ Validation Failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Validate EVE Trader Data Integrity')
    parser.add_argument('--show-all', action='store_true', help='Show all rows regardless of count')
    args = parser.parse_args()
    
    validate_data_integrity(show_all=args.show_all)
