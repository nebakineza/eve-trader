import sqlalchemy
import os
from sqlalchemy import text

# Configuration
DB_HOST = "192.168.14.105"
DB_PORT = "5432"
DB_USER = "eve_user"
DB_PASS = "afterburn118921"
DB_NAME = "eve_market_data"

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

def verify_bridge():
    print("==========================================")
    print("   🌉 EVE Trader - Bridge Verification    ")
    print("==========================================")
    print(f"Target: {DB_HOST}:{DB_PORT}")
    
    try:
        print("🔗 Connecting to Body (Database)...")
        engine = sqlalchemy.create_engine(DATABASE_URL)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1")).scalar()
            if result == 1:
                print("✅ Connection Established!")
                print("   The Brain (WSL2) can talk to the Body (Debian).")
            else:
                print("❌ Connection active but returned unexpected result.")
                
            # Check for Hypertables (TimescaleDB)
            print("🔍 Checking TimescaleDB Extension...")
            timescale = conn.execute(text("SELECT extname FROM pg_extension WHERE extname = 'timescaledb'")).scalar()
            if timescale:
                print("✅ TimescaleDB is active.")
            else:
                print("⚠️  TimescaleDB extension NOT found.")
                
    except Exception as e:
        print(f"❌ Connection Failed: {e}")
        print("   Troubleshooting:")
        print("   1. Check Firewall on Debian (sudo ufw status)")
        print("   2. Check pg_hba.conf on DB Container")
        print("   3. Verify Password in .env")

if __name__ == "__main__":
    verify_bridge()
