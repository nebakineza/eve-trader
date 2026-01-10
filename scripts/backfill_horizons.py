#!/usr/bin/env python3
"""
Backfill / Resolve pending trades using ShadowManager logic.
Usage: python scripts/backfill_horizons.py
"""
import os
import sys
import logging
import time

# Ensure we can import from root
sys.path.append(os.getcwd())

from strategist.shadow_manager import ShadowManager
from storage.bridge import DataBridge

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BackfillHorizons")

def main():
    db_url = os.getenv("DATABASE_URL", "postgresql://eve_user:eve_password@localhost:5432/eve_market_data")
    if "postgresql" not in db_url:
        logger.error("Invalid DATABASE_URL for shadow manager")
        sys.exit(1)

    logger.info("Initializing ShadowManager for backfill...")
    shadow = ShadowManager(db_url)
    
    # Run settlement loop
    logger.info("Starting settlement...")
    count = shadow.settle_pending_trades()
    logger.info(f"Settled {count} pending trades.")

if __name__ == "__main__":
    main()
