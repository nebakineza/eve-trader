from fastapi import FastAPI
from typing import Dict, Any
import logging
import signal
import sys
import asyncio

class NexusCoordinator:
    """
    Nexus-Coordinator Module (The App)
    Purpose: The central nervous system that syncs all modules.
    """
    def __init__(self, system_mode: str = "SIMULATION", redis_url: str = "redis://localhost:6379/0"):
        self.app = FastAPI()
        self.logger = logging.getLogger("Nexus")
        self.system_mode = system_mode
        self.paper_wallet_file = "paper_wallet.json"
        
        # Redis for state management (Auto-Trade toggle)
        import redis
        self.redis_client = redis.from_url(redis_url)

        # Setup Signal Handlers for Graceful Shutdown
        self._setup_signal_handlers()

        # Start Dead Man's Switch (Data Liveness Monitor)
        asyncio.create_task(self.dead_mans_switch_loop())

    async def dead_mans_switch_loop(self):
        """
        Background task to monitor database liveness.
        If no new rows in market_history for > 15 minutes, restart Librarian.
        """
        import os
        try:
            # We need asyncpg here for non-blocking DB check
            import asyncpg
            db_url = os.getenv("DATABASE_URL", "postgresql://eve_user:eve_password@db:5432/eve_market_data")
            
            while True:
                try:
                    conn = await asyncpg.connect(db_url)
                    # Check the most recent timestamp in market_history
                    last_ts = await conn.fetchval("SELECT MAX(timestamp) FROM market_history")
                    await conn.close()
                    
                    if last_ts:
                        from datetime import datetime, timezone
                        # Ensure last_ts is timezone-aware if the DB returns it as such (likely yes)
                        # If not, assume UTC.
                        if last_ts.tzinfo is None:
                            last_ts = last_ts.replace(tzinfo=timezone.utc)
                            
                        now = datetime.now(timezone.utc)
                        delta = now - last_ts
                        
                        minutes_since_update = delta.total_seconds() / 60.0
                        
                        # Log heartbeat to Redis for Dashboard
                        self.redis_client.set("system:heartbeat:last_update_minutes", f"{minutes_since_update:.1f}")
                        
                        # Fix for Dashboard N/A: Also write the raw timestamp
                        self.redis_client.set("system:heartbeat:latest", str(now.timestamp()))

                        if minutes_since_update > 15:
                            self.logger.critical(f"💀 DEAD MAN'S SWITCH: No data for {minutes_since_update:.1f} mins. Triggering Librarian Restart.")
                            self.redis_client.publish('system:commands', 'RESTART_LIBRARIAN')
                            # In a real K8s setup, this might hit the K8s API. 
                            # Here we rely on an external agent listening to Redis or just alert.
                    
                except Exception as e:
                    self.logger.error(f"Dead Man's Switch Error: {e}")
                
                await asyncio.sleep(60) # Check every minute
                
        except ImportError:
            self.logger.error("asyncpg not installed. Dead Man's Switch disabled.")

    def _setup_signal_handlers(self):

        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _handle_shutdown(self, signum, frame):
        self.logger.critical(f"Received Signal {signum}. Initiating Graceful Shutdown...")
        try:
            # Broadcast Kill Switch
            self.redis_client.publish('system:signals', 'SYSTEM_HALT')
            
            # Also set a persistent flag just in case
            self.redis_client.set('system:status', 'HALT')
            
            self.logger.info("SYSTEM_HALT broadcasted to Redis.")
        except Exception as e:
            self.logger.error(f"Failed to broadcast halt signal: {e}")
        finally:
            self.logger.info("Nexus shutting down. Goodbye.")
            sys.exit(0)
        
    def _is_auto_trade_enabled(self) -> bool:
        """
        Checks Redis for the global 'AUTO_TRADE_ENABLED' flag.
        Default to False (Safety First) if key is missing.
        """
        try:
            val = self.redis_client.get("system:auto_trade_enabled")
            return val == b"true" or val == b"1"
        except Exception:
            return False

    def state_synchronizer(self, feature_metadata: Dict[str, Any]) -> bool:
        """
        Ensures the Oracle isn't making predictions based on stale data from the Librarian.
        """
        import time
        
        # Max allowed data age in seconds (5 minutes)
        MAX_AGE_SECONDS = 300 
        
        last_updated_ts = feature_metadata.get('last_updated_eof', 0)
        current_time = time.time()
        data_age = current_time - last_updated_ts
        
        if data_age > MAX_AGE_SECONDS:
            self.logger.warning(f"State Sync Failed: Data is {data_age:.1f}s old (Max: {MAX_AGE_SECONDS}s). Halting Strategist.")
            return False
            
        return True

    def command_dispatcher(self, strategy_decision: Dict[str, Any]):
        """
        Packages the Strategist's decision into a standardized JSON packet or logs to paper wallet.
        """
        # 1. Check Global Override
        if not self._is_auto_trade_enabled():
            self.logger.info(f"MANUAL OVERRIDE: Blocked Action {strategy_decision}. Sending Alert.")
            # Publish alert to Redis for Dashboard consumption
            try:
                alert_payload = json.dumps(strategy_decision)
                self.redis_client.publish('nexus:alerts', alert_payload)
                self.redis_client.lpush('nexus:alert_history', alert_payload) # Keep a history
                self.redis_client.ltrim('nexus:alert_history', 0, 99) # Keep last 100
            except Exception as e:
                self.logger.error(f"Failed to publish alert: {e}")
            return

        # 2. Execution Logic
        if self.system_mode == "SIMULATION":
            self._write_to_paper_wallet(strategy_decision)
        else:
            # TODO: Publish to RabbitMQ exchange for Live Execution
            pass

    def _write_to_paper_wallet(self, decision: Dict[str, Any]):
        """
        Simulation Mode: Appends decision to a local JSON file.
        """
        import json
        import os
        from datetime import datetime

        entry = {
            "timestamp": decision.get("timestamp", datetime.utcnow().isoformat()),
            "action": decision.get("action_id"), # 1=Buy, 2=Sell, 0=Hold
            "price": decision.get("price"),
            "parameter": decision.get("parameter"),
            "status": "SIMULATED_ORDER"
        }

        try:
            data = []
            if os.path.exists(self.paper_wallet_file):
                with open(self.paper_wallet_file, 'r') as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError:
                        data = []
            
            data.append(entry)
            
            with open(self.paper_wallet_file, 'w') as f:
                json.dump(data, f, indent=4)
                
            self.logger.info(f"SIMULATION: Wrote order to {self.paper_wallet_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to write to paper wallet: {e}")


    def performance_monitor(self, trade_id: str, real_outcome: float, predicted_outcome: float):
        """
        Tracks the bot's "Paper Trading" vs. "Real Trading" accuracy.
        Updates metrics for generic dashboard visibility.
        """
        # TODO: Log difference between real vs paper
        pass

    # API Routes would be defined here or in a separate routers file
    # for external control (Example):
    # @app.post("/start")
    # async def start_trading(): ...
