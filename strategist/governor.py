import os
import time
import random
import logging
import redis
from datetime import datetime, timezone

class BehavioralGovernor:
    """
    Behavioral Governor (Internal Policy & Safety)
    
    Enforces 'Human-Like' constraints to ensure system stability and API compliance.
    Includes:
    1. Order Rate Limiting (Caps actions/hour)
    2. Persistence (Prevents rapid-fire updates)
    3. Fatigue/Schedule (Simulates active hours and breaks)
    """

    def __init__(self, redis_url: str = None):
        if not redis_url:
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.r = redis.from_url(redis_url, decode_responses=True)
        self.logger = logging.getLogger("Governor")
        
        # Configuration (Safety & Schedule)
        self.ORDER_CAP_MIN = int(os.getenv("GOV_ORDER_CAP_MIN", "35"))
        self.ORDER_CAP_MAX = int(os.getenv("GOV_ORDER_CAP_MAX", "50"))
        
        # 3-7 minutes persistence
        self.PERSISTENCE_MIN_SEC = int(os.getenv("GOV_PERSISTENCE_MIN_SEC", "180"))
        self.PERSISTENCE_MAX_SEC = int(os.getenv("GOV_PERSISTENCE_MAX_SEC", "420"))

        # Schedule (UTC)
        self.SCHEDULE_START = int(os.getenv("GOV_SCHEDULE_START_HOUR", "8"))
        self.SCHEDULE_END = int(os.getenv("GOV_SCHEDULE_END_HOUR", "23"))

        # Fatigue (90m active -> 15m break)
        self.FATIGUE_ACTIVE_LIMIT_SEC = 90 * 60
        self.FATIGUE_BREAK_SEC = 15 * 60

    def _get_hour_bucket_key(self) -> str:
        # Key rotates every hour
        now_h = datetime.now(timezone.utc).strftime("%Y%m%d%H")
        return f"system:governor:order_count:{now_h}"

    def _get_active_session_start(self) -> float:
        val = self.r.get("system:governor:session_start")
        if not val:
            now = time.time()
            self.r.set("system:governor:session_start", now)
            return now
        return float(val)

    def should_allow_action(self, action_type: str = "order", item_id: int = None) -> (bool, str):
        """
        Master check: Schedule -> Fatigue -> Rate Limit -> Persistence.
        """
        now = datetime.now(timezone.utc)
        ts = now.timestamp()

        # 1. Schedule Check (Hard-coded active hours)
        # Note: Docks the bot outside these hours.
        if not (self.SCHEDULE_START <= now.hour < self.SCHEDULE_END):
            return False, f"Off-Hours (Schedule: {self.SCHEDULE_START:02d}:00 - {self.SCHEDULE_END:02d}:00 UTC)"

        # 2. Fatigue Simulation (Coffee Break)
        # Check if we are on forced break
        break_until = self.r.get("system:governor:on_break_until")
        if break_until:
            if ts < float(break_until):
                remaining = int(float(break_until) - ts)
                return False, f"Operator Fatigue/Break ({remaining}s remaining)"
            else:
                self.r.delete("system:governor:on_break_until")
                self.r.set("system:governor:session_start", ts) # Reset session

        # Check session duration
        session_start = self._get_active_session_start()
        if (ts - session_start) > self.FATIGUE_ACTIVE_LIMIT_SEC:
            # Trigger break
            break_end = ts + self.FATIGUE_BREAK_SEC
            self.r.set("system:governor:on_break_until", break_end)
            return False, "Triggering 15m Fatigue Break"

        # 3. Order Rate Limiting (Hourly Cap)
        if action_type == "order":
            bucket = self._get_hour_bucket_key()
            count = int(self.r.get(bucket) or 0)
            
            # Determine dynamic cap for this hour (randomized variance)
            # We hash the hour string to get a stable but random-looking cap for the hour
            h_seed = int(bucket.split(":")[-1])
            random.seed(h_seed)
            hourly_cap = random.randint(self.ORDER_CAP_MIN, self.ORDER_CAP_MAX)
            
            if count >= hourly_cap:
                return False, f"Hourly Order Cap Reached ({count}/{hourly_cap})"

        # 4. Minimum Persistence (Price Update Frequency)
        if item_id:
            last_update_key = f"system:governor:last_update:{item_id}"
            last_ts = self.r.get(last_update_key)
            if last_ts:
                # Randomized persistence per item/check
                elapsed = ts - float(last_ts)
                # We use a randomized required delay for "human" variance
                required = random.randint(self.PERSISTENCE_MIN_SEC, self.PERSISTENCE_MAX_SEC)
                if elapsed < required:
                    return False, f"Persistence Limit ({elapsed:.0f}s < {required}s)"

        return True, "OK"

    def register_action(self, action_type: str = "order", item_id: int = None):
        """
        Call this AFTER a successful action to update counters.
        """
        ts = time.time()
        
        if action_type == "order":
            bucket = self._get_hour_bucket_key()
            pipe = self.r.pipeline()
            pipe.incr(bucket)
            pipe.expire(bucket, 3600 * 2) # Keep for audit
            pipe.execute()
        
        if item_id:
            self.r.set(f"system:governor:last_update:{item_id}", ts)
