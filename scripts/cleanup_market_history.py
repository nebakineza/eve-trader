import argparse
import logging
import os
import time
from datetime import datetime

import psycopg2


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MarketHistoryCleanup")


def _db_url() -> str:
    # Prefer docker-compose injected URL; fall back to in-network DB name.
    return os.getenv("DATABASE_URL", "postgresql://eve_user:eve_pass@db:5432/eve_market_data")


def delete_older_than(*, keep_hours: int) -> int:
    db_url = _db_url()
    with psycopg2.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM market_history
                WHERE \"timestamp\" < NOW() - (%s * INTERVAL '1 hour');
                """,
                (int(keep_hours),),
            )
            deleted = cur.rowcount or 0
    return int(deleted)


def main() -> int:
    p = argparse.ArgumentParser(description="Delete market_history rows older than N hours")
    p.add_argument("--keep-hours", type=int, default=int(os.getenv("MARKET_HISTORY_KEEP_HOURS", "48")))
    p.add_argument("--interval-seconds", type=int, default=int(os.getenv("MARKET_HISTORY_CLEAN_INTERVAL_SECONDS", "3600")))
    p.add_argument("--loop", action="store_true", help="Run periodically")
    args = p.parse_args()

    if args.keep_hours <= 0:
        raise SystemExit("--keep-hours must be > 0")

    while True:
        try:
            deleted = delete_older_than(keep_hours=args.keep_hours)
            logger.info("Deleted %s rows older than %sh (ts=%s)", deleted, args.keep_hours, datetime.utcnow().isoformat() + "Z")
        except Exception as e:
            logger.warning("Cleanup failed: %s", e)

        if not args.loop:
            return 0
        time.sleep(max(10, int(args.interval_seconds)))


if __name__ == "__main__":
    raise SystemExit(main())
