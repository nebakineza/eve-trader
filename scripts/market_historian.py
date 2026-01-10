import argparse
import logging
import os
import time
from datetime import datetime, timedelta

import pandas as pd
import psycopg2
import sqlalchemy


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MarketHistorian")


def _floor_to_minutes(dt: datetime, minutes: int) -> datetime:
    dt = dt.replace(second=0, microsecond=0)
    floored_minute = dt.minute - (dt.minute % minutes)
    return dt.replace(minute=floored_minute)


def _sleep_until_next_bucket(bucket_minutes: int) -> None:
    now = datetime.utcnow()
    bucket_end = _floor_to_minutes(now, bucket_minutes) + timedelta(minutes=bucket_minutes)
    sleep_s = max(0.0, (bucket_end - now).total_seconds())
    time.sleep(sleep_s)


def _ensure_market_history_schema(conn) -> None:
    with conn.cursor() as cur:
        # Keep the existing OHLCV shape, but add the feature columns needed for sequence training.
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS market_history (
                timestamp TIMESTAMPTZ NOT NULL,
                region_id INT NOT NULL,
                type_id INT NOT NULL,
                open DOUBLE PRECISION,
                high DOUBLE PRECISION,
                low DOUBLE PRECISION,
                close DOUBLE PRECISION,
                volume BIGINT,
                avg_price DOUBLE PRECISION,
                max_velocity DOUBLE PRECISION,
                median_spread DOUBLE PRECISION,
                order_book_imbalance DOUBLE PRECISION,
                players_online INT,
                warfare_isk_destroyed DOUBLE PRECISION,
                PRIMARY KEY (timestamp, region_id, type_id)
            );
            """
        )

        # Add columns safely if the table already exists.
        cur.execute("ALTER TABLE market_history ADD COLUMN IF NOT EXISTS avg_price DOUBLE PRECISION;")
        cur.execute("ALTER TABLE market_history ADD COLUMN IF NOT EXISTS max_velocity DOUBLE PRECISION;")
        cur.execute("ALTER TABLE market_history ADD COLUMN IF NOT EXISTS median_spread DOUBLE PRECISION;")
        cur.execute("ALTER TABLE market_history ADD COLUMN IF NOT EXISTS order_book_imbalance DOUBLE PRECISION;")
        cur.execute("ALTER TABLE market_history ADD COLUMN IF NOT EXISTS players_online INT;")
        cur.execute("ALTER TABLE market_history ADD COLUMN IF NOT EXISTS warfare_isk_destroyed DOUBLE PRECISION;")


def _read_players_online(redis_url: str | None) -> int | None:
    if not redis_url:
        return None
    try:
        import redis

        r = redis.Redis.from_url(redis_url, decode_responses=True)
        v = r.get("system:macro:players")
        if v is None:
            return None
        players = int(float(v))
        if players < 0:
            return None
        return players
    except Exception:
        return None


def _read_warfare_isk_destroyed(redis_url: str | None) -> float | None:
    if not redis_url:
        return None
    try:
        import redis

        r = redis.Redis.from_url(redis_url, decode_responses=True)
        v = r.get("system:macro:warfare")
        if v is None:
            return None
        isk = float(v)
        if isk < 0:
            return None
        return isk
    except Exception:
        return None


def _fetch_snapshot_features(engine, window_start: datetime, window_end: datetime) -> pd.DataFrame:
    # NOTE: In this codebase, market_orders writes use a single timestamp per ingest tick.
    # We aggregate across timestamps within the last 5 minutes.
    query = """
        SELECT
            timestamp,
            region_id,
            type_id,
            is_buy_order,
            price,
            volume_remaining
        FROM market_orders
        WHERE timestamp >= %s AND timestamp < %s
    """

    # Many deployments use TIMESTAMP WITHOUT TIME ZONE for market_orders timestamp.
    # Use naive UTC to avoid tz comparison issues.
    start_naive = window_start.replace(tzinfo=None)
    end_naive = window_end.replace(tzinfo=None)

    return pd.read_sql_query(query, engine, params=(start_naive, end_naive))


def _compute_bucket_rows(
    df_orders: pd.DataFrame,
    bucket_timestamp: datetime,
    *,
    players_online: int | None,
    warfare_isk_destroyed: float | None,
) -> list[tuple]:
    if df_orders.empty:
        return []

    required_cols = {"timestamp", "region_id", "type_id", "is_buy_order", "price", "volume_remaining"}
    missing = required_cols - set(df_orders.columns)
    if missing:
        raise RuntimeError(f"market_orders missing expected columns: {sorted(missing)}")

    # Build per-(type_id, region_id, timestamp) book summary
    df_orders = df_orders.copy()
    df_orders["price"] = pd.to_numeric(df_orders["price"], errors="coerce")
    df_orders["volume_remaining"] = pd.to_numeric(df_orders["volume_remaining"], errors="coerce").fillna(0)
    df_orders = df_orders.dropna(subset=["price"])  # price is required

    summaries = []
    for (region_id, type_id, ts), g in df_orders.groupby(["region_id", "type_id", "timestamp"], sort=True):
        bids = g[g["is_buy_order"] == True]
        asks = g[g["is_buy_order"] == False]

        best_bid = float(bids["price"].max()) if not bids.empty else 0.0
        best_ask = float(asks["price"].min()) if not asks.empty else 0.0
        bid_vol = float(bids["volume_remaining"].sum()) if not bids.empty else 0.0
        ask_vol = float(asks["volume_remaining"].sum()) if not asks.empty else 0.0

        if best_bid > 0.0 and best_ask > 0.0:
            mid = (best_bid + best_ask) / 2.0
            spread = (best_ask - best_bid) / best_bid
        else:
            mid = best_ask if best_ask > 0.0 else (best_bid if best_bid > 0.0 else float("nan"))
            spread = float("nan")

        imbalance = float("nan")
        if ask_vol > 0.0:
            imbalance = bid_vol / ask_vol

        summaries.append(
            {
                "region_id": int(region_id),
                "type_id": int(type_id),
                "timestamp": ts,
                "mid": mid,
                "spread": spread,
                "imbalance": imbalance,
                "bid_vol": bid_vol,
                "ask_vol": ask_vol,
            }
        )

    if not summaries:
        return []

    df = pd.DataFrame(summaries)
    df = df.sort_values(["region_id", "type_id", "timestamp"], ascending=True)

    rows = []
    for (region_id, type_id), g in df.groupby(["region_id", "type_id"], sort=True):
        g = g.sort_values("timestamp")
        mid = pd.to_numeric(g["mid"], errors="coerce")
        mid = mid.dropna()
        if mid.empty:
            continue

        open_ = float(mid.iloc[0])
        close_ = float(mid.iloc[-1])
        high_ = float(mid.max())
        low_ = float(mid.min())

        avg_price = float(mid.mean())

        # Velocity: max abs pct change between consecutive snapshot mids
        pct = mid.pct_change()
        max_velocity = float(pct.abs().max()) if len(pct) > 1 else 0.0
        if pd.isna(max_velocity):
            max_velocity = 0.0

        spread = pd.to_numeric(g["spread"], errors="coerce").dropna()
        median_spread = float(spread.median()) if not spread.empty else 0.0

        imbalance = pd.to_numeric(g["imbalance"], errors="coerce").dropna()
        order_book_imbalance = float(imbalance.median()) if not imbalance.empty else 1.0

        # Represent the book depth at the end of the window (avoid double counting across timestamps)
        last_bid_vol = float(g["bid_vol"].iloc[-1])
        last_ask_vol = float(g["ask_vol"].iloc[-1])
        volume = int(max(0.0, last_bid_vol + last_ask_vol))

        rows.append(
            (
                bucket_timestamp,
                int(region_id),
                int(type_id),
                open_,
                high_,
                low_,
                close_,
                volume,
                avg_price,
                max_velocity,
                median_spread,
                order_book_imbalance,
                (int(players_online) if players_online is not None else None),
                (float(warfare_isk_destroyed) if warfare_isk_destroyed is not None else None),
            )
        )

    return rows


def write_market_history_tick(
    *,
    database_url: str,
    redis_url: str | None,
    region_id: int | None,
    bucket_minutes: int,
    lookback_minutes: int,
) -> int:
    now = datetime.utcnow()
    bucket_end = _floor_to_minutes(now, bucket_minutes)
    bucket_start = bucket_end - timedelta(minutes=lookback_minutes)

    conn = psycopg2.connect(database_url)
    conn.autocommit = True
    try:
        _ensure_market_history_schema(conn)
        engine = sqlalchemy.create_engine(database_url)
        df_orders = _fetch_snapshot_features(engine, bucket_start, bucket_end)
        if region_id is not None and not df_orders.empty:
            df_orders = df_orders[df_orders["region_id"] == region_id]

        players_online = _read_players_online(redis_url)
        warfare_isk_destroyed = _read_warfare_isk_destroyed(redis_url)
        rows = _compute_bucket_rows(
            df_orders,
            bucket_start.replace(tzinfo=None),
            players_online=players_online,
            warfare_isk_destroyed=warfare_isk_destroyed,
        )
        if not rows:
            logger.info(
                "No rows to write for bucket %s (window %s -> %s)",
                bucket_start.isoformat(),
                bucket_start.isoformat(),
                bucket_end.isoformat(),
            )
            return 0

        insert_sql = """
            INSERT INTO market_history (
                timestamp, region_id, type_id,
                open, high, low, close, volume,
                avg_price, max_velocity, median_spread, order_book_imbalance,
                players_online,
                warfare_isk_destroyed
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (timestamp, region_id, type_id) DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume,
                avg_price = EXCLUDED.avg_price,
                max_velocity = EXCLUDED.max_velocity,
                median_spread = EXCLUDED.median_spread,
                order_book_imbalance = EXCLUDED.order_book_imbalance,
                players_online = EXCLUDED.players_online,
                warfare_isk_destroyed = EXCLUDED.warfare_isk_destroyed
        """

        with conn.cursor() as cur:
            cur.executemany(insert_sql, rows)

        logger.info("✅ Wrote %d market_history rows for bucket %s", len(rows), bucket_start.isoformat())
        return len(rows)
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate market_orders into market_history feature buckets.")
    parser.add_argument(
        "--database-url",
        default=os.getenv("DATABASE_URL", "postgresql://eve_user:eve_pass@db:5432/eve_market_data"),
        help="Postgres DSN (defaults to DATABASE_URL)",
    )
    parser.add_argument(
        "--redis-url",
        default=os.getenv("REDIS_URL", ""),
        help="Redis URL for macro features (defaults to REDIS_URL)",
    )
    parser.add_argument("--region-id", type=int, default=None, help="Optional region_id filter")
    parser.add_argument("--bucket-minutes", type=int, default=5)
    parser.add_argument("--lookback-minutes", type=int, default=5)
    parser.add_argument("--loop", action="store_true", help="Run forever, ticking on bucket boundaries")

    args = parser.parse_args()

    if not args.loop:
        write_market_history_tick(
            database_url=args.database_url,
            redis_url=(args.redis_url or None),
            region_id=args.region_id,
            bucket_minutes=args.bucket_minutes,
            lookback_minutes=args.lookback_minutes,
        )
        return

    logger.info("Starting historian loop (bucket=%dm, lookback=%dm)", args.bucket_minutes, args.lookback_minutes)

    # Run once immediately so operators don't wait up to a full bucket interval
    # before seeing market_history advance.
    try:
        write_market_history_tick(
            database_url=args.database_url,
            redis_url=(args.redis_url or None),
            region_id=args.region_id,
            bucket_minutes=args.bucket_minutes,
            lookback_minutes=args.lookback_minutes,
        )
    except Exception as e:
        logger.error("Historian initial tick failed: %s", e)

    while True:
        _sleep_until_next_bucket(args.bucket_minutes)
        try:
            write_market_history_tick(
                database_url=args.database_url,
                redis_url=(args.redis_url or None),
                region_id=args.region_id,
                bucket_minutes=args.bucket_minutes,
                lookback_minutes=args.lookback_minutes,
            )
        except Exception as e:
            logger.error("Historian tick failed: %s", e)


if __name__ == "__main__":
    main()
