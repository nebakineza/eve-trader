import argparse
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import redis
import sqlalchemy
from sqlalchemy import text


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DailyAlpha")


@dataclass(frozen=True)
class AlphaResult:
    updated_at: str
    n_resolved: int
    horizon_minutes: int
    fee_rate: float
    geometric_mean_return: float
    max_drawdown: float


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _compute_max_drawdown(equity_curve: np.ndarray) -> float:
    if equity_curve.size == 0:
        return 0.0
    peaks = np.maximum.accumulate(equity_curve)
    dd = (equity_curve / np.where(peaks == 0.0, 1.0, peaks)) - 1.0
    return float(dd.min())


def compute_last_24h_alpha(*, db_engine, horizon_minutes: int, fee_rate: float) -> AlphaResult:
    since = datetime.now(timezone.utc) - timedelta(hours=24)

    query = text(
        """
        SELECT
            t.timestamp,
            t.type_id,
            t.signal_type,
            t.actual_price_at_time AS entry_price,
            mh.close AS exit_price
        FROM shadow_trades t
        LEFT JOIN LATERAL (
            SELECT close
            FROM market_history
            WHERE type_id = t.type_id
              AND timestamp >= t.timestamp + (:horizon || ' minutes')::interval
            ORDER BY timestamp ASC
            LIMIT 1
        ) mh ON TRUE
        WHERE t.timestamp >= :since
          AND t.virtual_outcome IN ('WIN','LOSS')
        ORDER BY t.timestamp ASC
        """
    )

    with db_engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"since": since, "horizon": int(horizon_minutes)})

    if df.empty:
        return AlphaResult(
            updated_at=_now_iso(),
            n_resolved=0,
            horizon_minutes=int(horizon_minutes),
            fee_rate=float(fee_rate),
            geometric_mean_return=0.0,
            max_drawdown=0.0,
        )

    df["entry_price"] = pd.to_numeric(df["entry_price"], errors="coerce")
    df["exit_price"] = pd.to_numeric(df["exit_price"], errors="coerce")
    df = df.dropna(subset=["entry_price", "exit_price", "signal_type"])
    df = df[df["entry_price"] > 0]
    if df.empty:
        return AlphaResult(
            updated_at=_now_iso(),
            n_resolved=0,
            horizon_minutes=int(horizon_minutes),
            fee_rate=float(fee_rate),
            geometric_mean_return=0.0,
            max_drawdown=0.0,
        )

    # Per-trade net return (fee applied to profit component)
    buy = df["signal_type"].astype(str).str.upper().eq("BUY")
    df["gross_r"] = np.where(
        buy.to_numpy(),
        (df["exit_price"].to_numpy(dtype=float) - df["entry_price"].to_numpy(dtype=float))
        / df["entry_price"].to_numpy(dtype=float),
        (df["entry_price"].to_numpy(dtype=float) - df["exit_price"].to_numpy(dtype=float))
        / df["entry_price"].to_numpy(dtype=float),
    )

    # Filter out clearly invalid returns first (data corruption / bad ticks).
    filter_lo = float(os.getenv("DAILY_ALPHA_RETURN_FILTER_LO", "-0.95"))
    filter_hi = float(os.getenv("DAILY_ALPHA_RETURN_FILTER_HI", "5.0"))
    df = df[np.isfinite(df["gross_r"]) & (df["gross_r"] > filter_lo) & (df["gross_r"] < filter_hi)].copy()
    if df.empty:
        return AlphaResult(
            updated_at=_now_iso(),
            n_resolved=0,
            horizon_minutes=int(horizon_minutes),
            fee_rate=float(fee_rate),
            geometric_mean_return=0.0,
            max_drawdown=0.0,
        )

    df["net_r"] = df["gross_r"].astype(float) * (1.0 - float(fee_rate))

    # Final winsorization: stabilizes metrics without masking systemic issues.
    winsor_lo = float(os.getenv("DAILY_ALPHA_RETURN_WINSOR_LO", "-0.90"))
    winsor_hi = float(os.getenv("DAILY_ALPHA_RETURN_WINSOR_HI", "2.0"))
    df["net_r"] = df["net_r"].clip(lower=winsor_lo, upper=winsor_hi)

    net_r = df["net_r"].to_numpy(dtype=float)

    # Geometric mean over per-trade net returns
    one_plus = 1.0 + net_r
    one_plus = np.where(np.isfinite(one_plus), one_plus, 1.0)
    one_plus = np.clip(one_plus, 1e-6, None)
    gmean = float(np.exp(np.mean(np.log(one_plus))) - 1.0) if one_plus.size else 0.0

    # Max drawdown: compute on hourly aggregated returns (more meaningful than per-trade compounding
    # when many trades overlap across instruments).
    try:
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        buckets = ts.dt.floor("1H")
        hourly = df.assign(bucket=buckets).groupby("bucket", dropna=True)["net_r"].mean().sort_index()
        if hourly.empty:
            max_dd = 0.0
        else:
            equity_h = np.cumprod(1.0 + hourly.to_numpy(dtype=float))
            max_dd = _compute_max_drawdown(equity_h)
    except Exception:
        max_dd = 0.0

    return AlphaResult(
        updated_at=_now_iso(),
        n_resolved=int(len(df)),
        horizon_minutes=int(horizon_minutes),
        fee_rate=float(fee_rate),
        geometric_mean_return=gmean,
        max_drawdown=max_dd,
    )


def write_to_redis(*, redis_client, key: str, result: AlphaResult) -> None:
    payload = {
        "updated_at": result.updated_at,
        "n_resolved": result.n_resolved,
        "horizon_minutes": result.horizon_minutes,
        "fee_rate": result.fee_rate,
        "geometric_mean_return": result.geometric_mean_return,
        "max_drawdown": result.max_drawdown,
    }
    redis_client.set(key, json.dumps(payload))


def run_once(*, database_url: str, redis_url: str, key: str, horizon_minutes: int, fee_rate: float) -> int:
    engine = sqlalchemy.create_engine(database_url)
    r = redis.Redis.from_url(redis_url, decode_responses=True)

    result = compute_last_24h_alpha(db_engine=engine, horizon_minutes=horizon_minutes, fee_rate=fee_rate)
    write_to_redis(redis_client=r, key=key, result=result)

    logger.info(
        "Daily Alpha updated: n=%d gmean=%.6f max_dd=%.6f",
        result.n_resolved,
        result.geometric_mean_return,
        result.max_drawdown,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute last-24h performance metrics from shadow_trades.")
    parser.add_argument(
        "--database-url",
        default=os.getenv("DATABASE_URL", "postgresql://eve_user:eve_pass@db:5432/eve_market_data"),
    )
    parser.add_argument("--redis-url", default=os.getenv("REDIS_URL", "redis://cache:6379/0"))
    parser.add_argument("--redis-key", default=os.getenv("DAILY_ALPHA_REDIS_KEY", "performance:last24h"))
    parser.add_argument("--horizon-minutes", type=int, default=int(os.getenv("DAILY_ALPHA_HORIZON_MINUTES", "60")))
    parser.add_argument("--fee-rate", type=float, default=float(os.getenv("DAILY_ALPHA_FEE_RATE", "0.025")))
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--interval-seconds", type=int, default=int(os.getenv("DAILY_ALPHA_INTERVAL_SECONDS", str(24 * 3600))))

    args = parser.parse_args()

    if not args.loop:
        return run_once(
            database_url=args.database_url,
            redis_url=args.redis_url,
            key=args.redis_key,
            horizon_minutes=args.horizon_minutes,
            fee_rate=args.fee_rate,
        )

    logger.info("Daily Alpha loop started (interval=%ss)", args.interval_seconds)
    while True:
        try:
            run_once(
                database_url=args.database_url,
                redis_url=args.redis_url,
                key=args.redis_key,
                horizon_minutes=args.horizon_minutes,
                fee_rate=args.fee_rate,
            )
        except Exception as e:
            logger.error("Daily Alpha run failed: %s", e)
        time.sleep(max(60, int(args.interval_seconds)))


if __name__ == "__main__":
    raise SystemExit(main())
