"""Cointegration (Engle–Granger) scanner for statistical arbitrage watchlists.

This module is intended to run on the Debian host where the DB + Redis live.
It:
- Pulls bucketed market_history price series for a configurable universe
- Runs Engle–Granger cointegration tests across candidate pairs
- Writes the resulting pairs to Redis: system:pairs:watchlist

This does not execute trades; it only discovers and publishes candidate pairs.

Env (defaults are conservative):
- DATABASE_URL: postgres connection string
- REDIS_URL: redis connection string
- LOOKBACK_HOURS: how far back to scan
- BUCKET_MINUTES: time-bucket size used for resampling
- MAX_TYPES: max number of type_ids to consider (pairwise is O(n^2))
- MIN_BUCKETS: minimum number of buckets required per series
- PVALUE_THRESHOLD: maximum p-value to accept as cointegrated
- OUT_KEY: redis key to write

Run:
  python3 -m librarian.analyst
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import sqlalchemy
from sqlalchemy import text


@dataclass(frozen=True)
class Config:
    database_url: str
    redis_url: str
    lookback_hours: int = 24
    bucket_minutes: int = 5
    max_types: int = 50
    min_buckets: int = 93
    pvalue_threshold: float = 0.05
    out_key: str = "system:pairs:watchlist"


def _redis(redis_url: str):
    import redis

    r = redis.Redis.from_url(redis_url, decode_responses=True)
    r.ping()
    return r


def _db_engine(database_url: str):
    # Strategist/Librarian sometimes use asyncpg URLs; we want sync.
    sync_url = database_url.replace("asyncpg", "psycopg2")
    return sqlalchemy.create_engine(sync_url)


def _load_bucketed_prices(cfg: Config) -> pd.DataFrame:
    engine = _db_engine(cfg.database_url)

    # Select candidate universe by activity.
    # We use row-count as a simple proxy; the caller can tighten MAX_TYPES.
    with engine.connect() as conn:
        universe_rows = conn.execute(
            text(
                """
                SELECT type_id
                FROM market_history
                WHERE timestamp >= NOW() - (:lookback || ' hours')::interval
                  AND type_id IS NOT NULL
                GROUP BY type_id
                ORDER BY COUNT(*) DESC
                LIMIT :max_types
                """
            ),
            {"lookback": int(cfg.lookback_hours), "max_types": int(cfg.max_types)},
        ).fetchall()

    type_ids = [int(r[0]) for r in universe_rows if r and r[0] is not None]
    if not type_ids:
        return pd.DataFrame()

    with engine.connect() as conn:
        df = pd.read_sql_query(
            text(
                """
                SELECT timestamp, type_id,
                       COALESCE(avg_price, close) AS price
                FROM market_history
                WHERE timestamp >= NOW() - (:lookback || ' hours')::interval
                  AND type_id = ANY(:type_ids)
                ORDER BY timestamp ASC
                """
            ),
            conn,
            params={"lookback": int(cfg.lookback_hours), "type_ids": type_ids},
        )

    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "type_id", "price"])

    bucket = f"{int(cfg.bucket_minutes)}min"
    df["t"] = df["timestamp"].dt.floor(bucket)

    # 1 row per (t, type_id)
    out = (
        df.groupby(["t", "type_id"], as_index=False)
        .agg(price=("price", "mean"))
        .sort_values(["t", "type_id"])
    )

    return out


def _coint_test(x: np.ndarray, y: np.ndarray) -> float:
    """Return Engle–Granger p-value for cointegration."""

    try:
        from statsmodels.tsa.stattools import coint  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "statsmodels is required for cointegration scanning. Install statsmodels on Debian. "
            f"Original error: {e}"
        )

    # statsmodels expects 1d arrays
    pvalue = float(coint(x, y)[1])
    return pvalue


def find_cointegrated_pairs(cfg: Config) -> list[dict[str, object]]:
    df = _load_bucketed_prices(cfg)
    if df.empty:
        return []

    pivot = df.pivot(index="t", columns="type_id", values="price").sort_index()

    # Require enough buckets per series.
    good_cols = []
    for c in pivot.columns:
        if int(pivot[c].dropna().shape[0]) >= int(cfg.min_buckets):
            good_cols.append(int(c))

    pivot = pivot[good_cols]
    if pivot.shape[1] < 2:
        return []

    # Simple fill to align time indices without introducing wild artifacts.
    pivot = pivot.ffill().bfill()

    cols = list(pivot.columns)
    results: list[dict[str, object]] = []

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a = int(cols[i])
            b = int(cols[j])
            x = pivot[a].to_numpy(dtype=float)
            y = pivot[b].to_numpy(dtype=float)

            # Ignore degenerate series
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                continue
            if np.std(x) <= 0 or np.std(y) <= 0:
                continue

            try:
                p = _coint_test(x, y)
            except Exception:
                continue

            if p <= float(cfg.pvalue_threshold):
                results.append({"a": a, "b": b, "pvalue": float(p)})

    results.sort(key=lambda r: float(r["pvalue"]))
    return results


def publish_watchlist(cfg: Config, pairs: list[dict[str, object]]) -> None:
    r = _redis(cfg.redis_url)
    payload = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "lookback_hours": int(cfg.lookback_hours),
        "bucket_minutes": int(cfg.bucket_minutes),
        "min_buckets": int(cfg.min_buckets),
        "max_types": int(cfg.max_types),
        "pvalue_threshold": float(cfg.pvalue_threshold),
        "pairs": pairs,
    }
    r.set(cfg.out_key, json.dumps(payload))


def main() -> int:
    cfg = Config(
        database_url=os.getenv(
            "DATABASE_URL",
            "postgresql://eve_user:eve_pass@localhost:5432/eve_market_data",
        ),
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        lookback_hours=int(os.getenv("LOOKBACK_HOURS", "24")),
        bucket_minutes=int(os.getenv("BUCKET_MINUTES", "5")),
        max_types=int(os.getenv("MAX_TYPES", "50")),
        min_buckets=int(os.getenv("MIN_BUCKETS", "93")),
        pvalue_threshold=float(os.getenv("PVALUE_THRESHOLD", "0.05")),
        out_key=os.getenv("OUT_KEY", "system:pairs:watchlist"),
    )

    pairs = find_cointegrated_pairs(cfg)
    publish_watchlist(cfg, pairs)
    print(f"Published {len(pairs)} cointegrated pairs to {cfg.out_key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
