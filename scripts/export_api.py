import os
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import sqlalchemy
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse


app = FastAPI(title="EVE Trader Data Export")


def _export_path() -> Path:
    return Path(os.getenv("EXPORT_FILE", "/app/data/training_data_cleaned.parquet"))


def _db_url() -> str:
    # Prefer explicit env from docker-compose; fall back to in-network DB.
    return os.getenv("DATABASE_URL", "postgresql://eve_user:eve_pass@db:5432/eve_market_data")


def _lookback_hours() -> int:
    try:
        return int(os.getenv("EXPORT_LOOKBACK_HOURS", "24"))
    except Exception:
        return 24


def _min_bytes() -> int:
    try:
        return int(os.getenv("EXPORT_MIN_BYTES", "500000"))  # ~0.5MB
    except Exception:
        return 500000


def _max_age_seconds() -> int:
    try:
        return int(os.getenv("EXPORT_MAX_AGE_SECONDS", "300"))  # 5 minutes
    except Exception:
        return 300


def _is_stale_or_small(p: Path) -> bool:
    if not p.exists():
        return True
    try:
        st = p.stat()
        if st.st_size < _min_bytes():
            return True
        age = datetime.utcnow() - datetime.utcfromtimestamp(st.st_mtime)
        if age.total_seconds() > _max_age_seconds():
            return True
    except Exception:
        return True
    return False


def _generate_export(p: Path) -> None:
    lookback = _lookback_hours()
    db_url = _db_url()
    engine = sqlalchemy.create_engine(db_url)

    # Pull all available market_history rows from last N hours.
    q = sqlalchemy.text(
        """
        SELECT *
        FROM market_history
        WHERE \"timestamp\" >= NOW() - (:lookback * INTERVAL '1 hour')
        ORDER BY \"timestamp\" ASC, type_id
        """
    )

    with engine.connect() as conn:
        df = pd.read_sql_query(q, conn, params={"lookback": int(lookback)})

    # Lightweight scrub in-memory (do not mutate DB).
    if not df.empty:
        if "avg_price" in df.columns and "close" in df.columns:
            price = df["avg_price"].fillna(df["close"])
        elif "close" in df.columns:
            price = df["close"]
        else:
            price = None

        if price is not None:
            df = df[price <= 49_000_000_000]
        if "volume" in df.columns:
            df = df[df["volume"].fillna(0) >= 5]

    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(p)


@app.get("/health")
def health():
    p = _export_path()
    return {"ok": True, "file": str(p), "exists": p.exists(), "size": p.stat().st_size if p.exists() else 0}


@app.get("/training_data_cleaned.parquet")
def download_training_data(refresh: bool = False):
    p = _export_path()
    if refresh or _is_stale_or_small(p):
        try:
            _generate_export(p)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Export generation failed: {e}")
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"Missing export file: {p}")
    return FileResponse(path=str(p), media_type="application/octet-stream", filename=p.name)
