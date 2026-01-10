"""Technical indicators for EVE Trader.

Implements a small, dependency-light indicator set:
- RSI(14)
- MACD(12, 26, 9)
- Bollinger Bands(20, 2σ)

Design goals:
- Works on historical Parquet frames (pandas DataFrames)
- Can also be used in streaming contexts by feeding rolling price histories

No strategy logic lives here; this module only computes signals.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI.

    Returns a series in [0, 100] (NaN for the initial warmup window).
    """

    close = pd.to_numeric(close, errors="coerce")
    delta = close.diff()

    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    # Wilder smoothing (EMA with alpha=1/period)
    avg_gain = gain.ewm(alpha=1 / float(period), adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / float(period), adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))

    # If avg_loss is 0, RSI is 100; if avg_gain is 0, RSI is 0.
    out = out.where(~avg_loss.eq(0.0), 100.0)
    out = out.where(~avg_gain.eq(0.0), 0.0)

    return out


def macd(
    close: pd.Series, *, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series]:
    """MACD line and signal line."""

    close = pd.to_numeric(close, errors="coerce")
    ema_fast = close.ewm(span=int(fast), adjust=False, min_periods=int(fast)).mean()
    ema_slow = close.ewm(span=int(slow), adjust=False, min_periods=int(slow)).mean()
    line = ema_fast - ema_slow
    sig = line.ewm(span=int(signal), adjust=False, min_periods=int(signal)).mean()
    return line, sig


def bollinger_bands(
    close: pd.Series, *, period: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series]:
    """Bollinger upper/lower bands."""

    close = pd.to_numeric(close, errors="coerce")
    ma = close.rolling(window=int(period), min_periods=int(period)).mean()
    std = close.rolling(window=int(period), min_periods=int(period)).std(ddof=0)
    upper = ma + float(num_std) * std
    lower = ma - float(num_std) * std
    return upper, lower


def add_indicators(
    df: pd.DataFrame,
    *,
    price_col: str = "market_price",
    group_col: str = "type_id",
    time_col: str = "t",
) -> pd.DataFrame:
    """Add RSI/MACD/Bollinger columns to a time-series feature frame.

    Output columns:
    - RSI
    - MACD
    - MACD_Signal
    - BB_Upper
    - BB_Low

    Assumes df has at least (group_col, time_col, price_col).
    """

    if df.empty:
        out = df.copy()
        for c in ["RSI", "MACD", "MACD_Signal", "BB_Upper", "BB_Low"]:
            out[c] = np.nan
        return out

    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
    out = out.sort_values([group_col, time_col])

    def _apply(g: pd.DataFrame) -> pd.DataFrame:
        close = g[price_col]
        g["RSI"] = rsi(close, period=14)
        macd_line, sig = macd(close, fast=12, slow=26, signal=9)
        g["MACD"] = macd_line
        g["MACD_Signal"] = sig
        upper, lower = bollinger_bands(close, period=20, num_std=2.0)
        g["BB_Upper"] = upper
        g["BB_Low"] = lower
        return g

    out = out.groupby(group_col, group_keys=False).apply(_apply)
    return out


@dataclass
class IndicatorState:
    """In-memory rolling indicator state for streaming feeds."""

    maxlen: int = 256
    prices: dict[int, deque[float]] | None = None

    def __post_init__(self) -> None:
        if self.prices is None:
            self.prices = {}

    def push(self, *, type_id: int, price: float) -> None:
        if type_id not in self.prices:
            self.prices[type_id] = deque(maxlen=int(self.maxlen))
        self.prices[type_id].append(float(price))

    def series(self, *, type_id: int) -> pd.Series:
        q = self.prices.get(type_id)
        if not q:
            return pd.Series([], dtype="float64")
        return pd.Series(list(q), dtype="float64")


def latest_indicators_from_prices(prices: Iterable[float]) -> dict[str, float | None]:
    """Compute last RSI/MACD/Bollinger values from a rolling price window."""

    s = pd.Series(list(prices), dtype="float64")
    if s.empty:
        return {"RSI": None, "MACD": None, "MACD_Signal": None, "BB_Upper": None, "BB_Low": None}

    r = rsi(s, period=14)
    m, ms = macd(s, fast=12, slow=26, signal=9)
    bb_u, bb_l = bollinger_bands(s, period=20, num_std=2.0)

    def _last(x: pd.Series) -> float | None:
        try:
            v = x.iloc[-1]
            if pd.isna(v):
                return None
            return float(v)
        except Exception:
            return None

    return {
        "RSI": _last(r),
        "MACD": _last(m),
        "MACD_Signal": _last(ms),
        "BB_Upper": _last(bb_u),
        "BB_Low": _last(bb_l),
    }
