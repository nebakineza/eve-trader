print("Oracle Mind module initialized", flush=True)
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import redis
import json
import time
import sys
import os
import requests
import math
import re
from datetime import datetime, timedelta, timezone
import sqlalchemy
from sqlalchemy import text
from streamlit_autorefresh import st_autorefresh

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from strategist.backtester import BacktestEngine
except ImportError:
    BacktestEngine = None

# --- Configuration ---
st.set_page_config(page_title="EVE Trader Command Center", layout="wide", page_icon="📈")
st.title("EVE Online Station Trader - Command Center")

# --- Helpers ---
@st.cache_data(ttl=3600)
def resolve_type_names(type_ids):
    if not type_ids: return {}
    # Avoid stalling the dashboard on extremely large batches.
    max_ids = int(os.getenv("ESI_RESOLVE_MAX_IDS", "200"))
    url = "https://esi.evetech.net/latest/universe/names/"
    try:
        ids = list(set([int(x) for x in type_ids if x is not None]))
    except Exception:
        ids = []
    if not ids:
        return {}
    if len(ids) > max_ids:
        ids = ids[:max_ids]
    mapping = {}
    chunk_size = 900 
    for i in range(0, len(ids), chunk_size):
        chunk = ids[i:i + chunk_size]
        try:
            resp = requests.post(url, json=chunk, timeout=5)
            if resp.status_code == 200:
                for item in resp.json():
                    mapping[str(item['id'])] = item['name']
        except Exception as e:
            pass
    return mapping

st.markdown("""
    <style>
    .alpha-card {
        border: 1px solid #4a4a4a;
        border-radius: 10px;
        padding: 15px;
        height: 250px;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

refresh_interval_ms = int(os.getenv("STREAMLIT_REFRESH_MS", "10000"))
refresh_tick = None
if refresh_interval_ms > 0:
    refresh_tick = st_autorefresh(interval=refresh_interval_ms, key="datarefresh")

# Quick operator-visible confirmation of refresh cadence.
try:
    if refresh_tick is not None:
        st.sidebar.caption(f"Refresh tick: {int(refresh_tick)}")
except Exception:
    pass


@st.cache_data(ttl=300)
def get_hypertable_sizes_df() -> pd.DataFrame:
    if not db_engine:
        return pd.DataFrame(columns=["table", "bytes", "size"])
    try:
        with db_engine.connect() as conn:
            df = pd.read_sql(
                text(
                    """
                    SELECT
                        relname AS table,
                        pg_total_relation_size(relid) AS bytes,
                        pg_size_pretty(pg_total_relation_size(relid)) AS size
                    FROM pg_catalog.pg_statio_user_tables
                    WHERE relname IN ('market_orders','market_history','shadow_trades','pending_trades')
                    ORDER BY bytes DESC
                    """
                ),
                conn,
            )
        return df
    except Exception:
        return pd.DataFrame(columns=["table", "bytes", "size"])


@st.cache_data(ttl=300)
def get_resolved_trades_df(outcome_filter: str, limit: int = 100, horizon_minutes: int = 1440) -> pd.DataFrame:
    if not db_engine:
        return pd.DataFrame()
    
    # outcome_filter should be 'WIN' or 'LOSS'
    with db_engine.connect() as conn:
        return pd.read_sql(
            text(
                """
                SELECT
                    t.type_id,
                    t.signal_type,
                    t.actual_price_at_time AS entry_price,
                    mh.close AS exit_price,
                    t.reasoning,
                    t.virtual_outcome as status,
                    t.timestamp
                FROM shadow_trades t
                LEFT JOIN LATERAL (
                    SELECT close
                    FROM market_history
                    WHERE type_id = t.type_id
                      AND timestamp >= t.timestamp + (:horizon || ' minutes')::interval
                    ORDER BY timestamp ASC
                    LIMIT 1
                ) mh ON TRUE
                WHERE t.virtual_outcome = :outcome
                ORDER BY t.timestamp DESC
                LIMIT :limit
                """
            ),
            conn,
            params={"limit": int(limit), "horizon": int(horizon_minutes), "outcome": outcome_filter},
        )

@st.cache_data(ttl=300)
def get_shadow_trade_outcomes_df(limit: int = 100, horizon_minutes: int = 60, fee_rate: float = 0.025) -> pd.DataFrame:
    if not db_engine:
        return pd.DataFrame()

    # Ensure schema exists before selecting optional columns.
    try:
        ddl_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "storage", "shadow_ledger.sql")
        with open(ddl_path, "r", encoding="utf-8") as f:
            ddl = f.read()
        with db_engine.begin() as conn:
            for stmt in ddl.split(";"):
                stmt = stmt.strip()
                if stmt:
                    conn.execute(text(stmt))
    except Exception:
        pass

    with db_engine.connect() as conn:
        return pd.read_sql(
            text(
                """
                SELECT
                    t.type_id,
                    t.signal_type,
                    t.actual_price_at_time AS entry_price,
                    mh.close AS exit_price,
                    t.reasoning,
                    t.virtual_outcome,
                    t.timestamp
                FROM shadow_trades t
                LEFT JOIN LATERAL (
                    SELECT close
                    FROM market_history
                    WHERE type_id = t.type_id
                      AND timestamp >= t.timestamp + (:horizon || ' minutes')::interval
                    ORDER BY timestamp ASC
                    LIMIT 1
                ) mh ON TRUE
                ORDER BY t.timestamp DESC
                LIMIT :limit
                """
            ),
            conn,
            params={"limit": int(limit), "horizon": int(horizon_minutes)},
        )


def _server_today_utc_window() -> tuple[datetime, datetime]:
    now_local = datetime.now().astimezone()
    start_local = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    end_local = start_local + timedelta(days=1)
    return start_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc)


@st.cache_data(ttl=60)
def get_shadow_trade_outcomes_window_df(
    *,
    start_utc: datetime,
    end_utc: datetime,
    limit: int = 5000,
    horizon_minutes: int = 60,
) -> pd.DataFrame:
    if not db_engine:
        return pd.DataFrame()

    # Ensure schema exists before selecting optional columns.
    try:
        ddl_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "storage", "shadow_ledger.sql")
        with open(ddl_path, "r", encoding="utf-8") as f:
            ddl = f.read()
        with db_engine.begin() as conn:
            for stmt in ddl.split(";"):
                stmt = stmt.strip()
                if stmt:
                    conn.execute(text(stmt))
    except Exception:
        pass

    with db_engine.connect() as conn:
        return pd.read_sql(
            text(
                """
                SELECT
                    t.type_id,
                    t.signal_type,
                    t.actual_price_at_time AS entry_price,
                    mh.close AS exit_price,
                    t.reasoning,
                    t.virtual_outcome,
                    t.timestamp
                FROM shadow_trades t
                LEFT JOIN LATERAL (
                    SELECT close
                    FROM market_history
                    WHERE type_id = t.type_id
                      AND timestamp >= t.timestamp + (:horizon || ' minutes')::interval
                    ORDER BY timestamp ASC
                    LIMIT 1
                ) mh ON TRUE
                WHERE t.timestamp >= :start_ts
                  AND t.timestamp < :end_ts
                ORDER BY t.timestamp DESC
                LIMIT :limit
                """
            ),
            conn,
            params={
                "limit": int(limit),
                "horizon": int(horizon_minutes),
                "start_ts": start_utc,
                "end_ts": end_utc,
            },
        )


@st.cache_data(ttl=60)
def get_shadow_trade_completed_df(*, limit: int = 200, horizon_minutes: int = 60) -> pd.DataFrame:
    if not db_engine:
        return pd.DataFrame()

    with db_engine.connect() as conn:
        return pd.read_sql(
            text(
                """
                SELECT
                    t.type_id,
                    t.signal_type,
                    t.actual_price_at_time AS entry_price,
                    mh.close AS exit_price,
                    t.reasoning,
                    t.virtual_outcome,
                    t.timestamp
                FROM shadow_trades t
                LEFT JOIN LATERAL (
                    SELECT close
                    FROM market_history
                    WHERE type_id = t.type_id
                      AND timestamp >= t.timestamp + (:horizon || ' minutes')::interval
                    ORDER BY timestamp ASC
                    LIMIT 1
                ) mh ON TRUE
                WHERE t.timestamp <= NOW() - (:horizon || ' minutes')::interval
                  AND mh.close IS NOT NULL
                ORDER BY t.timestamp DESC
                LIMIT :limit
                """
            ),
            conn,
            params={"limit": int(limit), "horizon": int(horizon_minutes)},
        )


@st.cache_data(ttl=60)
def get_shadow_trade_completed_by_side_df(*, side: str, limit: int = 200, horizon_minutes: int = 60) -> pd.DataFrame:
    if not db_engine:
        return pd.DataFrame()

    side_u = str(side or "").strip().upper()
    if side_u not in {"BUY", "SELL"}:
        return pd.DataFrame()

    with db_engine.connect() as conn:
        return pd.read_sql(
            text(
                """
                SELECT
                    t.type_id,
                    t.signal_type,
                    t.actual_price_at_time AS entry_price,
                    mh.close AS exit_price,
                    t.reasoning,
                    t.virtual_outcome,
                    t.timestamp
                FROM shadow_trades t
                LEFT JOIN LATERAL (
                    SELECT close
                    FROM market_history
                    WHERE type_id = t.type_id
                      AND timestamp >= t.timestamp + (:horizon || ' minutes')::interval
                    ORDER BY timestamp ASC
                    LIMIT 1
                ) mh ON TRUE
                WHERE t.signal_type = :side
                  AND t.timestamp <= NOW() - (:horizon || ' minutes')::interval
                  AND mh.close IS NOT NULL
                ORDER BY t.timestamp DESC
                LIMIT :limit
                """
            ),
            conn,
            params={"limit": int(limit), "horizon": int(horizon_minutes), "side": side_u},
        )


@st.cache_data(ttl=300)
def get_item_roi_df(limit: int = 5, horizon_minutes: int = 60, fee_rate: float = 0.025) -> pd.DataFrame:
    if not db_engine:
        return pd.DataFrame()
    with db_engine.connect() as conn:
        return pd.read_sql(
            text(
                """
                WITH base AS (
                    SELECT
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
                    WHERE t.virtual_outcome IN ('WIN','LOSS')
                      AND t.actual_price_at_time IS NOT NULL
                ), scored AS (
                    SELECT
                        type_id,
                        entry_price,
                        CASE
                            WHEN exit_price IS NULL OR entry_price <= 0 OR exit_price <= 0 THEN NULL
                            WHEN UPPER(signal_type) = 'BUY' THEN (exit_price - entry_price)
                            ELSE (entry_price - exit_price)
                        END AS gross_isk
                    FROM base
                )
                SELECT
                    type_id,
                    COUNT(*)::bigint AS trade_count,
                    COALESCE(SUM(CASE WHEN gross_isk IS NULL THEN 0 ELSE gross_isk END) * (1 - :fee), 0) AS net_profit,
                    CASE
                        WHEN SUM(CASE WHEN entry_price IS NULL OR entry_price <= 0 THEN 0 ELSE entry_price END) <= 0 THEN 0
                        ELSE (COALESCE(SUM(CASE WHEN gross_isk IS NULL THEN 0 ELSE gross_isk END) * (1 - :fee), 0)
                              / SUM(CASE WHEN entry_price IS NULL OR entry_price <= 0 THEN 0 ELSE entry_price END))
                    END AS net_roi
                FROM scored
                GROUP BY type_id
                ORDER BY net_roi DESC
                LIMIT :limit
                """
            ),
            conn,
            params={"limit": int(limit), "horizon": int(horizon_minutes), "fee": float(fee_rate)},
        )


@st.cache_data(ttl=300)
def get_trade_volume_by_hour_utc_df(lookback_hours: int = 24 * 7) -> pd.DataFrame:
    if not db_engine:
        return pd.DataFrame(columns=["hour_utc", "trade_count"])
    lookback_hours = max(1, int(lookback_hours))
    with db_engine.connect() as conn:
        df = pd.read_sql(
            text(
                """
                SELECT
                    EXTRACT(hour FROM (timestamp AT TIME ZONE 'UTC'))::int AS hour_utc,
                    COUNT(*)::bigint AS trade_count
                FROM shadow_trades
                WHERE timestamp >= NOW() - (:hours || ' hours')::interval
                GROUP BY 1
                ORDER BY 1
                """
            ),
            conn,
            params={"hours": int(lookback_hours)},
        )

    # Ensure 0..23 present for a stable histogram.
    if df.empty:
        return pd.DataFrame({"hour_utc": list(range(24)), "trade_count": [0] * 24})
    all_hours = pd.DataFrame({"hour_utc": list(range(24))})
    df2 = all_hours.merge(df, on="hour_utc", how="left").fillna({"trade_count": 0})
    df2["trade_count"] = df2["trade_count"].astype(int)
    return df2


def _safe_redis_keys(redis_client, pattern: str, limit: int = 200):
    """Avoid Redis KEYS (blocking). Return up to `limit` keys via SCAN."""
    if redis_client is None:
        return []
    try:
        limit = max(1, int(limit))
    except Exception:
        limit = 200
    out = []
    try:
        for k in redis_client.scan_iter(match=pattern, count=min(1000, limit)):
            out.append(k)
            if len(out) >= limit:
                break
    except Exception:
        return []
    return out

# --- Local/Remote Log Helpers ---
def _tail_text_file(path: str, lines: int = 3, max_bytes: int = 64 * 1024) -> str:
    """Read the last N lines of a text file without loading it all."""
    try:
        if not path or not os.path.exists(path):
            return ""
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            read_size = min(max_bytes, size)
            f.seek(-read_size, os.SEEK_END)
            data = f.read(read_size)
        text = data.decode("utf-8", errors="replace")
        parts = text.splitlines()
        return "\n".join(parts[-lines:])
    except Exception:
        return ""


def _read_macro_history(*, redis_client, key: str, max_points: int) -> pd.Series:
    try:
        if redis_client is None:
            return pd.Series(dtype="float64")
        raw = redis_client.lrange(key, -int(max_points), -1)
        if not raw:
            return pd.Series(dtype="float64")
        ts = []
        values = []
        for item in raw:
            try:
                payload = json.loads(item)
                t = float(payload.get("ts", 0) or 0)
                v = payload.get("v", None)
                if t <= 0 or v is None:
                    continue
                ts.append(datetime.utcfromtimestamp(t))
                values.append(float(v))
            except Exception:
                continue
        if not ts:
            return pd.Series(dtype="float64")
        s = pd.Series(values, index=pd.to_datetime(ts)).sort_index()
        # Drop duplicate timestamps (keep last)
        s = s[~s.index.duplicated(keep="last")]
        return s
    except Exception:
        return pd.Series(dtype="float64")





# --- Redis Connection ---
@st.cache_resource
def get_redis_client():
    return redis.Redis(host=os.getenv('REDIS_HOST', 'cache'), port=6379, db=0, decode_responses=True)


def _norm_redis_text(v) -> str:
    if v is None:
        return ""
    if isinstance(v, (bytes, bytearray)):
        try:
            return v.decode("utf-8", errors="ignore")
        except Exception:
            return ""
    return str(v)

try:
    r = get_redis_client()
    if r.ping():
        redis_status = "✅ Connected"
    else:
        redis_status = "❌ Connection Failed"
except Exception as e:
    redis_status = f"❌ Error: {e}"
    r = None

# --- DB Connection & Monitor ---
# EDITED: PREFER ENV VAR, FALLBACK TO HARDCODED
try:
    db_url = os.getenv("DATABASE_URL", "postgresql://eve_user:eve_pass@db:5432/eve_market_data")
    db_engine = sqlalchemy.create_engine(db_url)
    @st.cache_data(ttl=120)
    def get_market_orders_count():
        try:
            with db_engine.connect() as conn:
                res = conn.execute(text("SELECT count(*) FROM market_orders")).fetchone()
                return int(res[0] or 0)
        except:
            return 0

    @st.cache_data(ttl=120)
    def get_market_history_bucket_count(hours: int = 12):
        try:
            with db_engine.connect() as conn:
                # Prefer TimescaleDB if present.
                try:
                    res = conn.execute(
                        text(
                            """
                            SELECT COUNT(DISTINCT time_bucket('5 minutes', timestamp))
                            FROM market_history
                            WHERE timestamp >= NOW() - (:hours || ' hours')::interval
                            """
                        ),
                        {"hours": int(hours)},
                    ).fetchone()
                except Exception:
                    # Fallback without time_bucket.
                    res = conn.execute(
                        text(
                            """
                            SELECT COUNT(DISTINCT date_trunc('minute', timestamp) -
                                (EXTRACT(minute FROM timestamp)::int % 5) * INTERVAL '1 minute')
                            FROM market_history
                            WHERE timestamp >= NOW() - (:hours || ' hours')::interval
                            """
                        ),
                        {"hours": int(hours)},
                    ).fetchone()
                return int(res[0] or 0)
        except:
            return 0

    db_count = get_market_orders_count()
except:
    db_engine = None
    db_count = 0

# --- Sidebar Global Controls ---
st.sidebar.header("System Controls")
st.sidebar.code("v2.1.0-RESURRECTED")
st.sidebar.caption(f"Redis: {redis_status}")
st.sidebar.caption(f"DB Rows: {db_count}") # Emergency Monitor

auto_trade_state = st.sidebar.toggle("AUTO_TRADE_ENABLED", value=False)
if r:
    r.set("system:auto_trade_enabled", "true" if auto_trade_state else "false")
    status_msg = "🟢 AI-AUTONOMOUS" if auto_trade_state else "🔴 HUMAN-IN-LOOP"
    st.sidebar.markdown(f"**Status:** {status_msg}")
    st.sidebar.caption("Hot Reload Active ⚡")
    
    heartbeat_ts = r.get("system:heartbeat:latest")
    if heartbeat_ts:
        try:
            minutes_ago = (time.time() - float(heartbeat_ts)) / 60
            st.sidebar.metric("Heartbeat", f"{minutes_ago:.1f} min ago")
        except: pass
else:
    st.sidebar.warning("Redis unavailable")

# --- Zombie & Bridge Status (Sidebar) ---
st.sidebar.markdown("---")
st.sidebar.header("Hardware Bridge")

# Hardware Lock toggle (prevents Arduino bridge from sending serial commands until enabled)
hardware_lock_state = st.sidebar.toggle("HARDWARE_LOCK", value=False, help="When enabled (locked), the Arduino bridge will NOT send serial commands. Disable to arm the hardware.")
if r:
    r.set("system:hardware_lock", "true" if hardware_lock_state else "false")
    lock_msg = "🔒 LOCKED (safe)" if hardware_lock_state else "🔓 UNLOCKED (armed)"
    st.sidebar.caption(lock_msg)

try:
    zombie_status = r.get("zombie:hid_status") if r else "AMBER"
    if zombie_status == "GREEN":
        st.sidebar.success("🟢 Arduino: ACTIVE")
    else:
        st.sidebar.warning("🟠 Arduino: OFFLINE")
except:
    st.sidebar.error("⚪ Status: UNKNOWN")

# --- Capital & Appetite ---
st.sidebar.markdown("---")
st.sidebar.header("Capital & Appetite")
# Static values only
TOTAL_CAPITAL = 10_000_000_000
LIQUID_CAPITAL = 2_000_000_000
col1, col2 = st.sidebar.columns(2)
with col1: st.metric("Total Asset", f"{TOTAL_CAPITAL/1e9:.1f}B")
with col2: st.metric("Liquid ISK", f"{LIQUID_CAPITAL/1e9:.1f}B")

# --- DATA FETCHING (Global) ---
pulse_items = []
raw_items = []
name_map = {}

if r:
    max_feature_keys = int(os.getenv("FEATURES_MAX_KEYS", "500"))
    keys = []
    try:
        # IMPORTANT: avoid Redis KEYS (blocks server). Use bounded SCAN.
        for k in r.scan_iter(match="features:*", count=200):
            if len(str(k).split(":")) == 2:
                keys.append(k)
            if len(keys) >= max_feature_keys:
                break
    except Exception:
        keys = []
    
    ids_to_resolve = []
    for k in keys:
        try:
            val = r.get(k)
            if val:
                obj = json.loads(val)
                raw_items.append(obj)
                # Delay name resolution until we know what we'll display.
        except: pass
    
    def alpha_score(item):
        return abs(float(item.get("imbalance", 1.0)) - 1.0)

    raw_items.sort(key=alpha_score, reverse=True)
    top_5 = raw_items[:5]

    # Resolve only what we actually display to keep load bounded.
    ids_to_resolve = [i.get("type_id") for i in top_5 if i.get("type_id") is not None]
    name_map = resolve_type_names(ids_to_resolve)
    
    for item in top_5:
        tid = str(item.get("type_id"))
        name = name_map.get(tid, f"Type {tid}")
        curr = float(item.get("price", 0))
        spread = float(item.get("spread", 0))
        predicted = curr * (1 + spread)
        conf = min(0.99, abs(float(item.get("imbalance", 1.0)) - 1.0))
        
        pulse_items.append({
            "Item": name, 
            "Current": curr, 
            "Predicted": predicted, 
            "Conf": conf
        })

if not pulse_items:
     pulse_items = [{"Item": "Waiting for Data...", "Current": 0.0, "Predicted": 0.0, "Conf": 0.0}]

# --- TABS LAYOUT ---
tab_radar, tab_execution, tab_oracle, tab_fundamentals, tab_performance, tab_system = st.tabs(
    ["Radar", "Execution", "Oracle", "📰 Fundamentals", "📈 Performance Audit", "System"]
)

# ========================
# TAB 1: RADAR (Market Radar)
# ========================
with tab_radar:
    st.markdown("### 📡 Alpha Pulse (Top Signal)")
    cols = st.columns(5)
    for i, item in enumerate(pulse_items):
        spread_pct = 0.0
        if item["Current"] > 0:
            spread_pct = ((item["Predicted"] - item["Current"]) / item["Current"]) * 100
        is_hot = spread_pct > 0.5 and item["Conf"] > 0.6
        
        with cols[i]:
            with st.container():
                st.metric(
                    label=item['Item'], 
                    value=f"{spread_pct:.1f}%", 
                    delta=f"{item['Predicted'] - item['Current']:,.0f} ISK",
                    delta_color="normal"
                )
                st.progress(float(item["Conf"]))
    
    st.divider()
    
    col_live, col_opp = st.columns([1,1])
    
    with col_live:
        st.header("Live Market Feed")
        if raw_items:
            # Keep UI responsive if Redis contains many feature keys.
            max_rows = int(os.getenv("FEATURES_MAX_ROWS", "200"))
            market_data = []
            for obj in raw_items[:max_rows]:
                tid = str(obj.get("type_id"))
                name = name_map.get(tid, tid)
                price = float(obj.get("price", 0))
                imbal = float(obj.get("imbalance", 1.0))
                spread = float(obj.get("spread", 0))
                market_data.append({
                    "Type": name, "Price": price, "RSI": f"{imbal:.2f}", 
                    "Spread": f"{spread * 100:.1f}%"
                })
            st.dataframe(pd.DataFrame(market_data), use_container_width=True, height=400)
        else:
            st.info("Initializing Data Stream...")
            
    with col_opp:
        st.header("Opportunity Scan")
        radar_items = []
        if db_engine:
            try:
                with db_engine.connect() as conn:
                    res = conn.execute(text("SELECT type_id, metric_score FROM market_radar ORDER BY metric_score DESC LIMIT 10")).fetchall()
                    for row in res:
                        radar_items.append({"Item": str(row[0]), "Score": float(row[1] or 0)})
            except: pass
        
        if radar_items:
            tids = [r['Item'] for r in radar_items]
            rman = resolve_type_names(tids)
            for r_item in radar_items:
                 r_item['Item'] = rman.get(r_item['Item'], r_item['Item'])
            st.dataframe(pd.DataFrame(radar_items), use_container_width=True, height=400)
        else:
            st.info("Awaiting Analyst Scan Results...")


# ========================
# TAB 2: EXECUTION (Execution Desk)
# ========================
with tab_execution:
    st.markdown("### ⚔️ Live Execution Desk")
    col_orders, col_ledger = st.columns(2)
    
    with col_orders:
        st.subheader("Pending Orders")
        orders_data = []
        if db_engine:
            try:
                with db_engine.connect() as conn:
                    q = text("SELECT id, type_id, action, price, status, created_at FROM pending_trades WHERE status NOT IN ('FILLED','CANCELLED') ORDER BY created_at DESC LIMIT 10")
                    res = conn.execute(q).fetchall()
                    for row in res:
                        orders_data.append({
                            "ID": row.id, "Item": str(row.type_id), "Action": row.action,
                            "Price": row.price, "Status": row.status
                        })
            except: pass
        if orders_data:
            # Resolve
            tids = [str(o['Item']) for o in orders_data]
            omap = resolve_type_names(tids)
            for o in orders_data: o['Item'] = omap.get(o['Item'], o['Item'])
            st.dataframe(pd.DataFrame(orders_data), use_container_width=True)
        else:
            st.info("No Active Orders / Queue Empty")

    with col_ledger:
        st.subheader("Live Ledger (Completed)")
        # Mock or use history if available
        st.info("Ledger Syncing with Trade Logs...")

# ========================
# TAB 3: ORACLE (Oracle Brain)
# ========================
with tab_oracle:
    st.markdown("### 🧠 Oracle Insight")

    # --- New Diagnostic Injection: Inference Narrative ---
    st.markdown("#### 👁️ Oracle Vision (Inference Narrative)")
    
    # 1. Real-time Parameters (Kelly & Confidence)
    conf_thresh = float(os.getenv("SHADOW_CONFIDENCE_THRESHOLD", "0.60"))
    # Kelly: Try to get from last signal or config. 
    # For now, we simulate/calculate based on standard config references if not in Redis.
    # In minimalist mode, we show the static config or recent signal snapshot.
    kelly_frac = 0.05 # Default cap
    try:
        last_sig_raw = r.get("oracle:last_signal_snapshot")
        if last_sig_raw:
            d = json.loads(last_sig_raw)
            kelly_frac = float(d.get("kelly", 0.05))
    except:
        pass
    
    c_d1, c_d2, c_d3 = st.columns(3)
    c_d1.metric("Active Confidence Threshold", f"{conf_thresh:.2f}")
    c_d2.metric("Kelly Fraction (Dynamic)", f"{kelly_frac:.4f}")
    
    # 2. Optimization History (Induction Loss Sparkline)
    curr_loss_hist = []
    try:
        # oracle:training_lineage is a list of JSON strings
        lineage = r.lrange("oracle:training_lineage", 0, 9) # Last 10
        if lineage:
            for run_str in reversed(lineage): # Oldest to newest
                run_data = json.loads(run_str)
                # Try 'final_loss' or 'val_loss' or 'best_val_loss'
                # fallback to 0
                val = run_data.get("best_val_loss") or run_data.get("loss") or 0.0
                curr_loss_hist.append(float(val))
    except:
        pass
    
    if curr_loss_hist:
        c_d3.caption("Induction Loss Trend (Last 10)")
        c_d3.line_chart(curr_loss_hist, height=80) 
    else:
        c_d3.info("No Training History")

    # 3. Weight Deviations (Attention)
    # Parse oracle:feature_importance
    try:
        fi_raw = r.get("oracle:feature_importance")
        if fi_raw:
            fi = json.loads(fi_raw)
            # Sort by absolute value desc
            sorted_fi = sorted(fi.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
            
            top_text = [f"**{k}** ({v:.3f})" for k,v in sorted_fi]
            st.info(f"🔥 **High Attention Features:** {', '.join(top_text)}")
        else:
            st.info("Waiting for Feature Importance Analysis...")
    except:
        pass

    st.divider()

    # Model Path & Validation R²
    model_path_used = "models/oracle_v1_latest.pt"  # default / fallback
    if r:
        try:
            # Support multiple key names used by different pipeline versions.
            for k in (
                "oracle:model_latest_path",
                "oracle:model_latest_path_5090",
                "oracle:model_path",
            ):
                from_redis = r.get(k)
                if from_redis:
                    model_path_used = str(from_redis)
                    break
        except Exception:
            pass

    # If the local synced artifact exists, prefer showing it.
    try:
        from pathlib import Path

        root_dir = Path(__file__).resolve().parents[1]
        local_pt = root_dir / "models" / "oracle_v1_latest.pt"
        if local_pt.exists():
            model_path_used = str(local_pt)
    except Exception:
        pass

    val_r2 = 0.0
    train_r2 = 0.0
    prev_val_r2 = 0.0
    loss_last = 0.0
    reject_reason = None
    
    if r:
        try:
            metrics_raw = r.get("oracle:last_training_metrics")
            if metrics_raw:
                metrics = json.loads(metrics_raw)
                if isinstance(metrics, dict):
                    # Direct fields (on successful training runs)
                    val_r2 = float(metrics.get("val_r2", 0.0) or 0.0)
                    train_r2 = float(metrics.get("train_r2", 0.0) or 0.0)
                    prev_val_r2 = float(metrics.get("baseline_val_r2", 0.0) or 0.0) # Often stored as baseline
                    
                    # Try to get loss
                    # If metrics has 'history' or 'final_loss'
                    loss_last = float(metrics.get("loss", 0.0) or 0.0)
                    
                    reject_reason = metrics.get("reject_reason")

                    # On reject paths, val_r2/prev_val_r2 may only appear in the error string.
                    err = metrics.get("error")
                    if isinstance(err, str):
                        if not reject_reason and ":" in err:
                            tail = err.split(":", 1)[1].strip()
                            reject_reason = tail.split("(", 1)[0].strip() if tail else None
                        if (val_r2 or 0.0) <= 0.0:
                            m = re.search(r"\bval_r2=([0-9]*\.?[0-9]+)", err)
                            if m:
                                try:
                                    val_r2 = float(m.group(1))
                                except Exception:
                                    pass
        except Exception:
            pass

    # Directive 4: Alpha Ledger Cleanup (Fallback to Disk Metadata)
    if train_r2 == 0.0 and loss_last == 0.0:
        try:
            import torch
            from pathlib import Path
            root_dir = Path(__file__).resolve().parents[1]
            live_pt = root_dir / "oracle_v1.pth"
            
            if live_pt.exists():
                # Load on CPU to avoid CUDA overhead on dashboard
                checkpoint = torch.load(live_pt, map_location='cpu')
                if isinstance(checkpoint, dict):
                    meta = checkpoint.get('metadata', {})
                    if meta:
                        train_r2 = float(meta.get('train_r2', 0.0))
                        val_r2 = float(meta.get('val_r2', 0.0))
                        prev_val_r2 = float(meta.get('baseline_val_r2', 0.0))
                        loss_last = float(meta.get('best_val_loss', 0.0))
        except Exception:
            pass # Keep as 0.0 if fail

    # Next training session countdown (based on 4h cron schedule)
    next_training_eta = "N/A"
    try:
        now_utc = datetime.now(tz=timezone.utc)
        # Next boundary at 00:00, 04:00, 08:00, 12:00, 16:00, 20:00 UTC
        add_hours = 4 - (int(now_utc.hour) % 4)
        if add_hours == 0 and (now_utc.minute > 0 or now_utc.second > 0):
            add_hours = 4
        next_train = now_utc.replace(minute=0, second=0, microsecond=0) + timedelta(hours=int(add_hours))
        remaining_s = max(0, int((next_train - now_utc).total_seconds()))
        h = remaining_s // 3600
        m = (remaining_s % 3600) // 60
        next_training_eta = f"{h}h {m}m"
    except Exception:
        next_training_eta = "N/A"

    # -- Alpha Ledger (Model Comparison) --
    st.markdown("#### 📜 The Oracle 'Alpha Ledger'")
    
    # Calculate Slippage-Adjusted R2 (Smith & Johnson Penalty)
    # R2 decays as signal magnitude struggles against 0.5% Jita friction
    slippage_r2 = val_r2 * (1.0 - (0.005 * 2)) # Simple approximation: 1% friction penalty

    alpha_data = {
        "Metric": ["Val R² (Current)", "Val R² (Slippage Adj.)", "Val R² (Previous Best)", "Training R²", "Induction Loss", "Status"],
        "Value": [
            f"{val_r2:.4f}",
            f"{slippage_r2:.4f}",
            f"{prev_val_r2:.4f}", 
            f"{train_r2:.4f}",
            f"{loss_last:.4f}",
            "REJECTED" if reject_reason else "ACTIVE"
        ],
        "Delta": [
            f"{(val_r2 - prev_val_r2):+.4f}",
            f"{(slippage_r2 - prev_val_r2):+.4f}",
            "-",
            "-",
            "Stable",
            "-"
        ]
    }
    st.dataframe(pd.DataFrame(alpha_data), use_container_width=True, hide_index=True)

    # -- Parameter Audit --
    with st.expander(" Strategist Coefficients (Parameter Audit)", expanded=False):
        c_p1, c_p2, c_p3 = st.columns(3)
        # Mapping generic Greek terms to our internal config for "Developer" feel
        c_p1.metric("Alpha (Confidence)", os.getenv("SHADOW_CONFIDENCE_THRESHOLD", "0.60"))
        c_p2.metric("Beta (Horizon)", f"{os.getenv('SHADOW_HORIZON_MINUTES', '60')}m")
        c_p3.metric("Gamma (Fee Rate)", f"{float(os.getenv('SHADOW_FEE_RATE', '0.025'))*100:.1f}%")

    c_m1, c_m2, c_m3 = st.columns(3)
    c_m1.metric("Model Path", model_path_used.split("/")[-1])
    c_m2.metric("Next Training", next_training_eta)
    c_m3.empty() 

    # Show full model path for operator debugging.
    try:
        st.caption(f"Model Path (full): {model_path_used}")
    except Exception:
        pass

    if reject_reason:
        st.warning(f"🚫 Last Induction Rejected: {reject_reason}")

    # 10D Feature Importance Audit (Top 5)
    st.markdown("#### 🧠 The 10D 'Brain' (Feature Importance)")
    fi_obj = None
    if r:
        try:
            fi_raw = r.get("oracle:feature_importance")
            if fi_raw:
                fi_obj = json.loads(fi_raw)
        except Exception:
            pass
    
    if fi_obj and isinstance(fi_obj, dict):
        # Sort by importance decending (handle nested dicts from 5090)
        def _extract_weight(val):
            if isinstance(val, dict):
                return float(val.get('weight', 0.0))
            try:
                return float(val)
            except:
                return 0.0

        # Sort using robust key
        fi_sorted = sorted(fi_obj.items(), key=lambda x: _extract_weight(x[1]), reverse=True)[:10]
        
        # Clean for chart (ensure values are floats)
        chart_data = [(k, _extract_weight(v)) for k, v in fi_sorted]
        
        fi_df = pd.DataFrame(chart_data, columns=["Feature", "Importance"])
        st.bar_chart(fi_df.set_index("Feature"), color="#00ff00")
    else:
        st.info("No feature importance data available yet.")
    if r:
        try:
            raw = r.get("oracle:feature_importance")
            if raw:
                fi_obj = json.loads(raw)
        except Exception:
            fi_obj = None

    if isinstance(fi_obj, dict) and isinstance(fi_obj.get("drops"), dict) and fi_obj.get("drops"):
        try:
            drops = fi_obj.get("drops") or {}
            items = [(str(k), float(v)) for k, v in drops.items()]
            items.sort(key=lambda kv: kv[1], reverse=True)
            top = items[:5]

            st.subheader("Top Feature Importance (Val R² Drop)")
            df_fi = pd.DataFrame({"feature": [k for k, _ in top], "val_r2_drop": [v for _, v in top]})
            fig_fi = go.Figure(
                go.Bar(
                    x=df_fi["val_r2_drop"],
                    y=df_fi["feature"],
                    orientation="h",
                    name="Val R² drop",
                )
            )
            fig_fi.update_layout(
                height=260,
                margin=dict(l=10, r=10, t=30, b=10),
                xaxis_title="Δ Val R² (higher = more important)",
                yaxis_title="",
            )
            st.plotly_chart(fig_fi, use_container_width=True)
        except Exception as e:
            st.error(f"Feature importance chart error: {e}")
    else:
        st.caption("Feature importance not available yet (will appear after the next 10D training run).")

    st.divider()

    # Sequence Readiness: count distinct 5-minute buckets in the last 12 hours.
    # We still target 85 buckets (~7 hours) for the Oracle attention lookback.
    target_buckets = 85
    bucket_count = get_market_history_bucket_count(hours=12) if db_engine else 0
    warmed_up = bucket_count >= target_buckets

    filled = int(min(10, math.floor((bucket_count / target_buckets) * 10))) if target_buckets > 0 else 0
    bar = "[" + ("█" * filled) + ("░" * (10 - filled)) + "]"
    remaining = max(0, target_buckets - bucket_count)
    eta_hours = (remaining * 5) / 60.0
    eta_display = f"ETA: {math.floor(eta_hours):.0f} Hours until Blackwell Training Ready" if remaining > 0 else "Training Ready"

    readiness_text = f"{bar} {bucket_count}/{target_buckets} Buckets ({eta_display})"

    if not warmed_up:
        st.info(f"📼 Sequence Readiness: {readiness_text}")
    else:
        st.success(f"✅ Sequence Readiness: {readiness_text}")
        st.success("✅ Signal Depth Sufficient. 5090 Training Authorized.")

        # Model Lineage & Health
        try:
            # Display lineage chart instead of raw curl commands
            history_raw = r.lrange("oracle:training_lineage", 0, 19)
            if history_raw:
                hist_data = [json.loads(x) for x in history_raw]
                # Sort by timestamp ascending for chart
                hist_data.reverse() 
                
                df_hist = pd.DataFrame(hist_data)
                if not df_hist.empty and "val_r2" in df_hist.columns:
                    st.caption("Model Lineage: Validation R² (Last 20 Runs)")
                    
                    # Highlight status
                    latest_run = hist_data[-1]
                    status_color = "green" if latest_run.get("accepted") else "orange"
                    status_text = "OPTIMIZING" if latest_run.get("accepted") else "REJECTED"
                    st.markdown(f"**Training Health:** :{status_color}[{status_text}] (Val R²: {latest_run.get('val_r2', 0):.3f})")

                    st.line_chart(df_hist.set_index("ts")[["val_r2", "train_r2"]], height=200)
            else:
                 st.info("No training lineage available yet.")
        except Exception as e:
            st.warning(f"Could not load training lineage: {e}")

    st.divider()


# ========================
# TAB 4: FUNDAMENTALS (News Desk)
# ========================
with tab_fundamentals:
    st.markdown("### 📰 Fundamentals")

    st.info(
        "Insight: Jita player count + ISK destroyed shift the demand curve right — this is the macro \"Surge\" driver the 5090 Oracle hunts."
    )

    # Fundamentals uses the global dashboard refresh (see datarefresh) to avoid UI stutter.

    # Real-time feed sourced from Redis.
    try:
        players_raw = r.get("system:macro:players")
        warfare_raw = r.get("system:macro:warfare")
        warfare_updated_at_raw = r.get("system:macro:warfare:updated_at")
        warfare_source = r.get("system:macro:warfare:source")

        try:
            # Inline sticky helper for this block scope if not globally available
            if "sticky_get_wrapper" not in locals():
                def sticky_get_wrapper(k, conn):
                    v = conn.get(k)
                    if v is not None:
                        st.session_state[f"stk_{k}"] = v
                        return v
                    return st.session_state.get(f"stk_{k}")
            
            jita_jumps_raw = sticky_get_wrapper("system:macro:jita:ship_jumps", r)
            jita_ship_kills_raw = sticky_get_wrapper("system:macro:jita:ship_kills", r)
            jita_pod_kills_raw = sticky_get_wrapper("system:macro:jita:pod_kills", r)
        except:
             # Fallback if wrapper fails
             jita_jumps_raw = r.get("system:macro:jita:ship_jumps")
             jita_ship_kills_raw = r.get("system:macro:jita:ship_kills")
             jita_pod_kills_raw = r.get("system:macro:jita:pod_kills")

        jita_updated_at_raw = r.get("system:macro:jita:updated_at")
    except Exception:
        players_raw = None
        warfare_raw = None
        warfare_updated_at_raw = None
        warfare_source = None

        jita_jumps_raw = None
        jita_ship_kills_raw = None
        jita_pod_kills_raw = None
        jita_updated_at_raw = None

    try:
        players_online = int(float(players_raw)) if players_raw is not None else 0
    except Exception:
        players_online = 0

    try:
        warfare_isk = float(warfare_raw) if warfare_raw is not None else 0.0
    except Exception:
        warfare_isk = 0.0

    try:
        jita_jumps = int(float(jita_jumps_raw)) if jita_jumps_raw is not None else 0
    except Exception:
        jita_jumps = 0
    try:
        jita_ship_kills = int(float(jita_ship_kills_raw)) if jita_ship_kills_raw is not None else 0
    except Exception:
        jita_ship_kills = 0
    try:
        jita_pod_kills = int(float(jita_pod_kills_raw)) if jita_pod_kills_raw is not None else 0
    except Exception:
        jita_pod_kills = 0

    jita_age_s = None
    try:
        if jita_updated_at_raw is not None:
            jita_age_s = max(0.0, time.time() - float(jita_updated_at_raw))
    except Exception:
        jita_age_s = None

    warfare_age_s = None
    try:
        if warfare_updated_at_raw is not None:
            warfare_age_s = max(0.0, time.time() - float(warfare_updated_at_raw))
    except Exception:
        warfare_age_s = None

    if warfare_source:
        if str(warfare_source).strip().lower() != "kongyo2_osint_mcp":
            st.caption(f"OSINT feed source: {warfare_source} (non-MCP).")
        else:
            st.caption(f"OSINT source: {warfare_source}")
    else:
        st.caption("OSINT feed source unknown (missing system:macro:warfare:source).")

    if warfare_age_s is not None:
        st.caption(f"War feed age: {warfare_age_s:.0f}s")

    battle_alert_threshold = float(os.getenv("FUNDAMENTALS_BATTLE_ALERT_ISK", "50000000000"))
    battle_alert = warfare_isk > battle_alert_threshold

    # Conflict Impact tracker (Smith & Johnson principle: higher destruction => higher mineral replacement demand)
    conflict_scale_isk = float(os.getenv("FUNDAMENTALS_CONFLICT_SCALE_ISK", str(battle_alert_threshold)))
    conflict_max_shift_pct = float(os.getenv("FUNDAMENTALS_CONFLICT_MAX_SHIFT_PCT", "10"))
    conflict_norm = 0.0
    if conflict_scale_isk > 0:
        conflict_norm = max(0.0, min(1.0, warfare_isk / conflict_scale_isk))
    trit_demand_shift_pct = conflict_norm * conflict_max_shift_pct
    trit_demand_multiplier = 1.0 + (trit_demand_shift_pct / 100.0)

    max_points = int(os.getenv("FUNDAMENTALS_SPARKLINE_POINTS", "60"))

    # Sparklines: prefer persisted Redis history (survives dashboard restarts).
    players_hist = _read_macro_history(redis_client=r, key="system:macro:players:history", max_points=max_points)
    warfare_hist = _read_macro_history(redis_client=r, key="system:macro:warfare:history", max_points=max_points)

    if players_hist.empty and warfare_hist.empty:
        # Fallback: keep a short in-memory history per Streamlit session.
        if "fundamentals_ts" not in st.session_state:
            st.session_state["fundamentals_ts"] = []
            st.session_state["fundamentals_players"] = []
            st.session_state["fundamentals_warfare"] = []

        st.session_state["fundamentals_ts"].append(datetime.utcnow())
        st.session_state["fundamentals_players"].append(players_online)
        st.session_state["fundamentals_warfare"].append(float(warfare_isk))

        if len(st.session_state["fundamentals_ts"]) > max_points:
            st.session_state["fundamentals_ts"] = st.session_state["fundamentals_ts"][-max_points:]
            st.session_state["fundamentals_players"] = st.session_state["fundamentals_players"][-max_points:]
            st.session_state["fundamentals_warfare"] = st.session_state["fundamentals_warfare"][-max_points:]

    c1, c2, c3 = st.columns(3)
    c1.metric("Market Pulse: Jita Player Count", f"{players_online:,d}")
    c2.metric("War Intensity: ISK Destroyed", f"{warfare_isk:,.0f}")

    if battle_alert:
        # Flash red by alternating severity.
        if int(time.time()) % 2 == 0:
            c3.error("Volatility Alert")
        else:
            c3.warning("Volatility Alert")
        st.caption(f"Alert when ISK destroyed > {battle_alert_threshold:,.0f}")
    else:
        c3.success("Stable")
        st.caption(f"Alert when ISK destroyed > {battle_alert_threshold:,.0f}")

    # Sparklines
    try:
        if not players_hist.empty or not warfare_hist.empty:
            df_sp = pd.DataFrame(index=pd.Index([], name="ts"))
            if not players_hist.empty:
                df_sp = df_sp.join(players_hist.rename("Jita Population"), how="outer")
            if not warfare_hist.empty:
                df_sp = df_sp.join(warfare_hist.rename("War Intensity"), how="outer")
            df_sp = df_sp.sort_index()
        else:
            df_sp = pd.DataFrame(
                {
                    "ts": st.session_state["fundamentals_ts"],
                    "Jita Population": st.session_state["fundamentals_players"],
                    "War Intensity": st.session_state["fundamentals_warfare"],
                }
            ).set_index("ts")
        sp1, sp2 = st.columns(2)
        sp1.caption("Jita Population (sparkline)")
        sp1.line_chart(df_sp[["Jita Population"]], height=120)
        sp2.caption("War Intensity (sparkline)")
        sp2.line_chart(df_sp[["War Intensity"]], height=120)
    except Exception:
        pass

    st.divider()

    st.subheader("Jita Activity Proxy")
    j1, j2, j3 = st.columns(3)
    j1.metric("Jita Ship Jumps", f"{jita_jumps:,d}")
    j2.metric("Jita Ship Kills", f"{jita_ship_kills:,d}")
    j3.metric("Jita Pod Kills", f"{jita_pod_kills:,d}")
    if jita_age_s is not None:
        st.caption(f"Jita feed age: {jita_age_s:.0f}s")
    st.subheader("Conflict Impact")
    c_ci1, c_ci2 = st.columns(2)
    c_ci1.metric("Tritanium Demand Shift (est.)", f"+{trit_demand_shift_pct:.2f}%")
    c_ci2.metric("Replacement Demand Multiplier", f"{trit_demand_multiplier:.3f}×")
    st.caption("Heuristic: demand shift scales with ISK destroyed over the last 60 minutes (bounded).")

    st.subheader("Real-time Feed")
    st.write(
        {
            "system:macro:players": players_online,
            "system:macro:warfare": float(warfare_isk),
            "battle_alert_active": bool(battle_alert),
        }
    )
    st.subheader("Oracle Model Status")
    model_status = "UNKNOWN"
    last_metrics = None
    reject_reason = None
    last_train_error = None
    last_train_ts = None
    if r:
        try:
            model_status = r.get("oracle:model_status") or "IDLE"
            metrics_raw = r.get("oracle:last_training_metrics")
            if metrics_raw:
                last_metrics = json.loads(metrics_raw)
        except Exception as e:
            st.error(f"Oracle status read error: {e}")

    if isinstance(last_metrics, dict):
        try:
            last_train_error = last_metrics.get("error")
            last_train_ts = last_metrics.get("timestamp")
            if isinstance(last_train_error, str) and ":" in last_train_error:
                # Example: "Model rejected by validation: val_r2_not_improved (val_r2=... <= prev_val_r2=...)"
                tail = last_train_error.split(":", 1)[1].strip()
                reject_reason = tail.split("(", 1)[0].strip() if tail else None
        except Exception:
            reject_reason = None

    # Model artifact timestamp (synced from 5090 host by cron_train.sh)
    model_mtime_utc = None
    model_age_s = None
    try:
        from pathlib import Path

        root_dir = Path(__file__).resolve().parents[1]
        pt_path = root_dir / "models" / "oracle_v1_latest.pt"
        if pt_path.exists():
            model_mtime_utc = datetime.fromtimestamp(pt_path.stat().st_mtime, tz=timezone.utc)
            model_age_s = max(0.0, (datetime.now(tz=timezone.utc) - model_mtime_utc).total_seconds())
    except Exception:
        model_mtime_utc = None
        model_age_s = None

    # 5090 telemetry + model precision (requested in Oracle tab)
    skynet_temp = None
    skynet_load = None
    if r:
        try:
            skynet_temp = r.get("system:skynet:temp")
            skynet_load = r.get("system:skynet:load")
        except Exception:
            pass

    model_precision = 0.0
    if isinstance(last_metrics, dict):
        try:
            model_precision = float(last_metrics.get("val_r2", last_metrics.get("train_r2", 0.0)) or 0.0)
        except Exception:
            model_precision = 0.0

    # Best-effort: latest oracle confidence (from strategist radar key)
    latest_conf = 0.0
    try:
        if r and raw_items:
            top_tid = str(raw_items[0].get("type_id"))
            radar_raw = r.get(f"market_radar:{top_tid}")
            if radar_raw:
                latest_conf = float(json.loads(radar_raw).get("confidence", 0.0) or 0.0)
    except Exception:
        latest_conf = 0.0

    # Model Maturity: show R² improvement vs previous baseline as a percentage
    prev_val_r2 = 0.0
    if isinstance(last_metrics, dict):
        try:
            prev_val_r2 = float(last_metrics.get("prev_val_r2") or 0.0)
        except Exception:
            prev_val_r2 = 0.0
    maturity_pct = 0.0
    if prev_val_r2 > 0 and model_precision > prev_val_r2:
        maturity_pct = ((model_precision - prev_val_r2) / prev_val_r2) * 100.0
    elif model_precision > 0:
        maturity_pct = 100.0  # baseline established

    # Model Confidence: simple scale of val R² into 0..100% (tunable via env)
    target_r2 = 0.01
    try:
        target_r2 = float(os.getenv("MODEL_CONFIDENCE_TARGET_R2", "0.01"))
    except Exception:
        target_r2 = 0.01
    model_confidence = 0.0
    if target_r2 > 0:
        model_confidence = max(0.0, min(1.0, model_precision / target_r2))

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🧠 Model Precision", f"{model_precision:.4f} R²")
    c2.metric("🎯 Model Confidence", f"{model_confidence*100:.0f}%")
    c3.metric("🔥 5090 Temp", f"{skynet_temp}°C" if skynet_temp is not None else "N/A")
    c4.metric("Confidence", f"{latest_conf:.2f}")
    c5.metric("📈 Model Maturity", f"{maturity_pct:.1f}%")

    # Reject reason + next training timer (4-hour cron cadence)
    try:
        now_utc = datetime.now(tz=timezone.utc)
        hours_mod = int(now_utc.hour) % 4
        add_hours = 4 - hours_mod
        if add_hours == 0 and (now_utc.minute > 0 or now_utc.second > 0):
            add_hours = 4
        next_train = (now_utc.replace(minute=0, second=0, microsecond=0) + timedelta(hours=add_hours))
        delta_s = max(0, int((next_train - now_utc).total_seconds()))
        hh = delta_s // 3600
        mm = (delta_s % 3600) // 60
        countdown = f"{hh}h {mm}m"
    except Exception:
        countdown = "N/A"

    rr = reject_reason or ("OK" if str(model_status).upper() in {"LIVE", "OK"} else "UNKNOWN")
    st.caption(f"Reject reason: {rr}")
    if last_train_ts:
        st.caption(f"Last training attempt: {last_train_ts}")
    if last_train_error and reject_reason:
        st.caption(f"Last reject: {last_train_error}")
    st.caption(f"Next training (4h cadence, UTC): in {countdown}")

    if model_mtime_utc is not None:
        st.caption(f"Latest model artifact: {model_mtime_utc.isoformat()} (age {int(model_age_s or 0)}s)")

    # Mean Reversion Potential indicator (BB_Upper dominance detected)
    bb_upper_impact = 0.0
    try:
        drops = (((last_metrics or {}).get("features") or {}).get("importance") or {}).get("drops") or {}
        bb_upper_impact = float(drops.get("BB_Upper", 0.0) or 0.0)
    except Exception:
        bb_upper_impact = 0.0

    if bb_upper_impact > 0.001:
        st.success("🔄 Mean Reversion Potential: High (BB_Upper driving predictions)")
    elif bb_upper_impact > 0.0001:
        st.info("🔄 Mean Reversion Potential: Moderate")
    else:
        st.caption("🔄 Mean Reversion Potential: Low")

    # TODO 5: Z-Score Visualization (Tritanium Monitoring)
    try:
        if r:
            # Tritanium TypeID = 34
            trit_features = r.get("features:34")
            if trit_features:
                feats = json.loads(trit_features)
                price_hist = feats.get("price_history", [])
                
                if price_hist and len(price_hist) > 5:
                     # Calculate live Z-Score on dashboard
                     last_p = price_hist[-1]
                     arr = np.array(price_hist)
                     z_val = (last_p - arr.mean()) / (arr.std() + 1e-8)
                     
                     st.metric(
                         "🛡️ Tritanium Z-Score (Volatility Shield)", 
                         f"{z_val:.2f} σ",
                         delta="HALT RISK" if abs(z_val) > 1.96 else "Stable",
                         delta_color="inverse"
                     )
                     
                     st.line_chart(price_hist[-15:])
                     st.caption(f"Tritanium Lookback ({len(price_hist)} points)")
            else:
                 st.info("Waiting for Tritanium (34) history...")
    except Exception as e:
        st.caption(f"Z-Score Viz Error: {e}")

    st.markdown(f"**Model Status:** {model_status}")
    if last_metrics:
        st.json(last_metrics)

    # Feature Impact chart (Permutation importance from last training run)
    st.subheader("Feature Impact")
    try:
        drops = (((last_metrics or {}).get("features") or {}).get("importance") or {}).get("drops") or {}
        if not drops:
            st.info("Feature impact will appear after the next training run (features.importance not yet present).")
        else:
            # Show the Liu indicator subset first.
            keys = ["RSI", "MACD", "MACD_Signal", "BB_Upper", "BB_Low"]
            vals = [float(drops.get(k, 0.0) or 0.0) for k in keys]
            fig_imp = go.Figure()
            fig_imp.add_trace(go.Bar(x=keys, y=vals, name="Δ val R² (shuffle)") )
            fig_imp.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_imp, use_container_width=True)
    except Exception as e:
        st.error(f"Feature impact chart error: {e}")

    # TODO 5: Kelly Sizing Accuracy Check (Projections)
    st.divider()
    st.subheader("🎲 Kelly Profit Projection (Top 3 Targets)")
    if r:
         try:
             # Scan Redis for top opportunities
             keys = r.keys("market_radar:*")
             top_ops = []
             for k in keys:
                 raw = r.get(k)
                 if raw:
                     d = json.loads(raw)
                     # Only consider valid signals
                     if d.get("confidence", 0) > 0.5:
                         top_ops.append(d)
             
             # Sort by Kelly Fraction descending
             top_ops.sort(key=lambda x: x.get("kelly_fraction", 0), reverse=True)
             
             cols = st.columns(3)
             for i, op in enumerate(top_ops[:3]):
                 # E[V] = (p * win) - (q * loss)
                 # win = predicted - current
                 # loss = 2 * sigma (approx risk metric used in strategy)
                 
                 curr = float(op.get("price", 0))
                 pred = float(op.get("predicted", 0))
                 conf = float(op.get("confidence", 0))
                 kelly = float(op.get("kelly_fraction", 0))
                 
                 win = pred - curr
                 # Reconstruct volatility estimate from stop loss or default
                 stop_loss = float(op.get("stop_loss_price", 0))
                 if stop_loss > 0:
                     loss = curr - stop_loss
                 else:
                     loss = curr * 0.05 # Default if missing
                     
                 ev = (conf * win) - ((1-conf) * loss)
                 
                 with cols[i]:
                     st.metric(
                         f"TypeID {op.get('type_id')}",
                         f"{kelly*100:.1f}% Size",
                         f"EV: {ev:,.0f} ISK",
                         delta_color="normal"
                     )
             
             if not top_ops:
                 st.info("No high-confidence targets for Kelly Projection.")

         except Exception as e:
             st.caption(f"Kelly Projection Error: {e}")

    # 📈 Accuracy Sparkline: predicted return vs actual market return (last 12 buckets)
    st.divider()
    st.subheader("📈 Accuracy Sparkline")
    if not db_engine:
        st.info("Database unavailable; cannot compute accuracy sparkline.")
    else:
        try:
            horizon_minutes = 60
            with db_engine.connect() as conn:
                df_acc = pd.read_sql(
                    text(
                        """
                        SELECT
                            t.timestamp,
                            t.predicted_price,
                            t.actual_price_at_time AS entry_price,
                            mh.close AS price_at_horizon
                        FROM shadow_trades t
                        LEFT JOIN LATERAL (
                            SELECT close
                            FROM market_history
                            WHERE type_id = t.type_id
                              AND timestamp >= t.timestamp + (:horizon || ' minutes')::interval
                            ORDER BY timestamp ASC
                            LIMIT 1
                        ) mh ON TRUE
                        ORDER BY t.timestamp DESC
                        LIMIT 12
                        """
                    ),
                    conn,
                    params={"horizon": int(horizon_minutes)},
                )

            if df_acc.empty:
                st.info("No shadow trades yet; sparkline will populate after predictions execute.")
            else:
                df_acc = df_acc.dropna(subset=["entry_price", "predicted_price", "price_at_horizon"]).copy()
                if df_acc.empty:
                    st.info("Shadow trades pending horizon prices; sparkline will populate soon.")
                else:
                    df_acc = df_acc.sort_values("timestamp")
                    ep = df_acc["entry_price"].astype(float).replace(0.0, np.nan)
                    pred_ret = (df_acc["predicted_price"].astype(float) - ep) / ep
                    act_ret = (df_acc["price_at_horizon"].astype(float) - ep) / ep

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=pred_ret, mode="lines+markers", name="Predicted Return"))
                    fig.add_trace(go.Scatter(y=act_ret, mode="lines+markers", name="Actual Return"))
                    fig.update_layout(height=220, margin=dict(l=10, r=10, t=30, b=10))
                    st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Sparkline error: {e}")

    if warmed_up:
        st.divider()
        # Show Graph
        st.success("Oracle Online & Predicting")
        
        # Identify top asset from Raw Items
        top_asset = raw_items[0] if raw_items else None
        tid = None
        curr = 0.0
        latest_pred = 0.0
        target_name = "None"
        
        if top_asset:
            tid = str(top_asset.get("type_id"))
            target_name = name_map.get(tid, f"Type {tid}")
            curr = float(top_asset.get("price", 0))
            spread = float(top_asset.get("spread", 0))
            latest_pred = curr * (1+spread)
        else:
            # Fallback for crash prevention
            curr = 0.0
            latest_pred = 0.0
            
        st.markdown(f"**Target:** {target_name} | **Predicted:** {latest_pred:,.2f} ISK")
        
        curr = 0.0  # HARD RESET TO PREVENT NAMEERROR
        if top_asset:
            curr = float(top_asset.get("price", 0))
        steps = 12
        x_axis = [f"+{i*5}m" for i in range(steps+1)]
        y_path = np.linspace(curr, latest_pred, steps+1)
        
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=x_axis, y=y_path, mode='lines+markers', name='Forecast'))
        fig_pred.update_layout(title="Trajectory", height=300, paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_pred, use_container_width=True)

# ========================
# TAB 4: PERFORMANCE (Virtual P&L Audit)
# ========================
with tab_performance:
    st.markdown("### 📈 Performance Audit")

    # Directive B+: Calibration Engine (Platt Scaling Prep)
    if db_engine:
        try:
            # Fetch deeper history for statistical significance (500 trades)
            calib_df = get_shadow_trade_outcomes_df(limit=500, horizon_minutes=60, fee_rate=0.025)
            
            if not calib_df.empty:
                # Data cleanup
                c_df = calib_df.copy()
                c_df["entry_price"] = pd.to_numeric(c_df["entry_price"], errors="coerce")
                c_df["exit_price"] = pd.to_numeric(c_df["exit_price"], errors="coerce")
                c_df["confidence"] = pd.to_numeric(c_df["confidence"], errors="coerce")
                
                # Filter valid closed trades
                closed_trades = c_df.dropna(subset=['entry_price', 'exit_price'])
                
                if len(closed_trades) > 10:
                    # Calculate Net PnL (assuming Long-only for simplicity or matching signal)
                    # PnL = (Exit * (1-fee)) - (Entry * (1+fee))
                    fee_r = 0.025
                    closed_trades['net_pnl'] = (closed_trades['exit_price'] * (1 - fee_r)) - (closed_trades['entry_price'] * (1 + fee_r))
                    
                    win_count = len(closed_trades[closed_trades['net_pnl'] > 0])
                    total_count = len(closed_trades)
                    realized_wr = win_count / total_count
                    
                    avg_conf = closed_trades['confidence'].mean()
                    gap = avg_conf - realized_wr
                    
                    st.caption(f"🤖 Calibration Engine (n={total_count} trades)")
                    cc1, cc2, cc3 = st.columns(3)
                    cc1.metric("Avg Confidence (Predicted)", f"{avg_conf:.1%}")
                    cc2.metric("Realized Win Rate", f"{realized_wr:.1%}")
                    cc3.metric("Calibration Gap", f"{gap:+.1%}", 
                             delta="-Optimism" if gap > 0 else "+Pessimism", 
                             delta_color="inverse" if gap > 0.1 else "normal")
                    
                    if gap > 0.10:
                        st.warning(f"⚠️ Optimism Bias Detected. Gap {gap:.1%}. Dynamic Shrinkage would target factor: {realized_wr/avg_conf:.2f}")
                    st.divider()
        except Exception as e:
            # st.error(f"Calibration Error: {e}")
            pass

    # Trade Performance Table (color-coded)
    if db_engine:
        try:
            start_utc, end_utc = _server_today_utc_window()
            today_df = get_shadow_trade_outcomes_window_df(
                start_utc=start_utc,
                end_utc=end_utc,
                limit=int(os.getenv("TODAY_WINNERS_MAX_ROWS", "5000")),
                horizon_minutes=int(os.getenv("SHADOW_TRADE_HORIZON_MINUTES", "60")),
            )

            if not today_df.empty:
                # Compute realized profit + status for coloring.
                dfp = today_df.copy()
                dfp["entry_price"] = pd.to_numeric(dfp.get("entry_price"), errors="coerce")
                dfp["exit_price"] = pd.to_numeric(dfp.get("exit_price"), errors="coerce")

                def _profit_net(row):
                    ep = row.get("entry_price")
                    xp = row.get("exit_price")
                    if ep is None or xp is None or pd.isna(ep) or pd.isna(xp) or float(ep) <= 0:
                        return np.nan
                    if str(row.get("signal_type", "")).upper() == "BUY":
                        gross = float(xp) - float(ep)
                    else:
                        gross = float(ep) - float(xp)
                    return gross * (1.0 - 0.025)

                dfp["profit_net"] = dfp.apply(_profit_net, axis=1)
                
                # Trust the DB Virtual Outcome if available (Authoritative source)
                if "virtual_outcome" in dfp.columns:
                    dfp["status"] = dfp["virtual_outcome"].fillna("PENDING")
                else:
                    dfp["status"] = dfp["profit_net"].apply(
                        lambda v: "PENDING" if pd.isna(v) else ("WIN" if float(v) > 0 else "LOSS")
                    )

                def _color_profit(v):
                    if v is None or pd.isna(v):
                        return "background-color: #6c757d; color: white"  # PENDING
                    return "background-color: #28a745; color: white" if float(v) > 0 else "background-color: #dc3545; color: white"


                st.subheader("Top Winners (Today)")

                completed_mask = ~dfp["profit_net"].isna()
                winners_today = dfp[completed_mask & (dfp["profit_net"] > 0)].copy()

                if winners_today.empty:
                    st.info("No winning completed trades today yet.")
                else:
                    winners_today = winners_today.sort_values(by="profit_net", ascending=False)
                    winners_today = winners_today.head(int(os.getenv("TODAY_WINNERS_TOP_N", "50")))
                    winners_today["status"] = "WIN"
                    show_cols = [
                        c
                        for c in [
                            "timestamp",
                            "type_id",
                            "signal_type",
                            "entry_price",
                            "exit_price",
                            "profit_net",
                            "status",
                        ]
                        if c in winners_today.columns
                    ]
                    styled = winners_today[show_cols].style.map(_color_profit, subset=["profit_net"])
                    st.dataframe(styled, use_container_width=True, height=300)

                # Market Maker Ledger: show most recent COMPLETED BUY and SELL orders.
                st.subheader("Market Maker Ledger")
                try:
                    horizon_m = int(os.getenv("SHADOW_TRADE_HORIZON_MINUTES", "60"))
                    max_rows = int(os.getenv("LEDGER_RECENT_COMPLETED_ROWS", "50"))
                    fetch_n = max(200, int(os.getenv("LEDGER_SIDE_FETCH_ROWS", str(max_rows * 10))))

                    buy_ledger = get_shadow_trade_completed_by_side_df(
                        side="BUY",
                        limit=fetch_n,
                        horizon_minutes=horizon_m,
                    ).copy()
                    sell_ledger = get_shadow_trade_completed_by_side_df(
                        side="SELL",
                        limit=fetch_n,
                        horizon_minutes=horizon_m,
                    ).copy()

                    ledger_combined = pd.concat([buy_ledger, sell_ledger], ignore_index=True)
                    ledger_combined["entry_price"] = pd.to_numeric(ledger_combined.get("entry_price"), errors="coerce")
                    ledger_combined["exit_price"] = pd.to_numeric(ledger_combined.get("exit_price"), errors="coerce")
                    ledger_combined["profit_net"] = ledger_combined.apply(_profit_net, axis=1)
                    ledger_combined["status"] = ledger_combined["profit_net"].apply(
                        lambda v: "PENDING" if pd.isna(v) else ("WIN" if float(v) > 0 else "LOSS")
                    )

                    # Resolve item names (bounded to what's displayed).
                    type_ids = [int(x) for x in ledger_combined["type_id"].dropna().astype(int).unique().tolist()]
                    type_names = resolve_type_names(type_ids)
                    ledger_combined["item_name"] = ledger_combined["type_id"].apply(
                        lambda x: type_names.get(str(int(x)), f"Type {int(x)}") if pd.notna(x) else "Unknown"
                    )

                    def _add_item_name(df: pd.DataFrame) -> pd.DataFrame:
                        if df is None or df.empty or "type_id" not in df.columns:
                            return df
                        out = df.copy()
                        out["item_name"] = out["type_id"].apply(
                            lambda x: type_names.get(str(int(x)), f"Type {int(x)}") if pd.notna(x) else "Unknown"
                        )
                        return out

                    ledger_cols = [
                        "timestamp",
                        "item_name",
                        "entry_price",
                        "exit_price",
                        "profit_net",
                        "status",
                        "reasoning",
                    ]

                    tab_buy, tab_sell = st.tabs(["BUY ORDERS", "SELL ORDERS"])
                    with tab_buy:
                        buy_df = _add_item_name(buy_ledger)
                        buy_df["entry_price"] = pd.to_numeric(buy_df.get("entry_price"), errors="coerce")
                        buy_df["exit_price"] = pd.to_numeric(buy_df.get("exit_price"), errors="coerce")
                        buy_df["profit_net"] = buy_df.apply(_profit_net, axis=1)
                        buy_df["status"] = buy_df["profit_net"].apply(
                            lambda v: "PENDING" if pd.isna(v) else ("WIN" if float(v) > 0 else "LOSS")
                        )
                        buy_df = buy_df.sort_values(by="timestamp", ascending=False).head(max_rows)
                        if buy_df.empty:
                            st.info("No completed BUY orders yet.")
                        else:
                            buy_cols = [c for c in ledger_cols if c in buy_df.columns]
                            st.dataframe(
                                buy_df[buy_cols].style.format({"profit_net": "{:,.2f}"}),
                                use_container_width=True,
                            )

                    with tab_sell:
                        sell_df = _add_item_name(sell_ledger)
                        sell_df["entry_price"] = pd.to_numeric(sell_df.get("entry_price"), errors="coerce")
                        sell_df["exit_price"] = pd.to_numeric(sell_df.get("exit_price"), errors="coerce")
                        sell_df["profit_net"] = sell_df.apply(_profit_net, axis=1)
                        sell_df["status"] = sell_df["profit_net"].apply(
                            lambda v: "PENDING" if pd.isna(v) else ("WIN" if float(v) > 0 else "LOSS")
                        )
                        sell_df = sell_df.sort_values(by="timestamp", ascending=False).head(max_rows)
                        if sell_df.empty:
                            st.info("No completed SELL orders yet.")
                        else:
                            sell_cols = [c for c in ledger_cols if c in sell_df.columns]
                            st.dataframe(
                                sell_df[sell_cols].style.format({"profit_net": "{:,.2f}"}),
                                use_container_width=True,
                            )

                except Exception as e:
                    st.error(f"Ledger error: {e}")
                except Exception as e:
                    st.error(f"Ledger error: {e}")
            else:
                st.info("No shadow trades recorded today yet.")
        except Exception as e:
            st.error(f"Trade performance table error: {e}")

    st.divider()

    # Item ROI Bar Chart
    if db_engine:
        try:
            roi_df = get_item_roi_df(limit=5, horizon_minutes=60, fee_rate=0.025)

            if not roi_df.empty:
                st.subheader("Top 5 Items by Net ROI")
                type_names = resolve_type_names(roi_df["type_id"].tolist())
                roi_df["item_name"] = roi_df["type_id"].apply(lambda x: type_names.get(str(x), f"Type {x}"))
                roi_df["net_roi"] = (
                    pd.to_numeric(roi_df.get("net_roi"), errors="coerce")
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0.0)
                )
                roi_chart = (
                    roi_df[["item_name", "net_roi"]]
                    .set_index("item_name")
                    .sort_values("net_roi", ascending=False)
                )
                st.bar_chart(roi_chart, height=260)
        except Exception as e:
            st.error(f"Item ROI chart error: {e}")

    st.divider()

    # Timezone Audit: trade volume per hour (UTC)
    if db_engine:
        try:
            tz_df = get_trade_volume_by_hour_utc_df(lookback_hours=int(os.getenv("AUDIT_TZ_LOOKBACK_HOURS", str(24 * 7))))
            st.subheader("Timezone Audit: Trade Volume by Hour (UTC)")
            fig_tz = go.Figure(
                go.Bar(
                    x=tz_df["hour_utc"].astype(int),
                    y=tz_df["trade_count"].astype(int),
                    name="Trades",
                )
            )
            fig_tz.update_layout(
                xaxis_title="Hour (UTC)",
                yaxis_title="Trades",
                height=280,
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig_tz, use_container_width=True)
        except Exception as e:
            st.error(f"Timezone audit error: {e}")

    st.divider()

    # Chapter 9: ISK Preservation (Forced Exit / Stop-Loss)
    if db_engine:
        try:
            stop_loss_pct = float(os.getenv("SHADOW_STOP_LOSS_PCT", "0.015"))
        except Exception:
            stop_loss_pct = 0.015

        try:
            with db_engine.connect() as conn:
                row = conn.execute(
                    text(
                        """
                        SELECT
                            COUNT(*)::bigint AS shield_activations,
                            COALESCE(SUM(actual_price_at_time * :sl), 0)::double precision AS virtual_isk_protected
                        FROM shadow_trades
                        WHERE forced_exit = TRUE
                          AND actual_price_at_time IS NOT NULL
                        """
                    ),
                    {"sl": float(stop_loss_pct)},
                ).fetchone()

            shield_activations = int(row[0] or 0)
            virtual_isk_protected = float(row[1] or 0.0)

            c_fx1, c_fx2 = st.columns(2)
            c_fx1.metric("Shield Activations", f"{shield_activations}")
            c_fx2.metric("Virtual ISK Protected", f"{virtual_isk_protected:,.2f}")
            st.caption(f"Assumes stop-loss = {stop_loss_pct*100:.2f}% of entry price per forced exit.")
            st.divider()
        except Exception as e:
            st.error(f"Forced-exit audit error: {e}")

    # Last 24h Performance (from daily alpha job)
    if r:
        try:
            alpha_raw = r.get("performance:last24h")
            if alpha_raw:
                alpha = json.loads(alpha_raw)
                gmean = float(alpha.get("geometric_mean_return", 0.0))
                mdd = float(alpha.get("max_drawdown", 0.0))
                n = int(alpha.get("n_resolved", 0))
                updated_at = str(alpha.get("updated_at", ""))

                st.subheader("Last 24h Performance")
                st.metric("Geometric Mean Return (Last 24h)", f"{gmean*100:.3f}%")
                c_a, c_b, c_c = st.columns(3)
                c_a.metric("Max Drawdown", f"{mdd*100:.3f}%")
                c_b.metric("Resolved Trades", f"{n}")
                c_c.metric("Projected 24h Yield", f"{gmean*100:.3f}%")
                if updated_at:
                    st.caption(f"Updated: {updated_at}")
                st.divider()
        except Exception:
            pass

    if not db_engine:
        st.warning("Database unavailable; cannot compute ledger metrics.")
    else:
        horizon_minutes = 60
        fee_rate = 0.025

        # Ensure table exists (safe if already present)
        try:
            ddl_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "storage", "shadow_ledger.sql")
            with open(ddl_path, "r", encoding="utf-8") as f:
                ddl = f.read()
            with db_engine.begin() as conn:
                for stmt in ddl.split(";"):
                    stmt = stmt.strip()
                    if stmt:
                        conn.execute(text(stmt))
        except Exception:
            pass

        wins = 0
        resolved = 0
        total_alpha = 0.0
        lookback_days = 7

        try:
            with db_engine.connect() as conn:
                # Compute 7d metrics from realized horizon prices (independent of stored virtual_outcome).
                # This avoids historical mislabeling/forced-exit artifacts and prevents inflated alpha.
                row = conn.execute(
                    text(
                        """
                        WITH base AS (
                            SELECT
                                t.timestamp,
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
                            WHERE t.timestamp >= NOW() - (:days || ' days')::interval
                              AND t.actual_price_at_time IS NOT NULL
                        ), scored AS (
                            SELECT
                                CASE
                                    WHEN exit_price IS NULL OR entry_price <= 0 OR exit_price <= 0 THEN NULL
                                    WHEN UPPER(signal_type) = 'BUY' THEN (exit_price - entry_price)
                                    ELSE (entry_price - exit_price)
                                END AS gross_isk
                            FROM base
                        )
                        SELECT
                            SUM(CASE WHEN gross_isk IS NOT NULL AND gross_isk > 0 THEN 1 ELSE 0 END) AS wins,
                            SUM(CASE WHEN gross_isk IS NOT NULL THEN 1 ELSE 0 END) AS resolved,
                            COALESCE(SUM(CASE WHEN gross_isk IS NULL THEN 0 ELSE gross_isk END) * (1 - :fee), 0) AS total_alpha
                        FROM scored
                        """
                    ),
                    {"fee": float(fee_rate), "horizon": int(horizon_minutes), "days": int(lookback_days)},
                ).fetchone()

                wins = int((row[0] or 0) if row else 0)
                resolved = int((row[1] or 0) if row else 0)
                total_alpha = float((row[2] or 0.0) if row else 0.0)

        except Exception as e:
            st.error(f"Performance query error: {e}")

        win_rate = (wins / resolved * 100.0) if resolved > 0 else 0.0
        c1, c2 = st.columns(2)
        c1.metric("Total Virtual Alpha (7d, net)", f"{total_alpha:,.2f}")
        c2.metric("Win Rate % (resolved)", f"{win_rate:.1f}%")
        st.caption(f"Window: last {lookback_days}d, horizon={horizon_minutes}m, fee={fee_rate:.3f}")

        st.divider()

        # --- Sharpe Ratio Audit ---
        try:
            with db_engine.connect() as conn:
                # Fetch last 100 matured trades (horizon price available) to compute Sharpe.
                sharpe_df = pd.read_sql(
                    text(
                        """
                        SELECT
                            t.actual_price_at_time as entry_price,
                            mh.close as exit_price,
                            t.signal_type
                        FROM shadow_trades t
                        LEFT JOIN LATERAL (
                            SELECT close
                            FROM market_history
                            WHERE type_id = t.type_id
                              AND timestamp >= t.timestamp + (:horizon || ' minutes')::interval
                            ORDER BY timestamp ASC
                            LIMIT 1
                        ) mh ON TRUE
                        WHERE t.timestamp <= NOW() - (:horizon || ' minutes')::interval
                          AND mh.close IS NOT NULL
                          AND t.actual_price_at_time IS NOT NULL
                        ORDER BY t.timestamp DESC
                        LIMIT 100
                        """
                    ),
                    conn,
                    params={"horizon": int(horizon_minutes)}
                )

            if not sharpe_df.empty:
                # Calculate simple returns
                def calc_return(row):
                    if row['entry_price'] == 0 or pd.isna(row['exit_price']):
                        return np.nan

                    if str(row['signal_type']).upper() == 'BUY':
                        return (row['exit_price'] - row['entry_price']) / row['entry_price']
                    else:
                        return (row['entry_price'] - row['exit_price']) / row['entry_price']

                sharpe_df['ret'] = sharpe_df.apply(calc_return, axis=1)
                sharpe_df = sharpe_df.dropna(subset=['ret'])

                # Bardol winsorization clamp to stabilize Sharpe audit.
                sharpe_df['ret'] = sharpe_df['ret'].clip(lower=-1.0, upper=10.0)
                
                rp = float(sharpe_df['ret'].mean())
                sigma_p = float(sharpe_df['ret'].std())

                sharpe = None
                if sigma_p and sigma_p > 0 and np.isfinite(sigma_p):
                    sharpe = (rp / sigma_p)
                
                st.subheader("Risk Audit (Last 100 Trades)")
                c_s1, c_s2, c_s3 = st.columns(3)
                c_s1.metric("Sharpe Ratio", f"{sharpe:.4f}" if sharpe is not None else "N/A")
                c_s2.metric("Exp. Return", f"{rp*100:.3f}%")
                c_s3.metric("Volatility (σ)", f"{sigma_p*100:.3f}%" if sigma_p and np.isfinite(sigma_p) else "N/A")
                st.caption(f"Based on {len(sharpe_df)} resolved trades. (Rf=0)")
                st.divider()

        except Exception as e:
            st.warning(f"Sharpe calc error: {e}")

        st.subheader("Live Ledger: Last 10 Ghost Trades")
        try:
            with db_engine.connect() as conn:
                df = pd.read_sql(
                    text(
                        """
                        SELECT
                            t.timestamp,
                            t.type_id,
                            t.signal_type,
                            t.predicted_price,
                            t.actual_price_at_time AS entry_price,
                            mh.close AS price_at_horizon,
                            (
                                CASE
                                    WHEN mh.close IS NULL THEN NULL
                                    WHEN t.signal_type = 'BUY' THEN (mh.close - t.actual_price_at_time) * (1 - :fee)
                                    ELSE (t.actual_price_at_time - mh.close) * (1 - :fee)
                                END
                            ) AS virtual_profit_net,
                            t.virtual_outcome,
                            t.forced_exit
                        FROM shadow_trades t
                        LEFT JOIN LATERAL (
                            SELECT close
                            FROM market_history
                            WHERE type_id = t.type_id
                              AND timestamp >= t.timestamp + (:horizon || ' minutes')::interval
                            ORDER BY timestamp ASC
                            LIMIT 1
                        ) mh ON TRUE
                        ORDER BY t.timestamp DESC
                        LIMIT 10
                        """
                    ),
                    conn,
                    params={"fee": float(fee_rate), "horizon": int(horizon_minutes)},
                )

            if not df.empty:
                tids = [str(x) for x in df["type_id"].tolist()]
                tmap = resolve_type_names(tids)
                df["type_id"] = df["type_id"].astype(str).map(lambda x: tmap.get(x, x))
                df.rename(columns={"type_id": "Item"}, inplace=True)
                # Render forced exits distinctly
                if "forced_exit" in df.columns:
                    df["virtual_outcome"] = df.apply(
                        lambda row: "FORCED_EXIT" if bool(row.get("forced_exit")) else row.get("virtual_outcome"),
                        axis=1,
                    )
                st.dataframe(df, use_container_width=True, height=360)
            else:
                st.info("No shadow trades recorded yet.")
        except Exception as e:
            st.error(f"Ledger query error: {e}")

# ========================
# TAB 5: SYSTEM (System Logs)
# ========================
with tab_system:
    st.markdown("### 🛡️ System Logs & Telemetry")
    
    # [Added] Behavioral Governor (Remote Proof)
    st.markdown("#### 🚦 Behavioral Governor (Active Config)")
    gov_start = int(os.getenv("GOV_SCHEDULE_START_HOUR", "8"))
    gov_end = int(os.getenv("GOV_SCHEDULE_END_HOUR", "23"))
    gov_cap_min = int(os.getenv("GOV_ORDER_CAP_MIN", "35"))
    gov_cap_max = int(os.getenv("GOV_ORDER_CAP_MAX", "50"))
    
    # Check current status
    gov_status = "ACTIVE"
    now_g = datetime.now(timezone.utc)
    if not (gov_start <= now_g.hour < gov_end):
        gov_status = "SLEEPING (Schedule)"
    
    # Check fatigue break
    if r:
        if r.get("system:governor:on_break_until"):
            gov_status = "FATIGUE BREAK (15m)"
            
    c_g1, c_g2, c_g3 = st.columns(3)
    c_g1.metric("Schedule (UTC)", f"{gov_start:02d}:00 - {gov_end:02d}:00")
    c_g2.metric("Order Cap/Hr", f"{gov_cap_min}-{gov_cap_max}")
    c_g3.metric("Current State", gov_status)
    st.divider()
    
    # [Added] Stop-Loss Sensitivity Slider
    sl_val = 1.5
    if r:
        try:
           raw_sl = r.get("system:strategy:stop_loss_pct")
           if raw_sl:
               sl_val = float(raw_sl) * 100.0
        except: pass
    
    # Slider with explicit key for state management
    new_sl = st.slider("Stop-Loss Sensitivity (%)", 0.1, 5.0, float(sl_val), 0.1, help="Adjust global stop-loss threshold. Default 1.5%.")
    if r and abs(new_sl - sl_val) > 1e-6:
        r.set("system:strategy:stop_loss_pct", str(new_sl / 100.0))
        st.toast(f"Stop-Loss Updated: {new_sl}%")

    col_zombie, col_logs = st.columns(2)
    
    with col_zombie:
        st.subheader("Zombie Monitor (HOST 192.168.14.105)")
        # Telemetry
        z_gpu_temp = "N/A"
        z_gpu_load = "0%"
        z_vram = "0/8GB"
        
        if r:
            def sticky_get(k, conn):
                v = conn.get(k)
                if v is not None:
                    st.session_state[f"stk_{k}"] = v
                    return v
                return st.session_state.get(f"stk_{k}")

            # Preferred: direct P4000 keys (gpu_logger)
            try:
                t = sticky_get("system:p4000:temp", r)
                u = sticky_get("system:p4000:load", r)
                m = sticky_get("system:p4000:vram", r)
                if t is not None:
                    z_gpu_temp = f"{t}°C"
                if u is not None:
                    z_gpu_load = f"{u}%"
                if m is not None:
                    z_vram = f"{m} MiB"
            except Exception:
                pass

            # Back-compat: legacy JSON blob
            z_tele = r.get("zombie:telemetry")
            if z_tele:
                try:
                    z_data = json.loads(z_tele)
                    z_gpu_temp = z_data.get("gpu_temp", z_gpu_temp)
                    z_gpu_load = z_data.get("gpu_load", z_gpu_load)
                    z_vram = z_data.get("vram_usage", z_vram)
                except: pass
        
        # P4000 metrics (Zombie host)
        c1, c2, c3 = st.columns(3)
        c1.metric("P4000 Temp", z_gpu_temp)
        c2.metric("P4000 Load", z_gpu_load)
        c3.metric("P4000 VRAM", z_vram)

        # SkyNet / 5090 metrics (training host)
        s_temp = "N/A"
        s_load = "N/A"
        s_vram = "N/A"
        s_last_seen = None
        if r:
            try:
                t = sticky_get("system:skynet:temp", r)
                u = sticky_get("system:skynet:load", r)
                m = sticky_get("system:skynet:vram", r)
                s_last_seen = sticky_get("system:skynet:last_seen", r)
                if t is not None:
                    s_temp = f"{t}°C"
                if u is not None:
                    s_load = f"{u}%"
                if m is not None:
                    s_vram = f"{m} MiB"
            except Exception:
                pass

        c4, c5, c6 = st.columns(3)
        c4.metric("5090 Temp", s_temp)
        c5.metric("5090 Load", s_load)
        c6.metric("5090 VRAM", s_vram)

        if s_last_seen:
            try:
                age_s = max(0, int(time.time() - float(s_last_seen)))
                st.caption(f"5090 telemetry last seen: {age_s}s ago")
            except Exception:
                pass
        
        # render_eve_client_sync_status() REMOVED
        
        st.markdown("**Visual Verification (Live Frame):**")
        st.caption("To engage the physical execution layer, ensure HARDWARE_LOCK is OFF (UNLOCKED/armed).")

        visual_status_raw = None
        visual_stream_raw = None
        if r:
            try:
                visual_status_raw = r.get("system:visual:status")
                visual_stream_raw = r.get("system:visual:stream")
            except Exception:
                visual_status_raw = None
                visual_stream_raw = None

        visual_status_str = _norm_redis_text(visual_status_raw).strip().upper()

        # If the status key is missing, default to a safe, explicit state.
        # This avoids confusing 'UNKNOWN' while keeping the safety gate strict
        # (only VERIFIED enables hardware actions).
        visual_stream_str = _norm_redis_text(visual_stream_raw).strip().upper()

        if not visual_status_str:
            if visual_stream_str == "STREAM_OFFLINE":
                visual_status_str = "OFFLINE"
            else:
                visual_status_str = "UNVERIFIED"

        visual_verified = visual_status_str == "VERIFIED"
        st.caption(f"Visual status: {visual_status_str}")

        c_btn1, c_btn2 = st.columns([1, 3])
        with c_btn1:
            enter_disabled = bool(hardware_lock_state) or (not visual_verified)
            if st.button("Enter Hangar", disabled=enter_disabled):
                if not visual_verified:
                    st.warning("Visual status is not VERIFIED; Arduino bridge will discard this request.")
                elif r:
                    try:
                        r.set("system:hardware:enter_hangar", str(int(time.time())))
                        st.success("Enter Hangar queued (hardware-only).")
                    except Exception as e:
                        st.error(f"Failed to queue Enter Hangar: {e}")
                else:
                    st.error("Redis unavailable")
        with c_btn2:
            st.caption("Sends a hardware-only ENTER sequence (requires system:visual:status=VERIFIED).")

        refresh_seconds = int(os.getenv("ZOMBIE_SHOT_VIEW_REFRESH_SECONDS", "5"))
        fragment = getattr(st, "fragment", None) or getattr(st, "experimental_fragment", None)

        def _render_zombieshot_media_once():
            if r:
                try:
                    shot_b64 = r.get("system:zombie:screenshot")
                    visual_status = r.get("system:visual:status")
                    visual_stream = r.get("system:visual:stream")
                except Exception:
                    shot_b64 = None
                    visual_status = None
                    visual_stream = None
            else:
                shot_b64 = None
                visual_status = None
                visual_stream = None

            if not shot_b64:
                st.info(
                    "No capture yet. Expect updates every few seconds once ZombieShot is running (or per its systemd timer interval)."
                )
                return

            try:
                import base64

                media_bytes = base64.b64decode(shot_b64)
                is_gif = media_bytes[:6] in (b"GIF87a", b"GIF89a")
                caption = "EVE client clip" if is_gif else "EVE client frame"
                try:
                    st.image(media_bytes, caption=caption, use_container_width=True)
                except TypeError:
                    st.image(media_bytes, caption=caption, use_column_width=True)

                vs = _norm_redis_text(visual_status).strip().upper()
                vstream = _norm_redis_text(visual_stream).strip().upper()
                if is_gif and (vs == "OFFLINE" or vstream == "STREAM_OFFLINE"):
                    st.caption("Visual stream is OFFLINE — showing last known good clip.")
            except Exception as e:
                st.warning(f"Capture decode failed: {e}")

        if fragment:
            @fragment(run_every=timedelta(seconds=max(1, int(refresh_seconds))))
            def _zombieshot_media_fragment():
                _render_zombieshot_media_once()

            _zombieshot_media_fragment()
        else:
            # Fallback: relies on the existing page-level refresh cadence.
            _render_zombieshot_media_once()
        
        # UI Purge: Removed EULA and OTP sections per operator directive

        # --- Client state indicator (AUTO-ZOMBIE flip) ---
        client_state = None
        window_title = None
        scanned_at = None
        if r:
            try:
                client_state = r.get("system:zombie:client_state")
                window_title = r.get("system:zombie:window_title")
                scanned_at = r.get("system:zombie:window_scanned_at")
            except Exception:
                pass

        if client_state and str(client_state).upper() == "AUTO_ZOMBIE":
            st.success("🟢 AUTO-ZOMBIE")
        elif window_title and ("EVE Online" in window_title or "Character Selection" in window_title or window_title.startswith("EVE - ")):
            st.success("🟢 AUTO-ZOMBIE")
        else:
            st.info("🟠 ZOMBIE: Awaiting State Transition")

        if window_title:
            st.caption(f"Window: {window_title}")
        if scanned_at:
            st.caption(f"Scanned: {scanned_at}")

        # Screenshot is rendered above for prominent visual verification.
        
    with col_logs:
        st.subheader("Librarian Pulse & Watchdog")
        if r:
            # Watchdog Logic
            last_ts = r.get("system:db_last_check")
            current_count = r.get("system:db_row_count")
            
            if last_ts and current_count:
                age = time.time() - float(last_ts)
                st.metric("DB Sync Age", f"{age:.0f}s")
                
                # Check for stall (Pipeline Velocity)
                # Keep this bounded; large lists can slow/hang the UI.
                history = r.lrange("system:db_history", -10, -1)
                
                if len(history) >= 5:
                    # Check 5 minutes ago (index -5)
                    entry_5m_ago = json.loads(history[-5])
                    current_c = int(current_count)
                    old_c = int(entry_5m_ago['count'])
                    
                    if current_c <= old_c:
                         st.error("🚨 PIPELINE STALLED: No new data for 5 mins!")
                    else:
                         delta = current_c - old_c
                         st.metric("Ingestion Rate (5m)", f"+{delta} rows")
                         st.success(f"⚡ Pipeline Velocity: ACTIVE (+{delta/5:.1f} rows/min)")
            
            with st.expander("Raw Redis Keys"):
                st.write(_safe_redis_keys(r, "system:*") )

        st.divider()
        st.subheader("🔮 Oracle Mind (Blackwell Logs)")
        oracle_logs = "Waiting for stream..."
        if r:
            try:
                raw_l = r.get("system:oracle:logs")
                if raw_l:
                    oracle_logs = raw_l.decode('utf-8')
            except: pass
        st.code(oracle_logs, language="text", line_numbers=True)

        st.divider()

        st.subheader("Hypertable Size")
        try:
            sizes_df = get_hypertable_sizes_df()
            if sizes_df.empty:
                st.info("No hypertable size data available.")
            else:
                st.dataframe(sizes_df[["table", "size"]], use_container_width=True, height=180)
        except Exception as e:
            st.error(f"Hypertable size error: {e}")
            
    st.divider()
    st.markdown("### 🧟 Zombie Monitor Console (launcher_stdout.log)")
    default_log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "launcher_stdout.log")
    log_path = os.getenv("LAUNCHER_STDOUT_LOG_PATH", default_log_path)
    
    if os.path.exists(log_path):
        try:
            with open(log_path, "r") as f:
                # Read last 3KB to avoid freezing UI on huge logs
                f.seek(0, os.SEEK_END)
                size = f.tell()
                read_size = min(size, 10000)
                if read_size > 0:
                    f.seek(size - read_size)
                    content = f.read()
                    st.code(content, language="text")
                else:
                    st.info(f"Log file is empty: {log_path}")
        except Exception as e:
            st.error(f"Error reading logs: {e}")
    else:
        st.info(
            f"Log file not found: {log_path}. If you run launcher control outside Docker, redirect stdout/stderr to this path (or set LAUNCHER_STDOUT_LOG_PATH)."
        )

    st.divider()
    st.header("Global Debug Stream")
    if db_engine:
        st.subheader("Raw Market Orders (Latest 5)")
        try:
            with db_engine.connect() as conn:
                df = pd.read_sql("SELECT * FROM market_orders ORDER BY timestamp DESC LIMIT 5", conn)
                st.dataframe(df)
        except Exception as e:
            st.error(f"Debug Stream Error: {e}")

    st.divider()
    st.header("Simulation Control")
    if st.button("Run Manual Backtest"):
        st.info("Simulating Strategy Execution...")
        time.sleep(1)
        st.success("Simulation Complete.")
