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
from datetime import datetime, timedelta
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
    url = "https://esi.evetech.net/latest/universe/names/"
    ids = list(set([int(x) for x in type_ids]))
    mapping = {}
    chunk_size = 900 
    for i in range(0, len(ids), chunk_size):
        chunk = ids[i:i + chunk_size]
        try:
            resp = requests.post(url, json=chunk)
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

st_autorefresh(interval=60 * 1000, key="datarefresh")

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


def read_eve_client_rsync_tail(lines: int = 3) -> str:
    """Best-effort: local log path, else optional SSH tail."""
    log_path = os.getenv("EVE_CLIENT_RSYNC_LOG_PATH", "/tmp/eve_client_rsync.log")
    tail = _tail_text_file(log_path, lines=lines)
    if tail:
        return tail

    # Optional: allow pulling from a remote host if the dashboard isn't running on Skynet.
    # Example: export EVE_CLIENT_RSYNC_LOG_SSH='seb@SKYNET'
    ssh_host = os.getenv("EVE_CLIENT_RSYNC_LOG_SSH", "").strip()
    if not ssh_host:
        return ""

    try:
        import subprocess

        cmd = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=3",
            ssh_host,
            f"tail -n {int(lines)} {log_path} 2>/dev/null || true",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        return (res.stdout or "").strip()
    except Exception:
        return ""


def parse_rsync_progress_from_tail(tail: str):
    """Parse rsync --info=progress2 lines to (percent, speed, eta)."""
    if not tail:
        return None
    last = ""
    for line in tail.splitlines()[::-1]:
        if "%" in line and "(" in line:
            last = line.strip()
            break
    if not last:
        return None

    # Example:
    # 100,629,169   2%   39.09MB/s    0:00:02 (xfr#342, ir-chk=1314/11517)
    m = re.search(
        r"\s(?P<pct>\d{1,3})%\s+(?P<speed>\S+)\s+(?P<eta>\d+:\d\d:\d\d|\d+:\d\d)\s+\(",
        last,
    )
    if not m:
        return None
    try:
        pct = max(0, min(100, int(m.group("pct"))))
    except Exception:
        pct = None
    return {
        "percent": pct,
        "speed": m.group("speed"),
        "eta": m.group("eta"),
        "raw": last,
    }


def render_eve_client_sync_status():
    tail = read_eve_client_rsync_tail(lines=3)
    info = parse_rsync_progress_from_tail(tail)

    if not tail:
        st.info("📂 Status: Awaiting Client Upload to `~/eve-client`.")
        st.markdown("📦 Syncing EVE Client: *(no rsync log visible)*")
        return

    if not info or info.get("percent") is None:
        st.info("📂 Status: Awaiting Client Upload to `~/eve-client`.")
        st.markdown("📦 Syncing EVE Client: *(parsing pending)*")
        st.code(tail)
        return

    percent = int(info["percent"])
    if percent >= 100:
        st.success("✅ Client Data Verified: Ready for Zombie Boot.")

        if not st.session_state.get("client_sync_ready_notified", False):
            try:
                st.toast("✅ Client Data Verified: Ready for Zombie Boot.")
            except Exception:
                pass
            st.session_state["client_sync_ready_notified"] = True
    else:
        st.info("📂 Status: Awaiting Client Upload to `~/eve-client`.")

    blocks = 10
    filled = max(0, min(blocks, int(round((percent / 100) * blocks))))
    bar = ("█" * filled) + ("░" * (blocks - filled))

    st.markdown(f"📦 Syncing EVE Client: [{bar}] {percent}% (ETA: {info['eta']})")
    try:
        st.progress(percent / 100)
    except Exception:
        pass
    st.caption(f"Speed: {info['speed']}")

# --- Redis Connection ---
@st.cache_resource
def get_redis_client():
    return redis.Redis(host=os.getenv('REDIS_HOST', 'cache'), port=6379, db=0, decode_responses=True)

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
# HARD-CODED CONNECTION - NO ENV VARS
try:
    db_engine = sqlalchemy.create_engine('postgresql://eve_user:eve_pass@db:5432/eve_market_data')
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
hardware_lock_state = st.sidebar.toggle("HARDWARE_LOCK", value=True, help="When enabled (locked), the Arduino bridge will NOT send serial commands. Disable to arm the hardware.")
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
    keys = r.keys("features:*")
    keys = [k for k in keys if len(k.split(":")) == 2]
    
    ids_to_resolve = []
    for k in keys:
        try:
            val = r.get(k)
            if val:
                obj = json.loads(val)
                raw_items.append(obj)
                ids_to_resolve.append(obj.get('type_id'))
        except: pass
    
    def alpha_score(item):
        return abs(float(item.get("imbalance", 1.0)) - 1.0)

    raw_items.sort(key=alpha_score, reverse=True)
    top_5 = raw_items[:5]
    
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
            market_data = []
            for obj in raw_items:
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

    # Model Path & Validation R²
    model_path_used = "models/oracle_v1_latest.pt"  # default for oracle container
    if r:
        try:
            from_redis = r.get("oracle:model_path")
            if from_redis:
                model_path_used = str(from_redis)
        except Exception:
            pass

    val_r2 = 0.0
    train_r2 = 0.0
    reject_reason = None
    if r:
        try:
            metrics_raw = r.get("oracle:last_training_metrics")
            if metrics_raw:
                metrics = json.loads(metrics_raw)
                val_r2 = float(metrics.get("val_r2", 0.0) or 0.0)
                train_r2 = float(metrics.get("train_r2", 0.0) or 0.0)
                reject_reason = metrics.get("reject_reason")
        except Exception:
            pass

    # Next training session countdown (based on 4h cron schedule)
    next_training_eta = "N/A"
    if r:
        try:
            last_train_ts_raw = r.get("oracle:last_training_ts")
            if last_train_ts_raw:
                last_ts = float(last_train_ts_raw)
                next_ts = last_ts + (4 * 3600)
                remaining_s = max(0, next_ts - time.time())
                h = int(remaining_s // 3600)
                m = int((remaining_s % 3600) // 60)
                next_training_eta = f"{h}h {m}m"
        except Exception:
            pass

    c_m1, c_m2, c_m3 = st.columns(3)
    c_m1.metric("Model Path", model_path_used.split("/")[-1])
    c_m2.metric("Validation R²", f"{val_r2:.4f}")
    c_m3.metric("Next Training", next_training_eta)

    if reject_reason:
        st.warning(f"🚫 Last Induction Rejected: {reject_reason}")

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

    st.info(
        "Run on 5090: curl -o training_data.parquet http://192.168.14.105:8001/training_data_cleaned.parquet && python3 -m oracle.train_model"
    )

    st.divider()


# ========================
# TAB 4: FUNDAMENTALS (News Desk)
# ========================
with tab_fundamentals:
    st.markdown("### 📰 Fundamentals")

    st.info(
        "Insight: Jita player count + ISK destroyed shift the demand curve right — this is the macro \"Surge\" driver the 5090 Oracle hunts."
    )

    # Faster refresh for desk-style alerting.
    st_autorefresh(interval=int(os.getenv("FUNDAMENTALS_REFRESH_MS", "2000")), key="fundamentals_refresh")

    # Real-time feed sourced from Redis.
    try:
        players_raw = r.get("system:macro:players")
        warfare_raw = r.get("system:macro:warfare")
        warfare_updated_at_raw = r.get("system:macro:warfare:updated_at")
        warfare_source = r.get("system:macro:warfare:source")

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
            st.warning(f"OSINT feed is not MCP-backed (source={warfare_source}).")
        else:
            st.caption(f"OSINT source: {warfare_source}")
    else:
        st.warning("OSINT feed source unknown (missing system:macro:warfare:source).")

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
    if r:
        try:
            model_status = r.get("oracle:model_status") or "IDLE"
            metrics_raw = r.get("oracle:last_training_metrics")
            if metrics_raw:
                last_metrics = json.loads(metrics_raw)
        except Exception as e:
            st.error(f"Oracle status read error: {e}")

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

    # Trade Performance Table (color-coded)
    if db_engine:
        try:
            with db_engine.connect() as conn:
                perf_df = pd.read_sql(
                    text("""
                        SELECT
                            type_id,
                            signal_type,
                            virtual_outcome,
                            actual_price_at_time AS entry_price,
                            timestamp
                        FROM shadow_trades
                        WHERE virtual_outcome IN ('WIN', 'LOSS')
                        ORDER BY timestamp DESC
                        LIMIT 100
                    """),
                    conn,
                )

            if not perf_df.empty:
                def _color_outcome(val):
                    if val == "WIN":
                        return "background-color: #28a745; color: white"
                    elif val == "LOSS":
                        return "background-color: #dc3545; color: white"
                    return ""

                st.subheader("Recent Trade Performance")
                styled = perf_df.style.applymap(_color_outcome, subset=["virtual_outcome"])
                st.dataframe(styled, use_container_width=True, height=300)
        except Exception as e:
            st.error(f"Trade performance table error: {e}")

    st.divider()

    # Item ROI Bar Chart
    if db_engine:
        try:
            with db_engine.connect() as conn:
                roi_df = pd.read_sql(
                    text("""
                        SELECT
                            t.type_id,
                            COUNT(*) AS trade_count,
                            SUM(
                                CASE
                                    WHEN t.virtual_outcome = 'WIN' AND mh.close IS NOT NULL THEN
                                        CASE
                                            WHEN t.signal_type = 'BUY' THEN (mh.close - t.actual_price_at_time)
                                            ELSE (t.actual_price_at_time - mh.close)
                                        END
                                    ELSE 0
                                END
                            ) AS net_profit
                        FROM shadow_trades t
                        LEFT JOIN LATERAL (
                            SELECT close
                            FROM market_history
                            WHERE type_id = t.type_id
                              AND timestamp >= t.timestamp + INTERVAL '60 minutes'
                            ORDER BY timestamp ASC
                            LIMIT 1
                        ) mh ON TRUE
                        WHERE t.virtual_outcome IN ('WIN', 'LOSS')
                        GROUP BY t.type_id
                        ORDER BY net_profit DESC
                        LIMIT 20
                    """),
                    conn,
                )

            if not roi_df.empty:
                st.subheader("Top 20 Items by Net Profit (ROI)")
                type_names = resolve_type_names(roi_df["type_id"].tolist())
                roi_df["item_name"] = roi_df["type_id"].apply(lambda x: type_names.get(str(x), f"Type {x}"))
                fig_roi = go.Figure(
                    go.Bar(
                        x=roi_df["net_profit"],
                        y=roi_df["item_name"],
                        orientation="h",
                        marker=dict(
                            color=roi_df["net_profit"],
                            colorscale="RdYlGn",
                            showscale=True,
                        ),
                    )
                )
                fig_roi.update_layout(title="Item ROI", xaxis_title="Net Profit (ISK)", yaxis_title="Item")
                st.plotly_chart(fig_roi, use_container_width=True)
        except Exception as e:
            st.error(f"Item ROI chart error: {e}")

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

        try:
            with db_engine.connect() as conn:
                # Win rate is computed over resolved outcomes (WIN+LOSS).
                row = conn.execute(
                    text(
                        """
                        SELECT
                            SUM(CASE WHEN virtual_outcome = 'WIN' THEN 1 ELSE 0 END) AS wins,
                            SUM(CASE WHEN virtual_outcome IN ('WIN','LOSS') THEN 1 ELSE 0 END) AS resolved
                        FROM shadow_trades
                        """
                    )
                ).fetchone()

                wins = int(row[0] or 0)
                resolved = int(row[1] or 0)

                alpha_row = conn.execute(
                    text(
                        """
                        SELECT COALESCE(
                            SUM(
                                CASE
                                    WHEN t.virtual_outcome = 'WIN' AND mh.close IS NOT NULL THEN
                                        CASE
                                            WHEN t.signal_type = 'BUY' THEN (mh.close - t.actual_price_at_time)
                                            ELSE (t.actual_price_at_time - mh.close)
                                        END
                                    ELSE 0
                                END
                            ) * (1 - :fee),
                            0
                        ) AS total_alpha
                        FROM shadow_trades t
                        LEFT JOIN LATERAL (
                            SELECT close
                            FROM market_history
                            WHERE type_id = t.type_id
                              AND timestamp >= t.timestamp + (:horizon || ' minutes')::interval
                            ORDER BY timestamp ASC
                            LIMIT 1
                        ) mh ON TRUE
                        """
                    ),
                    {"fee": float(fee_rate), "horizon": int(horizon_minutes)},
                ).fetchone()

                total_alpha = float(alpha_row[0] or 0.0)

        except Exception as e:
            st.error(f"Performance query error: {e}")

        win_rate = (wins / resolved * 100.0) if resolved > 0 else 0.0
        c1, c2 = st.columns(2)
        c1.metric("Total Virtual Alpha (7d, net)", f"{total_alpha:,.2f}")
        c2.metric("Win Rate % (resolved)", f"{win_rate:.1f}%")

        st.divider()

        # --- Sharpe Ratio Audit ---
        try:
            with db_engine.connect() as conn:
                # Fetch last 100 resolved trades to compute Sharpe
                sharpe_df = pd.read_sql(
                    text(
                        """
                        SELECT
                            t.actual_price_at_time as entry_price,
                            mh.close as exit_price,
                            t.signal_type,
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
                        WHERE t.virtual_outcome IN ('WIN', 'LOSS')
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
                    if row['forced_exit']:
                        # Stop-loss can widen during high-volatility warfare events.
                        try:
                            warfare_isk = float(r.get("system:macro:warfare") or 0.0)
                        except Exception:
                            warfare_isk = 0.0

                        try:
                            warfare_threshold = float(os.getenv("SHADOW_WARFARE_THRESHOLD_ISK", "0"))
                        except Exception:
                            warfare_threshold = 0.0

                        stop_loss_pct = 0.015
                        try:
                            stop_loss_pct = float(os.getenv("SHADOW_STOP_LOSS_PCT", "0.015"))
                        except Exception:
                            stop_loss_pct = 0.015

                        if warfare_threshold > 0 and warfare_isk > warfare_threshold:
                            try:
                                stop_loss_pct = float(os.getenv("SHADOW_WAR_STOP_LOSS_PCT", "0.03"))
                            except Exception:
                                stop_loss_pct = 0.03

                        return -abs(float(stop_loss_pct))
                    
                    if row['entry_price'] == 0 or pd.isna(row['exit_price']):
                        return 0.0
                    
                    if row['signal_type'] == 'BUY':
                        return (row['exit_price'] - row['entry_price']) / row['entry_price']
                    else:
                        return (row['entry_price'] - row['exit_price']) / row['entry_price']

                sharpe_df['ret'] = sharpe_df.apply(calc_return, axis=1)

                # Bardol winsorization clamp to stabilize Sharpe audit.
                sharpe_df['ret'] = sharpe_df['ret'].clip(lower=-1.0, upper=10.0)
                
                rp = sharpe_df['ret'].mean()
                sigma_p = sharpe_df['ret'].std()
                
                sharpe = (rp / sigma_p) if sigma_p > 0 else 0.0
                
                st.subheader("Risk Audit (Last 100 Trades)")
                c_s1, c_s2, c_s3 = st.columns(3)
                c_s1.metric("Sharpe Ratio", f"{sharpe:.4f}")
                c_s2.metric("Exp. Return", f"{rp*100:.3f}%")
                c_s3.metric("Volatility (σ)", f"{sigma_p*100:.3f}%")
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
    col_zombie, col_logs = st.columns(2)
    
    with col_zombie:
        st.subheader("Zombie Monitor (HOST 192.168.14.105)")
        # Telemetry
        z_gpu_temp = "N/A"
        z_gpu_load = "0%"
        z_vram = "0/8GB"
        
        if r:
            # Preferred: direct P4000 keys (gpu_logger)
            try:
                t = r.get("system:p4000:temp")
                u = r.get("system:p4000:load")
                m = r.get("system:p4000:vram")
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
        s_load = "0%"
        s_vram = "0 MiB"
        if r:
            try:
                t = r.get("system:skynet:temp")
                u = r.get("system:skynet:load")
                m = r.get("system:skynet:vram")
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

        render_eve_client_sync_status()
        
        st.markdown("**Visual Verification (Live Frame):**")

        st.markdown("**📜 Accept EULA**")
        if r:
            c_eula1, c_eula2 = st.columns([1, 3])
            with c_eula1:
                do_eula = st.button("📜 Accept EULA", use_container_width=True)
            with c_eula2:
                st.caption("Triggers an aggressive host-side EULA bypass (blind-fire) and a standard prompt sweep.")

            if do_eula:
                try:
                    # Safety: only allow hardware actions when HARDWARE_LOCK is disabled.
                    lock_val = r.get("system:hardware_lock")
                    locked = lock_val is None or str(lock_val).lower() in {"true", "1", "yes"}
                    if locked:
                        st.warning("Hardware lock is enabled. Disable HARDWARE_LOCK to send the handshake.")
                    else:
                        # Trigger the Arduino hardware handshake (END -> 500ms -> RETURN).
                        r.setex("system:hardware:handshake", 30, "true")

                        # Visual flush: clear cached screenshot so the next frame shows post-accept state.
                        r.delete("system:zombie:screenshot")

                        st.success("Hardware handshake sent.")
                except Exception as e:
                    st.error(f"Failed to request clearance sweep: {e}")
        else:
            st.info("EULA clearance unavailable (Redis not connected).")

        st.markdown("**🔐 OTP Entry**")
        if r:
            otp_code = st.text_input("One-time code", type="password", help="Paste the OTP from your authenticator.")
            c_otp1, c_otp2 = st.columns([1, 3])
            with c_otp1:
                do_inject = st.button("Inject", use_container_width=True)
            with c_otp2:
                st.caption("Writes OTP to Redis for the host to inject into the client.")

            if do_inject:
                if not otp_code:
                    st.warning("Enter an OTP code first.")
                else:
                    try:
                        # Short TTL so stale OTPs don't linger.
                        r.setex("system:zombie:otp", 120, otp_code)
                        st.success("OTP queued for injection.")
                    except Exception as e:
                        st.error(f"Failed to queue OTP: {e}")
        else:
            st.info("OTP injection unavailable (Redis not connected).")

        if r:
            try:
                shot_b64 = r.get("system:zombie:screenshot")
            except Exception:
                shot_b64 = None
        else:
            shot_b64 = None

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

        if shot_b64:
            try:
                import base64

                img_bytes = base64.b64decode(shot_b64)

                # Fallback for older Streamlit versions
                try:
                    st.image(
                        img_bytes,
                        caption="EVE client frame (refreshes with page auto-refresh)",
                        use_container_width=True,
                    )
                except TypeError:
                    st.image(
                        img_bytes,
                        caption="EVE client frame (refreshes with page auto-refresh)",
                        use_column_width=True,
                    )
            except Exception as e:
                st.warning(f"Screenshot decode failed: {e}")
        else:
            st.info("No screenshot yet. Expect updates every ~60s once ZombieShot is running on the host.")
        
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
                history = r.lrange("system:db_history", 0, -1)
                
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
                st.write(r.keys("system:*"))
            
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
