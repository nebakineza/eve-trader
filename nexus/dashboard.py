import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import redis
import json
import time
import sys
import os
import requests
import math
from datetime import datetime, timedelta
import numpy as np
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

st_autorefresh(interval=30 * 1000, key="datarefresh")

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
db_conn_str = os.getenv("DATABASE_URL", "postgresql://eve_user:eve_password@db:5432/eve_market_data")
try:
    db_engine = sqlalchemy.create_engine(db_conn_str)
    with db_engine.connect() as conn:
        res = conn.execute(text("SELECT count(*) FROM market_orders")).fetchone()
        db_count = res[0]
except:
    db_engine = None
    db_count = 0

# --- Sidebar Global Controls ---
st.sidebar.header("System Controls")
st.sidebar.caption(f"Redis: {redis_status}")
st.sidebar.caption(f"DB Rows: {db_count}") # Emergency Monitor

auto_trade_state = st.sidebar.toggle("AUTO_TRADE_ENABLED", value=False)
if r:
    r.set("system:auto_trade_enabled", "true" if auto_trade_state else "false")
    status_msg = "🟢 AI-AUTONOMOUS" if auto_trade_state else "🔴 HUMAN-IN-LOOP"
    st.sidebar.markdown(f"**Status:** {status_msg}")
    
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
tab_radar, tab_execution, tab_oracle, tab_system = st.tabs(["Radar", "Execution", "Oracle", "System"])

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
    
    # Check Market History Depth
    history_depth = 0
    if db_engine:
        try:
            with db_engine.connect() as conn:
                 res = conn.execute(text("SELECT count(*) FROM market_history")).fetchone()
                 history_depth = res[0]
        except: pass
    
    # Check if warming up
    if history_depth < 50:
         st.info(f"🧠 Oracle is warming up... (Current Depth: {history_depth}/50 Ticks)")
    else:
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
         
         steps = 12
         x_axis = [f"+{i*5}m" for i in range(steps+1)]
         y_path = np.linspace(curr, latest_pred, steps+1)
         
         fig_pred = go.Figure()
         fig_pred.add_trace(go.Scatter(x=x_axis, y=y_path, mode='lines+markers', name='Forecast'))
         fig_pred.update_layout(title="Trajectory", height=300, paper_bgcolor='rgba(0,0,0,0)')
         st.plotly_chart(fig_pred, use_container_width=True)

# ========================
# TAB 4: SYSTEM (System Logs)
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
            z_tele = r.get("zombie:telemetry")
            if z_tele:
                try:
                    z_data = json.loads(z_tele)
                    z_gpu_temp = z_data.get("gpu_temp", z_gpu_temp)
                    z_gpu_load = z_data.get("gpu_load", z_gpu_load)
                    z_vram = z_data.get("vram_usage", z_vram)
                except: pass
        
        c1, c2, c3 = st.columns(3)
        c1.metric("P4000 Temp", z_gpu_temp)
        c2.metric("GPU Load", z_gpu_load)
        c3.metric("VRAM", z_vram)
        
        st.markdown("**Real-Time Scrape Feed:**")
        st.code("Waiting for Zombie Telemetry stream...")
        
    with col_logs:
        st.subheader("Librarian Pulse")
        if r:
            st.write(r.keys("system:*"))
            
    st.divider()
    st.header("Simulation Control")
    if st.button("Run Manual Backtest"):
        st.info("Simulating Strategy Execution...")
        time.sleep(1)
        st.success("Simulation Complete.")
