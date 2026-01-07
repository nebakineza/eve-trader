import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time
import os
import torch
import numpy as np

# Configuration
st.set_page_config(page_title="EVE Oracle Training Monitor", layout="wide", page_icon="🧠")
st.title("🧠 Oracle (RTX 5090) - Training Monitor")

# Placeholder for Metrics
col1, col2, col3 = st.columns(3)
placeholder_loss = col1.empty()
placeholder_acc = col2.empty()
placeholder_epoch = col3.empty()

# Charts
chart_loss = st.empty()
chart_convergence = st.empty()

# Simulation of Log Reading
# In a real scenario, the trainer.py would write to a CSV log file.
LOG_FILE = "training_log.csv"

def generate_mock_log():
    # Only for demonstration if no log exists
    if not os.path.exists(LOG_FILE):
        df = pd.DataFrame(columns=["Epoch", "Batch", "Loss", "Accuracy"])
        df.to_csv(LOG_FILE, index=False)

generate_mock_log()

st.sidebar.header("Training Controls")
if st.sidebar.button("Refresh Logs"):
    st.rerun()

auto_refresh = st.sidebar.checkbox("Auto-Refresh (2s)", value=True)

try:
    if os.path.exists(LOG_FILE):
        df = pd.read_csv(LOG_FILE)
        
        if not df.empty:
            # Latest Metrics
            latest = df.iloc[-1]
            placeholder_loss.metric("Current Loss", f"{latest['Loss']:.4f}")
            placeholder_acc.metric("Accuracy", f"{latest['Accuracy']*100:.1f}%")
            placeholder_epoch.metric("Epoch", f"{int(latest['Epoch'])}")
            
            # Loss Chart
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(x=df.index, y=df['Loss'], mode='lines', name='Loss'))
            fig_loss.update_layout(title="Training Loss Function", xaxis_title="Step", yaxis_title="Loss")
            chart_loss.plotly_chart(fig_loss, use_container_width=True)
            
            # Convergence/Accuracy Chart
            fig_acc = go.Figure()
            fig_acc.add_trace(go.Scatter(x=df.index, y=df['Accuracy'], mode='lines', name='Accuracy', line=dict(color='green')))
            fig_acc.update_layout(title="Model Convergence", xaxis_title="Step", yaxis_title="Accuracy")
            chart_convergence.plotly_chart(fig_acc, use_container_width=True)
        else:
            st.info("Waiting for training to start...")
    else:
        st.warning(f"Log file {LOG_FILE} not found.")

except Exception as e:
    st.error(f"Error reading logs: {e}")

if auto_refresh:
    time.sleep(2)
    st.rerun()
