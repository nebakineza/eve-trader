from __future__ import annotations

import logging
import math
import os
import sys
from datetime import datetime, timezone
from typing import Any

import torch
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Add project root
sys.path.append(os.getcwd())

from oracle.model import OracleTemporalTransformer
from oracle.processors import apply_stationarity
from oracle.config_loader import GLOBAL_CONFIG

app = FastAPI(title="Oracle Inference Service")
logger = logging.getLogger("OracleService")

model: OracleTemporalTransformer | None = None
MODEL_META: dict[str, Any] | None = None
DEVICE = torch.device("cpu")
REQUESTED_DEVICE = os.getenv("ORACLE_DEVICE", "cpu")


def _select_device(requested: str) -> str:
    requested = (requested or "cpu").strip().lower()
    if requested.startswith("cuda"):
        try:
            if torch.cuda.is_available():
                return requested
        except Exception:
            pass
        return "cpu"
    return requested


def _model_path() -> str:
    return os.getenv("ORACLE_MODEL_PATH", "models/oracle_v1_latest.pt")


def _load_checkpoint(path: str, *, map_location: str):
    obj = torch.load(path, map_location=map_location)
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"], obj
    return obj, None


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _build_sequence_from_features(
    features: dict[str, Any], *, input_dim: int, seq_len: int = 2
) -> tuple[list[list[float]], list[int], list[int]]:
    price = _as_float(features.get("price"), 0.0)
    last_close = _as_float(features.get("last_close"), 0.0)
    if last_close > 0 and price > 0:
        velocity = (price - last_close) / last_close
    else:
        velocity = 0.0

    imbalance = _as_float(features.get("imbalance"), 1.0)

    # Optional technical indicators (if the Librarian/stream provides them)
    # Keep these transforms aligned with training normalization.
    rsi_raw = _as_float(features.get("RSI"), 50.0)
    rsi_scaled = (rsi_raw - 50.0) / 50.0

    macd_raw = _as_float(features.get("MACD"), 0.0)
    macd_sig_raw = _as_float(features.get("MACD_Signal"), 0.0)
    bb_u_raw = _as_float(features.get("BB_Upper"), 0.0)
    bb_l_raw = _as_float(features.get("BB_Low"), 0.0)

    p_safe = price if price and price > 0 else 1.0
    macd_rel = macd_raw / p_safe
    macd_sig_rel = macd_sig_raw / p_safe
    bb_u_rel = (bb_u_raw / p_safe) - 1.0
    bb_l_rel = (bb_l_raw / p_safe) - 1.0

    players_raw = _as_float(features.get("players_online"), 0.0)
    if players_raw < 0:
        players_raw = 0.0
    # Match training scaling: log1p(players) / log1p(100000)
    try:
        players_scaled = math.log1p(players_raw) / math.log1p(100000.0)
    except Exception:
        players_scaled = 0.0

    warfare_raw = _as_float(features.get("warfare_isk_destroyed"), 0.0)
    if warfare_raw < 0:
        warfare_raw = 0.0
    war_scale = _as_float(os.getenv("ORACLE_WARFARE_LOG_SCALE_ISK", "10000000000000"), 10000000000000.0)
    if war_scale <= 0:
        war_scale = 10000000000000.0
    try:
        war_scaled = math.log1p(warfare_raw) / math.log1p(war_scale)
    except Exception:
        war_scaled = 0.0

    # Macro-Inertia: Demand_Shift = ISK_Destroyed_60m / Global_Market_Cap
    # Use the same stable log-ratio mapping as training.
    global_cap = _as_float(
        features.get("global_market_cap")
        if features.get("global_market_cap") is not None
        else os.getenv("ORACLE_GLOBAL_MARKET_CAP_ISK", "1000000000000000"),
        1000000000000000.0,
    )
    if global_cap <= 0:
        global_cap = 1.0
    try:
        demand_shift_scaled = math.log1p(warfare_raw) / math.log1p(global_cap)
    except Exception:
        demand_shift_scaled = 0.0

    vec = [
        velocity,
        imbalance,
        rsi_scaled,
        macd_rel,
        macd_sig_rel,
        bb_u_rel,
        bb_l_rel,
        float(players_scaled),
        float(war_scaled),
        float(demand_shift_scaled),
    ]

    if input_dim > len(vec):
        vec = vec + [0.0] * (input_dim - len(vec))
    elif input_dim < len(vec):
        vec = vec[:input_dim]

    seq = [vec[:] for _ in range(int(seq_len))]

    ts = features.get("timestamp")
    dt = None
    try:
        if ts is not None:
            dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
    except Exception:
        dt = None
    if dt is None:
        dt = datetime.now(tz=timezone.utc)

    hour = int(dt.hour)
    day = int(dt.weekday())
    hours = [hour] * int(seq_len)
    days = [day] * int(seq_len)
    return seq, hours, days


def _normalize_payload(
    payload: Any, *, input_dim: int
) -> tuple[list[list[float]], list[int], list[int]]:
    if isinstance(payload, dict):
        return _build_sequence_from_features(payload, input_dim=input_dim)

    if isinstance(payload, list):
        if len(payload) == 0:
            return _build_sequence_from_features({}, input_dim=input_dim)

        first = payload[0]
        if isinstance(first, dict):
            return _build_sequence_from_features(first, input_dim=input_dim)

        if isinstance(first, list):
            seq: list[list[float]] = []
            for row in payload:
                if not isinstance(row, list):
                    continue
                vec = [_as_float(v, 0.0) for v in row]
                if input_dim > len(vec):
                    vec = vec + [0.0] * (input_dim - len(vec))
                elif input_dim < len(vec):
                    vec = vec[:input_dim]
                seq.append(vec)
            if not seq:
                seq, _, _ = _build_sequence_from_features({}, input_dim=input_dim)
            hours = [0] * len(seq)
            days = [0] * len(seq)
            return seq, hours, days

        vec = [_as_float(v, 0.0) for v in payload]
        if input_dim > len(vec):
            vec = vec + [0.0] * (input_dim - len(vec))
        elif input_dim < len(vec):
            vec = vec[:input_dim]
        seq = [vec, vec]
        return seq, [0, 0], [0, 0]

    return _build_sequence_from_features({}, input_dim=input_dim)

@app.on_event("startup")
async def load_model():
    global model, DEVICE, MODEL_META
    try:
        device_str = _select_device(REQUESTED_DEVICE)
        DEVICE = torch.device(device_str)

        # Load weights if available
        model_path = _model_path()
        checkpoint_meta = None
        if os.path.exists(model_path):
            state_dict, checkpoint_meta = _load_checkpoint(model_path, map_location="cpu")
            if isinstance(checkpoint_meta, dict):
                MODEL_META = checkpoint_meta
            input_dim = 7
            if isinstance(checkpoint_meta, dict):
                try:
                    input_dim = int(checkpoint_meta.get("input_dim", input_dim))
                except Exception:
                    pass
            model = OracleTemporalTransformer(input_dim=input_dim)
            model.load_state_dict(state_dict, strict=True)
            logger.info(
                f"Loaded model weights from {model_path} (device={DEVICE}, input_dim={input_dim})"
            )
        else:
            model = OracleTemporalTransformer(input_dim=7)
            logger.warning(f"No model weights found at {model_path}. Using initialized random weights (Untrained).")

        model.to(DEVICE)
        model.eval()
    except Exception as e:
        logger.error(f"Failed to load model: {e}")


@app.get("/health")
async def health():
    mp = _model_path()
    return {
        "ok": True,
        "model_loaded": model is not None,
        "model_path": mp,
        "model_exists": os.path.exists(mp),
        "requested_device": REQUESTED_DEVICE,
        "cuda_available": bool(getattr(torch, "cuda", None) and torch.cuda.is_available()),
        "model_target": (MODEL_META or {}).get("target") if isinstance(MODEL_META, dict) else None,
        "model_output_activation": (MODEL_META or {}).get("output_activation") if isinstance(MODEL_META, dict) else None,
    }

class MarketFeatures(BaseModel):
    # flexible input: list or dict
    features: list | dict
    hour: int = 0
    day: int = 0
    # New: History for FracDiff
    price_history: list[float] | None = None
    volume_history: list[float] | None = None


@app.post("/predict")
async def predict(data: MarketFeatures):
    global model

    # 1. STATIONARITY INJECTION (FracDiff)
    frac_price_val = 0.0
    frac_vol_val = 0.0
    
    # If client sends history, compute FracDiff on the fly
    if data.price_history and len(data.price_history) >= 15:
        # Create minimal dataframe
        df_hist = pd.DataFrame({'close': data.price_history})
        if data.volume_history and len(data.volume_history) == len(data.price_history):
             df_hist['volume'] = data.volume_history
             
        # Apply transformation
        df_fd = apply_stationarity(df_hist)
        
        # Extract latest value
        if 'frac_close' in df_fd.columns:
            # fillna might happen in processor, but ensure float
            latest = df_fd['frac_close'].iloc[-1]
            if not pd.isna(latest):
                frac_price_val = float(latest)
                
        if 'frac_volume' in df_fd.columns:
            latest = df_fd['frac_volume'].iloc[-1]
            if not pd.isna(latest):
                frac_vol_val = float(latest)
    
    # Inject FracDiff values into feature dictionary if it's a dict
    # (If it's a list, the client code (Strategist) must have already packed it, 
    # but currently we only use dict-based inference from Strategist)
    if isinstance(data.features, dict):
        data.features['frac_diff_price'] = frac_price_val
        data.features['frac_diff_volume'] = frac_vol_val

    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    try:
        input_dim = getattr(getattr(model, "embedding", None), "in_features", None)
        if input_dim is None:
            input_dim = 2

        seq, hours, days = _normalize_payload(data.features, input_dim=int(input_dim))

        x = torch.tensor([seq], dtype=torch.float32, device=DEVICE)
        hour = torch.tensor([hours], dtype=torch.long, device=DEVICE)
        day = torch.tensor([days], dtype=torch.long, device=DEVICE)

        with torch.no_grad():
            output = model(x, hour, day)

        pred_tensor = output.get("predicted_price")
        conf_tensor = output.get("confidence")
        status = output.get("status", "OK")

        pred_raw = float(pred_tensor[0][0].item()) if pred_tensor is not None else 0.0
        confidence = float(conf_tensor[0][0].item()) if conf_tensor is not None else 0.0

        # Decode model output based on checkpoint metadata.
        # - target=return: output is a relative return over horizon; convert to absolute price using current price
        # - otherwise: treat output as absolute price
        target = (MODEL_META or {}).get("target") if isinstance(MODEL_META, dict) else None
        pred_price = pred_raw
        if target == "return":
            activation = (MODEL_META or {}).get("output_activation") if isinstance(MODEL_META, dict) else None
            activation = (activation or "").strip().lower()
            # Newer checkpoints may apply a bounded activation during training.
            # Keep inference consistent without breaking older checkpoints.
            if activation == "tanh":
                pred_raw = math.tanh(pred_raw)

            current = 0.0
            if isinstance(data.features, dict):
                current = _as_float(data.features.get("price"), 0.0)
            if current and current > 0:
                pred_price = float(current * (1.0 + pred_raw))

        # Fallback if unusable
        if pred_price <= 0.0 and isinstance(data.features, dict):
            pred_price = _as_float(data.features.get("price"), 0.0)

        return {
            "predicted_price": pred_price,
            "confidence": confidence,
            "status": status,
            "forecast_sequence": [float(pred_price * 0.99), float(pred_price * 1.005), pred_price],
        }
    except Exception as e:
        logger.error(f"Inference Error: {e}")
        return {
            "predicted_price": 0.0, 
            "confidence": 0.0, 
            "status": "ERROR",
            "forecast_sequence": [] 
        }

class PredictionRequest(BaseModel):
    # Standard 7 params
    velocity: float
    imbalance: float
    rsi: float
    macd: float
    macd_signal: float
    bb_upper: float
    bb_lower: float
    
    # Macro context
    players_online: float | None = None
    isk_destroyed_last_hour: float | None = None
    global_market_cap: float | None = None

    # Temporal context
    timestamp: float | None = None
    price_history: list[float] | None = None # NEW: For FracDiff
    volume_history: list[float] | None = None # NEW: For FracDiff

@app.post("/predict")
def predict(req: PredictionRequest):
    global model, MODEL_META
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # 1. Apply Stationarity if history provided
    frac_price = 0.0
    frac_vol = 0.0
    
    if req.price_history and len(req.price_history) > 10:
        s_price = pd.Series(req.price_history)
        # Apply strict FracDiff
        s_diff = apply_stationarity(pd.DataFrame({'close': s_price}))['frac_close']
        frac_price = float(s_diff.iloc[-1]) # Last value is current state
        
    if req.volume_history and len(req.volume_history) > 10:
        s_vol = pd.Series(req.volume_history)
        s_diff = apply_stationarity(pd.DataFrame({'volume': s_vol}))['frac_volume']
        frac_vol = float(s_diff.iloc[-1])

    # 2. Add as extra features if model expects them (12D)
    # This is a patched injection for the 12D upgrade
    # We append them to the standard vector
    
    # ... (existing prediction logic but now using  etc)
    # Since I cannot easily splice the exact middle of the file with 'cat', 
    # I am creating this stub to indicate where the logic goes.
    # I will perform a proper 'edit' next.
    return {"status": "stub"}
