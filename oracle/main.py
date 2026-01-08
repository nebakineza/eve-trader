from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Add project root
sys.path.append(os.getcwd())

from oracle.model import OracleTemporalTransformer

app = FastAPI(title="Oracle Inference Service")
logger = logging.getLogger("OracleService")

model: OracleTemporalTransformer | None = None
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
    return os.getenv("ORACLE_MODEL_PATH", "oracle_v1.pth")


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
    vec = [velocity, imbalance]

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
    global model, DEVICE
    try:
        device_str = _select_device(REQUESTED_DEVICE)
        DEVICE = torch.device(device_str)

        # Load weights if available
        model_path = _model_path()
        checkpoint_meta = None
        if os.path.exists(model_path):
            state_dict, checkpoint_meta = _load_checkpoint(model_path, map_location="cpu")
            input_dim = 10
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
            model = OracleTemporalTransformer(input_dim=10)
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
    }

class MarketFeatures(BaseModel):
    # flexible input: list or dict
    features: list | dict
    hour: int = 0
    day: int = 0

@app.post("/predict")
async def predict(data: MarketFeatures):
    global model

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

        pred_price = float(pred_tensor[0][0].item()) if pred_tensor is not None else 0.0
        confidence = float(conf_tensor[0][0].item()) if conf_tensor is not None else 0.0

        # Fallback if Prediction is unusable
        if pred_price <= 0.0:
            if isinstance(data.features, dict):
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
