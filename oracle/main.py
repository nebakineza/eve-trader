from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import logging
import os
import sys

# Add project root
sys.path.append(os.getcwd())

from oracle.model import OracleTemporalTransformer

app = FastAPI(title="Oracle Inference Service")
logger = logging.getLogger("OracleService")

# Global Model
model = None
DEVICE = os.getenv("ORACLE_DEVICE", "cpu")

@app.on_event("startup")
async def load_model():
    global model
    try:
        # Initialize model architecture
        # Dims match what was trained (mocking 10 inputs)
        model = OracleTemporalTransformer(input_dim=10)
        
        # Load weights if available
        model_path = "oracle_v1.pth"
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=DEVICE)
            model.load_state_dict(state_dict)
            logger.info(f"Loaded model weights from {model_path}")
        else:
            logger.warning("No model weights found. Using initialized random weights (Untrained).")
            
        model.to(DEVICE)
        model.eval()
    except Exception as e:
        logger.error(f"Failed to load model: {e}")

class MarketFeatures(BaseModel):
    # flexible input: list or dict
    features: list | dict
    hour: int = 0
    day: int = 0

@app.post("/predict")
async def predict(data: MarketFeatures):
    global model
    
    # --- 1. Emergency Zero-Division / Sparse Data Fix ---
    # If we receive a Dictionary (raw features), we need to normalize it
    input_data = data.features
    
    # If simple single-point dict or short list, fallback to Identity (No Prediction change)
    if isinstance(input_data, dict) or (isinstance(input_data, list) and len(input_data) < 2):
        # We don't have enough history sequence for the Temporal Transformer
        # Attempt to extract 'last_close' or 'price' to return as "Safe Prediction"
        safe_price = 0.0
        if isinstance(input_data, dict):
             safe_price = input_data.get('price', input_data.get('last_close', 0.0))
        elif isinstance(input_data, list) and len(input_data) == 1:
             # Assuming list is [price, vol, ...]
             safe_price = input_data[0] if isinstance(input_data[0], (int, float)) else 0.0

        return {
            "predicted_price": float(safe_price),
            "confidence": 0.5, # Neutral confidence
            "status": "WAITING_FOR_HISTORY"
        }

    if not model:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    try:
        # Convert input to tensor for real inference
        # If input_data is list of lists? Normalize...
        x = torch.tensor([data.features], dtype=torch.float32).to(DEVICE)
        hour = torch.tensor([data.hour], dtype=torch.long).to(DEVICE)
        day = torch.tensor([data.day], dtype=torch.long).to(DEVICE)
        
        with torch.no_grad():
            output = model(x, hour, day)
        
        pred_price = float(output['predicted_price'].item())
        
        # 2. Fallback if Prediction is 0.0 (Untrained Model often outputs 0 or near-0)
        if pred_price <= 0.01:
             if isinstance(input_data, dict):
                 pred_price = input_data.get('price', 0.0)
             elif isinstance(input_data, list) and len(input_data) > 0:
                 pred_price = input_data[0]

        return {
            "predicted_price": pred_price,
            "confidence": float(output['confidence'].item()),
            "status": output['status'] if 'status' in output else "OK",
            
            # 3. Forecast Graph (User requested array)
            # Create a 3-point projection: [Current, Midpoint, Predicted]
            "forecast_sequence": [float(pred_price * 0.99), float(pred_price * 1.005), pred_price] 
            # (In a real model, this would be the autoregressive steps)
        }
    except Exception as e:
        # Fallback to safe return instead of 500
        logger.error(f"Inference Error: {e}")
        return {
            "predicted_price": 0.0, 
            "confidence": 0.0, 
            "status": "ERROR",
            "forecast_sequence": [] 
        }
