import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

# Path hacking to import from local project
sys.path.append(os.getcwd())

from oracle.model import OracleTemporalTransformer
from oracle.trainer import OracleTrainer, MarketDataset
from torch.utils.data import DataLoader

def train_on_5090():
    print("🚀 Starting Remote Training on RTX 5090 (Blackwell/sm_120)...")
    
    # 1. Configuration
    # Use internal LAN IP for DB
    DB_URL = os.getenv("DATABASE_URL", "postgresql://eve_user:eve_password@192.168.14.105:5432/eve_market_data")
    
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"   Target Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
    
    if DEVICE == "cpu":
        print("⚠️  Warning: CUDA not found. Training will be slow.")

    # 2. Connect & data pull
    try:
        print("   Connecting to Database (192.168.14.105)...")
        engine = create_engine(DB_URL)
        
        # Pull last 30 days
        query = """
        SELECT timestamp, type_id, open, high, low, close, volume 
        FROM market_history 
        WHERE timestamp > NOW() - INTERVAL '30 days'
        ORDER BY timestamp ASC
        """
        df = pd.read_sql(query, engine)
        print(f"   Loaded {len(df)} rows from History.")
        
    except Exception as e:
        print(f"❌ DB Connection Error: {e}")
        return

    # 3. Preprocess with Rolling Z-Score Normalization (Simulated)
    # Ideally, we call 'analyst.feature_factory.normalize_rolling_z_score'
    # But for a script we can inline or import if PYTHONPATH allows
    
    print("   Applying Rolling Z-Score Normalization...")
    # Mocking single-item logic
    plex_id = 44992 
    df_item = df[df['type_id'] == plex_id].sort_values('timestamp').copy()
    
    if df_item.empty:
        print("   No data for PLEX. Using random mock data for demonstration.")
        data_np = np.random.randn(1000, 10)
    else:
        # Inline Normalization
        window = 24
        for col in ['open', 'high', 'low', 'close', 'volume']:
            roll_mean = df_item[col].rolling(window=window).mean()
            roll_std = df_item[col].rolling(window=window).std().replace(0, 1)
            df_item[f'{col}_z'] = (df_item[col] - roll_mean) / roll_std
            
        # Fill NaN from rolling w/ 0
        df_item.fillna(0, inplace=True)
            
        # Select Features
        z_cols = [f'{c}_z' for c in ['open', 'high', 'low', 'close', 'volume']]
        data_np = df_item[z_cols].values
        
        # Pad to 10 dims
        padding = np.zeros((len(data_np), 5))
        data_np = np.hstack([data_np, padding])

    # 4. Initialize Model
    model = OracleTemporalTransformer(input_dim=10)
    model.to(DEVICE)
    model.train()
    
    dataset = MarketDataset(data_np, seq_length=24*7) # 1 week window
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    trainer = OracleTrainer(model, learning_rate=1e-3)
    
    # 5. Train Loop
    epochs = 5
    for epoch in range(epochs):
        loss = trainer.train_epoch(loader)
        print(f"   Epoch {epoch+1}/{epochs} | Loss: {loss:.6f}")
        
    # 6. Save Artifacts
    print("💾 Saving Model...")
    torch.save(model.state_dict(), "oracle_v1.pth")
    print("   Saved oracle_v1.pth")
    
    # Export to ONNX
    # Important: Use dynamic axes or fixed batch size depending on deployment
    dummy_input = (
        torch.randn(1, 24*7, 10).to(DEVICE), 
        torch.randint(0, 24, (1, 24*7)).to(DEVICE),
        torch.randint(0, 7, (1, 24*7)).to(DEVICE)
    )
    torch.onnx.export(model, dummy_input, "oracle_v1.onnx", 
                      input_names=['market_features', 'hours', 'days'],
                      output_names=['predictions'],
                      verbose=False)
    print("   Exported oracle_v1.onnx.")

if __name__ == "__main__":
    train_on_5090()
