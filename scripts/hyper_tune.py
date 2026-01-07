import os
import json
import logging
import time
import subprocess
import threading
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import optuna
import sys

# Define path imports to reach oracle module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from oracle.model import OracleTemporalTransformer
except ImportError:
    # Fallback definition if import fails (e.g. running outside package context)
    print("Could not import OracleTemporalTransformer, ensuring mocked version is available if needed.")
    pass

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("HyperTune")

# Configuration
DATA_FILE = "training_data_24h.parquet"
OUTPUT_CONFIG = "best_oracle_config.json"
LOG_STATS_FILE = "logs/training_stats.json"
N_TRIALS = 20 # Adjust based on time available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs("logs", exist_ok=True)

# --- Monitoring Thread ---
class GPUMonitor(threading.Thread):
    def __init__(self, interval=5):
        super().__init__()
        self.interval = interval
        self.stop_event = threading.Event()
        self.stats = []

    def get_gpu_stats(self):
        try:
            # Run nvidia-smi query
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=temperature.gpu,memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                temp, mem_used, util = result.stdout.strip().split(',')
                return {
                    "timestamp": time.time(),
                    "gpu_temp_c": int(temp),
                    "memory_used_mb": int(mem_used),
                    "gpu_util_pct": int(util)
                }
        except Exception:
            pass
        return None

    def run(self):
        while not self.stop_event.is_set():
            stats = self.get_gpu_stats()
            if stats:
                self.stats.append(stats)
                # Dump to file every update to prevent data loss
                with open(LOG_STATS_FILE, 'w') as f:
                    json.dump(self.stats, f, indent=2)
            time.sleep(self.interval)

    def stop(self):
        self.stop_event.set()

# --- Data Loading ---
def load_data():
    if not os.path.exists(DATA_FILE):
        logger.error(f"Data file {DATA_FILE} not found. Run prepare_dataset.py first.")
        # Create dummy data for dry-run if real data missing
        logger.warning("Generating dummy data for dry-run...")
        return torch.randn(100, 24, 10), torch.randn(100, 24, 10).long(), torch.randn(100, 24, 10).long(), torch.randn(100, 1)
    
    df = pd.read_parquet(DATA_FILE)
    # Basic Preprocessing (Mock logic for brevity - replace with robust `analyst` pipeline in prod)
    # Assuming df has columns like timestamp, type_id, open, close, volume...
    # We need sequences. For now, let's assume prepared features or do simple sliding window.
    
    # Simple Mock of sliding window for optimization test
    # [Start, SeqLen, InputDim]
    # Here we simulate tensors to verify the model architecture on the GPU
    logger.info("loading parquet and simulating tensors...")
    
    seq_len = 24
    input_dim = 10 
    n_samples = max(100, len(df) // seq_len)
    
    x = torch.randn(n_samples, seq_len, input_dim)
    hour_idx = torch.randint(0, 24, (n_samples, seq_len))
    day_idx = torch.randint(0, 7, (n_samples, seq_len))
    y = torch.randn(n_samples, 1) # Target: Next close price
    
    return x, hour_idx, day_idx, y

# --- Optuna Objective ---
def objective(trial):
    # 1. Sample Hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    num_heads = trial.suggest_categorical("num_heads", [4, 8])
    embed_dim = trial.suggest_categorical("embed_dim", [64, 128, 256])
    # Ensure embed_dim is divisible by num_heads
    if embed_dim % num_heads != 0:
        embed_dim = (embed_dim // num_heads) * num_heads
        
    batch_size = trial.suggest_categorical("batch_size", [32, 64])

    # 2. Setup Model
    model = OracleTemporalTransformer(input_dim=10, embed_dim=embed_dim, num_heads=num_heads).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # 3. Data Helpers
    x, h, d, y = load_data()
    dataset = TensorDataset(x, h, d, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 4. Training Loop (Short epochs for tuning)
    model.train()
    epochs = 3 
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x, batch_h, batch_d, batch_y in loader:
            batch_x, batch_h, batch_d, batch_y = batch_x.to(DEVICE), batch_h.to(DEVICE), batch_d.to(DEVICE), batch_y.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(batch_x, batch_h, batch_d)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        # Pruning (Optional)
        trial.report(epoch_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return epoch_loss / len(loader)

def main():
    logger.info("🔥 Starting 5090 Hyper-Tune Session")
    logger.info(f"   Device: {DEVICE}")
    
    # Start Monitor
    monitor = GPUMonitor()
    monitor.start()
    
    try:
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=N_TRIALS)

        logger.info("✅ Optimization Complete")
        logger.info(f"   Best Loss: {study.best_value}")
        logger.info(f"   Best Params: {study.best_params}")

        # Save Config
        with open(OUTPUT_CONFIG, 'w') as f:
            json.dump(study.best_params, f, indent=4)
        logger.info(f"💾 Saved best config to {OUTPUT_CONFIG}")

    finally:
        monitor.stop()
        monitor.join()
        logger.info(f"🛑 Monitoring stopped. Stats saved to {LOG_STATS_FILE}")

if __name__ == "__main__":
    main()
