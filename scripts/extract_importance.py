import sys
import os
import torch
import numpy as np
import json

# Path hacking
sys.path.append(os.getcwd())
from oracle.model import OracleTemporalTransformer

def extract():
    try:
        model = OracleTemporalTransformer(input_dim=10)
        # Load absolute path if needed, assuming run from root with file in root or scripts
        path = "oracle_v1.pth"
        if not os.path.exists(path):
            # Try scripts dir
            path = "scripts/oracle_v1.pth"
        
        if not os.path.exists(path):
             # Try default save location
             path = "oracle_v1.pth"

        model.load_state_dict(torch.load(path, map_location='cpu'))
        
        # Extract weights from input embedding
        w = model.embedding.weight.detach().numpy()
        
        # Calculate importance as L1 norm of weights per feature
        importance = np.abs(w).sum(axis=0)
        
        # Normalize percent
        importance = importance / importance.sum()
        
        feature_names = ['open', 'high', 'low', 'close', 'volume', 'avg_price', 'max_velocity', 'median_spread', 'order_book_imbalance', 'padding']
        
        res = {k: float(v) for k, v in zip(feature_names, importance)}
        
        print("IMPORTANCE_JSON_START")
        print(json.dumps(res))
        print("IMPORTANCE_JSON_END")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    extract()
